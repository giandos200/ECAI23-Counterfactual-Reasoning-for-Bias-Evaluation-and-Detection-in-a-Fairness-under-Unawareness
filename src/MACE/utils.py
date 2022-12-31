from src.MACE.modelConversion import *
from pysmt.shortcuts import *
from pysmt.typing import *
from pprint import pprint

def getDistanceFormula(model_symbols, dataset_obj, factual_sample, norm_type, approach_string, norm_threshold):

  if 'mace' in approach_string.lower():
    variable_to_compute_distance_on = 'counterfactual'
  elif 'mint' in approach_string.lower():
    variable_to_compute_distance_on = 'interventional'


  def getAbsoluteDifference(symbol_1, symbol_2):
    return Ite(
      GE(Minus(ToReal(symbol_1), ToReal(symbol_2)), Real(0)),
      Minus(ToReal(symbol_1), ToReal(symbol_2)),
      Minus(ToReal(symbol_2), ToReal(symbol_1))
    )

  # normalize this feature's distance by dividing the absolute difference by the
  # range of the variable (only applies for non-hot variables)
  normalized_absolute_distances = []
  normalized_squared_distances = []

  # IMPORTANT CHANGE IN CODE (Feb 04, 2020): prior to today, actionable/mutable
  # features overlapped. Now that we have introduced 3 types of variables
  # (actionable and mutable, non-actionable but mutable, immutable and non-actionable),
  # we must re-write the distance function to depent on all mutable features only,
  # while before we wrote distance as a function over actionable/mutable features.

  mutable_attributes = dataset_obj.getMutableAttributeNames('kurz')
  one_hot_attributes = dataset_obj.getOneHotAttributesNames('kurz')
  non_hot_attributes = dataset_obj.getNonHotAttributesNames('kurz')

  # 1. mutable & non-hot
  for attr_name_kurz in np.intersect1d(mutable_attributes, non_hot_attributes):
    normalized_absolute_distances.append(
      Div(
        ToReal(
          getAbsoluteDifference(
            model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol'],
            factual_sample[attr_name_kurz]
          )
        ),
        # Real(1)
        ToReal(
          model_symbols[variable_to_compute_distance_on][attr_name_kurz]['upper_bound'] -
          model_symbols[variable_to_compute_distance_on][attr_name_kurz]['lower_bound']
        )
      )
    )
    normalized_squared_distances.append(
      Pow(
        Div(
          ToReal(
            Minus(
              model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol'],
              factual_sample[attr_name_kurz]
            )
          ),
          # Real(1)
          ToReal(
            model_symbols[variable_to_compute_distance_on][attr_name_kurz]['upper_bound'] -
            model_symbols[variable_to_compute_distance_on][attr_name_kurz]['lower_bound']
          )
        ),
        Real(2)
      )
    )

  # 2. mutable & integer-based & one-hot
  already_considered = []
  for attr_name_kurz in np.intersect1d(mutable_attributes, one_hot_attributes):
    if attr_name_kurz not in already_considered:
      siblings_kurz = dataset_obj.getSiblingsFor(attr_name_kurz)
      if 'cat' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
        # this can also be implemented as the abs value of sum of a difference
        # in each attribute, divided by 2
        normalized_absolute_distances.append(
          Ite(
            And([
              EqualsOrIff(
                model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol'],
                factual_sample[attr_name_kurz]
              )
              for attr_name_kurz in siblings_kurz
            ]),
            Real(0),
            Real(1)
          )
        )
        # TODO: What about this? might be cheaper than Ite.
        # normalized_absolute_distances.append(
        #   Minus(
        #     Real(1),
        #     ToReal(And([
        #       EqualsOrIff(
        #         model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol'],
        #         factual_sample[attr_name_kurz]
        #       )
        #       for attr_name_kurz in siblings_kurz
        #     ]))
        #   )
        # )

        # As the distance is 0 or 1 in this case, the 2nd power is same as itself
        normalized_squared_distances.append(normalized_absolute_distances[-1])

      elif 'ord' in dataset_obj.attributes_kurz[attr_name_kurz].attr_type:
        normalized_absolute_distances.append(
          Div(
            ToReal(
              getAbsoluteDifference(
                Plus([
                  model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol']
                  for attr_name_kurz in siblings_kurz
                ]),
                Plus([
                  factual_sample[attr_name_kurz]
                  for attr_name_kurz in siblings_kurz
                ]),
              )
            ),
            Real(len(siblings_kurz))
          )
        )
        # this can also be implemented as below:
        # normalized_absolute_distances.append(
        #   Div(
        #     ToReal(
        #       Plus([
        #         Ite(
        #           EqualsOrIff(
        #             model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol'],
        #             factual_sample[attr_name_kurz]
        #           ),
        #           Real(0),
        #           Real(1)
        #         )
        #         for attr_name_kurz in siblings_kurz
        #       ])
        #     ),
        #     Real(len(siblings_kurz))
        #   )
        # )
        normalized_squared_distances.append(
          Pow(
            Div(
              ToReal(
                Minus(
                  Plus([
                    model_symbols[variable_to_compute_distance_on][attr_name_kurz]['symbol']
                    for attr_name_kurz in siblings_kurz
                  ]),
                  Plus([
                    factual_sample[attr_name_kurz]
                    for attr_name_kurz in siblings_kurz
                  ]),
                )
              ),
              Real(len(siblings_kurz))
            ),
            Real(2)
          )
        )
      else:
        raise Exception(f'{attr_name_kurz} must include either `cat` or `ord`.')
      already_considered.extend(siblings_kurz)

  # # 3. compute normalized squared distances
  # # pysmt.exceptions.SolverReturnedUnknownResultError
  # normalized_squared_distances = [
  #   # Times(distance, distance)
  #   Pow(distance, Int(2))
  #   for distance in normalized_absolute_distances
  # ]
  # # TODO: deprecate?
  # # def getSquaredifference(symbol_1, symbol_2):
  # #   return Times(
  # #     ToReal(Minus(ToReal(symbol_1), ToReal(symbol_2))),
  # #     ToReal(Minus(ToReal(symbol_2), ToReal(symbol_1)))
  # #   )


  # 4. sum up over everything allowed...
  # We use 1 / len(normalized_absolute_distances) below because we only consider
  # those attributes that are mutable, and for each sibling-group (ord, cat)
  # we only consider 1 entry in the normalized_absolute_distances
  if norm_type == 'zero_norm':
    distance_formula = LE(
      Times(
        Real(1 / len(normalized_absolute_distances)),
        Plus([
          Ite(
            Equals(elem, Real(0)),
            Real(0),
            Real(1)
          ) for elem in normalized_absolute_distances
        ])
      ),
      Real(norm_threshold)
    )
  elif norm_type == 'one_norm':
    distance_formula = LE(
      Times(
        Real(1 / len(normalized_absolute_distances)),
        ToReal(Plus(normalized_absolute_distances))
      ),
      Real(norm_threshold)
    )
  elif norm_type == 'two_norm':
    distance_formula = LE(
      Times(
        Real(1 / len(normalized_squared_distances)),
        ToReal(Plus(normalized_squared_distances))
      ),
      Pow(
        Real(norm_threshold),
        Real(2)
      )
    )
  elif norm_type == 'infty_norm':
    distance_formula = LE(
      Times(
        Real(1 / len(normalized_absolute_distances)),
        ToReal(Max(normalized_absolute_distances))
      ),
      Real(norm_threshold)
    )
  else:
    raise Exception(f'{norm_type} not recognized as a valid `norm_type`.')

  return distance_formula

def getPySMTSampleFromDictSample(dict_sample, dataset_obj):
  pysmt_sample = {}
  for attr_name_kurz in dataset_obj.getInputOutputAttributeNames('kurz'):
    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
    if attr_name_kurz not in dataset_obj.getInputAttributeNames('kurz'):
      pysmt_sample[attr_name_kurz] = Bool(dict_sample[attr_name_kurz])
    elif attr_obj.attr_type == 'numeric-real':
      pysmt_sample[attr_name_kurz] = Real(float(dict_sample[attr_name_kurz]))
    else: # refer to loadData.VALID_ATTRIBUTE_TYPES
      pysmt_sample[attr_name_kurz] = Int(int(dict_sample[attr_name_kurz]))
  return pysmt_sample

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def getModelFormula(model_symbols, model_trained):
  if isinstance(model_trained, DecisionTreeClassifier):
    model2formula = lambda a,b : tree2formula(a,b)
  elif isinstance(model_trained, LogisticRegression):
    model2formula = lambda a,b : lr2formula(a,b)
  elif isinstance(model_trained, RandomForestClassifier):
    model2formula = lambda a,b : forest2formula(a,b)
  elif isinstance(model_trained, MLPClassifier):
    model2formula = lambda a,b : mlp2formula(a,b)

  return model2formula(
    model_trained,
    model_symbols)

def getCounterfactualFormula(model_symbols, factual_sample):
  return EqualsOrIff(
    model_symbols['output']['y']['symbol'],
    Not(factual_sample['y'])
  ) # meaning we want the decision to be flipped.

def getDictSampleFromPySMTSample(pysmt_sample, dataset_obj):
  dict_sample = {}
  for attr_name_kurz in dataset_obj.getInputOutputAttributeNames('kurz'):
    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
    try:
      if attr_name_kurz not in dataset_obj.getInputAttributeNames('kurz'):
        dict_sample[attr_name_kurz] = bool(str(pysmt_sample[attr_name_kurz]) == 'True')
      elif attr_obj.attr_type == 'numeric-real':
        dict_sample[attr_name_kurz] = float(eval(str(pysmt_sample[attr_name_kurz])))
      else: # refer to loadData.VALID_ATTRIBUTE_TYPES
        dict_sample[attr_name_kurz] = int(str(pysmt_sample[attr_name_kurz]))
    except:
      raise Exception(f'Failed to read value from pysmt sample. Debug me manually.')
  return dict_sample

def getPlausibilityFormula(model_symbols, dataset_obj, factual_sample, approach_string):
  # here is where the user specifies the following:
  #  1. data range plausibility
  #  2. data type plausibility
  #  3. actionability + mutability
  #  4. causal consistency

  ##############################################################################
  ## 1. data range plausibility
  ##############################################################################
  range_plausibility_counterfactual = And([
    And(
      GE(model_symbols['counterfactual'][attr_name_kurz]['symbol'], model_symbols['counterfactual'][attr_name_kurz]['lower_bound']),
      LE(model_symbols['counterfactual'][attr_name_kurz]['symbol'], model_symbols['counterfactual'][attr_name_kurz]['upper_bound'])
    )
    for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz')
  ])
  range_plausibility_interventional = And([
    And(
      GE(model_symbols['interventional'][attr_name_kurz]['symbol'], model_symbols['interventional'][attr_name_kurz]['lower_bound']),
      LE(model_symbols['interventional'][attr_name_kurz]['symbol'], model_symbols['interventional'][attr_name_kurz]['upper_bound'])
    )
    for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz')
  ])

  # IMPORTANT: a weird behavior of print(get_model(formula)) is that if there is
  #            a variable that is defined as a symbol, but is not constrained in
  #            the formula, then print(.) will not print the "verifying" value of
  #            that variable (as it can be anything). Therefore, we always use
  #            range plausibility constraints on ALL variables (including the
  #            interventional variables, even though they are only used for MINT
  #            and not MACE). TODO: find alternative method to print(model).
  range_plausibility = And([range_plausibility_counterfactual, range_plausibility_interventional])


  ##############################################################################
  ## 2. data type plausibility
  ##############################################################################
  onehot_categorical_plausibility = TRUE() # plausibility of categorical (sum = 1)
  onehot_ordinal_plausibility = TRUE() # plausibility ordinal (x3 >= x2 & x2 >= x1)

  if dataset_obj.is_one_hot:

    dict_of_siblings_kurz = dataset_obj.getDictOfSiblings('kurz')

    for parent_name_kurz in dict_of_siblings_kurz['cat'].keys():

      onehot_categorical_plausibility = And(
        onehot_categorical_plausibility,
        And(
          EqualsOrIff(
            Plus([
              model_symbols['counterfactual'][attr_name_kurz]['symbol']
              for attr_name_kurz in dict_of_siblings_kurz['cat'][parent_name_kurz]
            ]),
            Int(1)
          )
        ),
        And(
          EqualsOrIff(
            Plus([
              model_symbols['interventional'][attr_name_kurz]['symbol']
              for attr_name_kurz in dict_of_siblings_kurz['cat'][parent_name_kurz]
            ]),
            Int(1)
          )
        )
      )

    for parent_name_kurz in dict_of_siblings_kurz['ord'].keys():

      onehot_ordinal_plausibility = And(
        onehot_ordinal_plausibility,
        And([
          GE(
            ToReal(model_symbols['counterfactual'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx]]['symbol']),
            ToReal(model_symbols['counterfactual'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx + 1]]['symbol'])
          )
          for symbol_idx in range(len(dict_of_siblings_kurz['ord'][parent_name_kurz]) - 1) # already sorted
        ]),
        And([
          GE(
            ToReal(model_symbols['interventional'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx]]['symbol']),
            ToReal(model_symbols['interventional'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx + 1]]['symbol'])
          )
          for symbol_idx in range(len(dict_of_siblings_kurz['ord'][parent_name_kurz]) - 1) # already sorted
        ])
      )

      # # Also implemented as the following logic, stating that
      # # if x_j == 1, all x_i == 1 for i < j
      # # Friendly reminder that for ordinal variables, x_0 is always 1
      # onehot_ordinal_plausibility = And([
      #   Ite(
      #     EqualsOrIff(
      #       ToReal(model_symbols['counterfactual'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx_ahead]]['symbol']),
      #       Real(1)
      #     ),
      #     And([
      #       EqualsOrIff(
      #         ToReal(model_symbols['counterfactual'][dict_of_siblings_kurz['ord'][parent_name_kurz][symbol_idx_behind]]['symbol']),
      #         Real(1)
      #       )
      #       for symbol_idx_behind in range(symbol_idx_ahead)
      #     ]),
      #     TRUE()
      #   )
      #   for symbol_idx_ahead in range(1, len(dict_of_siblings_kurz['ord'][parent_name_kurz])) # already sorted
      # ])


  ##############################################################################
  ## 3. actionability + mutability
  #    a) actionable and mutable: both interventional and counterfactual value can change
  #    b) non-actionable but mutable: interventional value cannot change, but counterfactual value can
  #    c) immutable and non-actionable: neither interventional nor counterfactual value can change
  ##############################################################################
  actionability_mutability_plausibility = []
  for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):
    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]

    # a) actionable and mutable: both interventional and counterfactual value can change
    if attr_obj.mutability == True and attr_obj.actionability != 'none':

      if attr_obj.actionability == 'same-or-increase':
        actionability_mutability_plausibility.append(GE(
          model_symbols['counterfactual'][attr_name_kurz]['symbol'],
          factual_sample[attr_name_kurz]
        ))
        actionability_mutability_plausibility.append(GE(
          model_symbols['interventional'][attr_name_kurz]['symbol'],
          factual_sample[attr_name_kurz]
        ))
      elif attr_obj.actionability == 'same-or-decrease':
        actionability_mutability_plausibility.append(LE(
          model_symbols['counterfactual'][attr_name_kurz]['symbol'],
          factual_sample[attr_name_kurz]
        ))
        actionability_mutability_plausibility.append(LE(
          model_symbols['interventional'][attr_name_kurz]['symbol'],
          factual_sample[attr_name_kurz]
        ))
      elif attr_obj.actionability == 'any':
        continue

    # b) mutable but non-actionable: interventional value cannot change, but counterfactual value can
    elif attr_obj.mutability == True and attr_obj.actionability == 'none':

      # IMPORTANT: when we are optimizing for nearest CFE, we completely ignore
      #            the interventional symbols, even though they are defined. In
      #            such a world, we also don't have any assumptions about the
      #            causal structure, and therefore, causal_consistency = TRUE()
      #            later in the code. Therefore, a `mutable but actionable` var
      #            (i.e., a variable that can change due to it's ancerstors) does
      #            not even exist. Thus, non-actionable variables are supported
      #            by restricing the counterfactual symbols.
      # TODO: perhaps a better way to structure this code is to completely get
      #       rid of interventional symbols when calling genSATExp.py with MACE.
      if 'mace' in approach_string.lower():
        actionability_mutability_plausibility.append(EqualsOrIff(
          model_symbols['counterfactual'][attr_name_kurz]['symbol'],
          factual_sample[attr_name_kurz]
        ))
      elif 'mint' in approach_string.lower():
        actionability_mutability_plausibility.append(EqualsOrIff(
          model_symbols['interventional'][attr_name_kurz]['symbol'],
          factual_sample[attr_name_kurz]
        ))

    # c) immutable and non-actionable: neither interventional nor counterfactual value can change
    else:

      actionability_mutability_plausibility.append(EqualsOrIff(
        model_symbols['counterfactual'][attr_name_kurz]['symbol'],
        factual_sample[attr_name_kurz]
      ))
      actionability_mutability_plausibility.append(EqualsOrIff(
        model_symbols['interventional'][attr_name_kurz]['symbol'],
        factual_sample[attr_name_kurz]
      ))

  actionability_mutability_plausibility = And(actionability_mutability_plausibility)


  ##############################################################################
  ## 4. causal consistency
  ##############################################################################
  if 'mace' in approach_string.lower():
    causal_consistency = TRUE()
  elif 'mint' in approach_string.lower():
    causal_consistency = getCausalConsistencyConstraints(model_symbols, dataset_obj, factual_sample)


  return And(
    range_plausibility,
    onehot_categorical_plausibility,
    onehot_ordinal_plausibility,
    actionability_mutability_plausibility,
    causal_consistency
  )

def getCausalConsistencyConstraints(model_symbols, dataset_obj, factual_sample):
  if dataset_obj.dataset_name == 'german':
    return getGermanCausalConsistencyConstraints(model_symbols, factual_sample)
  elif dataset_obj.dataset_name == 'random':
    return getRandomCausalConsistencyConstraints(model_symbols, factual_sample)
  elif dataset_obj.dataset_name == 'mortgage':
    return getMortgageCausalConsistencyConstraints(model_symbols, factual_sample)
  elif dataset_obj.dataset_name == 'twomoon':
    return getTwoMoonCausalConsistencyConstraints(model_symbols, factual_sample)
  elif dataset_obj.dataset_name == 'test':
    return getTestCausalConsistencyConstraints(model_symbols, factual_sample)