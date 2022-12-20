import pandas as pd
import numpy as np
import copy

from loadData import Dataset

VALID_ATTRIBUTE_DATA_TYPES = { \
  'numeric-int', \
  'numeric-real', \
  'binary', \
  'categorical', \
  'sub-categorical', \
  'ordinal', \
  'sub-ordinal'}
VALID_ATTRIBUTE_NODE_TYPES = { \
  'meta', \
  'input', \
  'output'}
VALID_ACTIONABILITY_TYPES = { \
  'none', \
  'any', \
  'same-or-increase', \
  'same-or-decrease'}
VALID_MUTABILITY_TYPES = { \
  True, \
  False}

def getOneHotEquivalent(data_frame_non_hot, attributes_non_hot):

  # TODO: see how we can switch between feature_names = col names for kurz and long (also maybe ordered)

  data_frame = copy.deepcopy(data_frame_non_hot)
  attributes = copy.deepcopy(attributes_non_hot)

  def setOneHotValue(val):
    return np.append(np.append(
      np.zeros(val - 1),
      np.ones(1)),
      np.zeros(num_unique_values - val)
    )

  def setThermoValue(val):
    return np.append(
      np.ones(val),
      np.zeros(num_unique_values - val)
    )

  for col_name in data_frame.columns.values:

    if attributes[col_name].attr_type not in {'categorical', 'ordinal'}:
      continue

    old_col_name_long = col_name
    new_col_names_long = []
    new_col_names_kurz = []

    old_attr_name_long = attributes[old_col_name_long].attr_name_long
    old_attr_name_kurz = attributes[old_col_name_long].attr_name_kurz
    old_attr_type = attributes[old_col_name_long].attr_type
    old_node_type = attributes[old_col_name_long].node_type
    old_actionability = attributes[old_col_name_long].actionability
    old_mutability = attributes[old_col_name_long].mutability
    old_lower_bound = attributes[old_col_name_long].lower_bound
    old_upper_bound = attributes[old_col_name_long].upper_bound

    num_unique_values = int(old_upper_bound - old_lower_bound + 1)

    assert old_col_name_long == old_attr_name_long

    new_attr_type = 'sub-' + old_attr_type
    new_node_type = old_node_type
    new_actionability = old_actionability
    new_mutability = old_mutability
    new_parent_name_long = old_attr_name_long
    new_parent_name_kurz = old_attr_name_kurz


    if attributes[col_name].attr_type == 'categorical': # do not do this for 'binary'!

      new_col_names_long = [f'{old_attr_name_long}_cat_{i}' for i in range(num_unique_values)]
      new_col_names_kurz = [f'{old_attr_name_kurz}_cat_{i}' for i in range(num_unique_values)]
      print(f'Replacing column {col_name} with {{{", ".join(new_col_names_long)}}}')
      tmp = np.array(list(map(setOneHotValue, list(data_frame[col_name].astype(int).values))))
      data_frame_dummies = pd.DataFrame(data=tmp, columns=new_col_names_long)

    elif attributes[col_name].attr_type == 'ordinal':

      new_col_names_long = [f'{old_attr_name_long}_ord_{i}' for i in range(num_unique_values)]
      new_col_names_kurz = [f'{old_attr_name_kurz}_ord_{i}' for i in range(num_unique_values)]
      print(f'Replacing column {col_name} with {{{", ".join(new_col_names_long)}}}')
      tmp = np.array(list(map(setThermoValue, list(data_frame[col_name].astype(int).values))))
      data_frame_dummies = pd.DataFrame(data=tmp, columns=new_col_names_long)

    # Update data_frame
    data_frame = pd.concat([data_frame.drop(columns = old_col_name_long), data_frame_dummies], axis=1)

    # Update attributes
    del attributes[old_col_name_long]
    for col_idx in range(len(new_col_names_long)):
      new_col_name_long = new_col_names_long[col_idx]
      new_col_name_kurz = new_col_names_kurz[col_idx]
      attributes[new_col_name_long] = DatasetAttribute(
        attr_name_long = new_col_name_long,
        attr_name_kurz = new_col_name_kurz,
        attr_type = new_attr_type,
        node_type = new_node_type,
        actionability = new_actionability,
        mutability = new_mutability,
        parent_name_long = new_parent_name_long,
        parent_name_kurz = new_parent_name_kurz,
        lower_bound = data_frame[new_col_name_long].min(),
        upper_bound = data_frame[new_col_name_long].max())

  return data_frame, attributes

class DatasetAttribute(object):

  def __init__(
    self,
    attr_name_long,
    attr_name_kurz,
    attr_type,
    node_type,
    actionability,
    mutability,
    parent_name_long,
    parent_name_kurz,
    lower_bound,
    upper_bound):

    if attr_type not in VALID_ATTRIBUTE_DATA_TYPES:
      raise Exception("`attr_type` must be one of %r." % VALID_ATTRIBUTE_DATA_TYPES)

    if node_type not in VALID_ATTRIBUTE_NODE_TYPES:
      raise Exception("`node_type` must be one of %r." % VALID_ATTRIBUTE_NODE_TYPES)

    if actionability not in VALID_ACTIONABILITY_TYPES:
      raise Exception("`actionability` must be one of %r." % VALID_ACTIONABILITY_TYPES)

    if mutability not in VALID_MUTABILITY_TYPES:
      raise Exception("`mutability` must be one of %r." % VALID_MUTABILITY_TYPES)

    if lower_bound > upper_bound:
      raise Exception("`lower_bound` must be <= `upper_bound`")

    if attr_type in {'sub-categorical', 'sub-ordinal'}:
      assert parent_name_long != -1, 'Parent ID set for non-hot attribute.'
      assert parent_name_kurz != -1, 'Parent ID set for non-hot attribute.'
      if attr_type == 'sub-categorical':
        assert lower_bound == 0
        assert upper_bound == 1
      if attr_type == 'sub-ordinal':
        # the first elem in thermometer is always on, but the rest may be on or off
        assert lower_bound == 0 or lower_bound == 1
        assert upper_bound == 1
    else:
      assert parent_name_long == -1, 'Parent ID set for non-hot attribute.'
      assert parent_name_kurz == -1, 'Parent ID set for non-hot attribute.'

    if attr_type in {'categorical', 'ordinal'}:
      assert lower_bound == 1 # setOneHotValue & setThermoValue assume this in their logic

    if attr_type in {'binary', 'categorical', 'sub-categorical'}: # not 'ordinal' or 'sub-ordinal'
      # IMPORTANT: surprisingly, it is OK if all sub-ordinal variables share actionability
      #            think about it, if each sub- variable is same-or-increase, along with
      #            the constraints that x0_ord_1 >= x0_ord_2, all variables can only stay
      #            the same or increase. It works :)
      assert actionability in {'none', 'any'}, f"{attr_type}'s actionability can only be in {'none', 'any'}, not `{actionability}`."

    if node_type != 'input':
      assert actionability == 'none', f'{node_type} attribute is not actionable.'
      assert mutability == False, f'{node_type} attribute is not mutable.'

    # We have introduced 3 types of variables: (actionable and mutable, non-actionable but mutable, immutable and non-actionable)
    if actionability != 'none':
      assert mutability == True
    # TODO: above/below seem contradictory... (2020.04.14)
    if mutability == False:
      assert actionability == 'none'

    if parent_name_long == -1 or parent_name_kurz == -1:
      assert parent_name_long == parent_name_kurz == -1

    self.attr_name_long = attr_name_long
    self.attr_name_kurz = attr_name_kurz
    self.attr_type = attr_type
    self.node_type = node_type
    self.actionability = actionability
    self.mutability = mutability
    self.parent_name_long = parent_name_long
    self.parent_name_kurz = parent_name_kurz
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound

class MACE:
    def __init__(self, dataset, model, outcome, numvars, catvars, eps: 1e-3, norm_type='one_norm', random_seed=42):
        self.numvars = numvars
        self.catvars = catvars
        self.eps = eps
        self.norm = norm_type
        self.SEED = random_seed
        if len(catvars) >0 :
            self.one_hot = True
        input_cols, output_col = numvars+catvars, outcome.columns.tolist()
        attributes_non_hot = {}
        attributes_non_hot[output_col] = DatasetAttribute(
          attr_name_long=output_col,
          attr_name_kurz='y',
          attr_type='binary',
          node_type='output',
          actionability='none',
          mutability=False,
          parent_name_long=-1,
          parent_name_kurz=-1,
          lower_bound=outcome[output_col].min(),
          upper_bound=outcome[output_col].max())
        for col_idx, col_name in enumerate(input_cols):
          if col_name in numvars:
            attr_type = 'numeric-real'
            actionability = 'any'
            mutability = True
          elif col_name in catvars:
            attr_type = 'categorical'
            actionability = 'any'
            mutability = True
          else:
            raise ValueError
          attributes_non_hot[col_name] = DatasetAttribute(
            attr_name_long=col_name,
            attr_name_kurz=f'x{col_idx}',
            attr_type=attr_type,
            node_type='input',
            actionability=actionability,
            mutability=mutability,
            parent_name_long=-1,
            parent_name_kurz=-1,
            lower_bound=dataset[col_name].min(),
            upper_bound=dataset[col_name].max())
        if self.one_hot:
          data_frame, attributes = getOneHotEquivalent(dataset, attributes_non_hot)
        else:
          data_frame, attributes = dataset, attributes_non_hot

        self.dataset_obj = Dataset(data_frame, attributes, return_one_hot=self.one_hot, dataset_name=None)
        self.model = model
        all_prediction = model.predict(dataset)
        self.positive_pred = all_prediction[all_prediction==1]
        self.negative_pred = all_prediction[all_prediction==0]


    def generate_counterfactuals(self, sample, total_CFs, desired_class, verbose):
      candidate_factuals = sample.T.to_dict()
      if desired_class == 0:
        observable_data = self.negative_pred.T.to_dict()
      else:
        observable_data = self.positive_pred.T.to_dict()
      for factual_sample_index, factual_sample in candidate_factuals.items():
        factual_sample['y'] = bool(self.model.predict(sample[factual_sample_index]))

