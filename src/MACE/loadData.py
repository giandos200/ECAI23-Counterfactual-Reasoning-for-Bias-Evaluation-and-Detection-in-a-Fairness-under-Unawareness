import copy
import numpy as np
import pandas as pd
# from debug import ipsh
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

class Dataset(object):

  # TODO: getOneHotEquivalent can be a class method, and this object can store
  # both one-hot and non-hot versions!

  def __init__(self, data_frame, attributes, is_one_hot, dataset_name = None):

    self.dataset_name = dataset_name

    self.is_one_hot = is_one_hot

    attributes_long = attributes
    data_frame_long = data_frame
    self.data_frame_long = data_frame_long # i.e., data_frame is indexed by attr_name_long
    self.attributes_long = attributes_long # i.e., attributes is indexed by attr_name_long

    attributes_kurz = dict((attributes[key].attr_name_kurz, value) for (key, value) in attributes_long.items())
    data_frame_kurz = copy.deepcopy(data_frame_long)
    data_frame_kurz.columns = self.getAllAttributeNames('kurz')
    self.data_frame_kurz = data_frame_kurz # i.e., data_frame is indexed by attr_name_kurz
    self.attributes_kurz = attributes_kurz # i.e., attributes is indexed by attr_name_kurz

    # assert that data_frame and attributes match on variable names (long)
    assert len(np.setdiff1d(
      data_frame.columns.values,
      np.array(self.getAllAttributeNames('long'))
    )) == 0

    # assert attribute type matches what is in the data frame
    for attr_name in np.setdiff1d(
      self.getInputAttributeNames('long'),
      self.getRealBasedAttributeNames('long'),
    ):
      unique_values = np.unique(data_frame_long[attr_name].to_numpy())
      # all non-numerical-real values should be integer or {0,1}
      for value in unique_values:
        assert value == np.floor(value)
      if is_one_hot and attributes_long[attr_name].attr_type != 'numeric-int': # binary, sub-categorical, sub-ordinal
        try:
          assert \
            np.array_equal(unique_values, [0,1]) or \
            np.array_equal(unique_values, [1,2]) or \
            np.array_equal(unique_values, [1]) # the first sub-ordinal attribute is always 1
            # race (binary) in compass is encoded as {1,2}
        except:
          ipsh()

    # # assert attributes and is_one_hot agree on one-hot-ness (i.e., if is_one_hot,
    # # then at least one attribute should be encoded as one-hot (w/ parent reference))
    # tmp_is_one_hot = False
    # for attr_name in attributes.keys():
    #   attr_obj = attributes[attr_name]
    #   # this simply checks to make sure that at least one elem is one-hot encoded
    #   if attr_obj.parent_name_long != -1 or attr_obj.parent_name_kurz != -1:
    #     tmp_is_one_hot = True
    # # TODO: assert only if there is a cat/ord variable!
    # assert is_one_hot == tmp_is_one_hot, "Dataset object and actual attributes don't agree on one-hot"

    self.assertSiblingsShareAttributes('long')
    self.assertSiblingsShareAttributes('kurz')

  def getAttributeNames(self, allowed_node_types, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check attr_name
    for attr_name in self.attributes_long.keys():
      attr_obj = self.attributes_long[attr_name]
      if attr_obj.node_type not in allowed_node_types:
        continue
      if long_or_kurz == 'long':
        names.append(attr_obj.attr_name_long)
      elif long_or_kurz == 'kurz':
        names.append(attr_obj.attr_name_kurz)
      else:
        raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
    return np.array(names)

  def getAllAttributeNames(self, long_or_kurz = 'kurz'):
    return self.getAttributeNames({'meta', 'input', 'output'}, long_or_kurz)

  def getInputOutputAttributeNames(self, long_or_kurz = 'kurz'):
    return self.getAttributeNames({'input', 'output'}, long_or_kurz)

  def getMetaInputAttributeNames(self, long_or_kurz = 'kurz'):
    return self.getAttributeNames({'meta', 'input'}, long_or_kurz)

  def getMetaAttributeNames(self, long_or_kurz = 'kurz'):
    return self.getAttributeNames({'meta'}, long_or_kurz)

  def getInputAttributeNames(self, long_or_kurz = 'kurz'):
    return self.getAttributeNames({'input'}, long_or_kurz)

  def getOutputAttributeNames(self, long_or_kurz = 'kurz'):
    return self.getAttributeNames({'output'}, long_or_kurz)

  def getBinaryAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check binary
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.node_type == 'input' and attr_obj.attr_type == 'binary':
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
    return np.array(names)

  def getActionableAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check actionability
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.node_type == 'input' and attr_obj.actionability != 'none':
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
    return np.array(names)

  def getNonActionableAttributeNames(self, long_or_kurz = 'kurz'):
    a = self.getInputAttributeNames(long_or_kurz)
    b = self.getActionableAttributeNames(long_or_kurz)
    return np.setdiff1d(a,b)

  def getMutableAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check mutability
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.node_type == 'input' and attr_obj.mutability != False:
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
    return np.array(names)

  def getNonMutableAttributeNames(self, long_or_kurz = 'kurz'):
    a = self.getInputAttributeNames(long_or_kurz)
    b = self.getMutableAttributeNames(long_or_kurz)
    return np.setdiff1d(a,b)

  def getIntegerBasedAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check attr_type
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.attr_type == 'numeric-int':
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
    return np.array(names)

  def getRealBasedAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check attr_type
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.attr_type == 'numeric-real':
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
    return np.array(names)

  def assertSiblingsShareAttributes(self, long_or_kurz = 'kurz'):
    # assert elems of dictOfSiblings share attr_type, node_type, parent, actionability, and mutability
    dict_of_siblings = self.getDictOfSiblings(long_or_kurz)
    for parent_name in dict_of_siblings['cat'].keys():
      siblings = dict_of_siblings['cat'][parent_name]
      assert len(siblings) > 1
      for sibling in siblings:
        if long_or_kurz == 'long':
          self.attributes_long[sibling].attr_type = self.attributes_long[siblings[0]].attr_type
          self.attributes_long[sibling].node_type = self.attributes_long[siblings[0]].node_type
          self.attributes_long[sibling].actionability = self.attributes_long[siblings[0]].actionability
          self.attributes_long[sibling].mutability = self.attributes_long[siblings[0]].mutability
          self.attributes_long[sibling].parent_name_long = self.attributes_long[siblings[0]].parent_name_long
          self.attributes_long[sibling].parent_name_kurz = self.attributes_long[siblings[0]].parent_name_kurz
        elif long_or_kurz == 'kurz':
          self.attributes_kurz[sibling].attr_type = self.attributes_kurz[siblings[0]].attr_type
          self.attributes_kurz[sibling].node_type = self.attributes_kurz[siblings[0]].node_type
          self.attributes_kurz[sibling].actionability = self.attributes_kurz[siblings[0]].actionability
          self.attributes_kurz[sibling].mutability = self.attributes_kurz[siblings[0]].mutability
          self.attributes_kurz[sibling].parent_name_long = self.attributes_kurz[siblings[0]].parent_name_long
          self.attributes_kurz[sibling].parent_name_kurz = self.attributes_kurz[siblings[0]].parent_name_kurz
        else:
          raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')

  def getSiblingsFor(self, attr_name_long_or_kurz):
    # If attr_name_long is given, we will return siblings_long (the same length)
    # but not siblings_kurz. Same for the opposite direction.
    assert \
      'cat' in attr_name_long_or_kurz or 'ord' in attr_name_long_or_kurz, \
      'attr_name must include either `cat` or `ord`.'
    if attr_name_long_or_kurz in self.getInputOutputAttributeNames('long'):
      attr_name_long = attr_name_long_or_kurz
      dict_of_siblings_long = self.getDictOfSiblings('long')
      for parent_name_long in dict_of_siblings_long['cat']:
        siblings_long = dict_of_siblings_long['cat'][parent_name_long]
        if attr_name_long_or_kurz in siblings_long:
          return siblings_long
      for parent_name_long in dict_of_siblings_long['ord']:
        siblings_long = dict_of_siblings_long['ord'][parent_name_long]
        if attr_name_long_or_kurz in siblings_long:
          return siblings_long
    elif attr_name_long_or_kurz in self.getInputOutputAttributeNames('kurz'):
      attr_name_kurz = attr_name_long_or_kurz
      dict_of_siblings_kurz = self.getDictOfSiblings('kurz')
      for parent_name_kurz in dict_of_siblings_kurz['cat']:
        siblings_kurz = dict_of_siblings_kurz['cat'][parent_name_kurz]
        if attr_name_long_or_kurz in siblings_kurz:
          return siblings_kurz
      for parent_name_kurz in dict_of_siblings_kurz['ord']:
        siblings_kurz = dict_of_siblings_kurz['ord'][parent_name_kurz]
        if attr_name_long_or_kurz in siblings_kurz:
          return siblings_kurz
    else:
      raise Exception(f'{attr_name_long_or_kurz} not recognized as a valid `attr_name_long_or_kurz`.')

  def getDictOfSiblings(self, long_or_kurz = 'kurz'):
    if long_or_kurz == 'long':

      dict_of_siblings_long = {}
      dict_of_siblings_long['cat'] = {}
      dict_of_siblings_long['ord'] = {}

      for attr_name_long in self.getInputAttributeNames('long'):
        attr_obj = self.attributes_long[attr_name_long]
        if attr_obj.attr_type == 'sub-categorical':
          if attr_obj.parent_name_long not in dict_of_siblings_long['cat'].keys():
            dict_of_siblings_long['cat'][attr_obj.parent_name_long] = [] # initiate key-value pair
          dict_of_siblings_long['cat'][attr_obj.parent_name_long].append(attr_obj.attr_name_long)
        elif attr_obj.attr_type == 'sub-ordinal':
          if attr_obj.parent_name_long not in dict_of_siblings_long['ord'].keys():
            dict_of_siblings_long['ord'][attr_obj.parent_name_long] = [] # initiate key-value pair
          dict_of_siblings_long['ord'][attr_obj.parent_name_long].append(attr_obj.attr_name_long)

      # sort sub-arrays
      for key in dict_of_siblings_long['cat'].keys():
        dict_of_siblings_long['cat'][key] = sorted(dict_of_siblings_long['cat'][key], key = lambda x : int(x.split('_')[-1]))

      for key in dict_of_siblings_long['ord'].keys():
        dict_of_siblings_long['ord'][key] = sorted(dict_of_siblings_long['ord'][key], key = lambda x : int(x.split('_')[-1]))

      return dict_of_siblings_long

    elif long_or_kurz == 'kurz':

      dict_of_siblings_kurz = {}
      dict_of_siblings_kurz['cat'] = {}
      dict_of_siblings_kurz['ord'] = {}

      for attr_name_kurz in self.getInputAttributeNames('kurz'):
        attr_obj = self.attributes_kurz[attr_name_kurz]
        if attr_obj.attr_type == 'sub-categorical':
          if attr_obj.parent_name_kurz not in dict_of_siblings_kurz['cat'].keys():
            dict_of_siblings_kurz['cat'][attr_obj.parent_name_kurz] = [] # initiate key-value pair
          dict_of_siblings_kurz['cat'][attr_obj.parent_name_kurz].append(attr_obj.attr_name_kurz)
        elif attr_obj.attr_type == 'sub-ordinal':
          if attr_obj.parent_name_kurz not in dict_of_siblings_kurz['ord'].keys():
            dict_of_siblings_kurz['ord'][attr_obj.parent_name_kurz] = [] # initiate key-value pair
          dict_of_siblings_kurz['ord'][attr_obj.parent_name_kurz].append(attr_obj.attr_name_kurz)

      # sort sub-arrays
      for key in dict_of_siblings_kurz['cat'].keys():
        dict_of_siblings_kurz['cat'][key] = sorted(dict_of_siblings_kurz['cat'][key], key = lambda x : int(x.split('_')[-1]))

      for key in dict_of_siblings_kurz['ord'].keys():
        dict_of_siblings_kurz['ord'][key] = sorted(dict_of_siblings_kurz['ord'][key], key = lambda x : int(x.split('_')[-1]))

      return dict_of_siblings_kurz

    else:

      raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')

  def getOneHotAttributesNames(self, long_or_kurz = 'kurz'):
    tmp = self.getDictOfSiblings(long_or_kurz)
    names = []
    for key1 in tmp.keys():
      for key2 in tmp[key1].keys():
        names.extend(tmp[key1][key2])
    return np.array(names)

  def getNonHotAttributesNames(self, long_or_kurz = 'kurz'):
    a = self.getInputAttributeNames(long_or_kurz)
    b = self.getOneHotAttributesNames(long_or_kurz)
    return np.setdiff1d(a,b)

  def getVariableRanges(self):
    return dict(zip(
      self.getInputAttributeNames('kurz'),
      [
        self.attributes_kurz[attr_name_kurz].upper_bound -
        self.attributes_kurz[attr_name_kurz].lower_bound
        for attr_name_kurz in self.getInputAttributeNames('kurz')
      ],
    ))

  def printDataset(self, long_or_kurz = 'kurz'):
    if long_or_kurz == 'long':
      for attr_name_long in self.attributes_long:
        print(self.attributes_long[attr_name_long].__dict__)
    elif long_or_kurz == 'kurz':
      for attr_name_kurz in self.attributes_kurz:
        print(self.attributes_kurz[attr_name_kurz].__dict__)
    else:
      raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')

  def getBalancedDataFrame(self):
    balanced_data_frame = copy.deepcopy(self.data_frame_kurz)

    meta_cols = self.getMetaAttributeNames()
    input_cols = self.getInputAttributeNames()
    output_col = self.getOutputAttributeNames()[0]

    # assert only two classes in label (maybe relax later??)
    assert np.array_equal(
      np.unique(balanced_data_frame[output_col]),
      np.array([0, 1]) # only allowing {0, 1} labels
    )

    # get balanced dataframe (take minimum of the count, then round down to nearest 250)
    unique_values_and_count = balanced_data_frame[output_col].value_counts()
    number_of_subsamples_in_each_class = unique_values_and_count.min() // 250 * 250
    balanced_data_frame = pd.concat([
        balanced_data_frame[balanced_data_frame.loc[:,output_col] == 0].sample(number_of_subsamples_in_each_class, random_state = RANDOM_SEED),
        balanced_data_frame[balanced_data_frame.loc[:,output_col] == 1].sample(number_of_subsamples_in_each_class, random_state = RANDOM_SEED),
    ]).sample(frac = 1, random_state = RANDOM_SEED)
    # balanced_data_frame = pd.concat([
    #     balanced_data_frame[balanced_data_frame.loc[:,output_col] == 0],
    #     balanced_data_frame[balanced_data_frame.loc[:,output_col] == 1],
    # ]).sample(frac = 1, random_state = RANDOM_SEED)

    return balanced_data_frame, meta_cols, input_cols, output_col

  # (2020.04.15) perhaps we need a memoize here... but I tried calling this function
  # multiple times in a row from another file and it always returned the same slice
  # of data... weird.
  def getTrainTestSplit(self, preprocessing = None, with_meta = False):

    # When working only with normalized data in [0, 1], data ranges must change to [0, 1] as well
    # otherwise, in computing normalized distance we will normalize with intial ranges again!
    # pseudonym (2020.05.17) does this work with cat/ord and sub-cat/sub-ord data???
    def setBoundsToZeroOne():
      for attr_name_kurz in self.getNonHotAttributesNames('kurz'):
        attr_obj = self.attributes_kurz[attr_name_kurz]
        attr_obj.lower_bound = 0.0
        attr_obj.upper_bound = 1.0

        attr_obj = self.attributes_long[attr_obj.attr_name_long]
        attr_obj.lower_bound = 0.0
        attr_obj.upper_bound = 1.0

    # Normalize data: bring everything to [0, 1] - implemented for when feeding the model to DiCE
    def normalizeData(X_train, X_test):
      for attr_name_kurz in self.getNonHotAttributesNames('kurz'):
        attr_obj = self.attributes_kurz[attr_name_kurz]
        lower_bound = attr_obj.lower_bound
        upper_bound =attr_obj.upper_bound
        X_train[attr_name_kurz] = (X_train[attr_name_kurz] - lower_bound) / (upper_bound - lower_bound)
        X_test[attr_name_kurz] = (X_test[attr_name_kurz] - lower_bound) / (upper_bound - lower_bound)

      setBoundsToZeroOne()

      return X_train, X_test

    # TODO: This should be used with caution... it messes things up in MACE as ranges
    # will differ between factual and counterfactual domains
    def standardizeData(X_train, X_test):
      x_mean = X_train.mean()
      x_std = X_train.std()
      for index in x_std.index:
        if '_ord_' in index or '_cat_' in index:
          x_mean[index] = 0
          x_std[index] = 1
      X_train = (X_train - x_mean) / x_std
      X_test = (X_test - x_mean) / x_std
      return X_train, X_test

    balanced_data_frame, meta_cols, input_cols, output_col = self.getBalancedDataFrame()

    if with_meta:
      all_data = balanced_data_frame.loc[:,np.array((input_cols, meta_cols)).flatten()]
      all_true_labels = balanced_data_frame.loc[:,output_col]
      if preprocessing is not None:
        assert with_meta == False, 'This feature is not built yet...'

      X_train, X_test, y_train, y_test = train_test_split(
        all_data,
        all_true_labels,
        train_size=.7,
        random_state = RANDOM_SEED)

      # ordering of next two lines matters (shouldn't overwrite input_cols); silly code... :|
      U_train = X_train[self.getMetaAttributeNames()]
      U_test = X_test[self.getMetaAttributeNames()]
      X_train = X_train[self.getInputAttributeNames()]
      X_test = X_test[self.getInputAttributeNames()]
      y_train = y_train # noop
      y_test = y_test # noop

      return X_train, X_test, U_train, U_test, y_train, y_test
    else:
      all_data = balanced_data_frame.loc[:,input_cols]
      all_true_labels = balanced_data_frame.loc[:,output_col]

      X_train, X_test, y_train, y_test = train_test_split(
        all_data,
        all_true_labels,
        train_size=.7,
        random_state = RANDOM_SEED)

      # TODO (2020.05.18): this should be updated so as NOT to update meta variables
      if preprocessing == 'standardize':
        X_train, X_test = standardizeData(X_train, X_test)
      elif preprocessing == 'normalize':
        X_train, X_test = normalizeData(X_train, X_test)

      return X_train, X_test, y_train, y_test