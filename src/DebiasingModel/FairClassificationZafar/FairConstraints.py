from sklearnex import patch_sklearn
patch_sklearn(verbose=True)
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict
import numpy as np
import DebiasedModel.FairClassificationZafar.utils as ut
# import utils as ut
import random
import time
import numba as nb
nb.njit()
from numba import jit
import DebiasedModel.FairClassificationZafar.loss_funcs as lf # loss funcs that can be optimized subject to various constraints
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


class FairConstraints(BaseEstimator, ClassifierMixin):
    """Fairness constraints is an in-processing technique re-express fairness constraints
         via a convex relaxation.

         References:
             .. Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rogriguez, and Krishna P
                Gummadi. "Fairness Constraints: Mechanisms for Fair Classification." In
                Artificial Intelligence and Statistics, 2017.

         """

    def __init__(self, sensitive_feature: str = None, output: str = None, setting: str = 'c', c: float = None, seed: int = 42):
        """
        Args
        :param sensitive_feature (str): name of protected attribute
        :param output (str): label name
        :param eta (double, optional): fairness penalty parameter
        """
        self.c = c
        self.sensitive_attr = sensitive_feature
        self.output = output
        self.setting = setting
        self.seed = seed
        random.seed(seed)  # set the random seed so that the random permutations can be reproduced again
        np.random.seed(seed)

        if setting == 'gamma':
            if self.c:
                self.mode = {"accuracy": 1, "gamma": float(self.c)}
            else:
                self.mode = {"accuracy": 1, 'gamma': 0.5} #default parameters zafar accuracy
        elif setting == 'c':
            if not self.c:
                self.c = 0.001
            self.mode = {"fairness": 1}
        elif setting == 'baseline':
            self.mode = {}
        else:
            raise Exception("Don't know how to handle setting %s" % setting)

    @staticmethod
    def train_classifier(x, y, control, sensitive_attrs, mode, sensitive_attrs_to_cov_thresh,seed):
        loss_function = lf._logistic_loss
        return ut.train_model(
            x, y, control, loss_function,
            mode.get('fairness', 0),
            mode.get('accuracy', 0),
            mode.get('separation', 0),
            sensitive_attrs,
            sensitive_attrs_to_cov_thresh,
            seed,
            mode.get('gamma', None))

    @staticmethod
    @jit(nopython=True)
    def _add_intercept_and_dotProduct(X, w):
        m, n = X.shape
        R = []
        #intercept = np.ones(m).reshape(m, 1)
        #X = np.concatenate((intercept, X), axis=1)
        for i in range(m):
            v = w[0]
            for j in range(n):
                v += X[i][j] * w[j+1]
            R.append(v)#X[i].dot(w)
        return np.array(R)#np.dot(X, w)

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def dec_fun(X, w):
        m, n = X.shape
        R = []
        #intercept = np.ones(m).reshape(m, 1)
        #X = np.concatenate((intercept, X), axis=1)
        for i in range(m):
            v = w[0]
            for j in range(n):
                v += X[i][j] * w[j+1]
            if v > 0:
                v = 1
            else:
                v = 0
            R.append(v)#X[i].dot(w)
        return np.array(R)#np.dot(X, w)

    @staticmethod
    def add_intercept(X):
        m, n = X.shape
        intercept = np.ones(m).reshape(m, 1)
        return np.concatenate((intercept, X), axis=1)

    def fit(self, X, y):
        self.classes_ = np.unique(y[self.output].values)
        # splitting for calibrating probabilities
        X, X_calib, y, y_calib = train_test_split(X,y,test_size=0.2,random_state=self.seed, stratify=y)
        x_train = X
        y_train = y[self.output].replace(0,-1)
        s_control = {self.sensitive_attr: y[self.sensitive_attr]}

        # X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
        x_train = self.add_intercept(x_train)
        #x_test = self.add_intercept(x_test)

        thresh = {}
        if self.setting == 'c':
            thresh = dict((k, float(self.c)) for (k, v) in s_control.items())
        sensitive_attrs = [self.sensitive_attr]
        self.w = self.train_classifier(x_train, y_train, s_control,
                         sensitive_attrs, self.mode,
                         thresh, self.seed)
        self.Calib = CalibratedClassifierCV(base_estimator=self, method='sigmoid', cv='prefit')
        self.Calib.fit(X_calib, y_calib[self.output])
        return 0

    def predict(self, X):
        # predictions = np.sign(self.decision_function(X))
        # return (predictions>0).astype(int)
        return self.dec_fun(X, self.w)

    def predict_proba(self, X):
        # sigmoid = lambda x : 1 / (1 + np.exp(-x))
        # return sigmoid(self.decision_function(X))
        return self.Calib.predict_proba(X)

    def decision_function(self, X):
        # X_ = self.add_intercept(X)
        # return np.dot(X_, self.w)
        return self._add_intercept_and_dotProduct(X, self.w)

# if __name__=='__main__':
#     from sklearnex import patch_sklearn
#
#     patch_sklearn(verbose=2)
#
#     from sklearn.compose import ColumnTransformer
#     from sklearn.model_selection import train_test_split, GridSearchCV, KFold
#     from sklearn.pipeline import Pipeline
#     from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
#     from sklearn.svm import SVC
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.metrics import accuracy_score
#     from tqdm import tqdm
#     from src.metrics.metrics import DifferenceEqualOpportunity,DifferenceAverageOdds
#
#     import pandas as pd
#     import random
#     from src.metrics.metrics import *
#     from src.utils.dataloader import dataLoader
#     df, target, sf, outcome, numvars, categorical = dataLoader('german-age')
#     results = {'acc':[],'dsp':[],'deo':[],'dao':[],'di':[]}
#     x=5
#     d = set()
#     for seed in tqdm(range(100)):
#         _, x_test = \
#             train_test_split(range(df.shape[0]), test_size=0.1,
#                              random_state=seed)
#         d.update(x_test)
#     assert d.__len__() == df.shape[0]
#     for seed in tqdm(range(100)):
#         random.seed(seed)
#         np.random.seed(seed)
#         x_train, x_test, y_train, y_test = train_test_split(df,
#                                                             target,
#                                                             test_size=0.1,
#                                                             random_state=seed,
#                                                             stratify=target)
#
#         # categorical = x_train.columns.difference(numvars)
#         # We create the preprocessing pipelines for both numeric and categorical data.
#         numeric_transformer = Pipeline(
#             steps=[('scaler', StandardScaler())])
#
#         categorical_transformer = Pipeline(
#             steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
#
#         transformations = ColumnTransformer(
#             transformers=[
#                 ('num', numeric_transformer, numvars),
#                 ('cat', categorical_transformer, categorical)])
#         zafar = FairConstraints(sf, outcome,'c',0.001,seed)
#         pipeline = Pipeline(steps=[('preprocessor', transformations),
#                                     ('classifier', zafar)])
#         # x_train, x_test = pipeline.fit_transform(x_train), pipeline.fit_transform(x_test)
#         pipeline.fit(x_train,y_train)
#         # clf = GridSearchCV(ferm, param_grid, n_jobs=1)
#         # pipeline.fit(x_train,y_train)
#         # y_pred = pipeline.predict(x_train)
#         # ACC = accuracy_score(y_train[outcome], y_pred)
#         # print("Accuracy_train: {}".format(ACC))
#         y_pred = pipeline.predict(x_test)
#         ACC = accuracy_score(y_test[outcome],y_pred)
#         results['acc'].append(ACC)
#         print("Accuracy_test: {}".format(ACC))
#         DSP = DifferenceStatisticalParity(y_pred, y_test, sf, outcome, 1, 0, [0, 1])
#         results['dsp'].append(DSP)
#         print("DSP: {}".format(DSP))
#         DEO = DifferenceEqualOpportunity(y_pred, y_test,sf, outcome, 1, 0, [0, 1])
#         results['deo'].append(DEO)
#         print("DEO: {}".format(DEO))
#         DAO = DifferenceAverageOdds(y_pred, y_test, sf, outcome, 1, 0, [0, 1])
#         results['dao'].append(DAO)
#         print("DAO: {}".format(DAO))
#         DImp = DI(y_pred, y_test, sf, outcome, 1, 0, [0, 1])
#         print("DI: {}".format(DImp))
#         results['di'].append(DImp)
#     for k,i in results.items():
#         r_mean = np.array(i).mean()
#         r_std = np.array(i).std()
#         print(f"{k} mean: {r_mean} and +/-: {r_std}")