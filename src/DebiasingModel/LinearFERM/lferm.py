import numpy as np
import sys
from collections import namedtuple
from sklearn.base import BaseEstimator, ClassifierMixin

class Linear_FERM(BaseEstimator,ClassifierMixin):
    # The linear FERM algorithm
    def __init__(self, model, sensitive_feature, output):
        self.sensitive_f = sensitive_feature
        #self.values_of_sensible_feature = [0,1]
        self.output = output
        self.model = model
        self.u = None
        self.max_i = None

    def new_representation(self, examples):
        if self.u is None:
            sys.exit('Model not trained yet!')
            return 0

        new_examples = np.array([ex - self.u * (ex[self.max_i] / self.u[self.max_i]) for ex in examples])
        new_examples = np.delete(new_examples, self.max_i, 1)
        return new_examples

    def predict_proba(self, X):
        new_examples = self.new_representation(X)
        prediction = self.model.predict_proba(new_examples)
        return prediction

    def predict(self, X):
        new_examples = self.new_representation(X)
        prediction = self.model.predict(new_examples)
        return prediction

    def fit(self, X, y):
        # Evaluation of the empirical averages among the groups
        self.meanY = y[self.sensitive_f].mean()
        self.stdY = y[self.sensitive_f].std()
        self.val0 = (0-self.meanY)/self.stdY
        self.val1 = (1-self.meanY)/self.stdY

        tmp = [ex for idx, ex in enumerate(X)
               if y[self.output].iloc[idx] == 1 and y[self.sensitive_f].iloc[idx] == 1]
        average_A_1 = np.mean(tmp, 0)
        tmp = [ex for idx, ex in enumerate(X)
               if y[self.output].iloc[idx] == 1 and y[self.sensitive_f].iloc[idx] == 0]
        average_not_A_1 = np.mean(tmp, 0)

        # Evaluation of the vector u (difference among the two averages)
        self.u = -(average_A_1 - average_not_A_1)
        self.max_i = np.argmax(self.u)

        # Application of the new representation
        newdata = np.array([ex - self.u * (ex[self.max_i] / self.u[self.max_i]) for ex in X])
        newdata = np.delete(newdata, self.max_i, 1)
        self.dataset = namedtuple('_', 'data, target')(newdata, y[self.output].values)

        # Fitting the linear model by using the new data
        if self.model:
            self.model.fit(self.dataset.data, self.dataset.target)

#
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split, GridSearchCV, KFold
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from tqdm import tqdm
# from src.metrics.metrics import DifferenceEqualOpportunity,DifferenceAverageOdds
#
#
# import pandas as pd
# import random
# from src.metrics.metrics import *
# random.seed(42)
# np.random.seed(42)
#
# df = pd.read_csv("../../data/German/German.tsv", sep='\t')
#
# numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'existingcredits', 'peopleliable']
#
# Sensitive_Features = ['gender','foreignworker']
# target = df[['classification',Sensitive_Features[0]]]
# #dict={'privilaged':('M',1), 'unprovilaged':('F',0)}
# mappingPrivUnpriv={'privilaged':'M', 'unprivilaged':'F'}
# target.replace(['M','F'],[1,0],inplace=True)
#
# df = df.drop(columns=Sensitive_Features)
# # Split data into train and test
# datasetX = df.drop("classification", axis=1)
# x_train, x_test, y_train, y_test = train_test_split(datasetX,
#                                                     target,
#                                                     test_size=0.1,
#                                                     random_state=42,
#                                                     stratify=target)
#
# categorical = x_train.columns.difference(numvars)
# # We create the preprocessing pipelines for both numeric and categorical data.
# numeric_transformer = Pipeline(
#     steps=[('scaler', StandardScaler())])
#
# categorical_transformer = Pipeline(
#     steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
#
# transformations = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numvars),
#         ('cat', categorical_transformer, categorical)])
# lferm = Linear_FERM(SVC(C = 0.01,kernel='linear'),'gender','classification')
# pipeline = Pipeline(steps=[('preprocessor', transformations),
#                            ('classifier', lferm)])
#
# pipeline.fit(x_train,y_train)
#
# y_pred = pipeline.predict(x_test)
# ACC = accuracy_score(y_test['classification'],y_pred)
# print("Accuracy: {}".format(ACC))
# DEO = DifferenceEqualOpportunity(y_pred, y_test,'gender', 'classification', 1, 0, [0, 1])
# print("DEO: {}".format(DEO))
# DAO = DifferenceAverageOdds(y_pred, y_test, 'gender', 'classification', 1, 0, [0, 1])
# print("DAO: {}".format(DAO))
# x=5