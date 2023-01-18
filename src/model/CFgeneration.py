from abc import ABC
import pickle
import pandas as pd
import os
from src.utils.dataloader import dataLoader
from src.MACE.MACE import MACE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score

from dice_ml import Data, Model, Dice

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class GenerateCF(ABC):
    def __init__(self, config):
        self.config = config

    def choseModel(self, model):
        if not hasattr(self,'modelSF'):
            if self.config['modelSF']:
                paramSF = self.config['modelSF']
            else:
                paramSF = {}
            from xgboost import XGBClassifier
            self.modelSF = XGBClassifier(**paramSF)
        #Logistic regression
        if self.config['models'][model]:
            params = self.config['models'][model]
        else:
            params = {}
        if model == 'LR':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(**params) # trovare modo per passere Dict di parametri
        #Linear fair Empirical Risk Minimization

        #eXtreme Gradient Boosting
        elif model == 'XGB':
            from xgboost import XGBClassifier
            self.model = XGBClassifier(**params)

        elif model == 'MLP':
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier(**params)

        elif model == 'DT':
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier(**params)

        elif model == 'RF':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**params)

        #SVM
        elif model == 'SVM':
            from sklearn.svm import SVC
            self.model = SVC(**params)

        elif model == 'LGB':
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(**params)

        elif model == 'AdvDeb':
            from src.DebiasingModel.AdversarialDebiasing.AdversarialDebiasing import AdversarialDebiasing
            self.model = AdversarialDebiasing(**params)

        elif model == 'LFERM':
            from sklearn.svm import SVC
            from src.DebiasingModel.LinearFERM.lferm import Linear_FERM
            self.model = Linear_FERM(SVC(**params),
                                     sensitive_feature=self.sensitiveFeature,
                                     output=self.outcomeFeature)
        elif model == 'FairC':
            from src.DebiasingModel.FairClassificationZafar.FairConstraints import FairConstraints
            self.model = FairConstraints(**params)

    def generate(self):
        #train_model
        for model in self.config['models'].keys():
            print(f'Starting model: {model}!')
            self.pipeline(model)
            self.initDice_MACE()
            CFgenStrategy = [('MACE', self.exp_MACE), ('genetic', self.exp_genetic), ('KDtree', self.exp_KD)]
            CFgenStrategyDict = {cFtype : expCF for cFtype, expCF in CFgenStrategy}
            if 'typeCF' not in self.config:
                iterCFgen = CFgenStrategy
            else:
                iterCFgen = [(typeCF, CFgenStrategyDict[typeCF]) for typeCF in self.config['typeCF']]
            # df_genetic = pd.DataFrame(columns=self.df.columns.to_list()+[self.sensitiveFeature])
            for cFtype, expCF in iterCFgen:
                risultati = {}
                risultati[cFtype] = []
                for index in tqdm(range(self.x_test.shape[0])):
                    try:
                        sample = self.x_test[index:index + 1]
                        y_real = self.y_test[index:index + 1]
                        result = self.pipe.predict(sample)
                        if result[0] == 1:
                            ds = 0
                        else:
                            ds = 1
                            # if expCF == self.exp_random:
                            # if self.config['data']  =='toy-dataset':
                            #     continue
                            # print('Counterfactuals Random Generation Initialized!')
                            # dice_exp = expCF.generate_counterfactuals(sample, total_CFs=self.config['NCF'], desired_class=ds,
                            #                                           verbose=True, random_seed=42)
                            print(f'Counterfactuals {cFtype.capitalize()} Generation Initialized!')
                            if expCF == self.exp_genetic:
                                print('Counterfactuals Genetic Generation Initialized!')
                                dice_exp = expCF.generate_counterfactuals(sample, total_CFs=self.config['NCF'],
                                                                          desired_class=ds, verbose=True,
                                                                          posthoc_sparsity_algorithm="binary")
                                CF = dice_exp.cf_examples_list[0].final_cfs_df
                            elif expCF == self.exp_KD:
                                dice_exp = expCF.generate_counterfactuals(sample, total_CFs=self.config['NCF'], desired_class=ds,
                                                                          verbose=True, posthoc_sparsity_algorithm="binary")
                                CF = dice_exp.cf_examples_list[0].final_cfs_df
                            elif expCF == self.exp_MACE:
                                CF = expCF.generate_counterfactuals(sample, total_CFs=self.config['NCF'], desired_class=ds,
                                                               verbose=True,)

                            if expCF != self.exp_KD and expCF != self.exp_MACE:
                                y_CF = CF[self.outcomeFeature]
                                CF = CF.drop(columns=self.outcomeFeature)
                            y_sens = self.pipe.predict(sample)
                            y_CF_sens = self.pipeSF.predict(CF)
                            risultati[cFtype].append((sample, y_real, result,  y_sens, y_CF_sens, CF))
                            # if cFtype == 'random' and self.config['data']  !='toy-dataset':
                            #     df_Random = pd.concat([df_Random, CF])
                            # elif cFtype == 'genetic':
                            #     df_genetic = pd.concat([df_genetic,CF])
                    except:
                        continue
                with open(self.generalPathModel+f'/{self.sensitiveFeature}/{cFtype.capitalize()}.pickle','wb') as file:
                    pickle.dump(risultati,file)


    def pipeline(self, model):
        if not hasattr(self, 'df'):
            self.df, self.target, self.sensitiveFeature, self.outcomeFeature, self.numvars, self.categorical = \
                dataLoader(self.config['data'])
        # self.outcomeFeature = outF
        # self.sensitiveFeature = SF
        # self.numvars = numvars
        if not hasattr(self, 'x_test'):
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df,self.target, test_size=0.1,
                                                                                    random_state=self.config['seed'],
                                                                                    stratify=self.target)
            self.feature_dim = self.x_train.shape[1]
            self.output_dim = 1
        self.choseModel(model)
        numeric_transformer = Pipeline(
            steps=[('scaler', StandardScaler())])

        categorical_transformer = Pipeline(
            steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

        transformations = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numvars),
                ('cat', categorical_transformer, self.categorical)])
        if not hasattr(self, 'pipeSF'):
            self.pipeSF = Pipeline(steps=[('preprocessor', transformations),
                                               ('classifier', self.modelSF)])
            self.pipeSF.fit(self.x_train, self.y_train[self.sensitiveFeature])
        self.pipe = Pipeline(steps=[('preprocessor', transformations),
                        ('classifier', self.model)])
        if model in ['AdvDeb', 'LFERM']:
            self.pipe.fit(self.x_train, self.y_train)
        else:
            self.pipe.fit(self.x_train, self.y_train[self.outcomeFeature])
        self.evaluate(model)

    def initDice_MACE(self):
        DF = pd.concat([self.df,self.target[self.outcomeFeature]],axis=1)
        self.d = Data(dataframe = DF, continuous_features=self.numvars, outcome_name=self.outcomeFeature)
        m = Model(model=self.pipe, backend="sklearn")
        # self.exp_random = Dice(self.d, m, method="random")
        self.exp_genetic = Dice(self.d, m, method="genetic")
        self.exp_KD = Dice(self.d, m, method="kdtree")
        self.exp_MACE = MACE(data=self.df, xtrain=self.x_train, model = self.pipe, outcome=self.target[self.outcomeFeature],
                         numvars=self.numvars, catvars=self.categorical,eps=1e-3,norm_type='one_norm',
                         random_seed=self.config['seed'])


    def evaluate(self,model):
        generalPath = os.environ['VIRTUAL_ENV'].split('venv')[0]

        ResultDir = self.config['savedir'].split('/')
        for dir in ResultDir:
            if dir == '':
                continue
            if generalPath[-1]=='/':
                generalPath = ''.join([generalPath,dir])
            else:
                generalPath = '/'.join([generalPath, dir])
            if not os.path.isdir(generalPath):
                os.makedirs(generalPath)
        SF_results = '/'.join([generalPath, 'XGB_res_{}_{}.txt'.format(self.config['data'].split('-')[0], self.sensitiveFeature)])
        if not os.path.isfile(SF_results):
            AUC = roc_auc_score(self.y_test[self.sensitiveFeature], self.pipeSF.predict_proba(self.x_test)[:, 1])
            ACC = accuracy_score(self.y_test[self.sensitiveFeature], self.pipeSF.predict(self.x_test))
            Recall = recall_score(self.y_test[self.sensitiveFeature], self.pipeSF.predict(self.x_test))
            Precision = precision_score(self.y_test[self.sensitiveFeature], self.pipeSF.predict(self.x_test))
            F1 = f1_score(self.y_test[self.sensitiveFeature], self.pipeSF.predict(self.x_test))
            print(f"\n_______________________ SensitiveCLF METRICS ____________________"
                  f"\nArea under ROCurve: {AUC}\nAccuracy : {ACC}\nPrecision : {Precision}\nRecall : {Recall}\n"
                  f"F1 score :{F1}", file=open(SF_results, "a"))
        from src.utils.metrics import DifferenceStatisticalParity, DifferenceEqualOpportunity, DifferenceAverageOdds
        self.generalPathModel = '/'.join([generalPath,model])
        if not os.path.isdir(self.generalPathModel):
            os.makedirs(self.generalPathModel)
        if not os.path.isdir('/'.join([self.generalPathModel,self.sensitiveFeature])):
            os.makedirs('/'.join([self.generalPathModel,self.sensitiveFeature]))
        modelRes = '/'.join([self.generalPathModel, '{}_res_{}_{}.txt'.format(model,self.config['data'].split('-')[0],
                                                                              self.outcomeFeature)])
        AUC = roc_auc_score(self.y_test[self.outcomeFeature], self.pipe.predict_proba(self.x_test)[:, 1])
        ACC = accuracy_score(self.y_test[self.outcomeFeature], self.pipe.predict(self.x_test))
        Recall = recall_score(self.y_test[self.outcomeFeature], self.pipe.predict(self.x_test))
        Precision = precision_score(self.y_test[self.outcomeFeature], self.pipe.predict(self.x_test))
        F1 = f1_score(self.y_test[self.outcomeFeature], self.pipe.predict(self.x_test))
        ############################Difference in Statistical Parity#######################################
        DSP = DifferenceStatisticalParity(self.pipe.predict(self.x_test), self.y_test, self.sensitiveFeature, self.outcomeFeature, 1, 0, [0, 1])
        ############################Difference in Equal Opportunity########################################
        DiffEqOpp = DifferenceEqualOpportunity(self.pipe.predict(self.x_test), self.y_test, self.sensitiveFeature, self.outcomeFeature, 1, 0, [0, 1])
        ############################Difference in Average Odds#############################################
        DiffAvvOds = DifferenceAverageOdds(self.pipe.predict(self.x_test), self.y_test, self.sensitiveFeature, self.outcomeFeature, 1, 0, [0, 1])
        print(f"\nArea under ROCurve: {AUC}\nAccuracy : {ACC}\nPrecision : {Precision}\nRecall : {Recall}\n"
              f"F1 score :{F1}",file=open(modelRes, "a"))
        print("\n_______________________ FAIRESS METRICS ____________________\n"
              "DSP: {}\nDEO: {}\nDAO: {}\n".format(DSP,DiffEqOpp,DiffAvvOds), file=open(modelRes, "a"))