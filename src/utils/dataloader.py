import os
import zipfile
import pandas as pd
import numpy as np

from urllib import request


def dataLoader(type):
    generalPath = os.environ['VIRTUAL_ENV'].split('venv')[0]
    # pathHealth = generalPath + 'data/Health/HHP_release3.zip'
    # pathCrime = generalPath + 'data/Crime/CommViolPredUnnormalizedData.txt'
    # pathDrug = generalPath + 'data/Drug/drug_consumption.data'
    pathAdult = generalPath + "data/Adult/adult.tsv"
    pathCrime = generalPath + 'data/Crime/CommViolPredUnnormalizedData.txt'
    pathGerman = generalPath + 'data/German/German.tsv'


    if type.lower() == 'crime-race':
        header = [
            'communityname', 'state', 'countyCode', 'communityCode', 'fold', 'population', 'householdsize',
            'racepctblack',
            'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
            'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst',
            'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap',
            'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad',
            'PctBSorMore',
            'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf',
            'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par',
            'PctKids2Par',
            'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', 'NumKidsBornNeverMar',
            'PctKidsBornNeverMar', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10',
            'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly',
            'PctNotSpeakEnglWell',
            'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous',
            'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant', 'PctHousOccup',
            'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb',
            'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'OwnOccQrange', 'RentLowQ', 'RentMedian', 'RentHighQ',
            'RentQrange', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', 'NumInShelters',
            'NumStreet', 'PctForeignBorn', 'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 'PctSameState85',
            'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq',
            'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite',
            'PctPolicBlack',
            'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
            'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
            'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop', 'murders',
            'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries',
            'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop',
            'ViolentCrimesPerPop', 'nonViolPerPop'
        ]
        df = pd.read_csv(pathCrime, sep=',', names=header)
        # remove features that are not predictive
        df.drop(['communityname', 'countyCode', 'communityCode', 'fold'], axis=1, inplace=True)
        # remove all other potential goal variables
        df.drop(
            [
                'murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults',
                'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft',
                'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'nonViolPerPop'
            ], axis=1, inplace=True
        )
        df.replace(to_replace='?', value=np.nan, inplace=True)
        # drop rows with missing labels
        df.dropna(axis=0, subset=['ViolentCrimesPerPop'], inplace=True)
        # drop columns with missing values
        df.dropna(axis=1, inplace=True)
        features, labels = df.drop('ViolentCrimesPerPop', axis=1), df['ViolentCrimesPerPop'].astype(float)
        continuous_vars = []
        categorical_columns = []
        for col in features.columns:
            if features[col].isnull().sum() > 0:
                features.drop(col, axis=1, inplace=True)
            else:
                if features[col].dtype == object:
                    categorical_columns += [col]
                else:
                    continuous_vars += [col]
        protected = np.less(
            features['racePctWhite'] / 5, features['racepctblack'] + features['racePctAsian'] + features['racePctHisp']
        )
        privilaged = 1-protected.astype(int)
        df = features
        target= (labels<labels.median()).astype(int).to_frame()
        target['race'] = privilaged.values
        df.drop(columns=['racePctWhite','racepctblack','racePctAsian','racePctHisp'],inplace=True)
        for i in ['racePctWhite','racepctblack','racePctAsian','racePctHisp']:
            continuous_vars.remove(i)
        #DF.shape == (1994, 102)
        return df, target, 'race', 'ViolentCrimesPerPop', continuous_vars, categorical_columns




    if type.lower() == 'adult-gender-biased':
        df = pd.read_csv(pathAdult, sep='\t')
        numvars = ['education-num', 'capital gain', 'capital loss', 'hours per week' , 'Age', 'fnlwgt']
        # df['workclass'].replace([' Private', ' Self-emp-not-inc', ' Self-emp-inc'], ' Private', inplace=True)
        # df['workclass'].replace([' Federal-gov', ' Local-gov', ' State-gov'], ' Public', inplace=True)
        # df['workclass'].replace([' Without-pay'], 'Unemployed', inplace=True)
        # df = df.drop(columns=['Age', 'race', 'relationship', 'fnlwgt', 'education', 'native-country'])
        # Sensitive_Features = ['gender', 'marital-status']
        df.replace([' Male', ' Female'], [1,0],inplace=True)
        target = df[['income','gender']]
        df.drop(columns=['income', 'gender'], inplace=True)
        categorical = df.columns.difference(numvars)
        return df, target, 'gender', 'income', numvars, categorical


    elif type.lower() == 'adult-ms-biased':
        df = pd.read_csv(pathAdult, sep='\t')
        numvars = [ 'education-num','capital gain', 'capital loss', 'hours per week', 'Age', 'fnlwgt']
        # df['workclass'].replace([' Private', ' Self-emp-not-inc', ' Self-emp-inc'], ' Private', inplace=True)
        # df['workclass'].replace([' Federal-gov', ' Local-gov', ' State-gov'], ' Public', inplace=True)
        # df['workclass'].replace([' Without-pay'], 'Unemployed', inplace=True)
        # df = df.drop(columns=['Age', 'race', 'relationship', 'fnlwgt', 'education', 'native-country'])
        #Sensitive_Features = ['gender', 'marital-status']
        df.replace(to_replace=[' Divorced', ' Married-AF-spouse', ' Married-civ-spouse',
                               ' Married-spouse-absent', ' Never-married', ' Separated',
                               ' Widowed'], value=['not married','married','married',
                                                   'married', 'not married', 'not married',
                                                   'not married'],inplace=True)
        df.replace(['married','not married'], [1,0], inplace=True)
        target = df[['income','marital-status']]
        df.drop(columns=['income','marital-status'], inplace=True)
        categorical = df.columns.difference(numvars)
        return df, target, 'marital-status', 'income', numvars, categorical

    if type.lower() == 'adult-gender':
        df = pd.read_csv(pathAdult, sep='\t')
        numvars = ['education-num', 'capital gain', 'capital loss', 'hours per week']  # , 'Age', 'fnlwgt']
        df['workclass'].replace([' Private', ' Self-emp-not-inc', ' Self-emp-inc'], ' Private', inplace=True)
        df['workclass'].replace([' Federal-gov', ' Local-gov', ' State-gov'], ' Public', inplace=True)
        df['workclass'].replace([' Without-pay'], 'Unemployed', inplace=True)
        df = df.drop(columns=['Age', 'race', 'relationship', 'fnlwgt', 'education', 'native-country'])
        # Sensitive_Features = ['gender', 'marital-status']
        df.replace([' Male', ' Female'], [1,0],inplace=True)
        target = df[['income','gender']]
        df.drop(columns=['income','marital-status', 'gender'], inplace=True)
        categorical = df.columns.difference(numvars)
        return df, target, 'gender', 'income', numvars, categorical


    elif type.lower() == 'adult-ms':
        df = pd.read_csv(pathAdult, sep='\t')
        numvars = [ 'education-num','capital gain', 'capital loss', 'hours per week']#, 'Age', 'fnlwgt']
        df['workclass'].replace([' Private', ' Self-emp-not-inc', ' Self-emp-inc'], ' Private', inplace=True)
        df['workclass'].replace([' Federal-gov', ' Local-gov', ' State-gov'], ' Public', inplace=True)
        df['workclass'].replace([' Without-pay'], 'Unemployed', inplace=True)
        df = df.drop(columns=['Age', 'race', 'relationship', 'fnlwgt', 'education', 'native-country'])
        #Sensitive_Features = ['gender', 'marital-status']
        df.replace(to_replace=[' Divorced', ' Married-AF-spouse', ' Married-civ-spouse',
                               ' Married-spouse-absent', ' Never-married', ' Separated',
                               ' Widowed'], value=['not married','married','married',
                                                   'married', 'not married', 'not married',
                                                   'not married'],inplace=True)
        df.replace(['married','not married'], [1,0], inplace=True)
        target = df[['income','marital-status']]
        df.drop(columns=['income','marital-status','gender'], inplace=True)
        categorical = df.columns.difference(numvars)
        return df, target, 'marital-status', 'income', numvars, categorical

    if type.lower() == 'german-gender':
        df = pd.read_csv(pathGerman, sep='\t')
        numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'existingcredits', 'peopleliable']
        Sensitive_Features = ['gender', 'foreignworker', 'age','statussex']
        target = df[['gender','classification']]
        target.replace(['M','F'], [1, 0], inplace=True)
        df = df.drop(columns=Sensitive_Features)
        # Split data into train and test
        df = df.drop("classification", axis=1)
        categorical = df.columns.difference(numvars)
        return df, target, 'gender', 'classification', numvars, categorical

    # if type.lower() == 'german-fw':
    #     df = pd.read_csv(pathGerman, sep='\t')
    #     numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'existingcredits', 'peopleliable']
    #     Sensitive_Features = ['gender', 'foreignworker']
    #     target = df[['foreignworker','classification']]
    #     target.replace(['no', 'yes'], [1, 0], inplace=True)
    #     df = df.drop(columns=Sensitive_Features)
    #     # Split data into train and test
    #     df = df.drop("classification", axis=1)
    #     categorical = df.columns.difference(numvars)
    #     return df, target, 'foreignworker', 'classification', numvars, categorical
    #
    elif type.lower() == 'german-age':
        '''
                AIF360 : https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/german_dataset.py
                By default, this code converts the 'age' attribute to a binary value
                where privileged is `age > 25` and unprivileged is `age <= 25` as
                proposed by Kamiran and Calders [1]_.
                References:
                    .. [1] F. Kamiran and T. Calders, "Classifying without
                       discriminating," 2nd International Conference on Computer,
                       Control and Communication, 2009.
                '''
        df = pd.read_csv(pathGerman, sep='\t')
        numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'existingcredits', 'peopleliable']
        Sensitive_Features = ['gender', 'foreignworker', 'age', 'statussex']
        df['age'] = df['age'].apply(lambda x: int(x >= 26))
        target = df[['age','classification']]
        # target.replace(['no', 'yes'], [1, 0], inplace=True)
        df = df.drop(columns=Sensitive_Features)
        # Split data into train and test
        df = df.drop("classification", axis=1)
        categorical = df.columns.difference(numvars)
        return df, target, 'age', 'classification', numvars, categorical

    elif type.lower() == 'toy-dataset':
        # Load toy test
        n_samples = 200 * 2
        n_samples_low = 20 * 2
        n_dimensions = 10
        X, y, sensible_feature, target = generate_toy_data(n_samples=n_samples,
                                                           n_samples_low=n_samples_low,
                                                           n_dimensions=n_dimensions)
        return X, y, sensible_feature, target, [str(i) for i in range(X.shape[1])], []

def generate_toy_data(n_samples, n_samples_low, n_dimensions):
    np.random.seed(42)
    varA = 0.8
    aveApos = [-1.0] * n_dimensions
    aveAneg = [1.0] * n_dimensions
    varB = 0.5
    aveBpos = [0.5] * int(n_dimensions / 2) + [-0.5] * int(n_dimensions / 2 + n_dimensions % 2)
    aveBneg = [0.5] * n_dimensions

    X = np.random.multivariate_normal(aveApos, np.diag([varA] * n_dimensions), n_samples)
    X = np.vstack([X, np.random.multivariate_normal(aveAneg, np.diag([varA] * n_dimensions), n_samples)])
    X = np.vstack([X, np.random.multivariate_normal(aveBpos, np.diag([varB] * n_dimensions), n_samples_low)])
    X = np.vstack([X, np.random.multivariate_normal(aveBneg, np.diag([varB] * n_dimensions), n_samples)])
    sensible_feature = [1] * (n_samples * 2) + [-1] * (n_samples + n_samples_low)
    #sensible_feature = np.array(sensible_feature)
    #sensible_feature.shape = (len(sensible_feature), 1)
    #X = np.hstack([X, sensible_feature])
    y = [1] * n_samples + [-1] * n_samples + [1] * n_samples_low + [-1] * n_samples
    y = np.array(list(zip(y,sensible_feature)))
    idx_A = list(range(0, n_samples * 2))
    idx_B = list(range(n_samples * 2, n_samples * 3 + n_samples_low))
    col = [str(i) for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=col)
    y = pd.DataFrame(y, columns=['target','SF'])
    y.replace(-1,0,inplace=True)

    # print('(y, sensible_feature):')
    # for el in zip(y, sensible_feature):
    #     print(el)
    return X, y, 'SF','target'
