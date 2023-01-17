import pickle
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
from tqdm import tqdm

from src.utils.dataloader import dataLoader
from src.utils.mainEvalSFclf import SFgridDict, loadMap

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder



def get_result_from_pickle(path):
    with open(path,"rb") as file:
        return pickle.load(file)

def makePipeline(dataset,SF,modelName):
    '''
    create and fit the sensitive feature classifier for evaluating
    counterfactual samples through CFlips metric.
    :param dataset: dataset name
    :param SF: sensitive information considered for the investigation
    :param modelName: sensitive feature classifier considered (i.e., between RF, MLP, and XGB).
    :return: pipeline fitted
    '''
    datal = loadMap[dataset][SF]
    if not modelName:
        modelName = 'XGB'
    model = SFgridDict[dataset][SF][modelName]
    df, target, sf, outcome, numvars, categorical = dataLoader(datal)
    x_train, x_test, y_train, y_test = train_test_split(df,
                                                        target,
                                                        test_size=0.1,
                                                        random_state=42,
                                                        stratify=target)
    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numvars),
            ('cat', categorical_transformer, categorical)])
    pipeline = Pipeline(steps=[('preprocessor', transformations),
                               ('classifier', model)])

    pipeline.fit(x_train, y_train[sf])
    return pipeline

def FlipTest(y_real_sens, y_pred_sens_CF,nCF=None):
    '''
    Count the number of Counterfactual Flip for a particular user at a determined number of CF
    :param y_real_sens: real sensitive information of the x sample
    :param y_pred_sens_CF: predicted sensitive information of the considered CF samples
    :param nCF: number of considered CF samples (between 1-100)
    :return: number of Flipped counterfactual
    '''
    if nCF:
        assert nCF<=y_pred_sens_CF.shape[0], "the number of CF ({}) is higher than the shape of generated " \
                                             "CF ({})".format(nCF,y_pred_sens_CF.shape)
        l=0
        # return np.mean((y_real_sens != y_pred_sens_CF[:nCF]).astype(int))
        for i in range(nCF):
            if y_real_sens != y_pred_sens_CF[i]:
                l+=1
        return l/nCF
        # l += sum((1 for i in range(nCF) if y_real_sens != y_pred_sens_CF[i]))
        # return l/nCF
    else:
        if y_real_sens != y_pred_sens_CF[0]:
            return 1
        else:
            return 0

def CFlips(Results, cfStrategy, SF, SFclf, type_sensitive):
    MaxCF = 100 # number of generated CF for each sample
    if type_sensitive == 'unprivileged':
        ySF = 0
    else:
        ySF = 1
    R = get_result_from_pickle(Results)[cfStrategy]['sample_CF']
    Results = []
    CFrangeFlip = [0]
    for i in tqdm(R):
        sample, y_real, result, _, _, CF = i
        if result != 0:
            continue
        else:
            if y_real[SF].item() !=ySF:
                continue
            else:
                y_sens = SFclf.predict(sample)
                y_CF_sens = SFclf.predict(CF)
                Results.append(tuple([sample, y_real, result, y_sens, y_CF_sens, CF]))
    for nCF in tqdm(range(MaxCF)):
        FlTest = 0
        totN = 0 # sample with nCF
        for i in Results:
            sample, y_real, result, y_sens, y_CF_sens, CF = i
            if y_real[SF].item() != y_sens:
                continue
            else:
                if CF.shape[0] < nCF:
                    continue
                else:
                    CF = CF[ : nCF+1]
                    Flip = FlipTest(y_real[SF].item(), y_CF_sens, nCF=nCF)
                    totN += 1
                    FlTest += Flip
        CFrangeFlip.append((FlTest / (totN + 1e-10)) * 100)

    return CFrangeFlip



def CFmetrics(Results, dataset, SF, CFmethod, SFclf = None):
    pipelineSFclf = makePipeline(dataset, SF,SFclf)
    if CFmethod.lower() == 'genetic':
        cfIndex = 0
    else:
        cfIndex = 1
    plotR = {}
    for model in Results.keys():
        print(f'Evaluating model: {model}')
        plotR[model] = {}
        pathR = Results[model][SF][cfIndex]
        type_sensitive = 'unprivileged'
        flippedList = CFlips(pathR, CFmethod, SF, pipelineSFclf, type_sensitive)
        plotR[model][type_sensitive] = flippedList
        type_sensitive = 'privileged'
        flippedList = CFlips(pathR, CFmethod, SF, pipelineSFclf, type_sensitive)
        plotR[model][type_sensitive] = flippedList

    X_axis = [0] + [i + 1 for i in range(0, 100)]
    for type_sensitive in ['unprivileged','privileged', 'delta']:
        legend = []
        if type_sensitive == 'delta':
            plt.title("{} {} $\Delta$CFlips ({})".format(dataset, SF, CFmethod.upper()))
            for model in Results.keys():
                legend.append(model)
                delta = [abs(unp-priv) for unp,priv in zip(plotR[model]['unprivileged'],plotR[model]['privileged'])]
                plt.plot(X_axis, delta, marker='.')
                print(f"Delta CFlips = {delta[-1]}, for model {model} ")
            plt.ylim(0, 100)
            plt.legend(legend)
            plt.xlabel("N° of CF (|$C_x$|))")
            plt.ylabel("$\Delta$CFlips($X^-$)")
            plt.savefig(f'figure/DeltaCFlips_k_ablation_{dataset}_{SF}_{CFmethod.upper()}_SFclf:{SFclf}.svg',
                        bbox_inches='tight')
            plt.show()
            plt.close()
        else:
            plt.title("{} {} CFlips for {} ({})".format(dataset,SF,type_sensitive,CFmethod.upper()))
            for model in Results.keys():
                legend.append(model)
                plt.bar(model, plotR[model][type_sensitive][-1])
            plt.ylim(0, 100)
            plt.legend(legend)
            plt.xlabel("N° of CF (|$C_x$|))")
            plt.ylabel("CFlips($X^-$)")
            plt.savefig(f'figure/BarChart_{dataset}_{SF}_{CFmethod.upper()}_{type_sensitive}_SFclf:{SFclf}.svg',
                        bbox_inches='tight')
            plt.show()
            plt.close()
