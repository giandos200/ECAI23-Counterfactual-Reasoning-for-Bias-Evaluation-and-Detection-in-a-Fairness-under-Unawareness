import pickle
import numpy as np
import pandas as pd

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
        steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

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



def ProxyFeatureDetection(modelSF, dataset, model, dict,cFtype, SF):
    datal = loadMap[dataset][SF]
    df, target, SF, Clf, numvars, categorical = dataLoader(datal)
    pipelineCLF_Sens = makePipeline(dataset, SF, modelSF)

    ctr = pipelineCLF_Sens['preprocessor']
    OHEcat = ctr.named_transformers_['cat'].get_feature_names_out().tolist()
    NewFeature = numvars + OHEcat + ['deltaProb']

    for k, dictR in dict.items():
        if k != model:
            continue
        print(f"Dataset: {dataset}, model: {k}")
        path = dictR[SF]
        risultati = get_result_from_pickle(path)
        # dfCF = pd.DataFrame([],columns=x_test.columns.to_list())
        # dfCFflipped = pd.DataFrame([], columns=NewFeature)
        dfCFFlippedlist = []
        for i in tqdm(risultati[cFtype]['sample_CF']):
            sample, y_real, result, y_sens, y_CF_sens, CF = i
            if result in [0,1]:
                if y_real[SF].item() == 1:
                    FlipSF = 0
                else:
                    FlipSF = 1
                if y_real[SF].item() == pipelineCLF_Sens.predict(sample):
                    Prob_x = pipelineCLF_Sens.predict_proba(sample)[:, 1]
                    n_sample = sample[numvars].values
                    sample = ctr.transform(sample)
                    sample[:, :len(numvars)] = n_sample
                    sample = np.hstack([sample[0], Prob_x])
                    y_CF_sens = pipelineCLF_Sens.predict(CF)
                    # nCF = CF[~(y_CF_sens == 1)]
                    CF = CF[y_CF_sens == FlipSF]
                    if CF.shape[0] > 0:
                        Prob_c_x = pipelineCLF_Sens.predict_proba(CF)[:, 1]
                        num_CF = CF[numvars].values
                        CF = ctr.transform(CF)
                        CF[:, :len(numvars)] = num_CF
                        CF = np.hstack([CF, Prob_c_x.reshape(-1, 1)])
                        epsilon = CF - sample
                        epsilon = pd.DataFrame(epsilon, columns=NewFeature)
                        dfCFFlippedlist.append(epsilon)
        dfCFflipped = pd.concat(dfCFFlippedlist)
        PearsonCorr = dfCFflipped.corr()
        # PearsonCorr.style.background_gradient(cmap='coolwarm').set_precision(3)
        print(PearsonCorr['deltaProb'].reindex(PearsonCorr['deltaProb'].abs().sort_values(ascending=False).index))
        bar = PearsonCorr['deltaProb'].reindex(PearsonCorr['deltaProb'].abs().sort_values(ascending=False).index)
        pos_Cor = (bar.values[1:].reshape(1, bar.shape[0] - 1).flatten() > 0)[::-1]
        neg_Cor = (bar.values[1:].reshape(1, bar.shape[0] - 1).flatten() < 0)[::-1]
        rgba_colors = np.zeros((bar.shape[0] - 1, 4))
        rgba_colors[:,0] = 0.92968
        rgba_colors[pos_Cor, 2] = 0.10546  # value of blue intensity divided by 256
        rgba_colors[neg_Cor, 2] = 1
        # normilzed only for rgba_color_range
        normalizedBar = bar.abs()[1:].apply(lambda x: (x - 0) / (bar.abs()[1:].max() - 0) * (1 - 0.3) + 0.3)
        rgba_colors[:, -1] = normalizedBar.values[::-1].reshape(1, bar.shape[0] - 1).flatten()
        plt.figure(figsize=(9, 11))
        plt.style.use('seaborn-dark')
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.barh(y=bar.axes[0].to_list()[1:][::-1], width=bar.values[1:][::-1], color=rgba_colors)
        plt.savefig(f'figure/proxyFeature_{dataset}_{SF}_{cFtype}_clf{model}_SFclf{modelSF}.svg', bbox_inches="tight")
        plt.xlabel(f"$\\rho(\epsilon,\delta)$")
        plt.axis('on')
        plt.show()
        plt.close()

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
