from src.utils.dataloader import dataLoader
import pandas as pd
from sklearn.manifold import TSNE
from collections import Counter
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import random
random.seed(42)
import matplotlib.pyplot as plt
#plt.style.use("ggplot")

def get_result_from_pickle(path):
    with open(path,"rb") as file:
        return pickle.load(file)

dataset = 'Adult'

SF = 'gender'

Results = {'LR': f"Results/{dataset}/LR/{SF}/Genetic.pickle",
               "SVM": f"Results/{dataset}/SVM/{SF}/Genetic.pickle",
               'LGB': f"Results/{dataset}/LGB/{SF}/Genetic.pickle",
               'XGB': f"Results/{dataset}/XGB/{SF}/Genetic.pickle",
               'Debiased': f"Results/{dataset}/AdvDeb/{SF}/Genetic.pickle",
               'lferm': f"Results/{dataset}/LFERM/{SF}/Genetic.pickle"}

def plotTSNE(n_components, X_Neg, equal, opposite, gender, title):
    fig = plt.figure()
    #fig = plt.figure(figsize=(20, 15), constrained_layout=True)

    if gender == 'male':
        opp = 'female'
        color = 'b'
        color2 = 'r'
    else:
        opp = 'male'
        color = 'r'
        color2 = 'b'
    if n_components==2:
        plt.scatter(X_Neg[:, 0], X_Neg[:, 1], c=color, marker="o", edgecolor='black', linewidth=1.5,s = 50, label=f'Negative {gender} sample')
        plt.scatter(equal[:, 0], equal[:, 1], c=color, marker="+", label=f'Positive CF {gender} samples')
        plt.scatter(opposite[:, 0], opposite[:, 1], c=color2, marker="+", label=f'Positive CF {opp} samples')
    elif n_components==3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_Neg[:, 0], X_Neg[:, 1], X_Neg[:, 2], c=color, marker="o",edgecolor='black', linewidth=1.5, s = 100, label=f'Negative {gender} sample')
        ax.scatter(equal[:, 0], equal[:, 1], equal[:, 2], c=color, marker="+", s=70, label=f'Positive CF {gender} samples')
        ax.scatter(opposite[:, 0], opposite[:, 1], opposite[:, 2], c=color2, marker="+", s=70, label=f'Positive CF {opp} samples')

    plt.grid(True)
    # plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f'figure/{title}.svg', bbox_inches='tight')
    # import tikzplotlib
    # tikzplotlib.save(f"{title}.tex")
    # plt.savefig('1.png', bbox_inches='tight')
    plt.show()
    plt.close

if __name__=='__main__':

    n_components = 3 # 2 or 3

    dataF, target, SF, CF, numvars, categorical = dataLoader('adult-gender-biased')

    title = str(n_components)+'D TSNE'
    for Investigation in ['XGB','Debiased']:
        risultati = get_result_from_pickle(Results[Investigation])
        for Sensitive,s in [('male',1),('female',0)]:
            df = dataF.copy()
            title_case = ' '.join([title,Investigation.upper(),Sensitive,'f(x)=0'])
            Rlist = risultati['genetic']['sample_CF']
            random.shuffle(Rlist)
            for u in tqdm(Rlist):
                sample, y_real, result, y_sens, y_CF_sens, CF = u
                if result.item()==1:
                    continue
                else:
                    if y_real['gender'].item()==s and y_sens.item()==s:
                        count = Counter(y_CF_sens)
                        oppos = int(1-s)
                        if Investigation in ['Debiased','lferm']:
                            if 45<count[oppos]<55:
                                break
                        else:
                            if s==1:
                                if count[s]>80:
                                    break
                            elif count[oppos]>80:
                                break
            for col in categorical:
                ohe = LabelEncoder()
                ohe.fit(df[col])
                df[col] = ohe.transform(df[col])
                sample[col] = ohe.transform(sample[col])
                CF[col] = ohe.transform(CF[col])

            zscore = StandardScaler()
            #zscore.fit(df)
            #sample = pd.DataFrame(zscore.transform(sample), columns=sample.columns)
            #CF = pd.DataFrame(zscore.transform(CF), columns=CF.columns)

            tsne = TSNE(n_components=n_components, method = 'exact', n_iter=10000, verbose=10, n_jobs=-1, random_state=42)
            # X_embedded = tsne.fit(df)
            X_tot = pd.concat([sample, CF])
            X_tot = tsne.fit_transform(X_tot)
            # X_Neg = tsne.fit_transform(sample)
            # CF_x =tsne.fit_transform(CF)
            X_tot = zscore.fit_transform(X_tot)
            X_Neg = X_tot[0:1]
            CF_x = X_tot[1:]

            equal = CF_x[y_CF_sens == y_sens.item()]
            opposite = CF_x[~(y_CF_sens == y_sens.item())]

            plotTSNE(n_components, X_Neg, equal, opposite, Sensitive, title_case)