import os
import sys
import pickle

if __name__ == '__main__':
    pathResult = sys.argv
    pathResult.extend(['Adult-debiased', 'gender', 'LR', 'MACE'])
    if pathResult.__len__()>4:
        dataset = pathResult[1].capitalize()
        SF = pathResult[2]
        model = pathResult[3].upper()
        CFstrategy = pathResult[4]
        Results = {}
        if os.path.isdir(f'Results/{dataset}/{model}'):
            Results[model] = dict()
            for currSF in os.listdir(f'Results/{dataset}/{model}'):
                if currSF.lower() != SF:
                    continue
                else:
                    if os.path.isdir(f'Results/{dataset}/{model}/{SF}'):
                        if CFstrategy == 'genetic':
                            Results[model][SF] = f'Results/{dataset}/{model}/{SF}/Genetic.pickle'
                        elif CFstrategy.lower() == 'mace':
                            Results[model][SF] = f'Results/{dataset}/{model}/{SF}/Mace.pickle'
                        else:
                            Results[model][SF] = f'Results/{dataset}/{model}/{SF}/KDtree.pickle'

        with open(Results[model][SF], "rb") as file:
            Risultati = pickle.load(file)
        NCF = 0

        NtotCFY0Unprotected = 0 + 1e-10
        NCFY0Unprotected = 0
        NtotCFY1Protected = 0 + 1e-10
        NCFY1Protected = 0
        for Cftype in Risultati.keys():
            NCF = 0
            maxCF = 0
            median = []
            dictCF = {i: 0 for i in range(101)}
            NtotCFY0Unprotected = 0 + 1e-10
            NCFY0Unprotected = 0
            NtotCFY1Protected = 0 + 1e-10
            NCFY1Protected = 0
            if Cftype == 'MACE':
                N_Sample_with_CF = Risultati[Cftype].__len__()
                R = Risultati[Cftype]
            else:
                N_Sample_with_CF = Risultati[Cftype]['sample_CF'].__len__()
                R = Risultati[Cftype]['sample_CF']
            print("CF Strategy Type: {}".format(Cftype))
            for i in R:
                sample, y_real, result, y_sens, y_CF_sens, CF = i
                if CF.shape[0] > 100:
                    CF = CF[:100]
                NCF += CF.shape[0]
                maxCF = max([maxCF, CF.shape[0]])
                median.append(CF.shape[0])
                dictCF[CF.shape[0]] += 1
                if result == 0 and (y_sens == 0) and (y_real[SF].item()) == 0:
                    NtotCFY0Unprotected += 1
                    NCFY0Unprotected += CF.shape[0]
                if result == 0 and (y_sens == 1) and (y_real[SF].item()) == 1:
                    NtotCFY1Protected += 1
                    NCFY1Protected += CF.shape[0]
            median.sort()

            print("N° of total CF for a sample: {}\n".format(NCF))
            print("Max number of CF generated: {}\n".format(maxCF))
            print("median of CF generated: {}\n".format(median[median.__len__() // 2] if median.__len__() > 0 else 0))
            print("mean of CF generated: {}\n".format(NCF / N_Sample_with_CF if NCF > 0 else 0))
            print("N° Sample with at least 1 CF: {}\n".format(N_Sample_with_CF))
            print("N° of CF with Y=0 and Y_sens=0: {}\n".format(NCFY0Unprotected))
            print("N° of CF with Y=0 and Y_sens=1: {}\n".format(NCFY1Protected))
            print("\n\n")
            # print(dictCF)