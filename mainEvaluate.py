from src.utils.CFoppMetrics import CFmetrics

import os
import sys

if __name__ == '__main__':
    pathResult = sys.argv
    if pathResult.__len__()>4:
        dataset = pathResult[1].capitalize()
        SF = pathResult[2]
        CFstrategy = pathResult[3]
        SFclf = pathResult[4].upper()
        Results = {}
        for model in os.listdir(f'Results/{dataset}'):
            if os.path.isdir(f'Results/{dataset}/{model}'):
                Results[model] = dict()
                for currSF in os.listdir(f'Results/{dataset}/{model}'):
                    if currSF.lower() != SF:
                        continue
                    else:
                        if os.path.isdir(f'Results/{dataset}/{model}/{SF}'):
                            Results[model][SF] = [f'Results/{dataset}/{model}/{SF}/Genetic.pickle',
                                              f'Results/{dataset}/{model}/{SF}/KDtree.pickle']

        print(f'Evaluating CFlips and DeltaCFlips for {dataset.upper()}, with {SF}'
              f'as sensitive information and XGB as sensitive feature classifier!')
        CFmetrics(Results, dataset, SF, CFstrategy,SFclf)


    else:
        for dataset in os.listdir('Results'):
            Results = {}
            sf = list()
            for model in os.listdir(f'Results/{dataset}'):
                if os.path.isdir(f'Results/{dataset}/{model}'):
                    Results[model] = dict()
                    for SF in os.listdir(f'Results/{dataset}/{model}'):
                        if SF not in sf:
                            sf.append(SF)
                        if os.path.isdir(f'Results/{dataset}/{model}/{SF}'):
                            Results[model][SF] = [f'Results/{dataset}/{model}/{SF}/Genetic.pickle',
                                                  f'Results/{dataset}/{model}/{SF}/KDtree.pickle']
            for SF in sf:
                print(f'Evaluating CFlips and DeltaCFlips for {dataset.upper()}, with {SF}'
                      f'as sensitive information and XGB as sensitive feature classifier!')
                for SFclf in ['XGB', 'RF', 'MLP']:
                    CFmetrics(Results, dataset, SF, 'genetic',SFclf)
                    CFmetrics(Results, dataset, SF, 'KDtree', SFclf)