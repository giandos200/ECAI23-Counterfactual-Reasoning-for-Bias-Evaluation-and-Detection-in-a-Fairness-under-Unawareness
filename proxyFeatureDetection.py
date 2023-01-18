from src.utils.CFoppMetrics import ProxyFeatureDetection

import os
import sys

if __name__ == '__main__':
    pathResult = sys.argv
    #pathResult.extend(['Adult-debiased', 'gender', 'MLP', 'genetic', 'MLP'])
    if pathResult.__len__()>5:
        dataset = pathResult[1].capitalize()
        SF = pathResult[2]
        model = pathResult[3].upper()
        CFstrategy = pathResult[4]
        SFclf = pathResult[5].upper()
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
                        else:
                            Results[model][SF] = f'Results/{dataset}/{model}/{SF}/KDtree.pickle'

        print(f'Detecting ProxyFeature for {dataset.upper()}, with {SF} as sensitive information, {model} as '
              f'decision boundary, and {CFstrategy} for generating counterfactual, and' 
              f'{SFclf} as sensitive feature classifier!')
        ProxyFeatureDetection(SFclf, dataset, model, Results, CFstrategy, SF)