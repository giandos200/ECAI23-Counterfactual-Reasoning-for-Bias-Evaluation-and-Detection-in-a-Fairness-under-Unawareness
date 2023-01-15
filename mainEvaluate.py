from src.utils.CFoppMetrics import CFmetrics
import os

if __name__ == '__main__':
    for dataset in os.listdir('Results'):
        Results = {}
        for model in os.listdir(f'Results/{dataset}'):
            if os.path.isdir(f'Results/{dataset}/{model}'):
                Results[model] = ''
                for SF in os.listdir(f'Results/{dataset}/{model}'):
                    if os.path.isdir(f'Results/{dataset}/{model}/{SF}'):
                        Results[model] = f'Results/{dataset}/{model}/{SF}/Genetic.pickle'
        CFmetrics(Results, 'genetic', SF)