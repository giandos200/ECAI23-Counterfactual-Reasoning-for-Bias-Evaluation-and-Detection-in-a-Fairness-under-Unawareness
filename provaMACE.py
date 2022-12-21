import sys
import yaml
from src.model.CFgeneration import GenerateCF


if __name__ == '__main__':
    print(sys.argv)
    pathConfig = 'config/Adult/config-adult-gender.yml'
    with open(f"{pathConfig}", 'r') as stream:
        try:
            file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    superPipe_generation = GenerateCF(file)
    superPipe_generation.generate()