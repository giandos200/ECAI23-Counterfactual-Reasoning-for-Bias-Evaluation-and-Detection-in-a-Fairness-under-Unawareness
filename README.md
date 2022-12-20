# IJCAI23


## Official repository of Counterfactual Fair Opportunity (Poster) - The Web Conference 2023 (WWW '23)

Authors: Giandomenico Cornacchia*, Vito Walter Anelli, Fedelucio Narducci, and Azzurra Ragone; 

<p float="left">
  <img src="figure/3D TSNE XGB male f(x)=0(1).svg" width="200" />
  <img src="figure/3D TSNE XGB female f(x)=0(1).svg" width="200" />
  <img src="figure/3D TSNE DEBIASED male f(x)=0(1).svg" width="200" /> 
  <img src="figure/3D TSNE DEBIASED female f(x)=0(1).svg" width="200" />
  </p>

## Setup Instructions

Following, the instruction to install the correct packages for running the experiments. run:
```bash
#(Python version used during experiments 3.8.10)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Training, test models and generate CF samples
To train, evaluate models, and generate counterfactual samples for each dataset and Classifier, run:

```bash
#sudo chmod +x run_all_generation.sh + User_password if Permission Denied 
./run_all_generation.sh
```
or chose one of the config file for a specific dataset, and run:
```bash
python3 -u mainGenerate.py {config/{dataset}/...}.yml
```

Result can be found in the folder Results/{dataset}/{model}/{SF}/Genetic.pickle

N.B.: Experiments can take several days for each dataset and model. To speed up inference in the Model_Y and Counterfactual Generator loop for some scikit-learn models (e.g. SVM), the ["skearnex"](https://github.com/intel/scikit-learn-intelex) library from intel was used. The metrics' results may vary slightly from the scikit-learn models.

### Counterfactual metrics evaluation
To evaluate the proposed metric for each dataset, simply run the following command:

```bash
python3 -u mainEvaluate.py
```

The script will display Counterfactual Fairness metrics in Table 3.

### Generate toy figures

To generate Figure 1, run the following command:

```bash
python3 -u T-SNE.py
```

Image are saved in folder ./figures/3D TSNE {model} {unpriv/priv} f(x)=0.svg

N.B.: Images can be slightly different, based on seed, python and libraries versions. Neverthless, they are only toys image that will return the same conceptual findings.