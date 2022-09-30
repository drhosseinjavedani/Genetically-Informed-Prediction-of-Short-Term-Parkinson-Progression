## Code for Genetically-Informed Prediction of Short-Term Parkinson’s Disease Progression 
Parkinson’s disease (PD) is a slowly progressing neurodegenerative disorder. Available treatments modify the symptoms of PD but do not halt its progression, which is characterized by varied motor and non-motor early symptoms and changes overtime. The heterogenous nature of PD presentation and progression hampers clinical research, resulting in long and expensive clinical trials prone to failure. Prediction of short-term PD progression could be useful for informing individual-level disease management, as well as shortening the time required to detect disease-modifying drug effects in clinical studies. 

Methods: PD progressors were defined by an increase in the Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS) at 12-, 24-, and 36- months post-baseline, after adjustment for treatment. Using only baseline features, PD progression was separately predicted across all timepoints and MDS-UPDRS subparts (part I (non-motor), II (motor), and III (motor physician exam)) in an optimized XGBoost framework. These progression predictions were combined into a meta-predictor for 12-month MDS UPDRS Total progression. Data from the Parkinson's Progression Markers Initiative (PPMI) were used for training with independent testing on the Parkinson's Disease Biomarkers Program (PDPB) cohort. 

Results: 12-month PD progression, as measured by an increase in MDS-UPDRS Total, was predicted with an F-measure 0.77, ROC AUC of 0.77, and PR AUC of 0.76 when tested on a hold-out PPMI set. When tested on PDPB (without the inclusion of neuroimaging data) we achieve a F-measure 0.75, ROC AUC of 0.74, and PR AUC of 0.73. Exclusion of genetic predictors led to the greatest loss in predictive accuracy; ROC AUC of 0.66, PR AUC of 0.66-0.68 for both hold-out PPMI and independent PDPB tests. 

Conclusions: Short-term PD progression can be well predicted with a combination of survey-based, neuroimaging, physician examination, and genetic predictors. Physician examination and polygenic risk scores provide the greatest predictive value. Dissection of the interplay between genetic risk, motor symptoms, non-motor symptoms, and longer-term expected rates of progression enable generalizable predictions. These predictions may enhance the efficiency of clinical trials by enriching them with individuals likely to demonstrate disease progression over the duration of a short clinical trial.

## Steps for using this project code

### Data 

1. Put [AMD-PD ](https://amp-pd.org/)
 datasets in /src/src/datasets folder (Only PDBP and PPMI)
2. Adjust data path in src/src/my_confs/conf_data_engineering

**_NOTE:_**  Data should be in **CSV** format.


### Install required packages

1. Create a python virtual environment, e.g.,

```
python3 -m venv .venv

```
2. Activate it, for example in POSIX
```
source activate/bin/.venv
```

Read more about venv in [Creation of virtual environments ](https://docs.python.org/3/library/venv.html
) 

3. install packages
```
python3 -m pip install -r requirements.txt
```

**_NOTE:_**  This project was only tested for Python 3.10.0, 3.8.5, and 3.9.5.


### Generate training sets

1. Include or exclude data and variables in src/src/my_confs/conf_build_train.py conf file.
2. Use functions train_builder_runner() or multiple_train_builder_runner() with proper arguments to generate all required training sets.

**_NOTE:_**  train_builder_runner() generate only one train data, and multiple_train_builder_runner() generate all.


### Training model and get results

1. From src/src/models/runner_model_auto.py use runner() or runner_multiple() with
proper arguments.


**_NOTE:_**  runner() can be used for case-by-case training; however, runner_multiple() can be used for all.

### Training meta predictor model and get results

After training models in the former step and generating all meta models.

1. Define the type of meta-features in src/src/my_confs/conf_build_model.py 
2. Run runner() from src/src/models/runner_model_meta_auto.py

### Gerneral configurations for envoirment variables

For general configurations of project use src/.env.


### Contact information

1. Dr. Hossein Javedani Sadaei at  <h.javedani@gmail.com>, <hjavedani@scripps.edu>
2. Prof. Ali Torkamani at <atorkama@scripps.edu>




