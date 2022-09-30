# Conf file creating machine learning pipelines
###################################################
###################################################
###################################################
###################################################

import os
import xgboost
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from zoish.feature_selectors.optunashap import OptunaShapFeatureSelector
from lohrasb.best_estimator import BaseModel
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv

# Read environment variables from .env file
load_dotenv()
ALGORITHM= os.environ.get("ALGORITHM")
MODEL = os.environ.get("MODEL")
FEATURE_SELECTOR = os.environ.get("FEATURE_SELECTOR")

# set error metrics 
XGB_CLS_METRIC = os.environ.get("XGB_CLS_METRIC")
LG_CLS_METRIC = os.environ.get("LG_CLS_METRIC")

# classification models XGBoost and
# Logistic Regression
XGB_CLS_MODEL=None
XGB_CLS_SHAP=None
LG_CLS_MODEL=None


def model_builder(estimator,estimator_params,metric):
    obj = BaseModel.bestmodel_factory.using_optuna(
            estimator=estimator,
            estimator_params=estimator_params,
            measure_of_accuracy=metric,
            verbose=3,
            n_jobs=-1,
            random_state=42,
            # optuna params
            # optuna study init params
            study=optuna.create_study(
                storage=None,
                sampler=TPESampler(),
                pruner=HyperbandPruner(),
                study_name=None,
                direction="maximize",
                load_if_exists=False,
                directions=None,
            ),
            # optuna optimization params
            study_optimize_objective=None,
            study_optimize_objective_n_trials=10,
            study_optimize_objective_timeout=600,
            study_optimize_n_jobs=-1,
            study_optimize_catch=(),
            study_optimize_callbacks=None,
            study_optimize_gc_after_trial=False,
            study_optimize_show_progress_bar=False,
        )
    return obj


XGB_PARAMS = {
            "gamma": [0, 5],
            "learning_rate": [0.1, 0.01],
            "max_depth": [3, 30],
            "n_estimators": [100, 1000],
            "scale_pos_weight": ["XGB_CLS_SCALE_POS_WEIGHT_LOW",\
                "XGB_CLS_SCALE_POS_WEIGHT_UP"],
            #"n_jobs": [-1],
        }


LG_PARAMS = {
            #"penalty": ["l2"],
            #"dual": [False],
            #"tol": [1e-4],
            #"C": [1, 1.5],
            #"fit_intercept": [True,False],
            #"intercept_scaling": [1],
            # "class_weightt": [None],
            #"random_state": [0],
            #"solver": ["lbfgs"],
            "max_iter": [100],
            "multi_class": ["auto"],
            # "verbose": [0],
            #"warm_start": [False],
            #"n_jobs": [-1],
            # "l1_ratio": [None],
        }


if MODEL == "XGB":
    # engine for predictive models
    # check this package
    # https://github.com/drhosseinjavedani/lohrasb

    # For XGBoost
    if MODEL == "XGB":
        XGB_PARAMS["objective"] = ["binary:logistic"]
        XGB_CLS_MODEL=model_builder(
                estimator = xgboost.XGBClassifier(),
                estimator_params = XGB_PARAMS,
                metric = XGB_CLS_METRIC)
    
    # For Logistic Regression
    if MODEL == "LG":
        LG_CLS_MODEL=model_builder(
            estimator = LogisticRegression(),
            estimator_params = LG_PARAMS,
            metric = LG_CLS_METRIC)



#################################################
#################################################


if FEATURE_SELECTOR == "XGB":
   
    # engine for selecting feature using
    # shap values
    # check this package
    # https://github.com/drhosseinjavedani/zoish
    XGB_CLS_SHAP = OptunaShapFeatureSelector(
        verbose=1,
        random_state=0,
        logging_basicConfig=None,
        # general argument setting
        n_features=25,
        list_of_obligatory_features_that_must_be_in_model=[
            #'on_other_pd_medications',
            #'on_levodopa',
            #'on_dopamine_agonist',
        ],
        list_of_features_to_drop_before_any_selection=[],
        # shap argument setting
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "objective": ["binary:logistc"],
            # "base_score": [None],
            # "booster": [None],
            # "colsample_bylevel": [None],
            # "colsample_bynode": [None],
            # "colsample_bytree": [None],
            # "enable_categorical": [False],
            "eval_metric": ['auc'],
            "gamma": [0.0, 1.0],
            # "grow_policy": [None],
            # "learning_rate": [0.01, 0.1],
            "max_bin": [2, 20],
            # "max_delta_step": [None],
            "max_depth": [3, 30],
            # "max_leaves": [None],
            # "min_child_weight": [0, 10],
            # "n_estimators": [100, 1000],
            # "n_jobs": [-1],
            # "num_parallel_tree": [None],
            # "predictor": [None],
            # "random_state": [0],
            "reg_alpha": [0, 10],
            "reg_lambda": [0, 10],
            # "sampling_method": [None],
            "scale_pos_weight": [XGB_CLS_SCALE_POS_WEIGHT_LOW,XGB_CLS_SCALE_POS_WEIGHT_UP],
            # "subsample": [1.0],
            # "tree_method": [None],
            # "validate_parameters": [None],
            # "verbosity": [None],
            # 'num_class' :[3]
        },
        # shap arguments
        model_output="raw",
        feature_perturbation="interventional",
        algorithm="auto",
        shap_n_jobs=-1,
        memory_tolerance=-1,
        feature_names=None,
        approximate=False,
        shortcut=False,
        plot_shap_summary=False,
        save_shap_summary_plot=False,
        path_to_save_plot="./summary_plot.png",
        shap_fig=plt.figure(),
        ## optuna params
        test_size=0.33,
        with_stratified=True,
        performance_metric="f1_multi",
        # optuna study init params
        study=optuna.create_study(
            storage=None,
            sampler=TPESampler(),
            pruner=HyperbandPruner(),
            study_name=None,
            direction="maximize",
            load_if_exists=False,
            directions=None,
        ),
        study_optimize_objective_n_trials=100,
    )

