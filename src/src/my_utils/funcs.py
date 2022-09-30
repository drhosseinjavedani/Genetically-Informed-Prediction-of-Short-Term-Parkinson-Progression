# Varius helper functions that use in different classes 
######################################################
######################################################
######################################################
######################################################

import os
import sys
from os import listdir, remove
import category_encoders as cen
import numpy as np
import optuna
import pandas as pd
import xgboost
import xgboost as xgb
from feature_engine import transformation as vt
from feature_engine.imputation.categorical import CategoricalImputer
from feature_engine.imputation.mean_median import MeanMedianImputer
from feature_engine.outliers.winsorizer import Winsorizer
from scipy.stats import ranksums
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.src.processors.feature_store import FeatureStore
from src.src.processors.pipeline_step import ScalerEncoder, Shap_parametric
from src.src.setup_logger import logger
from src.src.my_confs.conf_data_engineering import general_rules_features
from src.src.my_confs.conf_build_train import DATA_VERSION, ALGORITHM


def reset_pkl(files_wild_cards):
    """
    Reset pickle files
    """
    try:
        for f in listdir(general_rules_features["path_to_save"]):
            if files_wild_cards in str(f):
                print(
                    "reset_pkl is working and removing a file from",
                    general_rules_features["path_to_save"] + f,
                )
                remove(general_rules_features["path_to_save"] + str(f))

    except Exception as e:
        print(e)
    return

def ret_pip_param_for_opt(
    feature_store_path,
    float_cols,
    int_cols,
    cat_cols,
    n_features,
    RE_CAL_FEATURE,
    SCORE_TYPE,
    NUMBER_OF_TRIALS,
    lambda_par,
    alpha,
    subsample,
    colsample_bytree,
    scale_pos_weight,
    max_depth,
    min_child_weight,
    eta,
    gamma,
    grow_policy,
    sample_type,
    normalize_type,
    rate_drop,
    skip_drop,
    label,
    test_size,
    sampler,
    pruner,
    eval_metric,
    n_components,
):
    """
    return parameters for optimization
    """

    all_dict_pipe = {
        "imputers": [
            # float missing values imputers
            (
                "floatimputer",
                MeanMedianImputer(
                    imputation_method="mean", variables=float_cols
                ),
            ),
            # int missing values imputers
            (
                "intimputer",
                MeanMedianImputer(
                    imputation_method="median", variables=int_cols
                ),
            ),
            # category missing values imputers
            (
                "catimputer",
                CategoricalImputer(
                    imputation_method="missing", variables=cat_cols
                ),
            ),
        ],
        "transformers_general": [
            # categorical encoders
            ("order", cen.OrdinalEncoder()),
            # ('order', cen.HelmertEncoder()),
            # outlier resolvers
            ("win", Winsorizer()),
            # transformers
            ("yeo", vt.YeoJohnsonTransformer()),
            # feature problem resolvers
            # ('drop_duplicat',fts.DropDuplicateFeatures()),
            # numberic scaler
            ("standard", ScalerEncoder()),
        ],
        "feature_selector_shap": [
            # # feature selectors
            (
                "shap",
                Shap_parametric(
                    n_features=n_features,
                    model=xgboost.XGBClassifier(),
                    list_of_features=[],
                    params=None,
                    retrain=RE_CAL_FEATURE,
                    feature_store_path=feature_store_path,
                    score_type=SCORE_TYPE,
                    number_of_trials=NUMBER_OF_TRIALS,
                    lambda_par=lambda_par,
                    alpha=alpha,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    scale_pos_weight=scale_pos_weight,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    eta=eta,
                    gamma=gamma,
                    grow_policy=grow_policy,
                    sample_type=sample_type,
                    normalize_type=normalize_type,
                    rate_drop=rate_drop,
                    skip_drop=skip_drop,
                    label=label,
                    test_size=test_size,
                    sampler=sampler,
                    pruner=pruner,
                    eval_metric=eval_metric,
                ),
            )
        ],
        "remove_noise_from_samples": [
            # PCA
            ("pca", PCA(n_components=n_components))
        ],
    }

    SHAP_PIP = Pipeline(
        all_dict_pipe["imputers"]
        + all_dict_pipe["transformers_general"]
        + all_dict_pipe["feature_selector_shap"]
        # +
        # all_dict_pipe['remove_noise_from_samples']
    )

    return SHAP_PIP


def return_best_params(
    XT_train,
    y_train,
    SCORE_TYPE,
    NUMBER_OF_TRIALS,
    lambda_par,
    alpha,
    subsample,
    colsample_bytree,
    scale_pos_weight,
    max_depth,
    min_child_weight,
    eta,
    gamma,
    grow_policy,
    sample_type,
    normalize_type,
    rate_drop,
    skip_drop,
    label,
    sampler,
    pruner,
    eval_metric,
):
    """
    return best params 
    """
    def objective(trial):

        train_x, valid_x, train_y, valid_y = train_test_split(
            XT_train, y_train, stratify=y_train[label], test_size=0.25
        )

        dtrain = xgb.DMatrix(train_x, label=train_y)
        dvalid = xgb.DMatrix(valid_x, label=valid_y)

        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            # use exact for small dataset.
            "tree_method": "exact",
            "eval_metric": eval_metric,
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree"]),
            # L2 regularization weight.
            "lambda": trial.suggest_float(
                "lambda", lambda_par[0], lambda_par[1], log=True
            ),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", alpha[0], alpha[1], log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float(
                "subsample", subsample[0], subsample[1]
            ),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", colsample_bytree[0], colsample_bytree[1]
            ),
            "scale_pos_weight": trial.suggest_loguniform(
                "scale_pos_weight", scale_pos_weight[0], scale_pos_weight[1]
            ),
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree signifies the complexity of the tree.
            param["max_depth"] = trial.suggest_int(
                "max_depth", max_depth[0], max_depth[1], step=1
            )
            # minimum child weight, the larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int(
                "min_child_weight", min_child_weight[0], min_child_weight[1]
            )
            param["eta"] = trial.suggest_float("eta", eta[0], eta[1], log=True)
            # defines how selective the algorithm is.
            param["gamma"] = trial.suggest_float(
                "gamma", gamma[0], gamma[1], log=True
            )
            param["grow_policy"] = trial.suggest_categorical(
                grow_policy[0], grow_policy[1]
            )

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical(
                sample_type[0], sample_type[1]
            )
            param["normalize_type"] = trial.suggest_categorical(
                normalize_type[0], normalize_type[1]
            )
            param["rate_drop"] = trial.suggest_float(
                "rate_drop", rate_drop[0], rate_drop[1], log=True
            )
            param["skip_drop"] = trial.suggest_float(
                "skip_drop", skip_drop[0], skip_drop[1], log=True
            )

        # Add a callback for pruning.
        if ALGORITHM=='cls':
            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "validation-" + eval_metric
            )
            bst = xgb.train(
                param,
                dtrain,
                evals=[(dvalid, "validation")],
                callbacks=[pruning_callback],
            )
    
        preds = bst.predict(dvalid)
        pred_labels = np.rint(preds)
        classification_r = classification_report(valid_y, pred_labels)
        f1 = f1_score(valid_y, pred_labels)
        acc = accuracy_score(valid_y, pred_labels)
        pr = precision_score(valid_y, pred_labels)
        recall = recall_score(valid_y, pred_labels)
        roc = roc_auc_score(valid_y, pred_labels)
        tn, fp, fn, tp = confusion_matrix(
            valid_y, pred_labels, labels=[0, 1]
        ).ravel()
        print(classification_r)

        if SCORE_TYPE == "f1":
            accr = f1
        if SCORE_TYPE == "acc":
            accr = acc
        if SCORE_TYPE == "pr":
            accr = pr
        if SCORE_TYPE == "recall":
            accr = recall
        if SCORE_TYPE == "roc":
            accr = roc
        if SCORE_TYPE == "tn":
            accr = tn
        if SCORE_TYPE == "tp":
            accr = tp
        if SCORE_TYPE == "f1+acc":
            accr = f1 + acc

        return accr

    if isinstance(sampler, str):
        sampler = eval(sampler)
    if isinstance(pruner, str):
        pruner = eval(pruner)

    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner
    )
    study.optimize(objective, n_trials=NUMBER_OF_TRIALS, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial.params


def make_slopes(df, year, type):
    """
    Create slopes and labels
    """

    algo = ALGORITHM

    if year == "12":

        if type == "i":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["mo0nth12"] - df["mo0nth0"]
            max_slope=df_slope["slope1"].quantile(slop_cut)
            if algo =='cls' :
                df_slope["label_i"]=df_slope['slope1'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_i"] = df_slope["label_i"]
        if type == "ii":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["Mo1nth12"] - df["Mo1nth0"]
            max_slope=df_slope["slope1"].quantile(slop_cut)
            if algo =='cls' :
                df_slope["label_ii"]=df_slope['slope1'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_ii"] = df_slope["label_ii"]
        if type == "iii":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["SMo2nth12"] - df["SMo2nth0"]
            max_slope=df_slope["slope1"].quantile(slop_cut)
            if algo =='cls' :
                df_slope["label_iii"]=df_slope['slope1'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_iii"] = df_slope["label_iii"]
        if type == "total":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["mo0nth12"] - df["mo0nth0"]
            df_slope["slope2"] = df["Mo1nth12"] - df["Mo1nth0"]
            df_slope["slope3"] = df["SMo2nth12"] - df["SMo2nth0"]
            df_slope["slopetotal"] = (
                df_slope["slope1"] + df_slope["slope2"] + df_slope["slope3"]
            )
            max_slope=df_slope["slopetotal"].quantile(slop_cut)
            if algo =='cls' :
                df_slope["label_total"]=df_slope['slopetotal'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_total"] = df_slope["label_total"]

    if year == "18":
        if type == "i":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["mo0nth12"] - df["mo0nth0"]
            df_slope["slope1.5"] = df["mo0nth18"] - df["mo0nth06"]
            df_slope["slope"] = (
                df_slope["slope1"] + df_slope["slope1.5"] 
            )
            if algo =='cls':
                df_slope["label_i"]=df_slope['slope'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_i"] = df_slope["label_i"]
        if type == "ii":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["Mo1nth12"] - df["Mo1nth0"]
            df_slope["slope1.5"] = df["Mo1nth18"] - df["Mo1nth06"]
            df_slope["slope"] = (
                df_slope["slope1"] + df_slope["slope1.5"] 
            )
            if algo =='cls':
                df_slope["label_ii"]=df_slope['slope'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_ii"] = df_slope["label_ii"]
        if type == "iii":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["SMo2nth12"] - df["SMo2nth0"]
            df_slope["slope1.5"] = df["SMo2nth18"] - df["SMo2nth06"]
            df_slope["slope"] = (
                df_slope["slope1"] + df_slope["slope1.5"] 
            )
            if algo =='cls':
                df_slope["label_iii"]=df_slope['slope'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_iii"] = df_slope["label_iii"]

        if type == "total":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["mo0nth12"] - df["mo0nth0"]
            df_slope["slope1.5"] = df["mo0nth18"] - df["mo0nth06"]

            df_slope["slope3"] = df["Mo1nth12"] - df["Mo1nth0"]
            df_slope["slope3.5"] = df["Mo1nth18"] - df["Mo1nth06"]

            df_slope["slope5"] = df["SMo2nth12"] - df["SMo2nth0"]
            df_slope["slope5.5"] = df["SMo2nth18"] - df["SMo2nth06"]

            df_slope["slopetotal"] = (
                df_slope["slope1"]
                + df_slope["slope1.5"]
                + df_slope["slope3"]
                + df_slope["slope3.5"]
                + df_slope["slope5"]
                + df_slope["slope5.5"]
            )
            if algo =='cls':
                df_slope["label_total"]=df_slope['slopetotal'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_total"] = df_slope["label_total"]

    if year == "24":
        if type == "i":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["mo0nth12"] - df["mo0nth0"]
            df_slope["slope1.5"] = df["mo0nth18"] - df["mo0nth06"]
            df_slope["slope2"] = df["mo0nth24"] - df["mo0nth12"]
            df_slope["slope"] = (
                df_slope["slope1"] + df_slope["slope1.5"] + df_slope["slope2"]
            )
            if algo =='cls':
                df_slope["label_i"]=df_slope['slope'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_i"] = df_slope["label_i"]
        if type == "ii":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["Mo1nth12"] - df["Mo1nth0"]
            df_slope["slope1.5"] = df["Mo1nth18"] - df["Mo1nth06"]
            df_slope["slope2"] = df["Mo1nth24"] - df["Mo1nth12"]
            df_slope["slope"] = (
                df_slope["slope1"] + df_slope["slope1.5"] + df_slope["slope2"]
            )
            if algo =='cls':
                df_slope["label_ii"]=df_slope['slope'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_ii"] = df_slope["label_ii"]
        if type == "iii":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["SMo2nth12"] - df["SMo2nth0"]
            df_slope["slope1.5"] = df["SMo2nth18"] - df["SMo2nth06"]
            df_slope["slope2"] = df["SMo2nth24"] - df["SMo2nth12"]
            df_slope["slope"] = (
                df_slope["slope1"] + df_slope["slope1.5"] + df_slope["slope2"]
            )
            if algo =='cls':
                df_slope["label_iii"]=df_slope['slope'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_iii"] = df_slope["label_iii"]

        if type == "total":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["mo0nth12"] - df["mo0nth0"]
            df_slope["slope1.5"] = df["mo0nth18"] - df["mo0nth06"]
            df_slope["slope2"] = df["mo0nth24"] - df["mo0nth12"]

            df_slope["slope3"] = df["Mo1nth12"] - df["Mo1nth0"]
            df_slope["slope3.5"] = df["Mo1nth18"] - df["Mo1nth06"]
            df_slope["slope4"] = df["Mo1nth24"] - df["Mo1nth12"]

            df_slope["slope5"] = df["SMo2nth12"] - df["SMo2nth0"]
            df_slope["slope5.5"] = df["SMo2nth18"] - df["SMo2nth06"]
            df_slope["slope6"] = df["SMo2nth24"] - df["SMo2nth12"]

            df_slope["slopetotal"] = (
                df_slope["slope1"]
                + df_slope["slope1.5"]
                + df_slope["slope2"]
                + df_slope["slope3"]
                + df_slope["slope3.5"]
                + df_slope["slope4"]
                + df_slope["slope5"]
                + df_slope["slope5.5"]
                + df_slope["slope6"]
            )
            if algo =='cls':
                df_slope["label_total"]=df_slope['slopetotal'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_total"] = df_slope["label_total"]

    if year == "36":
        if type == "i":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["mo0nth12"] - df["mo0nth0"]
            df_slope["slope1.5"] = df["mo0nth18"] - df["mo0nth06"]
            df_slope["slope2"] = df["mo0nth24"] - df["mo0nth12"]
            df_slope["slope2.5"] = df["mo0nth30"] - df["mo0nth18"]
            df_slope["slope3"] = df["mo0nth36"] - df["mo0nth24"]
            df_slope["slope"] = (
                df_slope["slope1"]
                + df_slope["slope1.5"]
                + df_slope["slope2"]
                + df_slope["slope2.5"]
                + df_slope["slope3"]
            )
            if algo =='cls':
                df_slope["label_i"]=df_slope['slope'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_i"] = df_slope["label_i"]
        if type == "ii":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["Mo1nth12"] - df["Mo1nth0"]
            df_slope["slope1.5"] = df["Mo1nth18"] - df["Mo1nth06"]
            df_slope["slope2"] = df["Mo1nth24"] - df["Mo1nth12"]
            df_slope["slope2.5"] = df["Mo1nth30"] - df["Mo1nth18"]
            df_slope["slope3"] = df["Mo1nth36"] - df["Mo1nth24"]
            df_slope["slope"] = (
                df_slope["slope1"]
                + df_slope["slope1.5"]
                + df_slope["slope2"]
                + df_slope["slope2.5"]
                + df_slope["slope3"]
            )
            if algo =='cls':
                df_slope["label_ii"]=df_slope['slope'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_ii"] = df_slope["label_ii"]
        if type == "iii":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["SMo2nth12"] - df["SMo2nth0"]
            df_slope["slope1.5"] = df["SMo2nth18"] - df["SMo2nth06"]
            df_slope["slope2"] = df["SMo2nth24"] - df["SMo2nth12"]
            df_slope["slope2.5"] = df["SMo2nth30"] - df["SMo2nth18"]
            df_slope["slope3"] = df["SMo2nth36"] - df["SMo2nth24"]
            df_slope["slope"] = (
                df_slope["slope1"]
                + df_slope["slope1.5"]
                + df_slope["slope2"]
                + df_slope["slope2.5"]
                + df_slope["slope3"]
            )
            if algo =='cls':
                df_slope["label_iii"]=df_slope['slope'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_iii"] = df_slope["label_iii"]

        if type == "total":
            df_slope = pd.DataFrame()
            df_slope["slope1"] = df["mo0nth12"] - df["mo0nth0"]
            df_slope["slope1.5"] = df["mo0nth18"] - df["mo0nth06"]
            df_slope["slope2"] = df["mo0nth24"] - df["mo0nth12"]
            df_slope["slope2.5"] = df["mo0nth30"] - df["mo0nth18"]
            df_slope["slope3"] = df["mo0nth36"] - df["mo0nth24"]

            df_slope["slope4"] = df["Mo1nth12"] - df["Mo1nth0"]
            df_slope["slope4.5"] = df["Mo1nth18"] - df["Mo1nth06"]
            df_slope["slope5"] = df["Mo1nth24"] - df["Mo1nth12"]
            df_slope["slope5.5"] = df["Mo1nth30"] - df["Mo1nth18"]
            df_slope["slope6"] = df["Mo1nth36"] - df["Mo1nth24"]

            df_slope["slope7"] = df["SMo2nth12"] - df["SMo2nth0"]
            df_slope["slope7.5"] = df["SMo2nth18"] - df["SMo2nth06"]
            df_slope["slope8"] = df["SMo2nth24"] - df["SMo2nth12"]
            df_slope["slope8.5"] = df["SMo2nth30"] - df["SMo2nth18"]
            df_slope["slope9"] = df["SMo2nth36"] - df["SMo2nth24"]

            df_slope["slopetotal"] = (
                df_slope["slope1"]
                + df_slope["slope1.5"]
                + df_slope["slope2"]
                + df_slope["slope2.5"]
                + df_slope["slope3"]
                + df_slope["slope4"]
                + df_slope["slope4.5"]
                + df_slope["slope5"]
                + df_slope["slope5.5"]
                + df_slope["slope6"]
                + df_slope["slope7"]
                + df_slope["slope7.5"]
                + df_slope["slope8"]
                + df_slope["slope8.5"]
                + df_slope["slope9"]
            )
            if algo =='cls':
                df_slope["label_total"]=df_slope['slopetotal'].apply(lambda x: -1 if  pd.isnull(x) else (1 if x > 0 else 0) )
            df["label_total"] = df_slope["label_total"]

    return df


def uplift_scores(df, subject_list_yes, subject_list_no, score, month):
    """
    Medical adjustment function
    """
    df_2_copy = df.copy()
    df = df.copy()
    print(df.head())
    print(df.loc[df["month_of_visit"] == month].head())

    # df=df.loc[df['month_of_visit']==month]
    df_yes = df.loc[
        (
            df["subject_id"].isin(subject_list_yes) & (df["month_of_visit"])
            == month
        ),
        score,
    ]
    df_no = df.loc[
        (
            df["subject_id"].isin(subject_list_no) & (df["month_of_visit"])
            == month
        ),
        score,
    ]

    print(df_yes.head())
    print(df_no.head())

    if len(df_yes) > 0 and len(df_no) > 0:
        no_mean = df_no.mean()
        yes_mean = df_yes.mean()
        to_be_reducted = no_mean - yes_mean
        print(" the difference : ")
        print(to_be_reducted)
        print(" problem is  : ")
        print(score)
        print("mean before reduce is:")
        print(df.loc[(df["subject_id"].isin(subject_list_no)), score].mean())
        df.update(
            df.loc[(df["subject_id"].isin(subject_list_no)), score]
            - to_be_reducted
        )
        print("mean after reduce is:")
        print(df.loc[(df["subject_id"].isin(subject_list_no)), score].mean())

    else:
        print("one group has zero members")
    print(df_2_copy.columns.to_list() == df.columns.to_list())
    print(len(df_2_copy) == len(df))
    if df_2_copy.columns.to_list() != df.columns.to_list():
        raise ValueError("Adjusted datafrmae columns changed !")
    if len(df_2_copy) != len(df):
        raise ValueError("Adjusted datafrmae rows changed !")

    return df


def uplift_scores_all_together(
    df, cohort, drug_name, score, month, do_persist=True, name=None, test=False
):

    """
    Medical adjustment function
    """


    fs = FeatureStore()
    adj_coef = {}
    df_2_copy = df.copy()
    df = df.copy()
    # statistics ratio related to each time point
    print(
        df.loc[df["month_of_visit"] == month][drug_name].value_counts(
            dropna=False
        )
        / len(df.loc[df["month_of_visit"] == month])
        * 100
    )
    print(
        df.loc[df["month_of_visit"] == month][drug_name].value_counts(
            dropna=False
        )
        / len(df)
        * 100
    )

    df_yes = df.loc[(df["month_of_visit"] == month)]
    df_yes = df_yes.loc[(df_yes["subject_id"].str.startswith(cohort))]
    df_yes = df_yes.loc[(df_yes[drug_name] == "yes")]

    df_no = df.loc[(df["month_of_visit"] == month)]
    df_no = df_no.loc[(df_no["subject_id"].str.startswith(cohort))]
    df_no = df_no.loc[
        (df_no[drug_name] == "no")
        | (df_no[drug_name].isnull())
        | (df_no[drug_name] == np.nan)
        | (df_no[drug_name].isna())
    ]

    print(df_yes[["subject_id", score, "month_of_visit", drug_name]].head(100))
    print(df_no[["subject_id", score, "month_of_visit", drug_name]].head(100))

    if len(df_yes) > 0 and len(df_no) > 0:
        no_mean = df_no[score].mean()
        yes_mean = df_yes[score].mean()
        to_be_reducted = no_mean - yes_mean
        if not do_persist and not test:
            data_adj = fs.helper_load_from_data_save_path(
                name + "_" + month + "_" + score + ".pkl"
            )
            to_be_reducted = data_adj[(month, score)]
        print(f" the difference in {month} for {score} or {(month,score)} is: ")
        print(to_be_reducted)
        print(" problem is  : ")
        print(score)
        print("mean before reduce is:")
        print(
            df.loc[(df["subject_id"].isin(df_yes["subject_id"])), score].mean()
        )
        df.update(
            df.loc[(df["subject_id"].isin(df_no["subject_id"])), score]
            - to_be_reducted
        )
        print("mean after reduce is:")
        print(
            df.loc[(df["subject_id"].isin(df_no["subject_id"])), score].mean()
        )

    else:
        print("one group has zero members")
    print(df_2_copy.columns.to_list() == df.columns.to_list())
    print(len(df_2_copy) == len(df))
    if df_2_copy.columns.to_list() != df.columns.to_list():
        raise ValueError("Adjusted datafrmae columns changed !")
    if len(df_2_copy) != len(df):
        raise ValueError("Adjusted datafrmae rows changed !")

    adj_coef[(month, score)] = to_be_reducted
    if do_persist and not test:
        fs.helper_persist_to_data_save_path(
            adj_coef, name + "_" + month + "_" + score
        )

    return df



def find_in_med(
    df,
    month,
    drug_name,
    cohort_name,
    yes_or_no,
    ppmi_subjects,
    pdpb_subjects,
    bio_subjects,
    hbs_subjects,
    stpd3_subjects,
    supd3_subjects,
):

    df_copy = df.copy()

    if yes_or_no == "yes":

        if cohort_name == "pp":

            output_list = (
                df_copy.loc[(df_copy["month_of_visit"] == month)]
                .loc[df_copy["subject_id"].isin(ppmi_subjects)]
                .loc[(df_copy[drug_name] == "yes")]["subject_id"]
                .tolist()
            )

        if cohort_name == "pd":

            output_list = (
                df_copy.loc[(df_copy["month_of_visit"] == month)]
                .loc[df_copy["subject_id"].isin(pdpb_subjects)]
                .loc[(df_copy[drug_name] == "yes")]["subject_id"]
                .tolist()
            )

    if yes_or_no == "no":

        if cohort_name == "pp":

            output_list = (
                df_copy.loc[(df_copy["month_of_visit"] == month)]
                .loc[df_copy["subject_id"].isin(ppmi_subjects)]
                .loc[(df_copy[drug_name] == "no")]["subject_id"]
                .tolist()
            )

        if cohort_name == "pd":

            output_list = (
                df_copy.loc[(df_copy["month_of_visit"] == month)]
                .loc[df_copy["subject_id"].isin(pdpb_subjects)]
                .loc[(df_copy[drug_name] == "no")]["subject_id"]
                .tolist()
            )

            output_list = (
                df_copy.loc[(df_copy["month_of_visit"] == month)]
                .loc[df_copy["subject_id"].isin(supd3_subjects)]
                .loc[(df_copy[drug_name] == "no")]["subject_id"]
                .tolist()
            )

    return set(output_list)


def find_in_all_med(df, month, yes_or_no):
    """
    Return list of subjects that either
    use any medication or don't.
    """

    df_copy = df.copy()
    if yes_or_no == "yes":

        output_list = (
            df_copy.loc[(df_copy["month_of_visit"] == month)]
            .loc[
                (df_copy["on_levodopa"] == "yes")
                | (df_copy["on_dopamine_agonist"] == "yes")
                | (df_copy["on_other_pd_medications"] == "yes")
            ]["subject_id"]
            .tolist()
        )

    if yes_or_no == "no":

        output_list = (
            df_copy.loc[(df_copy["month_of_visit"] == month)]
            .loc[
                (
                    (df_copy["on_levodopa"] == "no")
                    & (df_copy["on_dopamine_agonist"] == "no")
                    & (df_copy["on_other_pd_medications"] == "no")
                )
                | (
                    (df_copy["on_levodopa"] == np.nan)
                    & (df_copy["on_dopamine_agonist"] == np.nan)
                    & (df_copy["on_other_pd_medications"] == np.nan)
                )
                | (
                    (df_copy["on_levodopa"] == "no")
                    & (df_copy["on_dopamine_agonist"] == np.nan)
                    & (df_copy["on_other_pd_medications"] == np.nan)
                )
                | (
                    (df_copy["on_levodopa"] == np.nan)
                    & (df_copy["on_dopamine_agonist"] == "no")
                    & (df_copy["on_other_pd_medications"] == np.nan)
                )
                | (
                    (df_copy["on_levodopa"] == np.nan)
                    & (df_copy["on_dopamine_agonist"] == np.nan)
                    & (df_copy["on_other_pd_medications"] == "no")
                )
                | (
                    (df_copy["on_levodopa"] == "no")
                    & (df_copy["on_dopamine_agonist"] == "no")
                    & (df_copy["on_other_pd_medications"] == np.nan)
                )
                | (
                    (df_copy["on_levodopa"] == "no")
                    & (df_copy["on_dopamine_agonist"] == np.nan)
                    & (df_copy["on_other_pd_medications"] == "no")
                )
                | (
                    (df_copy["on_levodopa"] == np.nan)
                    & (df_copy["on_dopamine_agonist"] == "no")
                    & (df_copy["on_other_pd_medications"] == "no")
                )
            ]["subject_id"]
            .tolist()
        )

    return set(output_list)


def find_in_all_med_check_backward(df, month):

    output_yes = []

    df_copy = df.copy()
    if (
        month == "m0"
        or month == "m0#2"
        or month == "m0#3"
        or month == "sc"
        or month == "m0_5"
    ):
        list_backward = ["m0"]
    if month == "m6" or month == "m06":
        list_backward = ["m0", "m06", "m6"]
    if month == "m12":
        list_backward = ["m0", "m06", "m6", "m12"]
    if month == "m18":
        list_backward = ["m0", "m06", "m6", "m12", "m18"]
    if month == "m24":
        list_backward = ["m0", "m06", "m6", "m12", "m18", "m24"]
    if month == "m30":
        list_backward = ["m0", "m06", "m6", "m12", "m18", "m24", "m30"]
    if month == "m36":
        list_backward = ["m0", "m06", "m6", "m12", "m18", "m24", "m30", "m36"]

    for month in list_backward:
        df_copy = df_copy.loc[(df_copy["month_of_visit"] == month)]
        df_copy = df_copy.loc[
            (df_copy["on_levodopa"] == "yes")
            | (df_copy["on_dopamine_agonist"] == "yes")
            | (df_copy["on_other_pd_medications"] == "yes")
        ]
        output_list = df_copy["subject_id"].tolist()
        output_yes.append(output_list)
    set_yes = set.union(*[set(x) for x in output_yes])
    df_copy = df.copy()
    all_subject = set(
        df_copy.loc[(df_copy["month_of_visit"] == month)][
            "subject_id"
        ].to_list()
    )
    set_no = all_subject.difference(set_yes)
    return set_yes, set_no


def df_at_base(df, list_of_months_at_base):
    """
    Return data at baseline
    """
    df_list = []
    for month in list_of_months_at_base:
        df = df.loc[(df["month_of_visit"] == month)]
        df_list.append(df)
    df = pd.concat(df_list, sort=False)
    return df


def df_after_drop(df, list_to_drop, list_not_drop):
    """
    Return data after dropping some features
    """
    col_to_drop = []
    df_cols = df.columns.to_list()
    for col in list_to_drop:
        for df_col in df_cols:
            if col in df_col:
                col_to_drop.append(df_col)
    df.drop(col_to_drop, axis="columns", inplace=True, errors="ignore")
    return df


def df_analyzer(df, df_name, list_of_bases, verbose):
    """
    Log various information in log files
    """
    print(f"The outputs will be persist on {df_name}.txt file.")

    try:
        os.remove(general_rules_features["path_to_logs"] + df_name + ".txt")
    except Exception as e:
        logger.info(f"the {df_name} does not exist! The error is {e}")

    original_stdout = (
        sys.stdout
    )  # Save a reference to the original standard output
    with open(
        general_rules_features["path_to_logs"] + df_name + ".txt", "a+"
    ) as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print("################################################")
        print("################################################")
        print("################################################")
        print(f"{df_name} build by these data with these bases:")
        for b in list_of_bases:
            print(f"{b} ")
            print("------------")
        print("################################################")
        print("################################################")

        print(f"{df_name} head:")
        print(df.head())
        print("################################################")

        print(f"{df_name} tail:")
        print(df.tail())

        print("################################################")

        print(f"{df_name} features or variable list:")
        print(*sorted(df.columns.to_list()), sep="\n")
        print("################################################")
        print(f"{df_name} types:")
        print(df.dtypes)
        print("################################################")

        print(f"{df_name} describe:")
        print(df.describe())
        print("################################################")

        print(f"{df_name} null % evaluation :")
        print(df.isnull().mean().sort_values(ascending=False))
        if verbose > 5:
            for col in df.columns.to_list():
                print(col)
                print("################################################")
                print("information about:", col)
                print("unique values:")
                try:
                    print(df[col].unique())
                except Exception as e:
                    print(e)

                print("number of unique:")
                print(df[col].nunique())
                print("value counts:")
                print(df[col].value_counts(dropna=False))
                datatype_of_col = df.dtypes[col]
                print(f"data type of {col} is {datatype_of_col}:")
        if verbose > 5:
            for col in df.columns.to_list():
                if 'label' in col and 'subject_id' in df.columns:
                    print(col )
                    print("################################################")
                    print("###########print data frame as markdown #########")
                    print(df[['subject_id',col]].to_markdown())
                    print("###########print data frame as string ###########")
                    print(df[['subject_id',col]].to_string())
                    df.to_csv(general_rules_features["path_to_logs"] + df_name +'.csv',\
                         index=False)  

        sys.stdout = original_stdout


def df_med_analyzer(dfb, dfa, df_name, list_of_drugs, label, verbose):
    """
    Log various information in log files
    """
    print(
        f"The outputs of medical analyzing will be persisted on {df_name}.txt file."
    )

    try:
        os.remove(general_rules_features["path_to_logs"] + df_name + ".txt")
    except Exception as e:
        logger.info(f"the {df_name} does not exist! The error is {e}")

    original_stdout = (
        sys.stdout
    )  # Save a reference to the original standard output
    with open(
        general_rules_features["path_to_logs"] + df_name + ".txt", "a+"
    ) as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print("################################################")
        print("################################################")
        print("################################################")
        print(" list of drugs are:")
        for b in list_of_drugs:
            print(f"{b} ")
            print("------------")

        print(f"{df_name} describe for data before correction:")
        print(dfb[list_of_drugs].describe())
        print("################################################")

        print(f"{df_name} describe for data after correction:")
        print(dfa[list_of_drugs].describe())
        print("################################################")

        print(f"{df_name} null % evaluation for data before correction:")
        print(dfb[list_of_drugs].isnull().mean().sort_values(ascending=False))
        if verbose > 5:
            for col in list_of_drugs:
                print(col)
                print("################################################")
                print("information about (before correction):", col)
                print("unique values:")
                try:
                    print(dfb[col].unique())
                except Exception as e:
                    print(e)

                print("number of unique:")
                print(dfb[col].nunique())
                print("value counts:")
                print(dfb[col].value_counts(dropna=False))
                datatype_of_col = dfb.dtypes[col]
                print(f"data type of {col} is {datatype_of_col}:")
        print(f"{df_name} null % evaluation for data after correction:")
        print(dfa[list_of_drugs].isnull().mean().sort_values(ascending=False))
        if verbose > 5:
            for col in list_of_drugs:
                print(col)
                print("################################################")
                print("information about (after correction):", col)
                print("unique values:")
                try:
                    print(dfa[col].unique())
                except Exception as e:
                    print(e)

                print("number of unique:")
                print(dfa[col].nunique())
                print("value counts:")
                print(dfa[col].value_counts(dropna=False))
                datatype_of_col = dfa.dtypes[col]
                print(f"data type of {col} is {datatype_of_col}:")

        if verbose > 5:
            for col in list_of_drugs:
                print(f"for drug named {col} for {label} : ")
                print("################################################")
                print("################################################")
                print("################################################")
                print("################################################")
                print("################################################")
                print("################################################")
                print("information about (before correction):", col)
                print(dfb[col].value_counts(dropna=False))
                try:
                    dfb.loc[dfb[col] != "yes", col] = 0
                    dfb.loc[dfb[col] == "yes", col] = 1
                    print(dfb[col].value_counts(dropna=False))
                    print(dfb[label[0]].value_counts(dropna=False))

                    sample1 = dfb[dfb[label[0]] == 1][col]
                    sample2 = dfb[dfb[label[0]] == 0][col]
                    print("sample1 mean")
                    print(sample1.mean())
                    print("sample1 sem")
                    print(sample1.sem())
                    print("sample2 mean")
                    print(sample2.mean())
                    print("sample2 sem")
                    print(sample2.sem())
                    print(ranksums(sample1, sample2))

                except Exception as e:
                    print(e)

            for col in list_of_drugs:
                print(f"for drug named {col} for {label} : ")
                print("################################################")
                print("################################################")
                print("################################################")
                print("################################################")
                print("################################################")
                print("################################################")
                print("information about (after correction):", col)
                dfa.loc[dfa[col] != "yes", col] = 0
                dfa.loc[dfa[col] == "yes", col] = 1
                print(dfa[col].value_counts(dropna=False))
                print(dfa[label[0]].value_counts(dropna=False))

                sample1 = dfa[dfa[label[0]] == 1][col]
                sample2 = dfa[dfa[label[0]] == 0][col]
                print("sample1 mean")
                print(sample1.mean())
                print("sample1 sem")
                print(sample1.sem())
                print("sample2 mean")
                print(sample2.mean())
                print("sample2 sem")
                print(sample2.sem())
                print(ranksums(sample1, sample2))

        sys.stdout = original_stdout

