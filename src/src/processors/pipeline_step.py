# Create some steps for ML pipeline
###################################################
###################################################
###################################################
###################################################

import random
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
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
from sklearn.preprocessing import StandardScaler
from mytakeda.src.processors.feature_store import FeatureStore
from mytakeda.src.takeda_confs.conf_data_engineering import general_rules_features


class ScalerEncoder(BaseEstimator, TransformerMixin):
    """
    Scaling features using Standard Scaler
    """
    def __init__(self):

        self.numeric = None
        self.con_scaler = None

        return

    def fit(self, X, y):
        X = X.copy()
        self.numeric = X.select_dtypes(
            include=["float", "int"]
        ).columns.tolist()
        if self.numeric is not None and len(self.numeric) > 0:
            self.con_scaler = StandardScaler()
            self.con_scaler.fit_transform(X[self.numeric])
        return self

    def transform(self, X):

        X = X.copy()
        X[self.numeric] = self.con_scaler.transform(X[self.numeric])
        return X


class Shap_parametric(BaseEstimator, TransformerMixin):
    """
    Feature selection using Shap values and 
    Optuna optimization engine
    """
    def __init__(
        self,
        n_features,
        model,
        list_of_features,
        params,
        retrain,
        feature_store_path,
        score_type,
        number_of_trials,
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
    ):

        self.config = None
        self.cols = None
        self.selected_cols = None
        self.n_features = n_features
        self.model = model
        self.grid_search = None
        self.importance_df = None
        self.list_of_features = list_of_features
        self.params = params
        self.final_lists = []
        self.final_list = set()
        self.fs = FeatureStore()
        self.retrain = retrain
        self.model_temp = xgb.XGBClassifier()
        self.feature_store_path = feature_store_path
        self.score_type = score_type
        self.number_of_trials = number_of_trials

        self.lambda_par = lambda_par
        self.alpha = alpha
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight

        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.eta = eta
        self.gamma = gamma
        self.grow_policy = grow_policy

        self.sample_type = sample_type
        self.normalize_type = normalize_type
        self.rate_drop = rate_drop
        self.skip_drop = skip_drop

        self.label = label
        self.test_size = test_size
        self.sampler = sampler
        self.pruner = pruner
        self.eval_metric = eval_metric

    def fit(self, X, y):

        self.cols = X.columns

        def objective(trial):

            train_x, valid_x, train_y, valid_y = train_test_split(
                X,
                y,
                stratify=y[self.label],
                test_size=self.test_size,
                random_state=0,
            )

            dtrain = xgb.DMatrix(train_x, label=train_y)
            dvalid = xgb.DMatrix(valid_x, label=valid_y)
            print(type(self.lambda_par[0]))

            param = {
                "verbosity": 0,
                "objective": "binary:logistic",
                # use exact for small dataset.
                "tree_method": "exact",
                # defines booster, gblinear for linear functions.
                "eval_metric": self.eval_metric,
                "booster": trial.suggest_categorical("booster", ["gbtree"]),
                # L2 regularization weight.
                "lambda": trial.suggest_float(
                    "lambda", self.lambda_par[0], self.lambda_par[1], log=True
                ),
                # L1 regularization weight.
                "alpha": trial.suggest_float(
                    "alpha", self.alpha[0], self.alpha[1], log=True
                ),
                # sampling ratio for training data.
                "subsample": trial.suggest_float(
                    "subsample", self.subsample[0], self.subsample[1]
                ),
                # sampling according to each tree.
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    self.colsample_bytree[0],
                    self.colsample_bytree[1],
                ),
                "scale_pos_weight": trial.suggest_loguniform(
                    "scale_pos_weight",
                    self.scale_pos_weight[0],
                    self.scale_pos_weight[1],
                ),
            }

            if param["booster"] in ["gbtree", "dart"]:
                # maximum depth of the tree, signifies the complexity of the tree.
                param["max_depth"] = trial.suggest_int(
                    "max_depth", self.max_depth[0], self.max_depth[1], step=1
                )
                # minimum child weight, the larger the term, more conservative the tree.
                param["min_child_weight"] = trial.suggest_int(
                    "min_child_weight",
                    self.min_child_weight[0],
                    self.min_child_weight[1],
                )
                param["eta"] = trial.suggest_float(
                    "eta", self.eta[0], self.eta[1], log=True
                )
                # defines how selective  algorithm is.
                param["gamma"] = trial.suggest_float(
                    "gamma", self.gamma[0], self.gamma[1], log=True
                )
                param["grow_policy"] = trial.suggest_categorical(
                    self.grow_policy[0], self.grow_policy[1]
                )

            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical(
                    self.sample_type[0], self.sample_type[1]
                )
                param["normalize_type"] = trial.suggest_categorical(
                    self.normalize_type[0], self.normalize_type[1]
                )
                param["rate_drop"] = trial.suggest_float(
                    "rate_drop", self.rate_drop[0], self.rate_drop[1], log=True
                )
                param["skip_drop"] = trial.suggest_float(
                    "skip_drop", self.skip_drop[0], self.skip_drop[1], log=True
                )
            # Add a callback for pruning.
            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "validation-" + self.eval_metric
            )
            bst = xgb.train(
                param,
                dtrain,
                evals=[(dvalid, "validation")],
                callbacks=[pruning_callback],
            )
            preds = bst.predict(dvalid)
            pred_labels = np.rint(preds)
            f1 = f1_score(valid_y, pred_labels)
            acc = accuracy_score(valid_y, pred_labels)
            pr = precision_score(valid_y, pred_labels)
            recall = recall_score(valid_y, pred_labels)
            roc = roc_auc_score(valid_y, pred_labels)
            tn, fp, fn, tp = confusion_matrix(
                valid_y, pred_labels, labels=[0, 1]
            ).ravel()

            classification_r = classification_report(valid_y, pred_labels)
            print(classification_r)
            if self.score_type == "f1":
                accr = f1
            if self.score_type == "acc":
                accr = acc
            if self.score_type == "pr":
                accr = pr
            if self.score_type == "recall":
                accr = recall
            if self.score_type == "roc":
                accr = roc
            if self.score_type == "tn":
                accr = tn
            if self.score_type == "tp":
                accr = tp
            if self.score_type == "f1+acc":
                accr = f1 + acc
            return accr

        if self.retrain:

            if isinstance(self.sampler, str):
                self.sampler = eval(self.sampler)
            if isinstance(self.pruner, str):
                self.pruner = eval(self.pruner)
            study = optuna.create_study(
                direction="maximize", sampler=self.sampler, pruner=self.pruner
            )
            study.optimize(
                objective, n_trials=self.number_of_trials, timeout=600
            )

            trial = study.best_trial

            self.grid_search = xgb.XGBClassifier(**trial.params)
            self.grid_search.fit(X, y)
            best_estimator = self.grid_search
            explainer = shap.TreeExplainer(best_estimator)
            shap_values = explainer.shap_values(X)
            shap_sum = np.abs(shap_values).mean(axis=0)
            self.importance_df = pd.DataFrame(
                [X.columns.tolist(), shap_sum.tolist()]
            ).T
            self.importance_df.columns = ["column_name", "shap_importance"]
            self.importance_df = self.importance_df.sort_values(
                "shap_importance", ascending=False
            )
            num_feat = min([self.n_features, self.importance_df.shape[0]])
            self.selected_cols = self.importance_df["column_name"][
                0:num_feat
            ].to_list()

            # persist this round feature to a pickle file
            filename = (
                self.feature_store_path
                + "_featurespkl_"
                + str(random.uniform(0, 1))
            )
            print("recalculate of features is working ...")
            print("recalculate of features is working ...")
            print("recalculate of features is working ...")
            print("recalculate of features is working ...")

            print(filename, set(self.selected_cols + self.list_of_features))
            self.fs.helper_persist_to_data_save_path(
                set(self.selected_cols + self.list_of_features), filename
            )
            self.final_lists.append(
                set(self.selected_cols + self.list_of_features)
            )
            try:
                for f in listdir(general_rules_features["path_to_save"]):
                    if self.feature_store_path + "_featurespkl_" in str(f):
                        another_set = self.fs.helper_load_from_data_save_path(f)
                        print("another_set", another_set)
                        self.final_lists.append(another_set)

            except Exception as e:
                print("----------")
                print(e)

            self.final_list = set.intersection(*self.final_lists)
            final_filename = (
                self.feature_store_path + "_featurespkl_" + "_final_features_of"
            )
            self.fs.helper_persist_to_data_save_path(
                self.final_list, final_filename
            )

            # retrain with selected features
            print(" xgboost is running .....")
            print(" xgboost is running .....")
            print(" xgboost is running .....")
            print(" xgboost is running .....")
            print(" xgboost is running .....")
            print(" xgboost is running .....")

            self.model_temp.fit(X[self.final_list], y)

            # print final features
            print("self.final_list", self.final_list)
            explainer = shap.TreeExplainer(self.model_temp)
            shap_values = explainer.shap_values(X[self.final_list])

            XX = X[self.final_list].copy()

            for col in XX.columns:
                if "code" in col:
                    try:
                        col_without_code = col.replace("code", "#")
                        XX.rename({col: col_without_code}, axis=1, inplace=True)
                    except Exception as e:
                        print(f"there is an error {e}")

            for col in XX.columns:
                if "pat quest" in col:
                    try:
                        col_without_pat = col.replace("pat quest", "pq")
                        XX.rename({col: col_without_pat}, axis=1, inplace=True)
                    except Exception as e:
                        print(f"there is an error {e}")

            for col in XX.columns:
                if "_" in col:
                    try:
                        col_without_under = col.replace("_", " ")
                        XX.rename(
                            {col: col_without_under}, axis=1, inplace=True
                        )
                    except Exception as e:
                        print(f"there is an error {e}")

            for col in XX.columns:
                if "code" in col:
                    try:
                        col_without_code = col.replace("code", "")
                        XX.rename({col: col_without_code}, axis=1, inplace=True)
                    except Exception as e:
                        print(f"there is an error {e}")

            for col in XX.columns:
                if "pat quest" in col:
                    try:
                        col_without_pat = col.replace("pat quest", "")
                        XX.rename({col: col_without_pat}, axis=1, inplace=True)
                    except Exception as e:
                        print(f"there is an error {e}")

            for col in XX.columns:
                if "_" in col:
                    try:
                        col_without_under = col.replace("_", " ")
                        XX.rename(
                            {col: col_without_under}, axis=1, inplace=True
                        )
                    except Exception as e:
                        print(f"there is an error {e}")

            shap_values = explainer.shap_values(XX)
            shap.summary_plot(shap_values, XX, max_display=self.n_features)

            plt.show()

            shap_sum = np.abs(shap_values).mean(axis=0)
            self.importance_df = pd.DataFrame(
                [X.columns.tolist(), shap_sum.tolist()]
            ).T
            self.importance_df.columns = ["column_name", "shap_importance"]
            self.importance_df = self.importance_df.sort_values(
                "shap_importance", ascending=False
            )
            self.importance_df = self.importance_df[
                self.importance_df["shap_importance"] > 0
            ]
            # display(self.importance_df)

            num_feat = min([self.n_features, self.importance_df.shape[0]])
            # display('num_feat',num_feat)
            self.selected_cols = self.importance_df["column_name"][
                0:num_feat
            ].to_list()

        if not self.retrain:

            print(
                "features will retrieve from former pkl ...",
                self.feature_store_path
                + "_featurespkl_"
                + "_final_features_of",
            )
            print(
                "features will retrieve from former pkl ...",
                self.feature_store_path
                + "_featurespkl_"
                + "_final_features_of",
            )
            print(
                "features will retrieve from former pkl ...",
                self.feature_store_path
                + "_featurespkl_"
                + "_final_features_of",
            )
            print(
                "features will retrieve from former pkl ...",
                self.feature_store_path
                + "_featurespkl_"
                + "_final_features_of",
            )

            for f in listdir(general_rules_features["path_to_save"]):
                if (
                    self.feature_store_path
                    + "_featurespkl_"
                    + "_final_features_of"
                    + ".pkl"
                    == str(f)
                ):
                    final_set = self.fs.helper_load_from_data_save_path(f)
                    print("final_set", final_set)
                    self.final_lists.append(final_set)

            self.final_list = set.intersection(*self.final_lists)

        self.final_list = sorted(self.final_list)

        return self

    def transform(self, X):

        print("transform is working ....")
        print("transform is working ....")
        print("transform is working ....")
        print("transform is working ....")
        print("transform is working ....")

        print("self.final_list:", self.final_list)

        return X[self.final_list]
