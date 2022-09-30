# Script for run meta predictors
###############################
###############################
###############################
###############################

from functools import reduce
import category_encoders as cen
import feature_engine.selection as fts
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
# Create feature store object
from src.src.processors.feature_store import FeatureStore
from src.src.processors.pipeline_step import ScalerEncoder, Shap_parametric
from zoish.feature_selectors.optunashap import OptunaShapFeatureSelector

from src.src.setup_logger import logger
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.src.my_confs.conf_build_model import (COLS_MONTH, COLS_TO_DROP,
                                           LIST_OF_META_FEATURES,
                                           PROBLAMATIC_FEATURES,
                                           RE_CAL_FEATURE, SCORE_TYPE,
                                           ALGORITHM)
from src.src.my_utils.funcs import reset_pkl
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.model_selection import KFold
from lohrasb.best_estimator import BaseModel
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from src.src.my_confs.conf_tunner import (
    MODEL,
    FEATURE_SELECTOR,
    XGB_CLS_MODEL,
    XGB_CLS_SHAP,
    LG_CLS_MODEL,
)

class MetaModelBuilder:
    def __init__(
        self,
        year,
        problem_type,
        train_cohort_inital,
        test_cohort_inital,
        threshold,
        main_base,
    ):
        self.fs = FeatureStore()
        self.cols_month = COLS_MONTH
        self.cols_to_drop = COLS_TO_DROP
        self.problematic_features = PROBLAMATIC_FEATURES
        self.number_of_trails = number_of_trails
        self.n_features = n_features
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
        self.n_components = n_components
        self.recal_feature = RE_CAL_FEATURE
        self.score_type = SCORE_TYPE
        self.eval_metric = eval_metric
        self.sampler = sampler
        self.pruner = pruner
        self.labels_to_drop = None
        self.main_base = main_base
        self.final_problem_path = None
        self.train_path = None
        self.test_path = None
        self.info_path = None
        self.subject_name_in_persist = None
        self.y_pred_train_name = None
        self.y_pred_test_name = None
        self.test_size = test_size
        self.df_train = None
        self.df_test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.problem_type = problem_type
        self.label = None
        self.year = year
        self.test_cohort_inital = test_cohort_inital
        self.train_cohort_inital = train_cohort_inital
        self.threshold = threshold
        self.int_cols = None
        self.cat_cols = None
        self.float_cols = None
        self.df_train_copy = None
        self.df_test_copy = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.list_of_meta_features = LIST_OF_META_FEATURES
        self.list_of_subjects_and_meta_features_paths_for_train = None
        self.list_of_subjects_and_meta_features_paths_for_test = None
        self.shared_train_subject = None
        self.shared_test_subject = None
        self.df_meta_test = None
        self.df_meta_train = None
        self.feature_selector = None
        self.model = None


    @property
    def feature_selector(self):
        logger.info("Getting value for feature_selector")
        return self._feature_selector

    @feature_selector.setter
    def feature_selector(self, value):
        logger.info("Setting value for feature_selector")
        self._feature_selector = value


    @property
    def model(self):
        logger.info("Getting value for model")
        return self._model

    @model.setter
    def model(self, value):
        logger.info("Setting value for model")
        self._model = value

    @property
    def df_meta_train(self):
        logger.info("Getting value for df_meta_train")
        return self._df_meta_train

    @df_meta_train.setter
    def df_meta_train(self, value):
        logger.info("Setting value for df_meta_train")
        self._df_meta_train = value

    @property
    def df_meta_test(self):
        logger.info("Getting value for df_meta_test")
        return self._df_meta_test

    @df_meta_test.setter
    def df_meta_test(self, value):
        logger.info("Setting value for df_meta_test")
        self._df_meta_test = value

    @property
    def shared_train_subject(self):
        logger.info("Getting value for shared train subject")
        return self._shared_train_subject

    @shared_train_subject.setter
    def shared_train_subject(self, value):
        logger.info("Setting value for shared train subject")
        self._shared_train_subject = value

    @property
    def shared_test_subject(self):
        logger.info("Getting value for shared test subject")
        return self._shared_test_subject

    @shared_test_subject.setter
    def shared_test_subject(self, value):
        logger.info("Setting value for shared test subject")
        self._shared_test_subject = value

    @property
    def list_of_subjects_and_meta_features_paths_for_train(self):
        logger.info(
            "Getting value for list of subjects and meta-features paths for the train"
        )
        return self._list_of_subjects_and_meta_features_paths_for_train

    @list_of_subjects_and_meta_features_paths_for_train.setter
    def list_of_subjects_and_meta_features_paths_for_train(self, value):
        logger.info(
            "Setting value for list of subjects and meta-features paths for the train"
        )
        self._list_of_subjects_and_meta_features_paths_for_train = value

    @property
    def list_of_subjects_and_meta_features_paths_for_test(self):
        logger.info(
            "Getting value for list of subjects and meta-features paths for test"
        )
        return self._list_of_subjects_and_meta_features_paths_for_test

    @list_of_subjects_and_meta_features_paths_for_test.setter
    def list_of_subjects_and_meta_features_paths_for_test(self, value):
        logger.info(
            "Setting the value for list of subjects and meta-features paths for test"
        )
        self._list_of_subjects_and_meta_features_paths_for_test = value

    @property
    def list_of_meta_features(self):
        logger.info("Getting value for list of meta-features")
        return self._list_of_meta_features

    @list_of_meta_features.setter
    def list_of_meta_features(self, value):
        logger.info("Setting value for list of meta-features")
        self._list_of_meta_features = value

    @property
    def test_size(self):
        logger.info("Getting value for test_size")
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        logger.info("Setting value for test_size")
        self._test_size = value

    @property
    def year(self):
        logger.info("Getting value for year")
        return self._year

    @year.setter
    def year(self, value):
        logger.info("Setting value for year")
        self._year = value

    @property
    def problem_type(self):
        logger.info("Getting value for problem_type")
        return self._problem_type

    @problem_type.setter
    def problem_type(self, value):
        logger.info("Setting value for problem_type")
        self._problem_type = value

    @property
    def train_cohort_inital(self):
        logger.info("Getting value for train_cohort_inital")
        return self._train_cohort_inital

    @train_cohort_inital.setter
    def train_cohort_inital(self, value):
        logger.info("Setting value for train_cohort_inital")
        self._train_cohort_inital = value

    @property
    def test_cohort_inital(self):
        logger.info("Getting value for test_cohort_inital")
        return self._test_cohort_inital

    @test_cohort_inital.setter
    def test_cohort_inital(self, value):
        logger.info("Setting value for test_cohort_inital")
        self._test_cohort_inital = value

    @property
    def fs(self):
        logger.info("Getting value for fs")
        return self._fs

    @fs.setter
    def fs(self, value):
        logger.info("Setting value for fs")
        self._fs = value

    @property
    def cols_month(self):
        logger.info("Getting value for cols_month")
        return self._cols_month

    @cols_month.setter
    def cols_month(self, value):
        logger.info("Setting value for cols_month")
        self._cols_month = value

    @property
    def cols_to_drop(self):
        logger.info("Getting value for cols_to_drop")
        return self._cols_to_drop

    @cols_to_drop.setter
    def cols_to_drop(self, value):
        logger.info("Setting value for cols_to_drop")
        self._cols_to_drop = value

    @property
    def problematic_features(self):
        logger.info("Getting value for problematic_features")
        return self._problematic_features

    @problematic_features.setter
    def problematic_features(self, value):
        logger.info("Setting value for problematic_features")
        self._problematic_features = value

    @property
    def number_of_trails(self):
        logger.info("Getting value for number_of_trails")
        return self._number_of_trails

    @number_of_trails.setter
    def number_of_trails(self, value):
        logger.info("Setting value for number_of_trails")
        self._number_of_trails = value

    @property
    def n_features(self):
        logger.info("Getting value for n_features")
        return self._n_features

    @n_features.setter
    def n_features(self, value):
        logger.info("Setting value for n_features")
        self._n_features = value

    @property
    def score_type(self):
        logger.info("Getting value for score_type")
        return self._score_type

    @score_type.setter
    def score_type(self, value):
        logger.info("Setting value for score_type")
        self._score_type = value

    @property
    def lambda_par(self):
        return self._lambda_par

    @lambda_par.setter
    def lambda_par(self, value):
        self._lambda_par = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def subsample(self):
        return self._subsample

    @subsample.setter
    def subsample(self, value):
        self._subsample = value

    @property
    def colsample_bytree(self):
        return self._colsample_bytree

    @colsample_bytree.setter
    def colsample_bytree(self, value):
        self._colsample_bytree = value

    @property
    def scale_pos_weight(self):
        return self._scale_pos_weight

    @scale_pos_weight.setter
    def scale_pos_weight(self, value):
        self._scale_pos_weight = value

    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, value):
        self._max_depth = value

    @property
    def min_child_weight(self):
        return self._min_child_weight

    @min_child_weight.setter
    def min_child_weight(self, value):
        self._min_child_weight = value

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, value):
        self._eta = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @property
    def grow_policy(self):
        return self._grow_policy

    @grow_policy.setter
    def grow_policy(self, value):
        self._grow_policy = value

    @property
    def sample_type(self):
        return self._sample_type

    @sample_type.setter
    def sample_type(self, value):
        self._sample_type = value

    @property
    def normalize_type(self):
        return self._normalize_type

    @normalize_type.setter
    def normalize_type(self, value):
        self._normalize_type = value

    @property
    def rate_drop(self):
        return self._rate_drop

    @rate_drop.setter
    def rate_drop(self, value):
        self._rate_drop = value

    @property
    def skip_drop(self):
        return self._skip_drop

    @skip_drop.setter
    def skip_drop(self, value):
        self._skip_drop = value

    @property
    def n_components(self):
        return self._n_components

    @n_components.setter
    def n_components(self, value):
        self._n_components = value

    @property
    def recal_feature(self):
        return self._recal_feature

    @recal_feature.setter
    def recal_feature(self, value):
        self._recal_feature = value

    @property
    def number_of_trials(self):
        return self._number_of_trials

    @number_of_trials.setter
    def number_of_trials(self, value):
        self._number_of_trials = value

    @property
    def score_type(self):
        return self._score_type

    @score_type.setter
    def score_type(self, value):
        self._score_type = value

    @property
    def eval_metric(self):
        return self._eval_metric

    @eval_metric.setter
    def eval_metric(self, value):
        self._eval_metric = value

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, value):
        self._sampler = value

    @property
    def pruner(self):
        return self._pruner

    @pruner.setter
    def pruner(self, value):
        self._pruner = value

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = "label_" + self.problem_type

    @property
    def main_base(self):
        return self._main_base

    @main_base.setter
    def main_base(self, value):
        self._main_base = value

    @property
    def labels_to_drop(self):
        return self._labels_to_drop

    @labels_to_drop.setter
    def labels_to_drop(self, value):
        self._labels_to_drop = value

    @property
    def final_problem_path(self):
        return self._final_problem_path

    @final_problem_path.setter
    def final_problem_path(self, value):
        self._final_problem_path = value

    @property
    def final_problem_path(self):
        return self._final_problem_path

    @final_problem_path.setter
    def final_problem_path(self, value):
        self._final_problem_path = value

    @property
    def train_path(self):
        return self._train_path

    @train_path.setter
    def train_path(self, value):
        self._train_path = value

    @property
    def test_path(self):
        return self._test_path

    @test_path.setter
    def test_path(self, value):
        self._test_path = value

    @property
    def info_path(self):
        return self._info_path

    @info_path.setter
    def info_path(self, value):
        self._info_path = value

    @property
    def subject_name_in_persist(self):
        return self._subject_name_in_persist

    @subject_name_in_persist.setter
    def subject_name_in_persist(self, value):
        self._subject_name_in_persist = value

    @property
    def y_pred_train_name(self):
        return self._y_pred_train_name

    @y_pred_train_name.setter
    def y_pred_train_name(self, value):
        self._y_pred_train_name = value

    @property
    def y_pred_test_name(self):
        return self._y_pred_test_name

    @y_pred_test_name.setter
    def y_pred_test_name(self, value):
        self._y_pred_test_name = value

    @property
    def df_train(self):
        return self._df_train

    @df_train.setter
    def df_train(self, value):
        self._df_train = value

    @property
    def df_test(self):
        return self._df_test

    @df_test.setter
    def df_test(self, value):
        self._df_test = value

    @property
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, value):
        self._X_train = value

    @property
    def X_test(self):
        return self._X_test

    @X_test.setter
    def X_test(self, value):
        self._X_test = value

    @property
    def y_test(self):
        return self._y_test

    @y_test.setter
    def y_test(self, value):
        self._y_test = value

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        self._y_train = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    @property
    def int_cols(self):
        return self._int_cols

    @int_cols.setter
    def int_cols(self, value):
        self._int_cols = value

    @property
    def float_cols(self):
        return self._float_cols

    @float_cols.setter
    def float_cols(self, value):
        self._float_cols = value

    @property
    def cat_cols(self):
        return self._cat_cols

    @cat_cols.setter
    def cat_cols(self, value):
        self._cat_cols = value

    @property
    def df_test_copy(self):
        return self._df_test_copy

    @df_test_copy.setter
    def df_test_copy(self, value):
        self._df_test_copy = value

    @property
    def df_train_copy(self):
        return self._df_train_copy

    @df_train_copy.setter
    def df_train_copy(self, value):
        self._df_train_copy = value

    @property
    def y_pred_train(self):
        return self._y_pred_train

    @y_pred_train.setter
    def y_pred_train(self, value):
        self._y_pred_train = value

    @property
    def y_pred_test(self):
        return self._y_pred_test

    @y_pred_test.setter
    def y_pred_test(self, value):
        self._y_pred_test = value

    def ret_labels_to_drop_str(self):
        if self.problem_type == "i":
            self.labels_to_drop = ["label_ii", "label_iii", "label_total"]
        if self.problem_type == "ii":
            self.labels_to_drop = ["label_i", "label_iii", "label_total"]
        if self.problem_type == "iii":
            self.labels_to_drop = ["label_i", "label_ii", "label_total"]
        if self.problem_type == "total":
            self.labels_to_drop = ["label_i", "label_ii", "label_iii"]
        return True

    def ret_final_problem_path_str(self):
        self.final_problem_path = (
            self.train_cohort_inital
            + "_"
            + self.main_base
            + "_"
            + str(self.threshold)
            + "_"
            + "label_total_"
            + self.year
        )
        print(self.final_problem_path)
        return True

    def ret_train_path_str(self):
        self.train_path = (
            self.train_cohort_inital
            + "_"
            + self.main_base
            + "_"
            + str(self.threshold)
            + "_"
            + self.label
            + "_"
            + self.year
        )
        print(self.train_path)
        return True

    def ret_test_path_str(self):
        self.test_path = (
            self.test_cohort_inital
            + "_"
            + self.main_base
            + "_"
            + str(self.threshold)
            + "_"
            + self.label
            + "_"
            + self.year
        )
        print(self.test_path)
        return True

    def ret_info_path_str(self):
        self.info_path = (
            self.train_cohort_inital
            + "_"
            + self.main_base
            + "_"
            + str(self.threshold)
            + "_"
            + self.label
            + "_"
            + self.year
            + "final_info"
        )
        print(self.info_path)
        return True

    def ret_subject_name_in_persist_str(self):
        self.subject_name_in_persist = self.info_path + "_subject_id"
        return True

    def ret_y_pred_train_name_str(self):
        self.y_pred_train_name = self.info_path + "_y_pred_train"
        print(self.y_pred_train_name)
        return True

    def ret_y_pred_test_name_str(self):
        self.y_pred_test_name = self.info_path + "_y_pred_test"
        print(self.y_pred_test_name)
        return True

    def ret_list_of_subjects_and_meta_features_paths_lst_for_train_test(self):
        list_train = []
        list_test = []
        main_base = self.main_base
        for item_str in self.list_of_meta_features:
            if "24" in item_str:
                year = "24"
            if "36" in item_str:
                year = "36"
            if "12" in item_str:
                year = "12"
            if "up1" in item_str:
                problem_type = "i"
            if "up2" in item_str:
                problem_type = "ii"
            if "up3" in item_str:
                problem_type = "iii"
            if "uptotal" in item_str:
                problem_type = "total"
            train_str = (
                "train_"
                + self.train_cohort_inital
                + "_"
                + main_base
                + "_"
                + str(self.threshold)
                + "_"
                + "label_"
                + problem_type
                + "_"
                + year
                + "_metha_info"
            )
            test_str = (
                "test_"
                + self.train_cohort_inital
                + "_"
                + main_base
                + "_"
                + str(self.threshold)
                + "_"
                + "label_"
                + problem_type
                + "_"
                + year
                + "_metha_info"
            )
            list_train.append(train_str)
            list_test.append(test_str)

        self.list_of_subjects_and_meta_features_paths_for_train = list_train
        self.list_of_subjects_and_meta_features_paths_for_test = list_test

        return True

    def ret_shared_meta_info_train_test(self):
        subj_train = []
        subj_test = []
        dfs_train = []
        dfs_test = []

        # read each dataframe from persisting and merge them
        # in the same time, collect shared subjects
        for train_path in self.list_of_subjects_and_meta_features_paths_for_train:
            train_pandas = pd.DataFrame()
            train_pandas = self.fs.helper_load_from_data_save_path(train_path + ".pkl")
            subj_train.append(set(train_pandas[train_path + "_subject_id"].to_list()))
            # drop subject_id if exists
            train_pandas.drop(
                ["subject_id"], axis="columns", inplace=True, errors="ignore"
            )
            train_pandas.rename(
                {train_path + "_subject_id": "subject_id"},
                axis="columns",
                inplace=True,
            )
            print(train_pandas.head())
            dfs_train.append(train_pandas[["subject_id", train_path + "_y_pred"]])

        # merge all train beforehand read dfs
        self.df_meta_train = reduce(
            lambda left, right: pd.merge(left, right, on=["subject_id"], how="inner"),
            dfs_train,
        )
        # find an intersection between all subjects in all meta feature dfs
        self.shared_train_subject = set.intersection(*subj_train)

        # filter df train meta to contain only shared subjects among themselve
        self.df_meta_train = self.df_meta_train.loc[
            self.df_meta_train["subject_id"].isin(self.shared_train_subject)
        ]

        # read each dataframe from persisting and merge them
        # in the same time, collect shared subjects
        for test_path in self.list_of_subjects_and_meta_features_paths_for_test:
            test_pandas = pd.DataFrame()
            test_pandas = self.fs.helper_load_from_data_save_path(test_path + ".pkl")
            subj_test.append(set(test_pandas[test_path + "_subject_id"].to_list()))
            # drop subject_id if exists
            test_pandas.drop(
                ["subject_id"], axis="columns", inplace=True, errors="ignore"
            )
            test_pandas.rename(
                {test_path + "_subject_id": "subject_id"},
                axis="columns",
                inplace=True,
            )
            print(test_pandas.head())
            dfs_test.append(test_pandas[["subject_id", test_path + "_y_pred"]])

        # merge all train beforehand read dfs
        self.df_meta_test = reduce(
            lambda left, right: pd.merge(left, right, on=["subject_id"], how="inner"),
            dfs_test,
        )
        # find an intersection between all subjects in all meta feature dfs
        self.shared_test_subject = set.intersection(*subj_test)

        # filter df train meta to contain only shared subjects among themselve
        self.df_meta_test = self.df_meta_test.loc[
            self.df_meta_test["subject_id"].isin(self.shared_test_subject)
        ]

        print(train_pandas.head())
        print(test_pandas.head())

        return True

    def ret_pip_param_for_opt(self, feature_store_path):

        # set the model and feature selector 
        # based on env variables

        if ALGORITHM=='cls':
            if MODEL=='LG':
                self.model = LG_CLS_MODEL
            if MODEL=='XGB':
                self.model = XGB_CLS_MODEL
            if FEATURE_SELECTOR=='XGB':
                self.feature_selector = XGB_CLS_SHAP

        imputer_list = []

        if len(self.cat_cols) > 0:
            # category missing values imputers
            imputer_list.append(
                (
                    "catimputer",
                    CategoricalImputer(
                        imputation_method="missing", variables=self.cat_cols
                    ),
                )
            )
        if len(self.float_cols) > 0:
            # float missing values imputers
            imputer_list.append(
                (
                    "floatimputer",
                    MeanMedianImputer(
                        imputation_method="mean", variables=self.float_cols
                    ),
                )
            )
        if len(self.int_cols) > 0:
            # int missing values imputers
            imputer_list.append(
                (
                    "intimputer",
                    MeanMedianImputer(
                        imputation_method="median", variables=self.int_cols
                    ),
                )
            )

        all_dict_pipe = {
            # imputers come from imputer_list
            "imputers": imputer_list,
            "transformers_general": [
                # categorical encoders
                ("order", cen.OrdinalEncoder()),
                # ('order', cen.HelmertEncoder()),
                # outlier resolvers
                ('win', Winsorizer()),
                # transformers
                ('yeo', vt.YeoJohnsonTransformer()),
                # feature problem resolvers
                ("drop_duplicat", fts.DropDuplicateFeatures()),
                # numberic scaler
                ("standard", ScalerEncoder()),
            ],
            "feature_selector_shap": [
                # # feature selectors
                ("shap", self.feature_selector)
            ],
            "remove_noise_from_samples": [
                # PCA
                ("pca", PCA(n_components=self.n_components))
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


    def prepare_train_test_predcit(self):

        if self.recal_feature:
            reset_pkl(self.train_path + "_featurespkl_")

        # extract subjects for 12 months
        print(self.final_problem_path)
        

        self.df_train = self.fs.helper_load_from_data_save_path(
            self.train_path + ".pkl"
        )
        self.df_test = self.fs.helper_load_from_data_save_path(self.test_path + ".pkl")

        # read train data and clean them
        # it is not suitable for self-training because the number of tests will be minimal

        if self.test_cohort_inital == self.train_cohort_inital:
            self.df_train, self.df_test, = train_test_split(
                self.df_train,
                test_size=self.test_size,
                stratify=self.df_train[self.label],
            )

        self.df_train.sort_values(by=["subject_id"], inplace=True)

        self.df_train.drop(
            self.cols_month + self.cols_to_drop + self.labels_to_drop,
            errors="ignore",
            axis=1,
            inplace=True,
        )
        print(*sorted(self.df_meta_train.columns.to_list()), sep="\n")
        print(self.df_meta_train.head())
        print(self.df_meta_train.shape)

        # add meta train to df train
        self.df_train = self.df_train.merge(
            self.df_meta_train, on=["subject_id"], how="inner"
        )
        # to make sure that df train only has the right subjects
        self.df_train = self.df_train.loc[
            self.df_train["subject_id"].isin(self.shared_train_subject)
        ]

        # make a copy of train
        self.df_train_copy = self.df_train.copy()

        # drop subject_ids
        self.df_train.drop(["subject_id"], errors="ignore", axis=1, inplace=True)
        print(*sorted(self.df_train.columns.to_list()), sep="\n")
        print(self.df_train.head())
        print(self.df_train.shape)
        for col in self.df_train:
            if "_y_pred" in col:
                print(self.df_train[col])

        # read test data and clean them
        self.df_test.sort_values(by=["subject_id"], inplace=True)
        print(self.df_test.head())
        self.df_test.drop(
            self.cols_month + self.cols_to_drop + self.labels_to_drop,
            errors="ignore",
            axis=1,
            inplace=True,
        )
        print(*sorted(self.df_meta_test.columns.to_list()), sep="\n")
        print(self.df_meta_test.shape)

        # add meta test to df train
        self.df_test = self.df_test.merge(
            self.df_meta_test, on=["subject_id"], how="inner"
        )
        print(self.df_test.head())

        # to make sure that df test only has right subjects
        self.df_test = self.df_test.loc[
            self.df_test["subject_id"].isin(self.shared_test_subject)
        ]
        print(self.shared_test_subject)
        print(self.df_test.head())
        print(self.df_test.shape)
        for col in self.df_test:
            if "_y_pred" in col:
                print(self.df_test[col])

        # make a copy of test
        self.df_test_copy = self.df_test.copy()

        # drop subject_ids
        self.df_test.drop(["subject_id"], errors="ignore", axis=1, inplace=True)
        print(*sorted(self.df_test.columns.to_list()), sep="\n")
        print(self.df_test.head())

        # train test split
        self.X_train = self.df_train.loc[
            :,
            (
                (self.df_train.columns != self.label)
                & (self.df_train.columns != "subject_id")
            ),
        ]
        self.y_train = self.df_train.loc[
            :,
            (
                (self.df_train.columns == self.label)
                & (self.df_train.columns != "subject_id")
            ),
        ]
        self.X_test = self.df_test.loc[
            :,
            (
                (self.df_test.columns != self.label)
                & (self.df_test.columns != "subject_id")
            ),
        ]
        self.y_test = self.df_test.loc[
            :,
            (
                (self.df_test.columns == self.label)
                & (self.df_test.columns != "subject_id")
            ),
        ]

        # renaming features that should be renames
        for col in self.X_test.columns.to_list():
            if "test_" in col:
                col_for_rename = col.replace("test_", "train_")
                col_for_rename = col_for_rename.replace(self.main_base, "")
                print(col_for_rename)
                self.X_test.rename(columns={col: col_for_rename}, inplace=True)

        for col in self.X_train.columns.to_list():
            if "train_" in col:
                col_for_rename = col.replace(self.main_base, "")
                print(col_for_rename)
                self.X_train.rename(columns={col: col_for_rename}, inplace=True)

        # create shared features
        shared_features = set(self.X_train.columns).intersection(
            set(self.X_test.columns)
        )
        print(self.X_train.describe())
        print(self.X_test.describe())
        shared_features = shared_features.difference(set(self.problematic_features))
        print(self.X_train.columns.to_list() == self.X_test.columns.to_list())

        # only use shared features
        self.X_train = self.X_train[shared_features]
        self.X_test = self.X_test[shared_features]

        self.int_cols = self.X_train.select_dtypes(include=["int"]).columns.tolist()
        self.float_cols = self.X_train.select_dtypes(include=["float"]).columns.tolist()
        self.cat_cols = self.X_train.select_dtypes(include=["object"]).columns.tolist()

        print("cat columns :")
        print(self.cat_cols)
        print("float columns :")
        print(self.float_cols)
        print("int columns :")
        print(self.int_cols)

        ## best estimator creation 
        # apply pipeline on X_train
        SHAP_PIP = self.ret_pip_param_for_opt(self.train_path)
        best_model = self.model
        XT_train = SHAP_PIP.fit_transform(self.X_train, self.y_train)
        best_model.fit(XT_train,self.y_train)

        # test
        XT_test = SHAP_PIP.transform(self.X_test)

        # calculate pred on test
        y = best_model.predict(XT_test)
        y = np.rint(y)
        self.y_pred_test = y
        # calculate pred on train
        self.y_pred_train = best_model.predict(XT_train)
        self.y_pred_train = np.rint(self.y_pred_train)

        # show results
        print("classification report for test ...")
        print("classification report for test ...")
        print("classification report for test ...")
        print("classification report for test ...")
        classification_r = classification_report(
            self.y_test,
            self.y_pred_test,
            labels=[0, 1],
            target_names=["non-progressors", "progressors"],
        )
        print(classification_r)
        print("confusion matrix  for test ...")
        print("confusion matrix  for test ...")
        print("confusion matrix  for test ...")
        print("confusion matrix  for test ...")
        cf_matrix = confusion_matrix(self.y_test, self.y_pred_test, labels=[0, 1])
        print(cf_matrix)
        print("f1_score ...")
        print(f1_score(self.y_test, self.y_pred_test))
        print("precision score ...")
        print(precision_score(self.y_test, self.y_pred_test))
        print("recall score ...")
        print(recall_score(self.y_test, self.y_pred_test))

        return True

    def persist(self):
        # # re-open the train and test files for putting information
        # # train first
        logger.info(
            f"train is not created yet; now it will be \
                created with base {self.info_path}"
        )
        train = pd.DataFrame()

        train[self.subject_name_in_persist] = self.df_train_copy["subject_id"]
        train[self.y_pred_train_name] = self.y_pred_train
        self.fs.helper_persist_to_data_save_path(train, "train_final" + self.info_path)

        # # test second
        logger.info(
            f"test is not created yet; now it will be created \
                with base {self.info_path}"
        )
        test = pd.DataFrame()

        test[self.subject_name_in_persist] = self.df_test_copy["subject_id"]
        test[self.y_pred_test_name] = self.y_pred_test
        self.fs.helper_persist_to_data_save_path(test, "test_final" + self.info_path)

    True


def runner():
    mmb = MetaModelBuilder(
        problem_type="ii",
        year="12",
        train_cohort_inital="pp",
        threshold=0.5,
        test_cohort_inital="su",
        main_base="df_after_adjustment_train_data",
    )
    mmb.ret_labels_to_drop_str()
    mmb.ret_final_problem_path_str()
    mmb.ret_train_path_str()
    mmb.ret_test_path_str()
    mmb.ret_info_path_str()
    mmb.ret_subject_name_in_persist_str()
    mmb.ret_y_pred_train_name_str()
    mmb.ret_y_pred_test_name_str()
    mmb.ret_list_of_subjects_and_meta_features_paths_lst_for_train_test()
    mmb.ret_shared_meta_info_train_test()
    mmb.prepare_train_test_predcit()
    mmb.persist()
    return True


if __name__ == "__main__":
    runner()
