# Script for run predictive model for each case
###############################################
###############################################
###############################################

from operator import concat
from tkinter import EXCEPTION
import category_encoders as cen
import feature_engine.selection as fts
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from zoish.feature_selectors.optunashap import OptunaShapFeatureSelector
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.outliers import Winsorizer
from feature_engine.transformation import YeoJohnsonTransformer 

# Create feature store object
from src.src.processors.feature_store import FeatureStore
from src.src.processors.pipeline_step import ScalerEncoder, Shap_parametric
from src.src.setup_logger import logger
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# importing pandas
from src.src.my_confs.conf_build_model import (
    COLS_MONTH,
    COLS_TO_DROP,
    SCORE_TYPE,
    ALGORITHM,
    
)
from src.src.my_confs.conf_tunner import (
    MODEL,
    FEATURE_SELECTOR,
    XGB_CLS_MODEL,
    XGB_CLS_SHAP,
    LG_CLS_MODEL,
)
from src.src.my_utils.funcs import reset_pkl
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.model_selection import KFold
from lohrasb.best_estimator import BaseModel
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


class ModelBuilder:
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
        self.subject_in_test = None
        self.subject_in_train = None
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
        self.duplicated_subjects = duplicated_subjects
        self.pipeline = None
        self.features = None
        self.model = None
        self.feature_selector = None

    @property
    def model(self):
        logger.info("Getting value for model")
        return self._model

    @model.setter
    def model(self, value):
        logger.info("Setting value for model")
        self._model = value

    @property
    def feature_selector(self):
        logger.info("Getting value for feature selector")
        return self._feature_selector

    @feature_selector.setter
    def feature_selector(self, value):
        logger.info("Setting value for feature_selector")
        self._feature_selector = value

    @property
    def duplicated_subjects(self):
        logger.info("Getting value for duplicated_subjects")
        return self._duplicated_subjects

    @duplicated_subjects.setter
    def duplicated_subjects(self, value):
        logger.info("Setting value for duplicated_subjects")
        self._duplicated_subjects = value

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
    def subject_in_train(self):
        return self._subject_in_train

    @subject_in_train.setter
    def subject_in_train(self, value):
        self._subject_in_train = value

    @property
    def subject_in_test(self):
        return self._subject_in_test

    @subject_in_test.setter
    def subject_in_test(self, value):
        self._subject_in_test = value

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
            + self._label
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
            # + self._label
            + self.label
            + "_"
            + self.year
            + "_trained_model"
        )
        print(self.info_path)
        return True

    def ret_subject_in_train(self):
        self.subject_in_train = "train_" + self.info_path + "_subject_id"
        return True

    def ret_subject_in_test(self):
        self.subject_in_test = "test_" + self.info_path + "_subject_id"
        return True

    def ret_y_pred_train_name_str(self):
        self.y_pred_train_name = "train_" + self.info_path + "_y_pred"
        print(self.y_pred_train_name)
        return True

    def ret_y_pred_test_name_str(self):
        self.y_pred_test_name = "test_" + self.info_path + "_y_pred"
        print(self.y_pred_test_name)
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
            # imputers comes from imputer_list
            "imputers": imputer_list,
            "transformers_general": [
                # categorical encoders
                ("order", cen.OrdinalEncoder()),
                #('order', cen.HelmertEncoder()),
                # outlier resolvers
                ('win', Winsorizer()),
                # transformers
                ('yeo', YeoJohnsonTransformer()),
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
            #+ all_dict_pipe['remove_noise_from_samples']
        )

        return SHAP_PIP

    def prepare_train_test_predcit(self):

        if self.recal_feature:
            reset_pkl(self.train_path + "_featurespkl_")

        # extract subjects 
        print(self.final_problem_path)
        final_problem_data = self.fs.helper_load_from_data_save_path(
            self.final_problem_path + ".pkl"
        )
        filtered_subjects_from_final_problem_data = final_problem_data[
            "subject_id"
        ].to_list()

        self.df_train = self.fs.helper_load_from_data_save_path(
            self.train_path + ".pkl"
        )
        self.df_test = self.fs.helper_load_from_data_save_path(self.test_path + ".pkl")
        print(self.df_test.head())
        print(self.df_test.columns)

        # drop duplicated_subjects in test

        self.df_test = self.df_test.loc[
            ~self.df_test["subject_id"].isin(duplicated_subjects)
        ]

        # read train data and clean them
        if ALGORITHM == "cls":
            if self.test_cohort_inital == self.train_cohort_inital:
                self.df_train, self.df_test, = train_test_split(
                    self.df_train,
                    test_size=self.test_size,
                    stratify=self.df_train[self.label],
                )

        self.df_train = self.df_train[
            self.df_train["subject_id"].isin(filtered_subjects_from_final_problem_data)
        ]

        # create a copy of data for later use
        self.df_train.sort_values(by=["subject_id"], inplace=True)
        self.df_train_copy = self.df_train.copy()

        self.df_train.drop(
            self.cols_month + self.cols_to_drop + self.labels_to_drop,
            errors="ignore",
            axis=1,
            inplace=True,
        )

        print(*sorted(self.df_train.columns.to_list()), sep="\n")

        # read test data and clean them
        self.df_test.sort_values(by=["subject_id"], inplace=True)
        self.df_test_copy = self.df_test.copy()
        self.df_test.drop(
            self.cols_month + self.cols_to_drop + self.labels_to_drop,
            errors="ignore",
            axis=1,
            inplace=True,
        )

        print(*sorted(self.df_test.columns.to_list()), sep="\n")

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

        print(self.X_train.head())
        print(self.X_test.head())
       
        # create shared features
        shared_features = set(self.X_train.columns).intersection(
            set(self.X_test.columns)
        )

        shared_features = shared_features.difference(set(self.problematic_features))
        print(self.X_train.columns.to_list() == self.X_test.columns.to_list())

        # only use shared features
        self.X_train = self.X_train[shared_features]
        self.X_test = self.X_test[shared_features]

        print(self.X_train.head())
        print(self.X_test.head())


        self.int_cols = self.X_train.select_dtypes(include=["int"]).columns.tolist()
        self.float_cols = self.X_train.select_dtypes(include=["float"]).columns.tolist()
        self.cat_cols = self.X_train.select_dtypes(include=["object"]).columns.tolist()

        print(self.cat_cols)
        print(self.float_cols)
        print(self.int_cols)

        # apply pipeline on X_train
        SHAP_PIP = self.ret_pip_param_for_opt(self.train_path)
        XT_train = SHAP_PIP.fit_transform(self.X_train, self.y_train)
        self.model.fit(XT_train, self.y_train)
        # test
        print(self.X_test.head())
        print(*sorted(self.X_test.columns),sep="\n")
        XT_test = SHAP_PIP.transform(self.X_test)
        print(self.X_test.head())
        print(*sorted(self.X_test.columns),sep="\n")

        # assign pipeline and fitted model for later
        # use
        self.pipeline = SHAP_PIP
        self.features = shared_features

        # calculate pred on test
        y = self.model.predict(XT_test)
        if ALGORITHM == "cls":
            y = np.rint(y)
            self.y_pred_test = y
        
        # calculate pred on train
        self.y_pred_train = self.model.predict(XT_train)
        self.y_pred_train = np.rint(self.y_pred_train)

        if ALGORITHM == "cls":
            # show results in train
            print("classification report for train ...")
            print("classification report for train ...")
            print("classification report for train ...")
            print("classification report for train ...")
           
            print(classification_r)
            print("confusion matrix  for train ...")
            print("confusion matrix  for train ...")
            print("confusion matrix  for train ...")
            print("confusion matrix  for train ...")
            cf_matrix = confusion_matrix(self.y_train, self.y_pred_train, labels=[0, 1])
            print(cf_matrix)
            print("f1_score ...")
            print(f1_score(self.y_train, self.y_pred_train))
            print("precision score ...")
            print(precision_score(self.y_train, self.y_pred_train))
            print("recall score ...")
            print(recall_score(self.y_train, self.y_pred_train))
            # show results test
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
            f"train is not created yet, now it will be created with \
                base {self.info_path}"
        )
        train = pd.DataFrame()

        train[self.subject_in_train] = self.df_train_copy["subject_id"]
        train[self.y_pred_train_name] = self.y_pred_train
        print(self.y_pred_train_name)
        self.fs.helper_persist_to_data_save_path(train, "train_" + self.info_path)

        # # test second
        logger.info(
            f"test is not created yet; now it will be created with \
                base {self.info_path}"
        )
        test = pd.DataFrame()

        test[self.subject_in_test] = self.df_test_copy["subject_id"]
        test[self.y_pred_test_name] = self.y_pred_test
        self.fs.helper_persist_to_data_save_path(test, "test_" + self.info_path)

        #  model persist
        logger.info(
            f" model meta is not created yet, now it will be created with \
                base {self.info_path}"
        )

        # get model name
        model_dict = {
            "pipeline": self.pipeline,
            "features": self.features,
            "model": self.model,
        }
        self.fs.helper_persist_to_data_save_path(
            model_dict, "pip_model_features_" + self.info_path
        )

    True

def runner():
    mb = ModelBuilder(
        problem_type="i",
        year="12",
        train_cohort_inital="pp",
        threshold=0.5,
        test_cohort_inital="pd",
        main_base="df_after_adjustment_train_data",
    )
    mb.ret_labels_to_drop_str()
    mb.ret_final_problem_path_str()
    mb.ret_train_path_str()
    mb.ret_test_path_str()
    mb.ret_info_path_str()
    mb.ret_subject_in_test()
    mb.ret_subject_in_train()
    mb.ret_y_pred_train_name_str()
    mb.ret_y_pred_test_name_str()
    mb.prepare_train_test_predcit()
    mb.persist()
    return True


def runner_multiple():
    year = ["12", "18", "24", "36"]

    threshold = [0.5]
    train_cohort_inital = ["pp"]
    test_cohort_inital = ["pd"]
    problem_type = ["i", "ii", "iii", "total"]

    if "su" in train_cohort_inital or "su" in test_cohort_inital:
        year = ["12", "18", "24"]
    else:
        year = ["12", "18", "24", "36"]
    main_base = ["df_after_adjustment_train_data"]

    for pt in problem_type:
        for y in year:
            for t in threshold:
                for tr_init in train_cohort_inital:
                    for ts_init in test_cohort_inital:
                        for mb in main_base:
                            mb = ModelBuilder(
                                problem_type=pt,
                                year=y,
                                train_cohort_inital=tr_init,
                                threshold=t,
                                test_cohort_inital=ts_init,
                                main_base=mb,
                            )
                            mb.ret_labels_to_drop_str()
                            mb.ret_final_problem_path_str()
                            mb.ret_train_path_str()
                            mb.ret_test_path_str()
                            mb.ret_info_path_str()
                            mb.ret_subject_in_test()
                            mb.ret_subject_in_train()
                            mb.ret_y_pred_train_name_str()
                            mb.ret_y_pred_test_name_str()
                            mb.prepare_train_test_predcit()
                            mb.persist()
    return True


if __name__ == "__main__":
    # run only one experiment
    runner()
    # run multiple experiments
    # runner_multiple()
