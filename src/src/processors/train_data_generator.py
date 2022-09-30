
# Generate training sets
###################################################
###################################################
###################################################
###################################################import os
import warnings
from functools import reduce
import numpy as np
import pandas as pd
from src.src.processors.feature_store import FeatureStore
from src.src.setup_logger import logger
from src.src.my_confs.conf_build_train import (
    DEBUG_FEATURES,
    COLS_MONTH_RENAME,
    DATA_VERSION,
    DATSCAN,
    LIST_ALL,
    LIST_BASE,
    LIST_LABELS,
    LIST_NOT_DROP,
    LIST_OF_COHORTS,
    LIST_OF_COHORTS_FOR_TRAIN,
    LIST_OF_UPDRS,
    LIST_ONLY_AT_M0,
    LIST_ONLY_AT_M0_TO_M96,
    LIST_PRS,
    LIST_SCORES,
    LIST_TO_DROP,
    MONTH_LABELS,
    NEVER_DROP_FEATURES,
    QUESTIONS_CODE,
    QUESTIONS_CODE_I,
    QUESTIONS_CODE_II,
    QUESTIONS_CODE_III,
    biospecimen_analyses_csf_abeta_tau_ptau_path,
    biospecimen_analyses_csf_beta_glucocerebrosidase_path,
    biospecimen_analyses_other_path,
    biospecimen_analyses_somaLogic_plasma_path,
    caffeine_history_path,
    case_control_other_at_baseline,
    data_for_participants,
    data_for_participants_second_part,
    datscan_sbr_path,
    datscan_visual_interpretation_path,
    demographics_path,
    dti_path,
    edu_prs_path,
    all_prs_path,
    enrollment_path,
    ess_path,
    family_history_pd_path,
    is_read_from_start,
    lbd_path,
    lbdpath_path,
    list_data_to_use,
    md_history,
    mds_updrs_part_i_path,
    mds_updrs_part_ii_path,
    mds_updrs_part_iii_path,
    min_threshold_of_case,
    mmse_path,
    moca_path,
    monitoring_time_line,
    mono_path,
    msq_path,
    pd_medical_history_path,
    pdq39_path,
    prscs_prs_path,
    pt_prs_path,
    rbd_path,
    schwab_path,
    smoking_and_alcohol_history_path,
    staged_data_path,
    subject_id,
    threshold,
    updrs_i_only_components,
    updrs_i_only_components_path,
    upsit_path,
)
from src.src.my_utils.funcs import (
    df_after_drop,
    df_analyzer,
    df_at_base,
    make_slopes,
    uplift_scores_all_together,
)

class BuildTrain:
    def __init__(
        self,
        cohort_inital,
        list_data_to_use,
        list_data_score,
        problem_type,
        year,
        threshold,
        verbose,

    ):

        self.fs = None
        self.is_read_from_start = is_read_from_start
        self.monitoring_time_line = monitoring_time_line
        self.min_threshold_of_case = min_threshold_of_case
        self.staged_data_path = staged_data_path
        self.data_for_participants = data_for_participants
        self.data_for_participants_second_part = (
            data_for_participants_second_part
        )
        self.case_control_other_at_baseline = case_control_other_at_baseline
        self.md_history = md_history
        self.updrs_i_only_components = updrs_i_only_components
        self.subject_id = subject_id

        self.updrs_i_path = mds_updrs_part_i_path
        self.updrs_ii_path = mds_updrs_part_ii_path
        self.updrs_iii_path = mds_updrs_part_iii_path
        self.prscs_prs_path = prscs_prs_path
        self.pt_prs_path = pt_prs_path
        self.edu_prs_path = edu_prs_path
        self.all_prs_path = all_prs_path 
        self.rbd = None
        self.rbd_path = rbd_path
        self.code_i_questions = QUESTIONS_CODE_I
        self.code_ii_questions = QUESTIONS_CODE_II
        self.code_iii_questions = QUESTIONS_CODE_III
        self.all_code_questions = QUESTIONS_CODE
        self.moca = None
        self.moca_path = moca_path
        self.ess = None
        self.ess_path = ess_path
        self.pdq39 = None
        self.pdq39_path = pdq39_path
        self.sch = None
        self.sch_path = schwab_path
        self.mono = None
        self.mono_path = mono_path
        self.datscan_questions = DATSCAN
        self.list_prs = LIST_PRS
        self.cols_month_rename = COLS_MONTH_RENAME
        self.list_of_uprs = LIST_OF_UPDRS
        self.list_to_drop = LIST_TO_DROP
        self.list_base = LIST_BASE
        self.list_only_at_m0 = LIST_ONLY_AT_M0
        self.list_only_at_m0_to_m96 = LIST_ONLY_AT_M0_TO_M96
        self.list_labels = LIST_LABELS
        self.list_all = LIST_ALL
        self.month_labels = MONTH_LABELS
        self.list_scores = LIST_SCORES
        self.list_participants_to_drop = LIST_PARTICIPANTS_TO_DROP
        self.all_data = None
        self.all_participants = None
        self.cohort_inital = cohort_inital
        self.subject_of_cohort = None
        self.list_data_to_use = list_data_to_use
        self.name_for_save = None
        self.medichal_path = pd_medical_history_path
        self.family_path = family_history_pd_path
        self.demographics_path = demographics_path
        self.enrollment_path = enrollment_path
        self.alchol_path = smoking_and_alcohol_history_path
        self.caffeine_path = caffeine_history_path
        self.bio_csf_abeta_path = biospecimen_analyses_csf_abeta_tau_ptau_path
        self.bio_csf_beta_path = (
            biospecimen_analyses_csf_beta_glucocerebrosidase_path
        )
        self.bio_other_path = biospecimen_analyses_other_path
        self.bio_plasma_path = biospecimen_analyses_somaLogic_plasma_path
        self.dat_scan_sbr_path = datscan_sbr_path
        self.dat_scan_visual_path = datscan_visual_interpretation_path
        self.dat_dti_path = dti_path
        self.dat_mmse_path = mmse_path
        self.clinichal_data = pd.DataFrame()
        self.prs_data = pd.DataFrame()
        self.list_not_drop = LIST_NOT_DROP
        self.mono_path = mono_path
        self.list_data_score = list_data_score
        self.score_data = pd.DataFrame()
        self.problem_type = problem_type
        self.year = year
        self.threshold = threshold
        self.df = None
        self.verbose = verbose
        self.data_version = None
        self.msq_path = msq_path
        self.lbd_path = lbd_path
        self.lbdpath_path = lbdpath_path
        self.upsit_path = upsit_path
        self.updrs_i_only_components_path = updrs_i_only_components_path
        self.do_persist = None
        self.adjust_correction = None
        self.filter=None
        self.never_drop_features = NEVER_DROP_FEATURES
        self.algorithm = None

    # Getters and Setters 
    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        logger.info("Setting value for algorithm")
        self._algorithm= value
    
    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, value):
        logger.info("Setting value for filter")
        self._filter = value
    @property
    def never_drop_features(self):
        return self._never_drop_features

    @never_drop_features.setter
    def never_drop_features(self, value):
        logger.info("Setting value for never_drop_features")
        self._never_drop_features = value

    @property
    def do_persist(self):
        return self._do_persist

    @do_persist.setter
    def do_persist(self, value):
        logger.info("Setting value for do_persist")
        self._do_persist = value

    @property
    def adjust_correction(self):
        return self._adjust_correction

    @adjust_correction.setter
    def adjust_correction(self, value):
        logger.info("Setting value for adjust_correction")
        self._adjust_correction = value

    @property
    def msq_path(self):
        return self._msq_path

    @msq_path.setter
    def msq_path(self, value):
        logger.info("Setting value for msq_path")
        self._msq_path = value

    @property
    def updrs_i_only_components_path(self):
        return self._updrs_i_only_components_path

    @updrs_i_only_components_path.setter
    def updrs_i_only_components_path(self, value):
        logger.info("Setting value for updrs_i_only_components_path")
        self._updrs_i_only_components_path = value

    @property
    def upsit_path(self):
        return self._upsit_path

    @upsit_path.setter
    def upsit_path(self, value):
        logger.info("Setting value for upsit_path")
        self._upsit_path = value

    @property
    def lbdpath_path(self):
        return self._lbdpath_path

    @lbdpath_path.setter
    def lbdpath_path(self, value):
        logger.info("Setting value for lbdpath_path")
        self._lbdpath_path = value

    @property
    def lbd_path(self):
        return self._lbd_path

    @lbd_path.setter
    def lbd_path(self, value):
        logger.info("Setting value for lbd_path")
        self._lbd_path = value

    @property
    def data_version(self):
        return self._data_version

    @data_version.setter
    def data_version(self, value):
        logger.info("Setting value for data_version")
        self._data_version = value

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, value):
        logger.info("Setting value for fs")
        self._fs = value

    @property
    def is_read_from_start(self):
        logger.info("Getting value for is_read_from_start")
        return self._is_read_from_start

    @is_read_from_start.setter
    def is_read_from_start(self, value):
        logger.info("Setting value for is_read_from_start")
        self._is_read_from_start = value

    @property
    def monitoring_time_line(self):
        logger.info("Getting value for monitoring_time_line")
        return self._monitoring_time_line

    @monitoring_time_line.setter
    def monitoring_time_line(self, value):
        logger.info("Setting value for monitoring_time_line")
        self._monitoring_time_line = value

    @property
    def min_threshold_of_case(self):
        logger.info("Getting value for min_threshold_of_case")
        return self._min_threshold_of_case

    @min_threshold_of_case.setter
    def min_threshold_of_case(self, value):
        logger.info("Setting value for min_threshold_of_case")
        self._min_threshold_of_case = value

    @property
    def threshold(self):
        logger.info("Getting value for threshold")
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        logger.info("Setting value for threshold")
        self._threshold = value

    @property
    def staged_data_path(self):
        logger.info("Getting value for staged_data_path")
        return self._staged_data_path

    @staged_data_path.setter
    def staged_data_path(self, value):
        logger.info("Setting value for staged_data_path")
        self._staged_data_path = value

    @property
    def data_for_participants(self):
        logger.info("Getting value for data_for_participants")
        return self._data_for_participants

    @data_for_participants.setter
    def data_for_participants(self, value):
        logger.info("Setting value for data_for_participants")
        self._data_for_participants = value

    @property
    def data_for_participants_second_part(self):
        logger.info("Getting value for data_for_participants_second_part")
        return self._data_for_participants_second_part

    @data_for_participants_second_part.setter
    def data_for_participants_second_part(self, value):
        logger.info("Setting value for data_for_participants_second_part")
        self._data_for_participants_second_part = value

    @property
    def case_control_other_at_baseline(self):
        logger.info("Getting value for case_control_other_at_baseline")
        return self._case_control_other_at_baseline

    @case_control_other_at_baseline.setter
    def case_control_other_at_baseline(self, value):
        logger.info("Setting value for case_control_other_at_baseline")
        self._case_control_other_at_baseline = value

    @property
    def md_history(self):
        logger.info("Getting value for md_history")
        return self._md_history

    @md_history.setter
    def md_history(self, value):
        logger.info("Setting value for md_history")
        self._md_history = value

    @property
    def updrs_i_only_components(self):
        logger.info("Getting value for updrs_i_only_components")
        return self._updrs_i_only_components

    @updrs_i_only_components.setter
    def updrs_i_only_components(self, value):
        logger.info("Setting value for updrs_i_only_components")
        self._updrs_i_only_components = value

    @property
    def subject_id(self):
        logger.info("Getting value for subject_id")
        return self._subject_id

    @subject_id.setter
    def subject_id(self, value):
        logger.info("Setting value for subject_id")
        self._subject_id = value

    @property
    def updrs_i_path(self):
        logger.info("Getting value for updrs_i_path")
        return self._updrs_i_path

    @updrs_i_path.setter
    def updrs_i_path(self, value):
        logger.info("Setting value for updrs_i_path")
        self._updrs_i_path = value

    @property
    def updrs_ii_path(self):
        logger.info("Getting value for updrs_ii_path")
        return self._updrs_ii_path

    @updrs_ii_path.setter
    def updrs_ii_path(self, value):
        logger.info("Setting value for updrs_ii_path")
        self._updrs_ii_path = value

    @property
    def updrs_iii_path(self):
        logger.info("Getting value for updrs_iii_path")
        return self._updrs_iii_path

    @updrs_iii_path.setter
    def updrs_iii_path(self, value):
        logger.info("Setting value for updrs_iii_path")
        self._updrs_iii_path = value

    @property
    def all_code_questions(self):
        logger.info("Getting value for all_code_questions")
        return self._all_code_questions

    @all_code_questions.setter
    def all_code_questions(self, value):
        logger.info("Setting value for all_code_questions")
        self._all_code_questions = value

    @property
    def moca(self):
        logger.info("Getting value for moca")
        return self._moca

    @moca.setter
    def moca(self, value):
        logger.info("Setting value for moca")
        self._moca = value

    @property
    def moca_path(self):
        logger.info("Getting value for moca_path")
        return self._moca_path

    @moca_path.setter
    def moca_path(self, value):
        logger.info("Setting value for moca_path")
        self._moca_path = value

    @property
    def ess(self):
        logger.info("Getting value for ess")
        return self._ess

    @ess.setter
    def ess(self, value):
        logger.info("Setting value for ess")
        self._ess = value

    @property
    def ess_path(self):
        logger.info("Getting value for ess_path")
        return self._ess_path

    @ess_path.setter
    def ess_path(self, value):
        logger.info("Setting value for ess_path")
        self._ess_path = value

    @property
    def pdq39(self):
        logger.info("Getting value for pdq39")
        return self._pdq39

    @pdq39.setter
    def pdq39(self, value):
        logger.info("Setting value for pdq39")
        self._pdq39 = value

    @property
    def pdq39_path(self):
        logger.info("Getting value for pdq39_path")
        return self._pdq39_path

    @pdq39_path.setter
    def pdq39_path(self, value):
        logger.info("Setting value for pdq39_path")
        self._pdq39_path = value

    @property
    def sch(self):
        logger.info("Getting value for sch")
        return self._sch

    @sch.setter
    def sch(self, value):
        logger.info("Setting value for sch")
        self._sch = value

    @property
    def medichal_path(self):
        logger.info("Getting value for medichal_path")
        return self._medichal_path

    @medichal_path.setter
    def medichal_path(self, value):
        logger.info("Setting value for medichal_path")
        self._medichal_path = value

    @property
    def family_path(self):
        logger.info("Getting value for family_path")
        return self._family_path

    @family_path.setter
    def family_path(self, value):
        logger.info("Setting value for family_path")
        self._family_path = value

    @property
    def demographics_path(self):
        logger.info("Getting value for demographics_path")
        return self._demographics_path

    @demographics_path.setter
    def demographics_path(self, value):
        logger.info("Setting value for demographics_path")
        self._demographics_path = value

    @property
    def enrollment_path(self):
        logger.info("Getting value for enrollment_path")
        return self._enrollment_path

    @enrollment_path.setter
    def enrollment_path(self, value):
        logger.info("Setting value for enrollment_path")
        self._enrollment_path = value

    @property
    def alchol_path(self):
        logger.info("Getting value for alchol_path")
        return self._alchol_path

    @alchol_path.setter
    def alchol_path(self, value):
        logger.info("Setting value for alchol_path")
        self._alchol_path = value

    @property
    def caffeine_path(self):
        logger.info("Getting value for caffeine_path")
        return self._caffeine_path

    @caffeine_path.setter
    def caffeine_path(self, value):
        logger.info("Setting value for caffeine_path")
        self._caffeine_path = value

    @property
    def bio_csf_abeta_path(self):
        logger.info("Getting value for bio_csf_abeta_path")
        return self._bio_csf_abeta_path

    @bio_csf_abeta_path.setter
    def bio_csf_abeta_path(self, value):
        logger.info("Setting value for bio_csf_abeta_path")
        self._bio_csf_abeta_path = value

    @property
    def bio_csf_beta_path(self):
        logger.info("Getting value for bio_csf_beta_path")
        return self._bio_csf_beta_path

    @bio_csf_beta_path.setter
    def bio_csf_beta_path(self, value):
        logger.info("Setting value for bio_csf_beta_path")
        self._bio_csf_beta_path = value

    @property
    def bio_other_path(self):
        logger.info("Getting value for bio_other_path")
        return self._bio_other_path

    @bio_other_path.setter
    def bio_other_path(self, value):
        logger.info("Setting value for bio_other_path")
        self._bio_other_path = value

    @property
    def bio_plasma_path(self):
        logger.info("Getting value for bio_plasma_path")
        return self._bio_plasma_path

    @bio_plasma_path.setter
    def bio_plasma_path(self, value):
        logger.info("Setting value for bio_plasma_path")
        self._bio_plasma_path = value

    @property
    def dat_scan_sbr_path(self):
        logger.info("Getting value for dat_scan_sbr_path")
        return self._dat_scan_sbr_path

    @dat_scan_sbr_path.setter
    def dat_scan_sbr_path(self, value):
        logger.info("Setting value for dat_scan_sbr_path")
        self._dat_scan_sbr_path = value

    @property
    def dat_scan_visual_path(self):
        logger.info("Getting value for dat_scan_visual_path")
        return self._dat_scan_visual_path

    @dat_scan_visual_path.setter
    def dat_scan_visual_path(self, value):
        logger.info("Setting value for dat_scan_visual_path")
        self._dat_scan_visual_path = value

    @property
    def dat_dti_path(self):
        logger.info("Getting value for dat_dti_path")
        return self._dat_dti_path

    @dat_dti_path.setter
    def dat_dti_path(self, value):
        logger.info("Setting value for dat_dti_path")
        self._dat_dti_path = value

    @property
    def dat_mmse_path(self):
        logger.info("Getting value for dat_mmse_path")
        return self._dat_mmse_path

    @dat_mmse_path.setter
    def dat_mmse_path(self, value):
        logger.info("Setting value for dat_mmse_path")
        self._dat_mmse_path = value

    @property
    def datscan_questions(self):
        logger.info("Getting value for datscan_questions")
        return self._datscan_questions

    @datscan_questions.setter
    def datscan_questions(self, value):
        logger.info("Setting value for datscan_questions")
        self._datscan_questions = value

    @property
    def list_prs(self):
        logger.info("Getting value for list_prs")
        return self._list_prs

    @list_prs.setter
    def list_prs(self, value):
        logger.info("Setting value for list_prs")
        self._list_prs = value

    @property
    def mono(self):
        logger.info("Getting value for mono")
        return self._mono

    @mono.setter
    def mono(self, value):
        logger.info("Setting value for mono")
        self._mono = value

    @property
    def cols_month_rename(self):
        logger.info("Getting value for cols_month_rename")
        return self._cols_month_rename

    @cols_month_rename.setter
    def cols_month_rename(self, value):
        logger.info("Setting value for cols_month_rename")
        self._cols_month_rename = value

    @property
    def list_to_drop(self):
        return self._list_to_drop

    @list_to_drop.setter
    def list_to_drop(self, value):
        logger.info("Setting value for list_to_drop")
        self._list_to_drop = value

    @property
    def list_base(self):
        logger.info("Getting value for list_base")
        return self._list_base

    @list_base.setter
    def list_base(self, value):
        logger.info("Setting value for list_base")
        self._list_base = value

    @property
    def list_scores(self):
        logger.info("Getting value for list_scores")
        return self._list_scores

    @list_scores.setter
    def list_scores(self, value):
        logger.info("Setting value for list_scores")
        self._list_scores = value

    @property
    def list_only_at_m0_to_m96(self):
        logger.info("Getting value for list_only_at_m0_to_m96")
        return self._list_only_at_m0_to_m96

    @list_only_at_m0_to_m96.setter
    def list_only_at_m0_to_m96(self, value):
        logger.info("Setting value for list_only_at_m0_to_m96")
        self._list_only_at_m0_to_m96 = value

    @property
    def list_labels(self):
        logger.info("Getting value for list_labels")
        return self._list_labels

    @list_labels.setter
    def list_labels(self, value):
        logger.info("Setting value for list_labels")
        self._list_labels = value

    @property
    def list_all(self):
        logger.info("Getting value for list_all")
        return self._list_all

    @list_all.setter
    def list_all(self, value):
        logger.info("Setting value for list_all")
        self._list_all = value

    @property
    def month_labels(self):
        logger.info("Getting value for month_labels")
        return self._month_labels

    @month_labels.setter
    def month_labels(self, value):
        logger.info("Setting value for month_labels")
        self._month_labels = value

    @property
    def list_participants_to_drop(self):
        logger.info("Getting value for list_participants_to_drop")
        return self._list_participants_to_drop

    @list_participants_to_drop.setter
    def list_participants_to_drop(self, value):
        logger.info("Setting value for list_participants_to_drop")
        self._list_participants_to_drop = value

    @property
    def all_data(self):
        logger.info("Getting value for all_data")
        return self._all_data

    @all_data.setter
    def all_data(self, value):
        logger.info("Setting value for all_data")
        self._all_data = value

    @property
    def all_participants(self):
        logger.info("Getting value for all_participants")
        return self._all_participants

    @all_participants.setter
    def all_participants(self, value):
        logger.info("Setting value for all_participants")
        self._all_participants = value

    @property
    def cohort_inital(self):
        return self._cohort_inital

    @cohort_inital.setter
    def cohort_inital(self, value):
        logger.info("Setting value for cohort_inital")
        self._cohort_inital = value

    @property
    def subject_of_cohort(self):
        logger.info("Getting value for subject_of_cohort")
        return self._subject_of_cohort

    @subject_of_cohort.setter
    def subject_of_cohort(self, value):
        logger.info("Setting value for subject_of_cohort")
        self._subject_of_cohort = value

    @property
    def list_data_to_use(self):
        logger.info("Getting value for list_data_to_use")
        return self._list_data_to_use

    @list_data_to_use.setter
    def list_data_to_use(self, value):
        logger.info("Setting value for list_data_to_use")
        self._list_data_to_use = value

    @property
    def name_for_save(self):
        logger.info("Getting value for name_for_save")
        return self._name_for_save

    @name_for_save.setter
    def name_for_save(self, value):
        logger.info("Setting value for name_for_save")
        self._name_for_save = value

    @property
    def clinichal_data(self):
        logger.info("Getting value for clinichal_data")
        return self._clinichal_data

    @clinichal_data.setter
    def clinichal_data(self, value):
        logger.info("Setting value for clinichal_data")
        self._clinichal_data = value

    @property
    def prs_data(self):
        logger.info("Getting value for prs_data")
        return self._prs_data

    @prs_data.setter
    def prs_data(self, value):
        logger.info("Setting value for prs_data")
        self._prs_data = value

    @property
    def list_not_drop(self):
        logger.info("Getting value for list_not_drop")
        return self._list_not_drop

    @list_not_drop.setter
    def list_not_drop(self, value):
        logger.info("Setting value for list_not_drop")
        self._list_not_drop = value

    @property
    def prscs_prs_path(self):
        logger.info("Getting value for prscs_prs_path")
        return self._prscs_prs_path

    @prscs_prs_path.setter
    def prscs_prs_path(self, value):
        logger.info("Setting value for prscs_prs_path")
        self._prscs_prs_path = value

    @property
    def pt_prs_path(self):
        logger.info("Getting value for pt_prs_path")
        return self._pt_prs_path

    @pt_prs_path.setter
    def pt_prs_path(self, value):
        logger.info("Setting value for pt_prs_path")
        self._pt_prs_path = value

    @property
    def edu_prs_path(self):
        logger.info("Getting value for edu_prs_path")
        return self._edu_prs_path

    @edu_prs_path.setter
    def edu_prs_path(self, value):
        logger.info("Setting value for edu_prs_path")
        self._edu_prs_path = value

    @property
    def all_prs_path(self):
        logger.info("Getting value for all_prs_path")
        return self._all_prs_path

    @all_prs_path.setter
    def all_prs_path(self, value):
        logger.info("Setting value for all_prs_path")
        self._all_prs_path = value

    @property
    def mono_path(self):
        logger.info("Getting value for mono_path")
        return self._mono_path

    @mono_path.setter
    def mono_path(self, value):
        logger.info("Setting value for mono_path")
        self._mono_path = value

    @property
    def list_data_score(self):
        logger.info("Getting value for list_data_score")
        return self._list_data_score

    @list_data_score.setter
    def list_data_score(self, value):
        logger.info("Setting value for list_data_score")
        self._list_data_score = value

    @property
    def score_data(self):
        logger.info("Getting value for score_data")
        return self._score_data

    @score_data.setter
    def score_data(self, value):
        logger.info("Setting value for score_data")
        self._score_data = value

    @property
    def problem_type(self):
        logger.info("Getting value for problem_type")
        return self._problem_type

    @problem_type.setter
    def problem_type(self, value):
        logger.info("Setting value for problem_type")
        self._problem_type = value

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, value):
        logger.info("Setting value for year")
        self._year = value

    @property
    def df(self):
        logger.info("Getting value for df")
        return self._df

    @df.setter
    def df(self, value):
        logger.info("Setting value for df")
        self._df = value

    @property
    def verbose(self):
        logger.info("Getting value for verbose")
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        logger.info("Setting value for verbose")
        self._verbose = value

    def prepare_setting_for_train(self):

        # to filter 
        self.filter = os.environ.get("FILTER")

        # for medical adjustments
        self.adjust_correction = os.environ.get("ADJUST_CORRECTION")

        if LIST_OF_COHORTS == LIST_OF_COHORTS_FOR_TRAIN:
            self.do_persist = True
        else:
            self.do_persist = False
        return True

    def populate_data(self):
        """
        Populate data 
        """
        self.data_version = DATA_VERSION
        staged_data_path = self.staged_data_path
        self.fs = FeatureStore()
        if self.is_read_from_start:
            self.fs.save_df_object_before_T()
        self.all_data = self.fs.helper_load_from_data_save_path(
            staged_data_path
        )

        if self.verbose > 5:
            if self.algorithm=='cls':
                _log='logs/'
            if self.data_version == "v1":
                _version = 'v1'

            try:
                os.remove(
                    "PATH TO LOGS"
                    + _log 
                    + _version
                    + "/pre_process_data.txt"
                )
            except Exception as e:
                logger.info(
                    f"pre_process_data file does not exist, it \
                        will be generated! This is error {e}"
                )
               
            for name in self.all_data.keys():
                self.fs.show_one_dataframe_from_persisted_features(name=name)
        return True

    def get_all_participants(self):
        """
        Get all participants 
        """
        if DATA_VERSION == "v1":
            print(self.all_data)
            participants = self.all_data[self.data_for_participants]["data"]
            self.participants = participants.loc[
                (
                    (participants["diagnosis"] == "idiopathic pd")
                    | (participants["diagnosis"] == "parkinson's disease")
                )
                & (
                    (participants["month_of_visit"] == "m0")
                    | (participants["month_of_visit"] == "sc")
                    | (participants["month_of_visit"] == "m0_5")
                )
            ]
        return True

    def get_list_of_cohort_ids(self):
        """
        Get list of cohorts
        """
        self.subject_of_cohort = set(
            self.participants[
                self.participants[self.subject_id].str.startswith(
                    self.cohort_inital
                )
            ][self.subject_id].to_list()
        )
        for subject in self.subject_of_cohort:
            if subject[0:2] != self.cohort_inital:
                raise ValueError(
                    f"This {subject} is not start with {self.cohort_inital}"
                )
        return True

    def get_clinichal_data(self):
        """
        Get Clinical data from data paths.
        """
        list_of_data_at_base_to_merge = []
        self.name_for_save = self.cohort_inital
        print(self.list_data_to_use)
        df_path=None
        for ldu in self.list_data_to_use:
            if ldu not in [
                "pt_prs",
                "prscs_prs",
                "edu_prs",
                "mono",
                "genetic_status_wgs",
            ]:
                if "moca" in ldu:
                    df_path = self.moca_path
                    str_data = "moca"
                if "schwab" in ldu:
                    df_path = self.sch_path
                    str_data = "schwab"
                if "rbd" in ldu:
                    df_path = self.rbd_path
                    str_data = "rbd"
                if "pdq39" in ldu:
                    df_path = self.pdq39_path
                    str_data = "pdq39"
                if "ess" in ldu:
                    df_path = self.ess_path
                    str_data = "ess"
                if "msq" in ldu:
                    df_path = self.msq_path
                    str_data = "msq"
                if "pd_medical_history" in ldu:
                    df_path = self.medichal_path
                    str_data = "pd_medical_history"
                if "family_history_pd" in ldu:
                    df_path = self.family_path
                    str_data = "family_history_pd"
                if "demographics" in ldu:
                    df_path = self.demographics_path
                    str_data = "demographics"
                if "enrollment" in ldu:
                    df_path = self.enrollment_path
                    str_data = "enrollment"
                if "smoking_and_alcohol_history" in ldu:
                    df_path = self.alchol_path
                    str_data = "smoking_and_alcohol_history"
                if "caffeine_history" in ldu:
                    df_path = self.caffeine_path
                    str_data = "caffeine_history"
                if "biospecimen_analyses_csf_abeta_tau_ptau" in ldu:
                    df_path = self.bio_csf_abeta_path
                    str_data = "biospecimen_abeta"
                if "biospecimen_analyses_csf_beta_glucocerebrosidase" in ldu:
                    df_path = self.bio_csf_beta_path
                    str_data = "biospecimen_beta"
                if "biospecimen_analyses_other" in ldu:
                    df_path = self.bio_other_path
                    str_data = "biospecimen_other"
                if "biospecimen_analyses_somaLogic_plasma" in ldu:
                    df_path = self.bio_plasma_path
                    str_data = "biospecimen_plasma"
                if "datscan_sbr" in ldu:
                    df_path = self.dat_scan_sbr_path
                    str_data = "datscan_sbr"
                if "lbd" in ldu:
                    df_path = self.lbd_path
                    str_data = "lbd"
                if "lbdpath" in ldu:
                    df_path = self.lbdpath_path
                    str_data = "lbdpath"
                if "upsit" in ldu:
                    df_path = self.upsit_path
                    str_data = "upsit"
                if "updrs_i_only_components" in ldu:
                    df_path = self.updrs_i_only_components_path
                    str_data = "updrs_i_only_components"
                if df_path is not None:
                    df = self.all_data[df_path]["data"]
                    # get only subjects of cohorts
                    df = df.loc[
                        df[self.subject_id].isin(self.subject_of_cohort)
                    ]

                    print(df.head())

                    # get subjects only at baseline
                    df = df.loc[
                        (
                            (df["month_of_visit"] == "m0")
                            | (df["month_of_visit"] == "sc")
                            | (df["month_of_visit"] == "m0_5")
                        )
                    ]

                    # remove some features that not useful
                    df = df.drop(
                        self.list_to_drop, errors="ignore", axis="columns"
                    )
                    # we will remove month_of_visit and we will add it later because
                    # we know that the data are filtered based on base at m0
                    df = df.drop(
                        ["month_of_visit"], errors="ignore", axis="columns"
                    )

                    print(str_data)
                    if len(df) >= self.min_threshold_of_case:
                        list_of_data_at_base_to_merge.append(df)
                        self.name_for_save = self.name_for_save + "_" + str_data
                        logger.info(
                            f"Data at {df_path} will be used for further \
                                analysis as requested !"
                        )
                    else:
                        raise ValueError(
                            f" Cohort start with {self.cohort_inital}  \
                                does not have enough subject records \
                                i.e., {min_threshold_of_case} \
                                in {str_data} only {len(df)} can be used."
                        )

                else:
                    logger.info(
                        f"{df_path} is None for Cohort start with {self.cohort_inital}!"
                    )

        try:

            self.clinichal_data = reduce(
                lambda left, right: pd.merge(
                    left,
                    right,
                    on=["subject_id"],
                    how="outer",
                    suffixes=("", "_y"),
                ),
                list_of_data_at_base_to_merge,
            )
            # remove duplicate columns
            self.clinichal_data.drop(
                self.clinichal_data.filter(regex="_y$").columns.tolist(),
                axis=1,
                inplace=True,
            )

            # add the month of visit again
            self.clinichal_data["month_of_visit"] = "m0"
            # drop possible duplicates after merging
            self.clinichal_data = self.clinichal_data.drop_duplicates(
                subset="subject_id", keep="first"
            )

        except Exception as e:
            raise ValueError(
                f" clinical or other data in {self.list_data_to_use}  \
                    can not be merged because of : {e}"
            )
        try:
            self.clinichal_data = df_at_base(self.clinichal_data, ["m0"])
            logger.info(
                'Filtering subjects records at "sc","m0#2","m0#3", or \
                "m0_5" will be used for further was successful :)'
            )
        except Exception as e:
            raise ValueError(
                f'Filtering subjects records at "sc","m0#2","m0#3",or \
                "m0_5" was not successful because of : {e}'
            )

        for mv in set(self.clinichal_data["month_of_visit"].to_list()):
            if mv not in ["m0", "sc", "m0#2", "m0#3", "m0_5"]:
                raise ValueError(
                    " Clinical data does not well filtered at \
                        baseline regarding the month of the visit! "
                )

        self.clinichal_data = df_after_drop(
            self.clinichal_data, self.list_to_drop, self.list_not_drop
        )

        for mv in set(self.clinichal_data.columns.to_list()):
            if mv in self.list_to_drop and mv not in self.list_not_drop:
                raise ValueError(
                    f" Clinical data should not have variables in \
                        {self.list_to_drop} that is also not in {self.list_not_drop} \
                            but it has! "
                )

        df_analyzer(
            self.clinichal_data,
            "Clinichal_data_and_demographics_data_"
            + self.year
            + "_"
            + self.problem_type
            + "_"
            + self.cohort_inital,
            self.list_data_to_use,
            verbose=10,
        )
        return True

    def get_prs_mono_data(self):
        """
        Get PRSs and mono genetic data from data paths.
        """
        list_of_data_at_base_to_merge = []
        for ldu in self.list_data_to_use:
            if ldu not in [
                "moca",
                "schwab",
                "rbd",
                "pdq39",
                "ess",
                "msq",
                "pd_medical_history",
                "family_history_pd",
                "demographics",
                "enrollment",
                "smoking_and_alcohol_history",
                "caffeine_history",
                "biospecimen_analyses_csf_abeta_tau_ptau",
                "biospecimen_analyses_csf_beta_glucocerebrosidase",
                "biospecimen_analyses_other",
                "biospecimen_analyses_somaLogic_plasma",
                "datscan_sbr",
                "datscan_visual_interpretation",
                "dti",
                "mmse",
                "lbd",
                "lbdpath",
                "upsit",
                "updrs_i_only_components",
            ]:
                if "pt_prs" in ldu:
                    df_path = self.pt_prs_path
                    str_data = "ptprs"
                if "prscs_prs" in ldu:
                    df_path = self.prscs_prs_path
                    str_data = "prscsprs"
                if "edu_prs" in ldu:
                    df_path = self.edu_prs_path
                    str_data = "eduprs"
                if "all_prs" in ldu:
                    df_path = self.all_prs_path
                    str_data = "allprs"
                if "mono" in ldu:
                    df_path = self.mono_path
                    str_data = "mono"
                if "genetic_status_wgs" in ldu:
                    df_path = self.mono_path
                    str_data = "genetic_status_wgs"

                if df_path is not None:
                    df = self.all_data[df_path]["data"]
                    df = df.loc[
                        df[self.subject_id].isin(self.subject_of_cohort)
                    ]
                    if len(df) > 0:
                        df["month_of_visit"] = "m0"
                        list_of_data_at_base_to_merge.append(df)
                        self.name_for_save = self.name_for_save + "_" + str_data
                    else:
                        raise ValueError(
                            f" Cohort start with {self.cohort_inital}  \
                                does not have records of {str_data}"
                        )
                else:
                    logger.info(
                        f"{df_path} is None for Cohort start with \
                            {self.cohort_inital}!"
                    )

        try:
            self.prs_data = reduce(
                lambda left, right: pd.merge(
                    left,
                    right,
                    on=["subject_id", "month_of_visit"],
                    how="outer",
                    suffixes=("", "_y"),
                ),
                list_of_data_at_base_to_merge,
            )
            # drop possible duplicates after merging
            self.prs_data = self.prs_data.drop_duplicates()
            # remove duplicate columns
            self.prs_data.drop(
                self.prs_data.filter(regex="_y$").columns.tolist(),
                axis=1,
                inplace=True,
            )

        except Exception as e:
            raise ValueError(
                f" PRSs and Monogenic data in {list_of_data_at_base_to_merge}  \
                    can not be merged because of: {e}"
            )
        list_of_data_that_really_used = set(self.list_data_to_use).difference(
            set(["moca", "medical_history", "family", "mayo", "demographics"])
        )
        df_analyzer(
            self.prs_data,
            "PRSs_and_monogenic_data_"
            + self.year
            + "_"
            + self.problem_type
            + "_"
            + self.cohort_inital,
            list_of_data_that_really_used,
            verbose=10,
        )
        return True

    def get_score_data(self):
        """
        Access MDS-UPDRS data sets and merge them
        with formerly proceed dataset
        """
        list_of_data_at_base_to_merge = []
        for lds in self.list_data_score:
            if "updrs_i" in lds:
                df_path = self.updrs_i_path
                str_data = "updrsi"
            if "updrs_ii" in lds:
                df_path = self.updrs_ii_path
                str_data = "updrsii"
            if "updrs_iii" in lds:
                df_path = self.updrs_iii_path
                str_data = "updrsiii"
            # check to see if the data path exist or not
            if df_path is not None:
                df = self.all_data[df_path]["data"]
                df = df.loc[df[self.subject_id].isin(self.subject_of_cohort)]
                if len(df) > 0:
                    list_of_data_at_base_to_merge.append(df)
                    self.name_for_save = self.name_for_save + "_" + str_data
                else:
                    raise ValueError(
                        f" Cohort start with {self.cohort_inital}  \
                            does not have records of {str_data}"
                    )
        try:
            self.score_data = reduce(
                lambda left, right: pd.merge(
                    left,
                    right,
                    on=["subject_id", "month_of_visit"],
                    how="outer",
                    suffixes=("", "_y"),
                ),
                list_of_data_at_base_to_merge,
            )
            # drop possible duplicates after merging
            self.score_data = self.score_data.drop_duplicates()
            # remove duplicate columns
            self.score_data.drop(
                self.score_data.filter(regex="_y$").columns.tolist(),
                axis=1,
                inplace=True,
            )

        except Exception as e:
            raise ValueError(f" Score  data can not merged because of : {e}")

        self.score_data = df_after_drop(
            self.score_data, self.list_to_drop, self.list_not_drop
        )

        df_analyzer(
            self.score_data,
            "UPDRs_score_data_before_cleaning_"
            + self.year
            + "_"
            + self.problem_type
            + "_"
            + self.cohort_inital,
            ["UPDRS part I", "UPDRS part II", "UPDRS part III"],
            verbose=10,
        )
        return True

    def do_score_data_clean(self):
        """
        Clean  MDS-UPDRS data sets. Use only 
        records that visit month is between baseline and
        36 months.
        """

        if self.data_version == "v1":
            acceptable_subjects = []
            self.score_data.loc[
                (self.score_data["month_of_visit"] == "sc")
                | (self.score_data["month_of_visit"] == "m0#2")
                | (self.score_data["month_of_visit"] == "m0#3")
                | (self.score_data["month_of_visit"] == "m0_5")
            ]["month_of_visit"] = "m0"
            sd = self.score_data.copy()
            if self.year == "12" or self.year == "24" or self.year == "36":
                all_subjects = sd["subject_id"].unique()
                for s in all_subjects:
                    if (
                        sd[sd["subject_id"] == s]["month_of_visit"].nunique()
                        >= 2
                        and "m0"
                        in sd[sd["subject_id"] == s]["month_of_visit"].to_list()
                        and "m06"
                        in sd[sd["subject_id"] == s]["month_of_visit"].to_list()
                        and "m12"
                        in sd[sd["subject_id"] == s]["month_of_visit"].to_list()
                        and (
                            (
                                sd[
                                    (sd["subject_id"] == s)
                                    & (sd["month_of_visit"] == "m0")
                                ]["mds_updrs_part_i_summary_score"].to_list()[0]
                                != np.nan
                                and sd[
                                    (sd["subject_id"] == s)
                                    & (sd["month_of_visit"] == "m0")
                                ]["mds_updrs_part_ii_summary_score"].to_list()[
                                    0
                                ]
                                != np.nan
                                and sd[
                                    (sd["subject_id"] == s)
                                    & (sd["month_of_visit"] == "m0")
                                ]["mds_updrs_part_iii_summary_score"].to_list()[
                                    0
                                ]
                                != np.nan
                            )
                            and (
                                sd[
                                    (sd["subject_id"] == s)
                                    & (sd["month_of_visit"] == "m06")
                                ]["mds_updrs_part_i_summary_score"].to_list()[0]
                                != np.nan
                                and sd[
                                    (sd["subject_id"] == s)
                                    & (sd["month_of_visit"] == "m06")
                                ]["mds_updrs_part_ii_summary_score"].to_list()[
                                    0
                                ]
                                != np.nan
                                and sd[
                                    (sd["subject_id"] == s)
                                    & (sd["month_of_visit"] == "m06")
                                ]["mds_updrs_part_iii_summary_score"].to_list()[
                                    0
                                ]
                                != np.nan
                            )
                            and (
                                sd[
                                    (sd["subject_id"] == s)
                                    & (sd["month_of_visit"] == "m12")
                                ]["mds_updrs_part_i_summary_score"].to_list()[0]
                                != np.nan
                                and sd[
                                    (sd["subject_id"] == s)
                                    & (sd["month_of_visit"] == "m12")
                                ]["mds_updrs_part_ii_summary_score"].to_list()[
                                    0
                                ]
                                != np.nan
                                and sd[
                                    (sd["subject_id"] == s)
                                    & (sd["month_of_visit"] == "m12")
                                ]["mds_updrs_part_iii_summary_score"].to_list()[
                                    0
                                ]
                                != np.nan
                            )
                        )
                    ):
                        acceptable_subjects.append(s)

        self.score_data = self.score_data[
            self.score_data["subject_id"].isin(acceptable_subjects)
        ]
        self.score_data = self.score_data.loc[
            (self.score_data["month_of_visit"] == "m0")
            | (self.score_data["month_of_visit"] == "m06")
            | (self.score_data["month_of_visit"] == "m12")
            | (self.score_data["month_of_visit"] == "m18")
            | (self.score_data["month_of_visit"] == "m24")
            | (self.score_data["month_of_visit"] == "m30")
            | (self.score_data["month_of_visit"] == "m36")
        ]
        df_analyzer(
            self.score_data,
            "UPDRs_score_data_after_cleaning_"
            + self.year
            + "_"
            + self.problem_type
            + "_"
            + self.cohort_inital,
            ["UPDRS part I", "UPDRS part II", "UPDRS part III"],
            verbose=10,
        )

        return True

    def merge_temporary_score_and_medical(self):
        """
        Merge formerly proceed data and datasets that 
        contains medical information.
        """
        medichal_df = self.all_data[self.medichal_path]["data"][
            [
                "subject_id",
                "month_of_visit",
                "on_levodopa",
                "on_dopamine_agonist",
                "on_other_pd_medications",
            ]
        ]
        print(medichal_df.head())
        self.score_data = self.score_data.merge(
            medichal_df, on=["subject_id", "month_of_visit"], how="left"
        )
        print(self.score_data.head())
        return True

    def do_medichal_adjust_score(self):
        """
        Medical correction happens here.
        """
        
        if self.data_version == "v1":
            print(self.score_data.head())
            for month in ["m0", "m06", "m12", "m18", "m24", "m30", "m36"]:
                for up in self.list_scores:
                    self.score_data = uplift_scores_all_together(
                        self.score_data,
                        self.cohort_inital,
                        "on_levodopa",
                        up,
                        month,
                        do_persist=self.do_persist,
                        name="adj_score_"
                        + self.year
                        + "_"
                        + self.problem_type
                        + "_",
                    )

        return True

    def do_score_data_transpose(self):
        """
        Prepare data for calculating scores.
        """
        

        sd = self.score_data.copy()
        sd = sd[["subject_id", "month_of_visit"] + self.list_scores].copy()
        print(
            sd[
                [
                    "subject_id",
                    "month_of_visit",
                    "mds_updrs_part_i_summary_score",
                ]
            ].head(7)
        )
        print(
            sd[
                [
                    "subject_id",
                    "month_of_visit",
                    "mds_updrs_part_ii_summary_score",
                ]
            ].head(7)
        )
        print(
            sd[
                [
                    "subject_id",
                    "month_of_visit",
                    "mds_updrs_part_iii_summary_score",
                ]
            ].head(7)
        )
        sd_cls = sd.pivot(
            index="subject_id",
            columns=["month_of_visit"],
            values=self.list_scores,
        )
        sd_cls = sd_cls.reset_index()
        print(sd_cls.head())
        sd_cls.columns = sd_cls.columns.droplevel()
        print(sd_cls.columns)
        print(self.cols_month_rename)
        sd_cls.columns = ["subject_id"] + self.cols_month_rename
        sd_cls.dropna(subset=["subject_id", "mo0nth0"], inplace=True)
        sd_cls_T = sd_cls.T.copy()

        # divid data into three datasets to fill nulls and then merge them again

        sd_cls_T_p1 = sd_cls_T[0:8].copy()
        sd_cls_T_p1=sd_cls_T_p1.fillna(sd_cls_T_p1[1:8].mean())
        print(sd_cls_T_p1.head(8))
        sd_cls_T_p2 = sd_cls_T[8:15].copy()
        sd_cls_T_p2=sd_cls_T_p2.fillna(sd_cls_T_p2.mean())
        print(sd_cls_T_p2.head(8))
        sd_cls_T_p3 = sd_cls_T[15:22].copy()
        sd_cls_T_p3=sd_cls_T_p3.fillna(sd_cls_T_p3.mean())
        print(sd_cls_T_p3.head(8))

        # combine all 
        sd_cls_T_all = pd.concat([sd_cls_T_p1,sd_cls_T_p2,sd_cls_T_p3])
        print(sd_cls_T_all.head(22))


        sd_cls = sd_cls_T_all.T.copy()
        print(sd_cls.head(22))
        sd_cls["month_of_visit"] = "m0"
        self.score_data = reduce(
            lambda left, right: pd.merge(
                left,
                right,
                on=["subject_id", "month_of_visit"],
                how="left",
                suffixes=("", "_y"),
            ),
            [sd_cls] + [self.score_data],
        )
        # drop possible duplicates after merging
        self.score_data = self.score_data.drop_duplicates()
        # remove duplicate columns
        self.score_data.drop(
            self.score_data.filter(regex="_y$").columns.tolist(),
            axis=1,
            inplace=True,
        )

        df_analyzer(
            self.score_data,
            "UPDRs_score_data_preparing_for_calculating_slopes_"
            + self.year
            + "_"
            + self.problem_type
            + "_"
            + self.cohort_inital,
            ["UPDRS part I + UPDRS part II + UPDRS part III"],
            verbose=10,
        )

        return True

    def calc_slopes(self):
        """
        Calculating scores.
        """

        
        if self.data_version == "v1":
            self.score_data = make_slopes(
                self.score_data, self.year, self.problem_type
            )
        df_analyzer(
            self.score_data,
            "UPDRs_score_data_after_calculating_slopes_"
            + self.year
            + "_"
            + self.problem_type
            + "_"
            + self.cohort_inital,
            ["UPDRS part I + UPDRS part II + UPDRS part III"],
            verbose=10,
        )
        return True

    def merged_all_drop_nulls(self):
        """
        Merge formerly proceed data with scores.
        """
        list_for_merge = [self.score_data, self.clinichal_data, self.prs_data]
        self.df = reduce(
            lambda left, right: pd.merge(
                left, right, on=["subject_id"], how="left", suffixes=("", "_y")
            ),
            list_for_merge,
        )
        # drop possible duplicates after merging
        self.df = self.df.drop_duplicates()
        self.df.drop(
            self.df.filter(regex="_y$").columns.tolist(), axis=1, inplace=True
        )

        print("clinichal_data")
        print(self.clinichal_data[["subject_id", "month_of_visit"]].head())
        print(self.clinichal_data.shape)

        print("prs_data")
        print(self.prs_data[["subject_id", "month_of_visit"]].head())
        print(self.prs_data.shape)

        print("score_data")
        print(self.score_data[["subject_id", "month_of_visit"]].head())
        print(self.score_data.shape)

        print("df")
        print(self.df[["subject_id", "month_of_visit"]].head())
        print(self.df.shape)

        self.df.dropna(subset=["label_" + self.problem_type], inplace=True)
        # only drop from those which should not be dropped
        print(self.df[self.never_drop_features])

        subset_df = self.df[self.never_drop_features].copy()
        self.df.dropna(
            axis="columns",
            thresh=(1 - self.threshold) * len(self.df),
            inplace=True,
        )
        self.df[self.never_drop_features] = subset_df.copy()
        print(self.df["label_" + self.problem_type].value_counts(dropna=False))
        self.df = df_after_drop(self.df, self.list_to_drop, self.list_not_drop)

        return True

    def code_categorichal_variables(self):
        self.df.replace(to_replace=["stage 0"], value=0, inplace=True)
        if len(self.df[self.df.eq("stage 0").any(1)]) > 1:
            raise ValueError(" this stage 1 is not replaced with a code :(")
        self.df.replace(to_replace=["stage 1"], value=1, inplace=True)
        if len(self.df[self.df.eq("stage 1").any(1)]) > 1:
            raise ValueError(" this stage 1 is not replaced with a code :(")
        self.df.replace(to_replace=["stage 2"], value=2, inplace=True)
        if len(self.df[self.df.eq("stage 2").any(1)]) > 1:
            raise ValueError(" this stage 2 is not replaced with a code :(")
        self.df.replace(to_replace=["stage 3"], value=3, inplace=True)
        if len(self.df[self.df.eq("stage 3").any(1)]) > 1:
            raise ValueError(" this stage 3 is not replaced with a code :(")
        self.df.replace(to_replace=["stage 4"], value=4, inplace=True)
        if len(self.df[self.df.eq("stage 4").any(1)]) > 1:
            raise ValueError(" this stage 4 is not replaced with a code :(")
        self.df.replace(to_replace=["stage 5"], value=5, inplace=True)
        if len(self.df[self.df.eq("stage 5").any(1)]) > 1:
            raise ValueError(" this stage 5 is not replaced with a code :(")

        self.df.replace(to_replace=["would never doze"], value=1, inplace=True)
        if len(self.df[self.df.eq("would never doze").any(1)]) > 1:
            raise ValueError(
                " this would never doze is not replaced with a code :("
            )
        self.df.replace(
            to_replace=["slight chance of dozing"], value=2, inplace=True
        )
        if len(self.df[self.df.eq("slight chance of dozing").any(1)]) > 1:
            raise ValueError(
                " this slight chance of dozing is not replaced with a code :("
            )
        self.df.replace(
            to_replace=["moderate chance of dozing"], value=3, inplace=True
        )
        if len(self.df[self.df.eq("moderate chance of dozing").any(1)]) > 1:
            raise ValueError(
                " This moderate chance of dozing is not replaced with a code :("
            )
        self.df.replace(
            to_replace=["high chance of dozing"], value=4, inplace=True
        )
        if len(self.df[self.df.eq("high chance of dozing").any(1)]) > 1:
            raise ValueError(
                " This high chance of dozing is not replaced with a code :("
            )

        self.df.replace(to_replace=["no"], value=0, inplace=True)
        if len(self.df[self.df.eq("no").any(1)]) > 1:
            raise ValueError(" this no is not replaced with a code :(")
        self.df.replace(to_replace=["yes"], value=1, inplace=True)
        if len(self.df[self.df.eq("yes").any(1)]) > 1:
            raise ValueError(" this yes is not replaced with a code :(")

        self.df.replace(to_replace=["never"], value=1, inplace=True)
        if len(self.df[self.df.eq("never").any(1)]) > 1:
            raise ValueError(" this never is not replaced with a code :(")
        self.df.replace(to_replace=["occasionally"], value=2, inplace=True)
        if len(self.df[self.df.eq("occasionally").any(1)]) > 1:
            raise ValueError(
                " This occasionally is not replaced with a code :("
            )
        self.df.replace(to_replace=["sometimes"], value=3, inplace=True)
        if len(self.df[self.df.eq("sometimes").any(1)]) > 1:
            raise ValueError(" this sometimes is not replaced with a code :(")
        self.df.replace(to_replace=["often"], value=4, inplace=True)
        if len(self.df[self.df.eq("often").any(1)]) > 1:
            raise ValueError(" this often is not replaced with a code :(")
        self.df.replace(
            to_replace=["always or cannot do at all"], value=5, inplace=True
        )
        if len(self.df[self.df.eq("always or cannot do at all").any(1)]) > 1:
            raise ValueError(
                " this always or cannot do at all is not replaced with a code :("
            )

        self.df.replace(to_replace=["normal"], value=1, inplace=True)
        if len(self.df[self.df.eq("normal").any(1)]) > 1:
            raise ValueError(" this normal is not replaced with a code :(")
        self.df.replace(to_replace=["slight"], value=2, inplace=True)
        if len(self.df[self.df.eq("slight").any(1)]) > 1:
            raise ValueError(" this slight is not replaced with a code :(")
        self.df.replace(to_replace=["mild"], value=3, inplace=True)
        if len(self.df[self.df.eq("mild").any(1)]) > 1:
            raise ValueError(" this mild is not replaced with a code :(")
        self.df.replace(to_replace=["moderate"], value=4, inplace=True)
        if len(self.df[self.df.eq("moderate").any(1)]) > 1:
            raise ValueError(" this moderate is not replaced with a code :(")
        self.df.replace(to_replace=["severe"], value=5, inplace=True)
        if len(self.df[self.df.eq("severe").any(1)]) > 1:
            raise ValueError(" this severe is not replaced with a code :(")
        self.df.replace(to_replace=["male"], value=0, inplace=True)
        if len(self.df[self.df.eq("male").any(1)]) > 1:
            raise ValueError(" this male is not replaced with a code :(")
        self.df.replace(to_replace=["female"], value=1, inplace=True)
        if len(self.df[self.df.eq("female").any(1)]) > 1:
            raise ValueError(" this female is not replaced with a code :(")

        return True

    def test_and_dump(self):
        """
        Check data quality and dump them.
        """
        self.name_for_save = (
            self.name_for_save
            + "_"
            + str(threshold)
            + "_label_"
            + self.problem_type
            + "_"
            + self.year
        )

        for col in self.df.columns:
            if "label" in col:
                if self.df[col].isnull().mean() > 0:
                    raise ValueError(
                        f" The final data in {col} has null values with \
                            {self.df[col].isnull().mean()} % which \
                                is not acceptable."
                    )
                else:
                    logger.info("Test 1 pass :)) no null values in labels!!!")
        
        
        for col in self.df.columns:
            if 'code_' not in col:
                if 'upd2' in col:
                    self.df.drop([col], axis = 1, inplace = True, errors='ignore')

        if self.adjust_correction == "True":
            prefix_df_for_save = "df_after_adjustment"
        else:
            prefix_df_for_save = "df_before_adjustment"
        
        if self.adjust_correction == 'True':
            str_part = self.cohort_inital+"_"
        else:
            str_part = self.cohort_inital+"_no_medical_adjustment"

        self.fs.helper_persist_to_data_save_path(
            self.df,
            str_part
            + prefix_df_for_save  # self.main_base
            + "_"  # self.main_base
            + "train_data"  # self.main_base
            + "_"
            + str(self.threshold)
            + "_"
            + "label_"
            + self.problem_type
            + "_"
            + self.year,
            flag_to_rewrite=True,
        )

        print(self.name_for_save)
        df_analyzer(
            self.df,
            str_part
            + prefix_df_for_save  # self.main_base
            + "_"  # self.main_base
            + "train_data"  # self.main_base
            + "_"
            + str(self.threshold)
            + "_"
            + "label_"
            + self.problem_type
            + "_"
            + self.year,
            ["All data"],
            verbose=10,
        )

        return True
    
    def hand_pickup_features(self):
        """
        Extra refining features for better predictions. 
        """
        for col in self.df.columns:
            if 'mds_updrs_part_i_pat_quest_sub_score' in col or \
                'mds_updrs_part_i_summary_score' in col :
                self.df.drop([col], axis=1, inplace=True, errors='ignore')
        
        return True

    def test_and_dump_extra_filters(self,filter='no_medicated'):
        """
        Test them
        
        """
        for col in self.df.columns:
            if "label" in col:
                if self.df[col].isnull().mean() > 0:
                    raise ValueError(
                        f" The final data in {col} has null values with \
                            {self.df[col].isnull().mean()} % which \
                                is not acceptable."
                    )
                else:
                    logger.info("Test 1 pass :)) no null values in labels!!!")

        if self.adjust_correction == "True":
            prefix_df_for_save = "df_after_adjustment"
        else:
            prefix_df_for_save = "df_before_adjustment"
        
        if self.adjust_correction == 'True':
            str_part = self.cohort_inital+"_"
        else:
            str_part = self.cohort_inital+"_no_medical_adjustment"

        # filters will apply here
        # case 1 : non-medicated subjects 
        # non-medicated subjects
        if filter=='no_medicated':
            all_medical_data = self.all_data['pd_medical_history']['data']
            all_medical_data = all_medical_data[['subject_id','on_levodopa','on_dopamine_agonist','on_other_pd_medications']]
            non_medicated_subjects = all_medical_data.loc[(all_medical_data['on_levodopa']=='no')
                                                    &(all_medical_data['on_dopamine_agonist']=='no')
                                                    &(all_medical_data['on_other_pd_medications']=='no')
                                                    ]['subject_id']
            # set non-medicated subjects
            non_medicated_subjects_list = set(non_medicated_subjects.to_list())
            # filter data frame
            self.df=self.df[self.df['subject_id'].isin(non_medicated_subjects_list)]

        # revise name based on filter status
        
        self.name_for_save = (
            self.name_for_save
            + filter # to apply filter
            + "_"
            + str(threshold)
            + "_label_"
            + self.problem_type
            + "_"
            + self.year
            + "_"
            + filter
        )
        self.fs.helper_persist_to_data_save_path(
            self.df,
            str_part
            + filter # to apply filter
            + "_"
            + prefix_df_for_save  # self.main_base
            + "_"  # self.main_base
            + "train_data"  # self.main_base
            + "_"
            + str(self.threshold)
            + "_"
            + "label_"
            + self.problem_type
            + "_"
            + self.year
            ,
            flag_to_rewrite=True,
        )

        print(self.name_for_save)
        df_analyzer(
            self.df,
            str_part
            + prefix_df_for_save  # self.main_base
            + "_"  # self.main_base
            + "train_data"  # self.main_base
            + "_"
            + str(self.threshold)
            + "_"
            + "label_"
            + self.problem_type
            + "_"
            + self.year,
            ["All data"],
            verbose=10,
        )

        return True

    def print_info(self):
        print("#######")
        print("#######")
        print("#######")
        print("#######")
        print("#######")
        print('columns for final df is :')
        print(*sorted(self.df.columns),sep='\n')

def train_builder_runner():
    """
    Run data pipeline for each case.  
    """
    bt = BuildTrain(
        cohort_inital="pp",
        list_data_to_use=list_data_to_use,
        list_data_score=["updrs_i", "updrs_ii", "updrs_iii"],
        problem_type="i",
        year="12",
        # % toleration for null values for each variable 
        threshold=0.5,
        verbose=10,
    )
    bt.prepare_setting_for_train()
    bt.populate_data()
    bt.get_all_participants()
    bt.get_list_of_cohort_ids()
    bt.get_clinichal_data()
    bt.get_prs_mono_data()
    bt.get_score_data()
    bt.do_score_data_clean()
    bt.merge_temporary_score_and_medical()
    if bt.adjust_correction == "True":
        bt.do_medichal_adjust_score()
    bt.do_score_data_transpose()
    bt.calc_slopes()
    bt.merged_all_drop_nulls()
    bt.code_categorichal_variables()
    bt.test_and_dump()
    if DEBUG_FEATURES:
        bt.hand_pickup_features()
    if bt.filter=="no_medicated":
        bt.test_and_dump_extra_filters(filter="no_medicated")
    bt.print_info()

    return True


def multiple_train_builder_runner():
    """
    Run data pipeline for all cases.  
    """
    cohort_inital = ["pd"]
    list_data = [list_data_to_use]
    list_data_score = [["updrs_i", "updrs_ii", "updrs_iii"]]
    problem_type = ["i", "ii", "iii", "total"]
    year = ["12", "18", "24", "36"]
    # % toleration for null values for each variable 
    threshold = [0.5]
    verbose = [10]

    for ci in cohort_inital:
        for ld in list_data:
            for lds in list_data_score:
                for pt in problem_type:
                    for y in year:
                        for t in threshold:
                            for v in verbose:
                                bt = BuildTrain(
                                    cohort_inital=ci,
                                    list_data_to_use=ld,
                                    list_data_score=lds,
                                    problem_type=pt,
                                    year=y,
                                    threshold=t,
                                    verbose=v,
                                )

                                bt.prepare_setting_for_train()
                                bt.populate_data()
                                bt.get_all_participants()
                                bt.get_list_of_cohort_ids()
                                bt.get_clinichal_data()
                                bt.get_prs_mono_data()
                                bt.get_score_data()
                                bt.do_score_data_clean()
                                bt.merge_temporary_score_and_medical()
                                if bt.adjust_correction == "True":
                                    bt.do_medichal_adjust_score()
                                bt.do_score_data_transpose()
                                bt.calc_slopes()
                                bt.merged_all_drop_nulls()
                                bt.code_categorichal_variables()
                                bt.test_and_dump()
                                if bt.filter=="no_medicated":
                                    bt.test_and_dump_extra_filters(filter="no_medicated")
    return True


if __name__ == "__main__":
    
    # build only one train data
    train_builder_runner()
    # build all train data for one data source
    # multiple_train_builder_runner()
