# Conf file for preparing train sets 
###################################################
###################################################
###################################################
###################################################

import os
from pathlib import Path
from dotenv import load_dotenv

DATA_VERSION = os.environ.get("DATA_VERSION")
LIST_OF_COHORTS = os.environ.get("LIST_OF_COHORTS")
LIST_OF_COHORTS_FOR_TRAIN = os.environ.get("LIST_OF_COHORTS_FOR_TRAIN")
BUILD_DATA_FROM_SCRATCH = os.environ.get("BUILD_DATA_FROM_SCRATCH")
ALGORITHM = os.environ.get("ALGORITHM")
print(DATA_VERSION)



if "pp" in LIST_OF_COHORTS and len(str(LIST_OF_COHORTS))==6:
    min_threshold_of_case = 500
    list_data_to_use = [
        
        "edu_prs",
        "all_prs",
        "mono",
        "moca",
        "schwab",
        "ess",
        "pd_medical_history",
        "family_history_pd",
        "demographics",
        "enrollment",
        "biospecimen_analyses_csf_abeta_tau_ptau",
        "datscan_sbr",
        "datscan_visual_interpretation",
        "dti",
        "mmse",
        "upsit",
    ]
if "pd" in LIST_OF_COHORTS and len(str(LIST_OF_COHORTS))==6:
    min_threshold_of_case = 500
    list_data_to_use = [
        "pt_prs",
        "edu_prs",
        "all_prs",
        "mono",
        "genetic_status_wgs",
        "moca",
        "schwab",
        "pdq39",
        "ess",
        "msq",
        "pd_medical_history",
        "family_history_pd",
        "demographics",
        "enrollment",
        "smoking_and_alcohol_history",
        "datscan_visual_interpretation",
        "dti",
        "mmse",
    ]


if DATA_VERSION == "v1":
    is_read_from_start = BUILD_DATA_FROM_SCRATCH
    min_threshold_of_case = 490
    threshold = 50
    monitoring_time_line = 12
    staged_data_path = "stage2_dictionary_sorted_summrized_dataframes.pkl"
    data_for_participants = "pd_medical_history"
    data_for_participants_second_part = "data"
    case_control_other_at_baseline = None
    md_history = "pd_medical_history"
    updrs_i_only_components = "updrs_i_only_components"
    subject_id = "subject_id"

    mds_updrs_part_i_path = "mds_updrs_part_i"
    mds_updrs_part_ii_path = "mds_updrs_part_ii"
    mds_updrs_part_iii_path = "mds_updrs_part_iii"

    prscs_prs_path = "prscs_prs"
    pt_prs_path = None
    edu_prs_path = "edu_prs"
    # not available for version 1
    all_prs_path = None
    mono_path = "mono"

    updrs_i_only_components_path = "updrs_i_only_components"
    moca_path = "moca"
    schwab_path = "schwab"
    rbd_path = "rbd"
    pdq39_path = "pdq39"
    ess_path = "ess"
    msq_path = "msq"
    mmse_path = "mmse"
    pd_medical_history_path = (
        #  has info in all months but mostly at base
        "pd_medical_history"
    )
    family_history_pd_path = (
        #  has info in all months but mostly at base
        "family_history_pd"
    )
    demographics_path = (
        #  has info in all months but mostly at base
        "demographics"
    )
    #  has info in all months but mostly at base
    enrollment_path = "enrollment"
    smoking_and_alcohol_history_path = "smoking_and_alcohol_history"
    #  error in name
    caffeine_history_path = "caffeine_history"
    biospecimen_analyses_csf_beta_glucocerebrosidase_path = (
        "biospecimen_analyses_csf_beta_glucocerebrosidase"
    )
    biospecimen_analyses_csf_abeta_tau_ptau_path = (
        "biospecimen_analyses_csf_abeta_tau_ptau"
    )
    biospecimen_analyses_other_path = "biospecimen_analyses_other"
    biospecimen_analyses_somaLogic_plasma_path = (
        "Biospecimen_analyses_SomaLogic_plasma"
    )
    datscan_sbr_path = "datscan_sbr"
    datscan_visual_interpretation_path = "datscan_visual_interpretation"
    dti_path = "dti"
    lbd_path = None
    lbdpath_path = None
    upsit_path = "upsit"
    #  in this version only use on_levodopa



   # List of initial variables, some of them will be dropped 
   # or updated while creating training set
   ####################################################### 
   ####################################################### 
   ####################################################### 
   ####################################################### 


    DATSCAN = [
        "sbr_caudate_r",
        "sbr_caudate_l",
        "sbr_putamen_r",
        "sbr_putamen_l",
    ]

    QUESTIONS_CODE_I = [
        "mds_updrs_part_i_primary_info_source",
        "code_upd2101_cognitive_impairment",
        "code_upd2102_hallucinations_and_psychosis",
        "code_upd2103_depressed_mood",
        "code_upd2104_anxious_mood",
        "code_upd2105_apathy",
        "code_upd2106_dopamine_dysregulation_syndrome_features",
        "upd2101_cognitive_impairment",
        "upd2102_hallucinations_and_psychosis",
        "upd2103_depressed_mood",
        "upd2104_anxious_mood",
        "upd2105_apathy",
        "upd2106_dopamine_dysregulation_syndrome_features",
        "mds_updrs_part_i_sub_score",
        "mds_updrs_part_i_pat_quest_primary_info_source",
        "code_upd2107_pat_quest_sleep_problems",
        "code_upd2108_pat_quest_daytime_sleepiness",
        "code_upd2109_pat_quest_pain_and_other_sensations",
        "code_upd2110_pat_quest_urinary_problems",
        "code_upd2111_pat_quest_constipation_problems",
        "code_upd2112_pat_quest_lightheadedness_on_standing",
        "code_upd2113_pat_quest_fatigue",
        "upd2107_pat_quest_sleep_problems",
        "upd2108_pat_quest_daytime_sleepiness",
        "upd2109_pat_quest_pain_and_other_sensations",
        "upd2110_pat_quest_urinary_problems",
        "upd2111_pat_quest_constipation_problems",
        "upd2112_pat_quest_lightheadedness_on_standing",
        "upd2113_pat_quest_fatigue",
        "mds_updrs_part_i_pat_quest_sub_score",
    ]

    QUESTIONS_CODE_II = [
        "mds_updrs_part_ii_primary_info_source",
        "code_upd2201_speech",
        "code_upd2202_saliva_and_drooling",
        "code_upd2203_chewing_and_swallowing",
        "code_upd2204_eating_tasks",
        "code_upd2205_dressing",
        "code_upd2206_hygiene",
        "code_upd2207_handwriting",
        "code_upd2208_doing_hobbies_and_other_activities",
        "code_upd2209_turning_in_bed",
        "code_upd2210_tremor",
        "code_upd2211_get_out_of_bed_car_or_deep_chair",
        "code_upd2212_walking_and_balance",
        "code_upd2213_freezing",
        "upd2201_speech",
        "upd2202_saliva_and_drooling",
        "upd2203_chewing_and_swallowing",
        "upd2204_eating_tasks",
        "upd2205_dressing",
        "upd2206_hygiene",
        "upd2207_handwriting",
        "upd2208_doing_hobbies_and_other_activities",
        "upd2209_turning_in_bed",
        "upd2210_tremor",
        "upd2211_get_out_of_bed_car_or_deep_chair",
        "upd2212_walking_and_balance",
        "upd2213_freezing",
    ]

    QUESTIONS_CODE_III = [
        "code_upd2301_speech_problems",
        "code_upd2302_facial_expression",
        "code_upd2303a_rigidity_neck",
        "code_upd2303b_rigidity_rt_upper_extremity",
        "code_upd2303c_rigidity_left_upper_extremity",
        "code_upd2303d_rigidity_rt_lower_extremity",
        "code_upd2303e_rigidity_left_lower_extremity",
        "code_upd2304a_right_finger_tapping",
        "code_upd2304b_left_finger_tapping",
        "code_upd2305a_right_hand_movements",
        "code_upd2305b_left_hand_movements",
        "code_upd2306a_pron_sup_movement_right_hand",
        "code_upd2306b_pron_sup_movement_left_hand",
        "code_upd2307a_right_toe_tapping",
        "code_upd2307b_left_toe_tapping",
        "code_upd2308a_right_leg_agility",
        "code_upd2308b_left_leg_agility",
        "code_upd2309_arising_from_chair",
        "code_upd2310_gait",
        "code_upd2311_freezing_of_gait",
        "code_upd2312_postural_stability",
        "code_upd2313_posture",
        "code_upd2314_body_bradykinesia",
        "code_upd2315a_postural_tremor_of_right_hand",
        "code_upd2315b_postural_tremor_of_left_hand",
        "code_upd2316a_kinetic_tremor_of_right_hand",
        "code_upd2316b_kinetic_tremor_of_left_hand",
        "code_upd2317a_rest_tremor_amplitude_right_upper_extremity",
        "code_upd2317b_rest_tremor_amplitude_left_upper_extremity",
        "code_upd2317c_rest_tremor_amplitude_right_lower_extremity",
        "code_upd2317d_rest_tremor_amplitude_left_lower_extremity",
        "code_upd2317e_rest_tremor_amplitude_lip_or_jaw",
        "code_upd2318_consistency_of_rest_tremor",
        "upd2301_speech_problems",
        "upd2302_facial_expression",
        "upd2303a_rigidity_neck",
        "upd2303b_rigidity_rt_upper_extremity",
        "upd2303c_rigidity_left_upper_extremity",
        "upd2303d_rigidity_rt_lower_extremity",
        "upd2303e_rigidity_left_lower_extremity",
        "upd2304a_right_finger_tapping",
        "upd2304b_left_finger_tapping",
        "upd2305a_right_hand_movements",
        "upd2305b_left_hand_movements",
        "upd2306a_pron_sup_movement_right_hand",
        "upd2306b_pron_sup_movement_left_hand",
        "upd2307a_right_toe_tapping",
        "upd2307b_left_toe_tapping",
        "upd2308a_right_leg_agility",
        "upd2308b_left_leg_agility",
        "upd2309_arising_from_chair",
        "upd2310_gait",
        "upd2311_freezing_of_gait",
        "upd2312_postural_stability",
        "upd2313_posture",
        "upd2314_body_bradykinesia",
        "upd2315a_postural_tremor_of_right_hand",
        "upd2315b_postural_tremor_of_left_hand",
        "upd2316a_kinetic_tremor_of_right_hand",
        "upd2316b_kinetic_tremor_of_left_hand",
        "upd2317a_rest_tremor_amplitude_right_upper_extremity",
        "upd2317b_rest_tremor_amplitude_left_upper_extremity",
        "upd2317c_rest_tremor_amplitude_right_lower_extremity",
        "upd2317d_rest_tremor_amplitude_left_lower_extremity",
        "upd2317e_rest_tremor_amplitude_lip_or_jaw",
        "upd2318_consistency_of_rest_tremor",
        "upd2da_dyskinesias_during_exam",
        "upd2db_movements_interfere_with_ratings",
        "code_upd2hy_hoehn_and_yahr_stage",
        "upd2hy_hoehn_and_yahr_stage",
        "upd23a_medication_for_pd",
        "upd23b_clinical_state_on_medication",
    ]

    QUESTIONS_CODE = QUESTIONS_CODE_III + QUESTIONS_CODE_II + QUESTIONS_CODE_I

    ALL_PRS = [
        "PGS000056",
        "PGS000123",
        "iwaki",
        "nalls",
        "jama",
        "educational_attainment_1225prs",
        "educational_attainment_739prs",
    ]

    MONO = [
        "has_known_GBA_mutation_in_WGS",
        "has_known_LRRK2_mutation_in_WGS",
        "has_known_PD_Mutation_in_WGS",
        "has_known_SNCA_mutation_in_WGS",
    ]

    
    MOCA = [
        "moca01_alternating_trail_making",
        "moca02_visuoconstr_skills_cube",
        "moca03_visuoconstr_skills_clock_cont",
        "moca04_visuoconstr_skills_clock_num",
        "moca05_visuoconstr_skills_clock_hands",
        "moca_visuospatial_executive_subscore",
        "moca06_naming_lion",
        "moca07_naming_rhino",
        "moca08_naming_camel",
        "moca_naming_subscore",
        "moca09_attention_forward_digit_span",
        "moca10_attention_backward_digit_span",
        "moca_attention_digits_subscore",
        "moca11_attention_vigilance",
        "moca12_attention_serial_7s",
        "moca13_sentence_repetition",
        "moca14_verbal_fluency_number_of_words",
        "moca15_verbal_fluency",
        "moca_language_subscore",
        "moca16_abstraction",
        "moca_abstraction_subscore",
        "moca17_delayed_recall_face",
        "moca18_delayed_recall_velvet",
        "moca19_delayed_recall_church",
        "moca20_delayed_recall_daisy",
        "moca21_delayed_recall_red",
        "moca_delayed_recall_subscore_optnl_cat_cue",
        "moca_delayed_recall_subscore_optnl_mult_choice",
        "moca_delayed_recall_subscore",
        "moca22_orientation_date_score",
        "moca23_orientation_month_score",
        "moca24_orientation_year_score",
        "moca25_orientation_day_score",
        "moca26_orientation_place_score",
        "moca27_orientation_city_score",
        "moca_orientation_subscore",
        "code_education_12years_complete",
        "education_12years_complete",
        "moca_total_score",
    ]

    ESS = [
        "ess_info_source",
        "code_ess0101_sitting_and_reading",
        "code_ess0102_watching_tv",
        "code_ess0103_sitting_inactive_in_public_place",
        "code_ess0104_passenger_in_car_for_hour",
        "code_ess0105_lying_down_to_rest_in_afternoon",
        "code_ess0106_sitting_and_talking_to_someone",
        "code_ess0107_sitting_after_lunch",
        "code_ess0108_car_stopped_in_traffic",
        "ess0101_sitting_and_reading",
        "ess0102_watching_tv",
        "ess0103_sitting_inactive_in_public_place",
        "ess0104_passenger_in_car_for_hour",
        "ess0105_lying_down_to_rest_in_afternoon",
        "ess0106_sitting_and_talking_to_someone",
        "ess0107_sitting_after_lunch",
        "ess0108_car_stopped_in_traffic",
        "ess_summary_score",
    ]

    PDQ39 = [
        "pdq39_01_doing_leisure_activity",
        "pdq39_02_looking_after_home",
        "pdq39_03_carrying_shopping_bags",
        "pdq39_04_walking_half_mile",
        "pdq39_05_walking_100_yards",
        "pdq39_06_getting_around_house",
        "pdq39_07_getting_around_in_public",
        "pdq39_08_need_someone_to_accompany",
        "pdq39_09_worried_about_falling",
        "pdq39_10_confined_to_house",
        "pdq39_11_showering",
        "pdq39_12_dressing",
        "pdq39_13_buttons_and_shoelaces",
        "pdq39_14_writing",
        "pdq39_15_cutting_food",
        "pdq39_16_spill_drink",
        "pdq39_17_depressed",
        "pdq39_18_lonely",
        "pdq39_19_weepy",
        "pdq39_20_angry",
        "pdq39_21_anxious",
        "pdq39_22_worried_about_future",
        "pdq39_23_hide_pd_from_people",
        "pdq39_24_avoid_eat_drink_in_public",
        "pdq39_25_embarassed_in_public",
        "pdq39_26_worried_about_reactions",
        "pdq39_27_close_personal_relations",
        "pdq39_28_support_from_spouse",
        "pdq39_29_support_from_family",
        "pdq39_30_sleep_in_day",
        "pdq39_31_problem_with_concentration",
        "pdq39_32_memory_is_failing",
        "pdq39_33_hallucinations",
        "pdq39_34_speaking",
        "pdq39_35_unable_to_communicate",
        "pdq39_36_felt_ignored",
        "pdq39_37_muscle_cramps",
        "pdq39_38_joint_pains",
        "pdq39_39_hot_or_cold",
        "pdq39_mobility_score",
        "pdq39_adl_score",
        "pdq39_emotional_score",
        "pdq39_stigma_score",
        "pdq39_social_score",
        "pdq39_cognition_score",
        "pdq39_communication_score",
        "pdq39_discomfort_score",
    ]

    MAYO = [
        "msq_info_source",
        "msq_interviewee_live_with_subject",
        "msq_interviewee_sleep_same_room",
        "msq_distracting_sleep_behaviors",
        "msq01_act_out_dreams",
        "msq01a_act_out_years",
        "msq01a_act_out_months",
        "msq01b_patient_injured",
        "msq01c_bedpartner_injured",
        "msq01d_told_dreams",
        "msq01e_dream_details_match",
        "msq02_legs_jerk",
        "msq03_restless_legs",
        "msq03a_leg_sensations_decrease",
        "msq03b_time_leg_sensations_worst",
        "msq04b_walked_asleep",
        "msq05_snorted_awake",
        "msq06_stop_breathing",
        "msq06a_treated_for_stop_breathing",
        "msq07_leg_cramps",
        "msq08_rate_of_alertness",
    ]

    SCH = ["mod_schwab_england_pct_adl_score"]

    COLS_MONTH_RENAME = [
        "mo0nth0",
        "mo0nth06",
        "mo0nth12",
        "mo0nth18",
        "mo0nth24",
        "mo0nth30",
        "mo0nth36",
        # 'mo0nth42',
        # 'mo0nth48',
        # 'mo0nth54',
        # 'mo0nth60',
        "Mo1nth0",
        "Mo1nth06",
        "Mo1nth12",
        "Mo1nth18",
        "Mo1nth24",
        "Mo1nth30",
        "Mo1nth36",
        # 'Mo1nth42',
        # 'Mo1nth48',
        # 'Mo1nth54',
        # 'Mo1nth60',
        "SMo2nth0",
        "SMo2nth06",
        "SMo2nth12",
        "SMo2nth18",
        "SMo2nth24",
        "SMo2nth30",
        "SMo2nth36",
        # 'SMo2nth42',
        # 'SMo2nth48',
        # 'SMo2nth54',
        # 'SMo2nth60'
    ]

    LIST_OF_UPDRS = [
        "mds_updrs_part_i_summary_score",
        "mds_updrs_part_ii_summary_score",
        "mds_updrs_part_iii_summary_score",
    ]

    LIST_TO_DROP = [
        #  I removed it because Prof. Sandor suggested
        "has_known_PD_Mutation_in_WGS",
        # "code_",
        # "code",
        # "_code",
        "month_of_visit_x",
        "visit_month",
        "visit_month_y",
        "visit_month_x",
        "GUID_x",
        "GUID",
        "GUID_x_x",
        "GUID_x_y",
        "GUID_y_x",
        "GUID_y_y",
        "PGS000056",
        "PGS000123",
        "iwaki",
        "jama",
        "visit_month_x_x",
        "visit_month_x_y",
        "visit_month_y_x",
        "visit_month_y_y",
        "informed_consent_months_after_baseline",
        "GUID_y",
        "sum_PRS",
        "SNP_N",
        "age_at_diagnosis",
        # "pt_prs_x",
        # "pt_prs_y",
        # "pt_prs",
    ]

    LIST_NOT_DROP = ["education_level_years"]

    LIST_BASE = [
        "releases_2021_v2_5release_0510_amp_pd_case_control",
    ]

    LIST_PRS = [
        "pt_prs",
        "edu_prs",
    ]

    LIST_ONLY_AT_M0 = [
        "caffeine_history",
    ]

    LIST_ONLY_AT_M0_TO_M96 = [
        "moca",
        "schwab",
        "pd_medical_history",
        "family_history_pd",
        "demographics",
        "enrollment",
        "smoking_and_alcohol_history",
    ]

    LIST_LABELS = [
        "mds_updrs_part_i",
        "mds_updrs_part_ii",
        "mds_updrs_part_iii",
    ]

    #  list of all dfs

    LIST_ALL = [
        LIST_BASE,
        LIST_PRS,
        LIST_ONLY_AT_M0,
        LIST_ONLY_AT_M0_TO_M96,
        LIST_LABELS,
    ]

    LIST_OUTPUTS = [12, 24, 36]

    MONTH_LABELS = [
        "m0",
        "m06",
        "m12",
        "m18",
        "m24",
        "m30",
        "m36",
        # 'm42',
        # 'm48',
        # 'm54',
        # 'm60',
        # 'm66',
        # 'm72',
        # 'm78',
        # 'm84',
        # 'm90',
        # 'm96',
    ]

    LIST_SCORES = [
        "mds_updrs_part_i_summary_score",
        "mds_updrs_part_ii_summary_score",
        "mds_updrs_part_iii_summary_score",
    ]


    # Medication feature info
    NEVER_DROP_FEATURES = [
        "on_levodopa",
        "on_dopamine_agonist",
        "on_other_pd_medications",
    ]

    # list of duplicated subjects
    duplicated_subjects = [
    'pdpdjv686aab',
    'pdpdbw494ghe',
    'hbpdinvfa733yef',
    'pdpdtn576by4',
    'lb07960',
    'lb07925',
    'lb07947',
    'lb07899',
    'lb07949',
    'pp42158',
    'pp51632',
    'pp50621',
    'hbpdinvne231zud',
    'hbpdinvru618mm4',
    'hbpdinvxc454fpx',
    'bf1091',
    'lb07049',
    'hbpdinvdz260aw1',
    'hbpdinvep649ze5',
    'lb06971',
    'hbpdinvxa119chn',
    'sypdhb484jy8',
    'hbpdinvwe905kpj',
    'lc700008',
    'su32541',
    'pp54916',
    'hbpdinvme305hyh',
    'bf1098',
    'bf1125',
    'pp53639',
    'pp59121',
    'pp50463',
    'lc60008',
    'pdpdvf405wvv',
    'pdpdgu974lxr',
    'sypdjv752tra',
    'pdpduh505fez',
    'pdpdkv712awz',
    'sypdhf039hpf',
    'lc1520006',
    'lc1110006',
    'lc620006',
    'lc1220006',
    'lc630006',
    'lc150006',
    'lc1510006',
    'lc920005',
    'lc450005',
    'lc40006',
    'lc550006',
    'lc220006',
    'lc650006',
    'lc400006',
    'sypdtb794jzm',
    'lc2680001',
    'lc1260001',
    'lc10080001',
    'lc4370001',
    'lc8770001',
    'lc370005',
    'lc520005',
    'lc790005',
    'lc680006',
    'lc800005',
    'lc360005',
    'lc810005',
    'lc6690001',
    'lc5340001',
    'lc2470001',
    'lc80001',
    'lc4850001',
    'lc980010',
    'lc40008',
    'lc170008',
    'lc3520011',
    'sypdcn152kwa',
    'lc210008',
    'lc150008',
    'su32824'
    ]

