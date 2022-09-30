# Conf file for data preprocessing and reading data
###################################################
###################################################
###################################################
###################################################

import os
from dotenv import load_dotenv

load_dotenv()
# get the version of data that is in .env
DATA_VERSION = os.environ.get("DATA_VERSION")
ALGORITHM = os.environ.get("ALGORITHM")

if ALGORITHM == 'cls':
    _logs="logs"
    _artifcats = "artifacts"
if DATA_VERSION == "v1":
    _version='v1'

RUN_REMOTE = False

if RUN_REMOTE:
    suffix_file_reader = "PATH TO REMOTE DIRECTORY OF DATA IF ANY"
else:
    suffix_file_reader = "PATH TO LOCAL DIRECTORY OF DATA"

# Path of config for logging
fname = suffix_file_reader + "src/logconfig.conf"

# ID
# Indexes for all data frame that join, merge, ..., use it.
ids = ["subject_id"]

# VERSION of feature store, each version will
# create a brand new feature store and all others
# past versions will be lost.

general_rules_features = {
    "version": 'VERSION',  # Version of rules
    "data_path": [
        suffix_file_reader + "src/datasets/AMP_PD_"+_version+"/clinichal_and_others",
        suffix_file_reader + "src/datasets/AMP_PD_"+_version+"/prs_and_mono",
    ],  # List of the data path
    # Extensions of data supported
    "data_extension": [".csv", ".xlsx", ".pkl", ".txt"],
    
    # Directory which processed data will dump
    "path_to_save": suffix_file_reader + "src/"+_artifcats+"/"+_version+"/",
    "path_to_logs": suffix_file_reader + "src/"+_logs+"/"+_version+"/",

    # Merge with label table if the data has the same id, or ...
    "merge_ids": ["subject_id", "month_of_visit"],
    "merge_ids_for_second_merge": ["subject_id", "month_of_visit"],
    
    "count_values": [
        ["subject_id"]
    ],  # Merge with label table if the data has the same id and month
    
    "output_rules": {
        "summary_before_T": {  # Summary of data before transformations
            "feature_list": True,  # A list of features
            "save_head": True,  # save a head data frame
            "save_tail": False,  # save a tail data frame
            "save_describe": True,  # Describe data frame
            "save_datatype": True,  # Datatype of each features
            "pr_nulls": True,  # Percantage of null values for each feature,
            "pr_unique_values": True,  # Percentage of unique values
        },
        "summary_After_T": {  # Summary of data after transformations
            "feature_list": True,  # A list of features
            "save_head": True,  # save a head data frame
            "save_tail": False,  # save a tail data frame
            "save_describe": True,  # Describe data frame
            "save_datatype": True,  # Datatype of each features
            "pr_nulls": True,  # Percantage of null values for each feature,
            "pr_unique_values": True,  # Percentage of unique values
        },
    },
}
