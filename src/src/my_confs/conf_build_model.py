# Conf file for creating predictive models 
###################################################
###################################################
###################################################
###################################################

import os
from dotenv import load_dotenv

load_dotenv()

# get the version of data that is set in .env
DATA_VERSION = os.environ.get("DATA_VERSION")
ALGORITHM = os.environ.get("ALGORITHM")

# if the version of data is old, use these configs
if DATA_VERSION == "v1":
    # Define type of meta-features by commenting and uncommenting 
    LIST_OF_META_FEATURES = [
        "y_pred_24_up1",
        "y_pred_24_up2",
        "y_pred_24_up3",
        # "y_pred_24_uptotal",
        "y_pred_36_up1",
        "y_pred_36_up2",
        "y_pred_36_up3",
        "y_pred_36_uptotal",
    ]

