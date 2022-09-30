# Read different file formats
###################################################
###################################################
###################################################
###################################################

import os
import pandas as pd
from src.src.setup_logger import logger


def read_data(data_path, sep=",", nrows=None):
    """Read data from a file path. It can handle three different types of raw data formats.
    1. CSV
    2. XLSX
    3. TXT

        Args:
        data_path: full path to data
        sep: the delimators used for making data separate
        nrow: number of rows to be retrieved

        Returns:
        A Pandas data frame object
    """

    _, file_extension = os.path.splitext(data_path)

    if file_extension == ".csv":

        # data = pd.read_csv(data_path, sep=sep, nrows=nrows, low_memory=False)
        data = pd.read_csv(data_path, sep=sep, nrows=nrows)

    if file_extension == ".xlsx":

        data = pd.read_excel(data_path, index_col=0, engine="openpyxl")
        # csv_file = csv_from_excel(data_path, data_path.replace('.xlsx','.csv'))
        # data = pd.read_csv(data_path, sep=sep, nrows=nrows, low_memory=False)

    if file_extension == ".txt":

        try:
            data = pd.read_fwf(data_path, infer_nrowsint=nrows)
        except Exception as e:
            logger.error(
                "Dataframes in this path th %s can not be read  %s ",
                data_path,
                e,
            )

    return data
