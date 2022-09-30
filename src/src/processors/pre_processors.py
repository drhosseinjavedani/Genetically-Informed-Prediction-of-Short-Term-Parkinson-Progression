# Solving some problems of raw data 
###################################################
###################################################
###################################################
###################################################

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from src.src.setup_logger import logger
from src.src.my_confs.conf_data_engineering import general_rules_features


class DataResolver(BaseEstimator, TransformerMixin):
    """Get a data frame and rename all its columns that has special characters with

    Args:
    Pandas Data Frame : A data frame of pandas data frame object.

    Returns:
    Same data frame with fixed columns names.
    """

    def __init__(self, params):

        self.params = params

    def fit(self, X, y):
        return self

    def transform(self, X):

        try:
            X = X.copy()
            X = X.loc[:, ~X.columns.duplicated()]

            X.reset_index(inplace=True, drop=True)
            if "index" in X.columns:
                X = X.drop(["index"], axis=1)
            if "Id" in X.columns:
                X = X.drop(["Id"], axis=1)
            if "num" in X.columns:
                X = X.drop(["num"], axis=1)

            categories_cols = X.select_dtypes(
                exclude=["float", "int"]
            ).columns.tolist()

            for col in categories_cols:
                if not X[col].str.isnumeric().any():
                    X[col] = X[col].str.replace("-", "")
                X[col] = X[col].str.replace("/", "_")
                X[col] = X[col].str.replace("m3", "m03")
                X[col] = X[col].str.replace("M03", "m03")
                X[col] = X[col].str.replace("M3", "m03")

                X[col] = X[col].str.replace("m9", "m09")
                X[col] = X[col].str.replace("M09", "m09")
                X[col] = X[col].str.replace("M9", "m09")

                X[col] = X[col].str.replace("m6", "m06")
                X[col] = X[col].str.replace("M6", "m06")
                X[col] = X[col].str.replace("M06", "m06")

                X[col] = X[col].str.replace("m012", "m12")
                X[col] = X[col].str.replace("M012", "m12")
                X[col] = X[col].str.replace("M12", "m12")

                X[col] = X[col].str.replace("m018", "m18")
                X[col] = X[col].str.replace("M018", "m18")
                X[col] = X[col].str.replace("M18", "m18")

                X[col] = X[col].str.replace("m024", "m24")
                X[col] = X[col].str.replace("M024", "m24")
                X[col] = X[col].str.replace("M24", "m24")

                X[col] = X[col].str.replace("m030", "m30")
                X[col] = X[col].str.replace("M030", "m30")
                X[col] = X[col].str.replace("M30", "m30")

                X[col] = X[col].str.replace("m036", "m36")
                X[col] = X[col].str.replace("M036", "m36")
                X[col] = X[col].str.replace("M36", "m36")

                X[col] = X[col].str.replace("m042", "m42")
                X[col] = X[col].str.replace("M042", "m42")
                X[col] = X[col].str.replace("M42", "m42")

                X[col] = X[col].str.replace("m048", "m48")
                X[col] = X[col].str.replace("M048", "m48")
                X[col] = X[col].str.replace("M48", "m48")

                X[col] = X[col].str.replace("m054", "m54")
                X[col] = X[col].str.replace("M054", "m54")
                X[col] = X[col].str.replace("M54", "m54")

                X[col] = X[col].str.replace("m060", "m60")
                X[col] = X[col].str.replace("M060", "m60")
                X[col] = X[col].str.replace("m060", "m60")

                X[col] = X[col].str.replace("m066", "m66")
                X[col] = X[col].str.replace("M066", "m66")
                X[col] = X[col].str.replace("m066", "m66")

                X[col] = X[col].str.replace("m072", "m72")
                X[col] = X[col].str.replace("M072", "m72")
                X[col] = X[col].str.replace("m072", "m72")

            logger.info(
                "Some treatments within data with the String format. Droping \
                    columns with specifics substr has finished :)"
            )
        except Exception as e:
            logger.error(
                "While treatments within data with the String format or droping \
                    columns with specifics something happened  %s:(",
                e,
            )

        return X


class NameResolver(BaseEstimator, TransformerMixin):
    """Get a data frame and rename all its columns that has special characters with

    Args:
    Pandas Data Frame : A data frame of pandas data frame object.

    Returns:
    Same data frame with fixed columns names.
    """

    def __init__(self, params):

        self.params = params

    def fit(self, X, y):
        return self

    def transform(self, X):

        try:
            X = X.copy()

            for col in X.columns:
                if X.dtypes[col] == np.object:
                    X[col] = X[col].str.lower()

            for chr in "!~*.%/ #-)(":
                for col in X.columns:
                    X.rename(
                        {col: col.replace(chr, "_")},
                        axis="columns",
                        inplace=True,
                    )

            X.rename({"subj_id": "subject_id"}, axis="columns", inplace=True)
            X.rename({"subj.id": "subject_id"}, axis="columns", inplace=True)
            X.rename({"_sampleID": "subject_id"}, axis="columns", inplace=True)
            X.rename({"Subj_id": "subject_id"}, axis="columns", inplace=True)
            X.rename(
                {"participant_id": "subject_id"}, axis="columns", inplace=True
            )
            X.rename(
                {"Participant_id": "subject_id"}, axis="columns", inplace=True
            )

            if "visit" in X.columns.to_list():
                if X.dtypes["visit"] != np.int:
                    X.rename(
                        {"visit": "month_of_visit"},
                        axis="columns",
                        inplace=True,
                    )

            if "Visit" in X.columns.to_list():
                if X.dtypes["Visit"] != np.int:
                    X.rename(
                        {"Visit": "month_of_visit"},
                        axis="columns",
                        inplace=True,
                    )
            if "visit_name" in X.columns.to_list():
                if X.dtypes["visit_name"] != np.int:
                    X.rename(
                        {"visit_name": "month_of_visit"},
                        axis="columns",
                        inplace=True,
                    )

            if "Visit_name" in X.columns.to_list():
                if X.dtypes["Visit_name"] != np.int:
                    X.rename(
                        {"Visit_name": "month_of_visit"},
                        axis="columns",
                        inplace=True,
                    )

            logger.info("Renaming with success :)")

        except Exception as e:
            logger.error("Renaming with errors %s :(", e)

        return X


class DataPipProcessors:
    def __init__(self, params):

        self.params = params

    def name_resolver(self, df):
        """Get a data frame and rename all its columns that has special characters with

        Args:
        Pandas Data Frame : A data frame of pandas data frame object.

        Returns:
        Same data frame with fixed columns names.
        """

        name_resolver = Pipeline([("name_resolver", NameResolver(df))])

        return name_resolver

    def data_resolver(self, df):
        """Get a data frame and rename all its columns that has special characters with

        Args:
        Pandas Data Frame : A data frame of pandas data frame object.

        Returns:
        Same data frame with fixed columns names.
        """

        data_resolver = Pipeline([("data_resolver", DataResolver(df))])

        return data_resolver

    def resolve_all_data_problems(self, df_dic):
        """Get a dictionary of data frames and rename all their
        columns that has special characters with

        Args:
        A dictionary of Data Frame : A dictionary of Data Frame
        of data frame of pandas data frame object.

        Returns:
        Same dictionary with fixed columns names.
        """
        df_dic_copy = {}
        try:
            for key, df in df_dic.items():
                # Resolve the name of data frames
                key = key.replace("-", "_")
                key = key.replace(" ", "")
                key = key.replace(".", "")
                key = key.lower()

                df = self.data_resolver(df).transform(df)
                df = self.name_resolver(df).transform(df)
                df.name = key
                df_dic_copy[key] = df

            logger.info(
                "Data cleaning columns names, removing spaces , etc, \
                     were successfull :) "
            )
        except Exception as e:
            logger.error(" Data cleaning has this error :( %s", e)

        # for key, df in df_dic_copy.items():
        #      print(key)
        #      print(df.head())

        return df_dic_copy

    def apply_renaming_rules(self, df_dic):
        """Rename all columns of all data frames that match with
        configurations defined in
        general_rules_features['renaming_rules_percentage']

        Args:
        Dictionary: of all data frames

        Returns:
        Dictionary: Same dictionary of renamed data frames
        """

        # Read config
        dic_of_names = general_rules_features["renaming_rules_percentage"]
        try:
            for selected_col, pr in dic_of_names.items():
                logger.info(
                    "searching for columns similar to  %s started !",
                    selected_col,
                )
                for name_df, df in df_dic.items():
                    if selected_col in df.columns:
                        for name_another_df, another_df in df_dic.items():
                            for another_col in another_df.columns:
                                if self.how_two_columns_similar(
                                    df,
                                    [selected_col],
                                    another_df,
                                    [another_col],
                                    pr,
                                ):
                                    logger.info(
                                        "Two dataframes %s and %s \
                                            in columns %s and %s ",
                                        name_df,
                                        name_another_df,
                                        selected_col,
                                        another_col,
                                    )
                                    another_df.rename(
                                        columns={another_col: selected_col},
                                        inplace=True,
                                    )
                                    break
        except Exception as e:
            logger.error("While renaming there is  %s :(", e)
        return df_dic

    def how_two_columns_similar(self, df1, col1, df2, col2, pr):
        """Check how two columns are similar.
        It recives some data frames with same values
        and different names and a percentages
        value to show if they are similar up to that
        percentage or not.

        Args:
        df1: Data frame 1
        col1: A column of data frame 1
        col2: A column of data frame 2
        pr: Float between 0 and 1, show in what percentage two columns are similar

        Returns:
        Boolean: True if the two columns are similar in
        respect to pr and False otherwise
        """
        output = False
        if col1 != col2:
            if df1[col1].values.dtype == df2[col2].values.dtype:
                ser1 = df1[col1].T.squeeze()
                ser2 = df2[col2].T.squeeze()
                set1 = set(ser1)
                set2 = set(ser2)
                inter_set = set1.intersection(set2)
                if len(inter_set) > pr * len(ser2):
                    output = True
        return output
