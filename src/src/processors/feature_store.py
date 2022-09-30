# Staged data processing and transforming class
###################################################
###################################################
###################################################
###################################################

import collections
import glob
import itertools
import os.path
import pickle
import sys
from functools import reduce
from os import path
import pandas as pd
from src.src.processors.pre_processors import DataPipProcessors
from src.src.setup_logger import logger
from src.src.my_confs.conf_data_engineering import general_rules_features, ids
from src.src.my_utils.read_data import read_data


class FeatureStore:
    def __init__(self):

        # Get the logger specified in the file
        logger.info("Feature store has started! ")
        # Create instance of DataPipProcessors
        self.dpp = DataPipProcessors(general_rules_features)
        return

    def list_of_data_paths(self):
        """Read configs and general_rules and return a list of data files.

        Args:
        Read args from data_configs.py, general_rules

        Returns:
        List of the data path in csv, xlsx,...
        """

        # Paths of all data directories
        data_path = general_rules_features["data_path"]
        # Extension of data to look for
        data_extension = general_rules_features["data_extension"]
        data_list = []  # Empty list of data

        # For loop to retrieve list of full paths of data files
        try:
            for dir in data_path:
                for ext in data_extension:
                    dir_with_ext = dir + "/*" + ext
                    data_list.append(glob.glob(dir_with_ext))
            data_list = list(itertools.chain(*data_list))
            self.helper_persist_to_data_save_path(
                data_list, "list_of_all_dataframes_read", flag_to_rewrite=True
            )
        except Exception as e:
            logger.error(
                "This path  can not be added to the list for reading :( The error is %s",
                e,
            )

        print("List of data : ")
        print(*sorted(data_list), sep="\n")

        return data_list

    def read_data_from_list_paths(self):
        """Read data from a list using internal function 'list_of_data_paths'

        Args:
        None: list of data paths comes from an internal function,
        i.e., list_of_data_paths

        Returns:
        Dictionary of pandas data frames, names of each one come
        form last part or base of the file path
        """
        error_counts = 0  # Store number of errors

        list_of_data_paths = (
            self.list_of_data_paths()
        )  # Paths of all data directories
        # Extension of data to look for
        data_extension = general_rules_features["data_extension"]
        df_dic = {}  # Empty dic of dataframes

        # For loop to retrieve list of full paths of data files
        for filename in list_of_data_paths:
            for ext in data_extension:
                try:
                    if filename.endswith(ext):
                        name, _ = os.path.splitext(
                            filename
                        )  # Extract filename from extension
                        # Get the base file name from path
                        base_filename = os.path.basename(name)
                        # Read_data function is in Utils.read_data module
                        df = read_data(filename, sep=",")
                        # Reset Index
                        try:
                            df = df.reset_index()
                        except Exception as e:
                            logger.error(
                                "Reindex for this id %s can not perform  :( %s",
                                id,
                                e,
                            )
                        df.name = base_filename
                        if "dictionary" not in base_filename:
                            df_dic[base_filename] = df
                            logger.info(
                                "New data frame added to data frames dictionary %s",
                                base_filename,
                            )
                            break
                except Exception as e:
                    error_counts = (
                        error_counts + 1
                    )  # Add one to the error counts
                    logger.error(
                        "An error occurred while reading %s the error is %s",
                        base_filename,
                        e,
                    )
        logger.info(
            "Data frames have been created  with %s number of errors :)",
            error_counts,
        )

        try:
            # how use new column names
            df_dic = self.dpp.resolve_all_data_problems(df_dic)
            self.helper_persist_to_data_save_path(
                df_dic, "stage1_dictionary_all_dataframes", flag_to_rewrite=True
            )
            logger.info(
                "Data frames have been renamed and persisted properly :)"
            )

        except Exception as e:
            logger.error(
                "Indexes of data have not been appropriately renamed %s :( ", e
            )

        return df_dic

    def dataframes_sorter_by_id(self):
        """Get access to data frames and then sort them by the number of
        unique ids. It will use read_data_from_list_paths to access
        data frames. The larger the number of ids, the lower
        index in the returned list. It also renames participant_id,
        subject_id, ..., into 'ID'


        Args:
        None: list of data frames comes from an internal function,
         i.e., read_data_from_list_paths
        None: id will be retrieved form data_configs.py, e.g.,

        Returns:
        A list of sorted pandas data frames by some other objects
        related to it same as:
            df_head = df.head() # Dataframe head
            df_tail = df.tail() # Dataframe tail
            df_feature_list = df.columns # Dataframe columns
            df_unique = df.value_counts(
                normalize=True) * 100 # Get value counts
            df_data_type = df.dtypes # Dataframe data type
            df_describe = df.describe() # Dataframe describe
            # Rules that come from config files
            rule = data_configs['general_rules_features']
            # Create an object of all information
            df_dic['objects']=[df_head,df_tail,df_feature_list,df_unique,
            df_data_type,df_describe,rule]

        """

        df_dic = self.read_data_from_list_paths()  # Read data
        number_of_unique_ids = (
            []
        )  # List that collects number of unique values of ids

        # Loop over data frames and ids that come from config

        for key, df in df_dic.items():
            for id in ids:
                if id in list(df.columns.values):
                    # Rename the participant_id, subject_id, ..., to ID
                    number_of_unique_ids.append((df, df[id].nunique()))
                    break

        # Sort list based on the second item of tuple
        number_of_unique_ids = sorted(number_of_unique_ids, key=lambda x: x[1])

        # Repopulate the df_dic, it is now an ordered dictionary
        df_dic = collections.OrderedDict()
        df_lst = []
        for (ixd, (df, nunique)) in enumerate(number_of_unique_ids):
            df_dic[df.name] = {}
            df_dic[df.name]["data"] = df
            df_dic[df.name]["name"] = df.name
            df_dic[df.name]["order"] = ixd
            df_dic[df.name]["df_head"] = df.head()  # Dataframe head
            df_dic[df.name]["df_tail"] = df.tail()  # Dataframe tail
            # Dataframe columns
            df_dic[df.name]["df_feature_list"] = df.columns

            # Create two sub list/subsets to check if they are part of
            # df columns or not
            subset_1 = set(general_rules_features["count_values"][0])
            col_set = set(df.columns.tolist())

            if subset_1.issubset(col_set):
                # Get value counts
                df_dic[df.name]["df_unique"] = df[subset_1].value_counts()

            df_dic[df.name]["df_data_type"] = df.dtypes  # Dataframe data type
            # Dataframe describe
            df_dic[df.name]["df_describe"] = df.describe()
            # Rules that come from data_configs.py
            df_dic[df.name]["rule"] = general_rules_features
            df_lst.append(df.name)

        len_df = str(len(df_dic.keys()))
        logger.info(
            "%s Data frames have been sorted based on the number of unique \
                ids with success %s:)",
            len_df,
            df_dic.keys(),
        )

        self.helper_persist_to_data_save_path(
            df_lst,
            "list_of_all_dataframes_sorted_by_number_of_ids",
            flag_to_rewrite=True,
        )

        return df_dic

    def save_df_object_before_T(self):
        """Get access to sorted data frames before transformation to
        create various objects from them.


        Args:
        None: list of sorted data frames comes from an internal function, i.e.,
         'dataframes_sorter_by_id'

        Returns:
        A saved object of various data frame objects, i.e., feature_list,
        head, tail, describes null percentage, percentage of unique
        values, data type, etc. All configs come from
        summary_before_T of data_configs.

        """
        # if the dumped file already exists and there saved
        # parameters are the same
        # no action will be taken.
        df_dic = (
            self.dataframes_sorter_by_id()
        )  # Get ordered dic of data frames
        file_name = "stage2_dictionary_sorted_summrized_dataframes"
        file_path = general_rules_features["path_to_save"] + file_name

        if path.exists(file_path + ".pkl"):
            # If the file exist, remove it
            try:
                os.remove(file_path + ".pkl")
            except Exception as e:
                logger.error("Something goes wrong :( %s", e)

        self.helper_persist_to_data_save_path(
            df_dic, file_name, flag_to_rewrite=True
        )
        logger.info(
            "A Dictionary of various data frames with their summaries \
                 has persisted in : %s",
            file_path,
        )

        return

    def read_from_persisted_features(self, flag="BT"):
        """Get access to sorted data frames and read them


        Args:
        flag: to read features from persisted pickles Before
        Transformation (BT) or After
        Transformation (AT), default value : BT

        Returns:
        An ordered object of various data frame objects, i.e.,
        feature_list, head, tail, describes null percentage,
        percentage of unique values, data type, etc. All configs come from
        summary_before_T of data_configs.
        """
        # Create an empty ordered dictionary
        df_dic = collections.OrderedDict()
        # Set file name
        file_name = "stage2_dictionary_sorted_summrized_dataframes" + ".pkl"
        # Try to read data from where it should be persisted before
        df_dic = self.helper_load_from_data_save_path(file_name)
        return df_dic

    def show_from_persisted_features(self, flag="BT"):
        """Get access to sorted data frames and read them

        Args:
        flag: to show features from persisted pickles Before
        Transformation (BT) or After
        Transformation (AT), default value : BT

        Returns:
        None: This method only shows various objects that
        exist in the returned object of reading
        by read_from_presisted_features function.
        """
        try:
            # Read data using internal function read_from_presisted_features
            df_dic = self.read_from_persisted_features(flag=flag)
            for key, value in df_dic.items():

                print("############################################")
                print("############################################")

                if flag == "BT":
                    print("############################################")

                    print("Data frame name:")
                    print(df_dic[key]["name"])
                    print("############################################")

                    print("Order (in term of number of IDs):")
                    print(df_dic[key]["order"])

                    if general_rules_features["output_rules"][
                        "summary_before_T"
                    ]["save_head"]:
                        print("############################################")

                        print("Data frame head:")
                        print(df_dic[key]["df_head"])

                    if general_rules_features["output_rules"][
                        "summary_before_T"
                    ]["save_tail"]:
                        print("############################################")

                        print("Data frame tail:")
                        print(df_dic[key]["df_head"])

                    if general_rules_features["output_rules"][
                        "summary_before_T"
                    ]["feature_list"]:
                        print("############################################")

                        print("Data frame features or variable list:")
                        print(*df_dic[key]["df_feature_list"], sep="\n")

                    # Some data frame does not have pair of (ID, Visit),
                    # so the below commands
                    # will generate the error.
                    try:
                        if general_rules_features["output_rules"][
                            "summary_before_T"
                        ]["pr_unique_values"]:
                            print(
                                "############################################"
                            )

                            print(
                                "Data frame number of unique \
                                values:"
                            )
                            print(df_dic[key]["df_unique"])
                    except Exception as e:
                        logger.warning(
                            f'This data frame :{df_dic[key]["name"]} does not have \
                                (ID, Visit) features !! The error is {e}'
                        )

                    if general_rules_features["output_rules"][
                        "summary_before_T"
                    ]["save_datatype"]:
                        print("############################################")

                        print("Data frame data types:")
                        print(df_dic[key]["df_data_type"])

                    if general_rules_features["output_rules"][
                        "summary_before_T"
                    ]["save_describe"]:
                        print("############################################")

                        print("Data frame describe:")
                        print(df_dic[key]["df_describe"])

        # Handle the error
        except Exception as e:
            logger.error("Problem while reading/ showing objects  !!! %s", e)
        return

    def show_one_dataframe_from_persisted_features(self, name, flag="BT"):
        """Get access to the presisted object and read one data frame

        Args:
        flag: to show features from persisted pickles Before
        Transformation (BT) or After
        Transformation (AT), default value : BT
        name: name of data frame

        Returns:
        None: This method only shows various objects that exist in
        returned of only one  data frame
        """
        try:
            # Read data using internal function read_from_presisted_features
            df_dic = self.read_from_persisted_features(flag=flag)
            original_stdout = (
                sys.stdout
            )  # Save a reference to the original standard output
            # with open('../../src/logs/pre_process_data.txt', 'a+') as f:
            with open(
                general_rules_features["path_to_logs"] + "pre_process_data.txt",
                "a+",
            ) as f:
                sys.stdout = (
                    f  # Change the standard output to the file we created.
                )
                for key, _ in df_dic.items():
                    if df_dic[key]["name"] == name:
                        print("The outputs will persist on output.txt file.")
                        print("############################################")
                        print("############################################")
                        print("############################################")

                        if flag == "BT":
                            print(
                                "############################################"
                            )

                            print("Data frame name:")
                            print(df_dic[key]["name"])
                            print(
                                "############################################"
                            )

                            print("Order (in term of number of IDs):")
                            print(df_dic[key]["order"])

                            if general_rules_features["output_rules"][
                                "summary_before_T"
                            ]["save_head"]:
                                print(
                                    "############################################"
                                )

                                print("Data frame head:")
                                print(df_dic[key]["df_head"])

                            if general_rules_features["output_rules"][
                                "summary_before_T"
                            ]["save_tail"]:
                                print(
                                    "############################################"
                                )

                                print("Data frame tail:")
                                print(df_dic[key]["df_head"])

                            if general_rules_features["output_rules"][
                                "summary_before_T"
                            ]["feature_list"]:
                                print(
                                    "############################################"
                                )

                                print("Data frame features or variable list:")
                                print(
                                    *sorted(df_dic[key]["df_feature_list"]),
                                    sep="\n",
                                )

                            # Some data frame does not have pair of (ID, Visit), \
                            # so the below commands
                            # will generate the error.
                            try:
                                if general_rules_features["output_rules"][
                                    "summary_before_T"
                                ]["pr_unique_values"]:
                                    print(
                                        "############################################"
                                    )

                                    print("Data frame number of unique values:")
                                    print(df_dic[key]["df_unique"])
                            except Exception as e:
                                logger.warning(
                                    f'This data frame :{df_dic[key]["name"]} does not have (ID, Visit)\
                                    features !! The error is {e}'
                                )

                            if general_rules_features["output_rules"][
                                "summary_before_T"
                            ]["save_datatype"]:
                                print(
                                    "############################################"
                                )

                                print("Data frame data types:")
                                print(df_dic[key]["df_data_type"])

                            if general_rules_features["output_rules"][
                                "summary_before_T"
                            ]["save_describe"]:
                                print(
                                    "############################################"
                                )

                                print("Data frame describe:")
                                print(df_dic[key]["df_describe"])

                            try:

                                for col in df_dic[name]["data"].columns:
                                    print(
                                        "############################################"
                                    )
                                    print("information about:", col)
                                    print("unique values:")
                                    print(df_dic[name]["data"][col].unique())
                                    print("number of unique:")
                                    print(df_dic[name]["data"][col].nunique())
                                    print("value counts:")
                                    print(
                                        df_dic[name]["data"][col].value_counts()
                                    )

                            except Exception as e:
                                logger.error(
                                    "Problem while showing feature \
                                         information :( %s",
                                    e,
                                )
            # Reset the standard output to its original value
            sys.stdout = original_stdout

        # Handle the error
        except Exception as e:
            logger.error("Problem while reading/ showing objects  !!! %s", e)
        return

    def data_merger_from_list(self, flag="BT", persist_read=False):
        """Get access to persisted data frames before transformation and merge
        using the information inside data_configs.py.


        Args:
        flag: use BT to only ready persisted objects before the transformation.

        Returns:
        None: This method will generate merged objects persisted in fstore.
        """
        df_frames = []

        if persist_read:
            # Read persisted object before transformation
            df_dic = self.read_from_persisted_features(flag=flag)
        else:
            # Read from files and data
            df_dic = self.dataframes_sorter_by_id()

        if general_rules_features["how_to_merge"]["select_data_frames"]:
            # This only adds name of the data frame
            for name in general_rules_features["how_to_merge"][
                "select_by_name"
            ]:
                # Add selected data frames to the list for further usage
                try:
                    df_frames.append(df_dic[name]["data"])
                    logger.info(
                        "Dataframes with this name %s  \
                            added to the list :)",
                        name,
                    )
                except Exception as e:
                    logger.error(
                        "Dataframes with this name %s \
                            can not be added to the list %s :(",
                        name,
                        e,
                    )

            # Log output
            logger.info(
                "A  list of data frames added  for \
                    performing merges later "
            )

        # Mergeing all dfs appended in df_frames_train
        try:
            # Select ids for merging, for example
            # ['subject_id', 'month_of_visit']
            sub_list_for_merge = general_rules_features["merge_ids"]
            # Check to see sub_list_for_merge is in df of df_frames to merge
            final_frames = []
            final_frames_names = []

            for df in df_frames:  # reversed(df_frames):
                first_check = True
                for m_id in sub_list_for_merge:
                    if df[m_id].nunique() < 4:
                        first_check = False
                if (
                    self.helper_check_sublist(sub_list_for_merge, df.columns)
                    and first_check
                ):
                    final_frames.append(df)
                    final_frames_names.append(df.name)

                    logger.info(
                        "One  Datafrmae with the name %s added to be merged \
                            for final with the number of unique, %s %s",
                        df.name,
                        sub_list_for_merge[0],
                        df[sub_list_for_merge[0]].nunique(),
                    )

            print(*final_frames_names, sep="\n")
            DF_IDs = reduce(
                lambda left, right: pd.merge(
                    left, right, on=sub_list_for_merge, how="left"
                ),
                final_frames,
            )
            # drop index
            self.helper_persist_to_data_save_path(
                final_frames_names,
                "stage3_first_round_marged_dataframes_names",
                flag_to_rewrite=True,
            )

            DF_IDs = DF_IDs.reset_index(drop=True)
            object_name = "stage3_first_round_marged_dataframes"
            self.helper_persist_to_data_save_path(
                DF_IDs, object_name, flag_to_rewrite=True
            )
        except Exception as e:
            logger.error("Problem while  merge data frames %s", e)

        return DF_IDs, object_name

    def data_merger_from_list_outside_data_configs(
        self, dfs_list, flag="BT", persist_read=False
    ):
        """Use a list of data frames and merge them out of data config.


        Args:
        flag: use BT to only ready persisted objects before the transformation.

        Returns:
        None: This method will generate merged objects persisted in fstore.
        """
        df_frames = []

        if persist_read:
            # Read persisted object before transformation
            df_dic = self.read_from_persisted_features(flag=flag)
        else:
            # Read from files and data
            df_dic = self.dataframes_sorter_by_id()

        # This only adds name of the data frame
        for name in dfs_list:
            # Add selected data frames to the list for further usage
            try:
                df_frames.append(df_dic[name]["data"])
                logger.info(
                    "Dataframes with this name %s  added to the list :)", name.
                )
            except Exception as e:
                logger.error(
                    "Dataframes with this name %s can not be added to the list %s :(",
                    name,
                    e,
                )

        # Log output
        logger.info("A  list of data frames added  for performing merges later ")

        # Mergeing all dfs appended in df_frames_train
        try:
            # Select ids for merging for example ['subject_id', 'month_of_visit']
            sub_list_for_merge = general_rules_features[
                "merge_ids_for_second_merge"
            ]
            # Check to see sub_list_for_merge is in df of df_frames to merge
            final_frames = []
            final_frames_names = []

            for df in df_frames:  # reversed(df_frames):
                first_check = True
                for m_id in sub_list_for_merge:
                    if df[m_id].nunique() < 1:
                        first_check = False
                if (
                    self.helper_check_sublist(sub_list_for_merge, df.columns)
                    and first_check
                ):
                    final_frames.append(df)
                    final_frames_names.append(df.name)

                    logger.info(
                        "One  Datafrmae with the name %s added to be merged \
                            for final with the number of unique, %s %s",
                        df.name,
                        sub_list_for_merge[0],
                        df[sub_list_for_merge[0]].nunique(),
                    )
            print(len(final_frames))
            print(*final_frames_names, sep="\n")
            if len(final_frames) == 1:
                print("final_frames", final_frames)
                DF_IDs = final_frames[0]

            else:
                DF_IDs = reduce(
                    lambda left, right: pd.merge(
                        left, right, on=sub_list_for_merge, how="left"
                    ),
                    final_frames,
                )
            # drop index
            DF_IDs = DF_IDs.reset_index(drop=True)
            object_name = "stage4_second_round_marged_dataframes"
            self.helper_persist_to_data_save_path(
                DF_IDs, object_name, flag_to_rewrite=True
            )
            self.helper_persist_to_data_save_path(
                final_frames_names,
                "stage4_second_round_marged_dataframes_names",
                flag_to_rewrite=True,
            )

        except Exception as e:
            logger.error("Problem while  merge data frames %s", e)

        return DF_IDs, object_name

    def data_merger_from_two_dataframes(
        self, df1, df2, ids, persist_read=False
    ):
        """Merge two data frames


        Args:
        flag: use BT to only ready persisted objects before the transformation.
        df1: First dataframe
        df2: Second dataframe
        ids: List of keys to use for merge

        Returns:
        Dataframe: A merged dataframe
        """
        try:
            final_frames = []
            final_frames.append(df1)
            final_frames.append(df2)
            DF_IDs = reduce(
                lambda left, right: pd.merge(left, right, on=ids, how="left"),
                final_frames,
            )
            logger.info("Two data frames merged :)")
            self.helper_persist_to_data_save_path(
                DF_IDs, "stage5_all_merged_dataframes", flag_to_rewrite=True
            )

        except Exception as e:
            logger.error("Problem while  merging two data frames %s", e)

        return DF_IDs

    def helper_persist_to_data_save_path(
        self, object, object_name, flag_to_rewrite=True
    ):
        """Persist object with object name into a pickle in feature store in
        this path : path_to_save


        Args:
        object: An object, Can be a data frame, pickle, data sets, ...
        flag_to_rewrite: If True, if the persisted object is already in the
        the path it will be rewritten.

        Returns:
        None: A success or fail message
        """

        try:
            # Find if the file already exists
            file_path = (
                general_rules_features["path_to_save"] + object_name + ".pkl"
            )

            if path.exists(file_path):
                if flag_to_rewrite:
                    logger.info(
                        "This object %s already exist, but it will be overwritten!",
                        object_name,
                    )

                    try:
                        # joblib.dump(object, file_path, compress = 1)
                        pickle.dump(object, open(file_path, "wb"))
                        logger.info("%s   has persisted  :)", object_name)
                    except Exception as e:
                        print(e)
                        logger.error(
                            "fail during persisting the  %s :(", object_name
                        )

                else:
                    logger.info("Nothing has changed %s", object_name)
            else:
                logger.info(
                    "This object %s does not persist, it will be persisted now!",
                    object_name,
                )

                try:
                    pickle.dump(object, open(file_path, "wb"))
                    logger.info("An object  df has persisted  %s :)", object)
                except Exception as e:
                    logger.error(
                        "fail during persisting the  %s :( The error is %s",
                        object_name,
                        e,
                    )

        except Exception as e:
            logger.error(
                "Problem while presisting object %s %s :(", object_name, e
            )

        return

    def helper_persist_to_data_save_path_csv(
        self, object, object_name, flag_to_rewrite=True
    ):
        """Persist object with df name into a csv in feature store in
        this path : path_to_save


        Args:
        object: An df, Can be a data frame, pickle, data sets, ...
        flag_to_rewrite: If True, if the persisted object is already in the
        the path it will be rewritten.

        Returns:
        None: A success or fail message
        """

        try:
            # Find if the file already exists
            file_path = (
                general_rules_features["path_to_save"] + object_name + ".csv"
            )

            if path.exists(file_path):
                if flag_to_rewrite:
                    logger.info(
                        "This object %s already exist, but it will be overwritten!",
                        object_name,
                    )

                    try:
                        # joblib.dump(object, file_path, compress = 1)
                        pd.object.to_csv(file_path)
                        logger.info("%s   has persisted  :)", object_name)
                    except Exception as e:
                        print(e)
                        logger.error(
                            "fail during persisting the  %s :(", object_name
                        )

                else:
                    logger.info("Nothing has changed %s", object_name)
            else:
                logger.info(
                    "This object %s does not persist, it will be persisted now!",
                    object_name,
                )

                try:
                    pd.object.to_csv(file_path)
                    logger.info("An object  df has persisted  %s :)", object)
                except Exception as e:
                    logger.error(
                        "fail during persisting the  %s :( The error is %s",
                        object_name,
                        e,
                    )

        except Exception as e:
            logger.error(
                "Problem while presisting object %s %s :(", object_name, e
            )

        return

    def helper_load_from_data_save_path(self, object_name):
        """Load object from persisted into memory. The object will be in
        this path : path_to_save


        Args:
        object_name: An string as a name for the object, Can be a
         data frame, pickle, data sets, ...

        Returns:
        None: A success or fail message
        """

        try:
            # Find if the file already exists
            directory = general_rules_features["path_to_save"]
            for filename in os.listdir(directory):
                if filename.endswith(".pkl"):
                    full_path_file = os.path.join(
                        directory, filename
                    )  # Full path of the object
                    if filename == object_name:
                        try:
                            object_to_return = pickle.load(
                                open(full_path_file, "rb")
                            )
                            # object_to_return = joblib.load(full_path_file)

                        except Exception as e:
                            print("error in load")
                            logger.error(
                                "fail, there is a problem accessing \
                                    pickle files, %s :(",
                                e,
                            )
                        break
        except Exception as e:
            logger.error(
                "fail while laoding object the  %s , %s:(", object_name, e
            )
        return object_to_return

    def helper_check_sublist(self, lst1, lst2):
        """Check if a list is a sublist of another one

        Args:
        lst1: sub list
        lst2: main list

        Returns:
        Boolean: True is lst2 contain elements of lst1
        """
        return set(lst1) <= set(lst2)

    def month_to_number(self, m1, m2):

        m1 = m1.replace("M", "")
        m1 = m1.replace("m", "")
        m2 = m2.replace("M", "")
        m2 = m2.replace("m", "")

        try:
            m1 = int(m1)
            m2 = int(m2)
        except Exception as e:
            print(e)

        return (m2 - m1) * 6

        # return m_num

    def data_merger(self, flag="BT", persist_read=False):
        """Get access to persisted data frames before transformation and merge
        using the information inside data_configs.py.


        Args:
        flag: use BT to only ready persisted objects before the transformation.

        Returns:
        None: This method will generate merged objects persisted in fstore.
        """
        df_frames = []

        if persist_read:
            # Read persisted object before transformation
            df_dic = self.read_from_persisted_features(flag=flag)
        else:
            # Read from files and data
            df_dic = self.dataframes_sorter_by_id()

        # Check to see if the list of data frames for merge will come form
        # a select_data_frames or not
        if not general_rules_features["how_to_merge"]["select_data_frames"]:
            # Reverse the order of a dictionary iteration because the last one
            # most likely has the biggest number of IDs
            for key, value in reversed(df_dic.items()):
                # If  base merge is already defined on configs, use it otherwise, do
                # next steps
                if general_rules_features["how_to_merge"]["base_marge"]:
                    # Select the base data frame by its name from the persisted file
                    # Create a copy of each data frame
                    df = df_dic[
                        general_rules_features["how_to_merge"]["base_marge"]
                    ]["data"].copy()
                    # drop index
                    df = df.reset_index(drop=True)
                    # Add df to df_frames_train
                    df_frames.append(df)
                    break
                # If the base is in None
                # Create a copy of each data frame
                df = df_dic[key]["data"].copy()
                # drop index
                df = df.reset_index(drop=True)
                df_frames.append(df)

            # If in the data_configs.py we select to merge from a list, this code will
            # be executed otherwise, the else
        elif general_rules_features["how_to_merge"]["select_data_frames"]:
            # This only adds name of the data frame
            for name in general_rules_features["how_to_merge"][
                "select_by_name"
            ]:
                # Add selected data frames to the list for further usage
                try:
                    df_frames.append(df_dic[name]["data"])
                    logger.info(
                        "Dataframes with this name %s  \
                            added to the list %s :)",
                        name,
                    )
                except Exception as e:
                    logger.error(
                        "Dataframes with this name %s can not be \
                            added to the list %s :(",
                        name,
                        e,
                    )

            # Log output
            logger.info(
                "A  list of data frames added  for performing merges later."
            )

        else:
            # Make df_frames =[]
            df_frames = []
            # Set the number of selected dfs to be less than
            # "no_count" in general_rules_features['how_to_merge']['no_count']
            number_of_selected_dfs = 0
            for key, value in df_dic.items():

                if (
                    number_of_selected_dfs
                    <= general_rules_features["how_to_merge"]["no_count"]
                ):

                    # Copy data frame
                    df = df_dic[key]["data"].copy()
                    # drop index
                    df = df.reset_index(drop=True)
                    df_name = df_dic[key]["name"]
                    # If data is in this list do not continue
                    if (
                        df
                        in general_rules_features[
                            "list_of_data_to_drop_manually"
                        ]
                    ):
                        break
                    # Add dfs to a list for marge later
                    try:
                        if not isinstance(df, pd.Series):
                            df_frames.append(df)
                    except Exception as e:
                        logger.error(
                            "Problem while preparing merge list  %s %s",
                            df_name,
                            e,
                        )

                    # Log output
                    logger.info(
                        "one data frame added to list for \
                            performing merges later  %s : ",
                        df_name,
                    )

                    # Add one to number_of_selected_dfs
                    number_of_selected_dfs = number_of_selected_dfs + 1

        # Mergeing all dfs appended in df_frames_train
        try:
            # Select ids for merging for example ['subject_id', 'month_of_visit']
            sub_list_for_merge = general_rules_features["merge_ids"]
            # Check to see sub_list_for_merge is in df of df_frames to merge
            final_frames = []
            for df in df_frames:
                if self.helper_check_sublist(sub_list_for_merge, df.columns):
                    final_frames.append(df)
                    logger.info(
                        "One  Datafrmae added to be merged for final, %s %s",
                        sub_list_for_merge[1],
                        df[sub_list_for_merge[1]].nunique(),
                    )

            print(final_frames)
            DF_IDs = reduce(
                lambda left, right: pd.merge(
                    left, right, on=sub_list_for_merge, how="left"
                ),
                final_frames,
            )
            # drop index
            DF_IDs = DF_IDs.reset_index(drop=True)
            object_name = "DF_IDs"
        except Exception as e:
            logger.error("Problem while  merge data frames %s", e)

        return DF_IDs, object_name


# This function takes a data frame
# as a parameter and returning list
# of column names whose contents
# are duplicates.


def helper_getDuplicateColumns(df):

    # Create an empty set
    duplicateColumnNames = set()

    # Iterate through all the columns
    # of data frame
    for x in range(df.shape[1]):

        # Take column at xth index.
        col = df.iloc[:, x]

        # Iterate through all the columns in
        # DataFrame from (x + 1)th index to
        # last index
        for y in range(x + 1, df.shape[1]):

            # Take column at yth index.
            otherCol = df.iloc[:, y]

            # Check if two columns at x & y
            # index are equal or not,
            # if equal then adding
            # to the set
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])

    # Return list of unique column names
    # whose contents are duplicates.
    return list(duplicateColumnNames)


def helper_month_to_num(lst):
    # Get a list of months and return a list of numbers of multiplication of 6
    # e.g., [m06 , m12, m48] ---> [1, 2, 8]
    month_to_num = []
    for month in lst:
        if month.lower() == "m0" or month.lower() == "sc":
            month_to_num.append(0)
        if month.lower() == "m06" or month.lower() == "m6":
            month_to_num.append(1)
        if month.lower() == "m12":
            month_to_num.append(2)
        if month.lower() == "m18":
            month_to_num.append(3)
        if month.lower() == "m24":
            month_to_num.append(4)
        if month.lower() == "m30":
            month_to_num.append(5)
        if month.lower() == "m36":
            month_to_num.append(6)
        if month.lower() == "m42":
            month_to_num.append(7)
        if month.lower() == "m48":
            month_to_num.append(8)
        if month.lower() == "m54":
            month_to_num.append(9)
        if month.lower() == "m60":
            month_to_num.append(10)
        if month.lower() == "m66":
            month_to_num.append(11)
        if month.lower() == "72":
            month_to_num.append(12)
        if month.lower() == "m78":
            month_to_num.append(13)
        if month.lower() == "m84":
            month_to_num.append(14)
        if month.lower() == "m90":
            month_to_num.append(15)
        if month.lower() == "m96":
            month_to_num.append(16)

    return month_to_num


def uplift_scores(df, subject_list_yes, subject_list_no, score):

    df = df.copy()

    df_yes = df.loc[(df["subject_id"].isin(subject_list_yes)), score]
    df_no = df.loc[(df["subject_id"].isin(subject_list_no)), score]

    # print('with medications')
    # print(df.loc[(df['subject_id'].isin(subject_list_yes)),'subject_id'].nunique())
    # print(df_yes.mean())
    # print(df_yes.std())

    # print('with no medications')
    # print(df.loc[(df['subject_id'].isin(subject_list_no)),'subject_id'].nunique())
    # print(df_no.mean())
    # print(df_no.std())

    if df_no.shape[0] > 0 and df_yes.shape[0] > 0:
        no_mean = df_no.mean()
        yes_mean = df_yes.mean()
        to_be_reducted = no_mean - yes_mean
        print(" the difference : ")
        print(to_be_reducted)
        print(" problem is  : ")
        print(score)
        df.update(
            df.loc[(df["subject_id"].isin(subject_list_no)), score]
            - to_be_reducted
        )
    else:
        print("one group has zero members")

    return df


def find_in_med(df, month, drug_name, pd_type, yes_or_no):

    df_copy = df.copy()

    if yes_or_no == "yes":

        if pd_type == "mix":

            output_list = (
                df_copy.loc[(df_copy["month_of_visit"] == month)]
                .loc[
                    (df_copy["diagnosis"] == "parkinson's disease")
                    | (df_copy["diagnosis"] == "idiopathic pd")
                ]
                .loc[(df_copy[drug_name] == "yes")]["subject_id"]
                .tolist()
            )

        if pd_type == "pd":

            output_list = (
                df_copy.loc[(df_copy["month_of_visit"] == month)]
                .loc[(df_copy["diagnosis"] == "parkinson's disease")]
                .loc[(df_copy[drug_name] == "yes")]["subject_id"]
                .tolist()
            )

        if pd_type == "pp":

            output_list = (
                df_copy.loc[(df_copy["month_of_visit"] == month)]
                .loc[(df_copy["diagnosis"] == "idiopathic pd")]
                .loc[(df_copy[drug_name] == "yes")]["subject_id"]
                .tolist()
            )

    if yes_or_no == "no":

        if pd_type == "mix":

            output_list = (
                df_copy.loc[(df_copy["month_of_visit"] == month)]
                .loc[
                    (df_copy["diagnosis"] == "parkinson's disease")
                    | (df_copy["diagnosis"] == "idiopathic pd")
                ]
                .loc[(df_copy[drug_name] == "no")]["subject_id"]
                .tolist()
            )

        if pd_type == "pd":

            output_list = (
                df_copy.loc[(df_copy["month_of_visit"] == month)]
                .loc[(df_copy["diagnosis"] == "parkinson's disease")]
                .loc[(df_copy[drug_name] == "no")]["subject_id"]
                .tolist()
            )

        if pd_type == "pp":

            output_list = (
                df_copy.loc[(df_copy["month_of_visit"] == month)]
                .loc[(df_copy["diagnosis"] == "idiopathic pd")]
                .loc[(df_copy[drug_name] == "no")]["subject_id"]
                .tolist()
            )

    return set(output_list)
