import os
import sys
sys.path.append('.')

import pandas as pd
import psycopg2

### ----- Set up project directory path names to load and save data ----- ###
FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
SENSITIVE_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, '../SENSITIVE')


class SurveyJoinAccountData(object):
    """
    Vinyl Me, Please survey and customer data processing class.

    Loads featurized survey data from .csv, subsets DataFrame to extract
    data to be used for model building, and adds target data (subscription
    status) from production PostgreSQL database by joining on customer email
    address.

    Requires:
        - survey data (.csv): Featurized .csv saved to SENSITIVE_DATA_DIRECTORY,
        which must be located outside of any git repo due to Personally
        Identifiable Information (PII).
        - production database: Instance of Vinyl Me, Please production database,
        online and reachable by Psycopg2.

    Returns:
        Saves featurized DataFrame to .csv to SENSITIVE_DATA_DIRECTORY. All
        PII should be removed, but still saving outside git repo to be safe.
    """


    def __init__(self, featurized_df_filename='featurized_survey_data.csv'):
        """Loads featurized data from SENSITIVE_DATA_DIRECTORY.

        Raises error if featurized data can't be found.

        Returns:
            df - DataFrame containing customer survey data.
        """

        featurized_df_filepath = os.path.join(SENSITIVE_DATA_DIRECTORY,
                                              featurized_df_filename)
        if os.path.exists(featurized_df_filepath):
            self.df = pd.read_csv(featurized_df_filepath,
                                  header=[0,1],
                                  low_memory=False)
            print(f"Loaded featurized survey data from "
                  f"{featurized_df_filepath}.\n")
        else:
            print('Failed to load featurized survey data!\n'
                  'Need to create featurized survey data using '
                  'src/features/build_survey_features.py\n')
            sys.exit()

        """Sets up PostgreSQL query.

        Query to be run on Vinyl Me, Please production database via Psycopg2
        by later function. Retrieves target data for modeling. Excludes
        customer accounts created within one month of when the data was
        collected. (Ends 2019-10-01 - DB dump 2019-11-02.)
        """

        self.conn = psycopg2.connect(database="vinyl", user="postgres",
                                     host="localhost", port="5435")
        self.cur = self.conn.cursor()

        self.db_query = '''SELECT
                              customer_email,
                              total_lifetime_revenue,
                              primary_status
                           FROM
                              mailchimp_list_fields
                           WHERE
                               customer_created_at < '2019-10-01'::date;
                        '''

    def subset_noobs(self, source_df):
        """Creates subsetted DataFrame with customers new to vinyl.

        Subsets featurized survey data to only include vinyl 'noobs', which are
        customers who answered the following question with the below answers:

        How long have you been buying records?
            - I just started
            - 6 - 12 months
            - 1-3 years
        Args:
            source_df - Full DataFrame with info on all customers.

        Returns:
            df_noobs - DataFrame only containing info on 'noobs'.
        """

        self.df_noobs = source_df[(source_df.iloc[:,33] <= 1)]
        print("Created df_noobs subset.")
        print(f"length total: {len(source_df)}")
        print(f"length noobs: {len(self.df_noobs)}\n")

    def create_model_df(self, source_df):
        """Subsets DataFrame to only retain data to be used for modeling.

        Args:
            source_df - DataFrame to be subsetted.

        Returns:
            df_model - DataFrame containing only data to be used for modeling.
        """

        model_column_list = [5, 9, 12, 13, 16, 33, 34]
        model_column_list = model_column_list + [i for i in range(63, 83)]
        model_column_list = model_column_list + [i for i in range(108, 120)]
        model_column_list = model_column_list + [i for i in range(125, 133)]
        model_column_list = model_column_list + [i for i in range(133, 146)]
        model_column_list = model_column_list + [i for i in range(180, 217)]
        model_column_list = model_column_list + [i for i in range(406, 414)]

        self.df_model = source_df.iloc[:, model_column_list]

    def create_dummy_cols(self):
        """Creates dummy (one-hot encoded) columns for several answer columns.

        THIS METHOD MUST BE RUN AS A FINAL PRE-PROCESSING STEP, AS IT WILL
        FLATTEN THE DATAFRAME MULTI-INDEX AND SHIFT COLUMN NUMBERS!

        Returns:
            Adds dummy columns for the following survey questions:
                Where do you live? - Original survey column 12
                What is your living arrangement?  - Original survey column 13
                Do you own/lease a vehicle? - Original survey column 16
        """

        # Creating dummy columns
        dummy_df_where_live = pd.get_dummies(self.df_model.iloc[:, 2],
                                             prefix='Where do you live?',
                                             prefix_sep='_')
        dummy_df_house = pd.get_dummies(self.df_model.iloc[:, 3],
                                     prefix='What is your living arrangement?',
                                     prefix_sep='_')
        dummy_df_car = pd.get_dummies(self.df_model.iloc[:, 4],
                                      prefix='Do you own/lease a vehicle?',
                                      prefix_sep='_')

        # Adding dummy columns to original DataFrame
        self.df_model = pd.concat([self.df_model,
                                  (dummy_df_where_live * 3)],
                                   axis=1)
        self.df_model = pd.concat([self.df_model,
                                  (dummy_df_house * 3)],
                                   axis=1)
        self.df_model = pd.concat([self.df_model,
                                  (dummy_df_car * 3)],
                                   axis=1)

        # Dropping original columns
        self.df_model.drop(self.df_model.columns[[2, 3, 4]],
                           axis=1,
                           inplace=True)

    def query_customer_status(self):
        """Runs PostgreSQL query defined in class __init__ method.

        Returns:
            df_status - DataFrame containing model target values.
        """

        print("Executing query...")
        self.cur.execute(self.db_query)
        results = self.cur.fetchall()
        self.colnames = [desc[0] for desc in self.cur.description]
        self.df_status = pd.DataFrame(results, columns=self.colnames)
        print("PostgreSQL query complete. Created df_status, colnames.\n")

    def join_survey_status(self, survey_data, target_data):
        """Joins survey data df and target data df on customer email address.

        Args:
            survey_data - DataFrame containing modeling-ready survey responses,
                with customer email address in the first column.
            target_data - DataFrame containing target data ($, acct status),
                with customer email address in the first column.

        Returns:
            df_merged - Combined DataFrame.
        """

        self.df_merged = pd.merge(survey_data,
                         target_data,
                         left_on=survey_data.iloc[:, 0],
                         right_on=target_data.iloc[:, 0],
                         how='inner')
        print("Joined survey_data and target_data on email address.")

    def col_status_encode(self, data_frame, col_idx=3):
        """ 
        Binary encodes account status DataFrame column, with canceled = 1.

        Args: 
            data_frame: DataFrame containing the account status column.
            col_idx (int): The index of the account status column.

        Returns: 
            DataFrame column with values encoded as follows:
                account status 'cancelled'  --> 1
                any other account status    --> 0
        """
        data_frame.iloc[:,col_idx] = data_frame.iloc[:,col_idx].apply(lambda x:
                                             1 if str(x) == 'cancelled' else 0)
        print(f"\nEncoded account status - column {col_idx}.\n")

    def remove_pii(self, data_frame, col_idx_lst=[0, 1, 2, 4]):
        """ 
        Removes remaining PII data.

        Args:
            data_frame: DataFrame containing PII columns.
            col_idx_list (list of int): Indicies of the PII columns to be
                removed.

        Returns:
            data_frame with PII columns removed.
        """

        data_frame.drop(data_frame.columns[col_idx_lst], axis=1, inplace=True)
        print(f"Removed PII data - columns {col_idx_lst}.\n")

    def save_to_csv(self, data_frame, file_name='modeling_data.csv'):
        """Saves DataFrame to .csv.

        Saves .csv to SENSITIVE_DATA_DIRECTORY, which must be located outside
        of any git repo due to risk of PII.

        Args:
            data_frame: DataFrame to be saved to .csv.
            file_name (str): Name of resulting .csv file.

        Returns:
            DataFrame saved as .csv to SENSITIVE_DATA_DIRECTORY.
        """

        modeling_data_filepath = os.path.join(SENSITIVE_DATA_DIRECTORY,
                                              file_name)
        data_frame.to_csv(modeling_data_filepath, index=False)
        print(f"Saved modeling-ready data to {modeling_data_filepath}.")


if __name__ == '__main__':
    survey_join = SurveyJoinAccountData()
    survey_join.subset_noobs(survey_join.df)
    survey_join.create_model_df(survey_join.df_noobs)
    survey_join.create_dummy_cols()
    survey_join.query_customer_status()
    survey_join.join_survey_status(survey_join.df_status, survey_join.df_model)
    survey_join.col_status_encode(survey_join.df_merged)
    survey_join.remove_pii(survey_join.df_merged)
    survey_join.save_to_csv(survey_join.df_merged)
