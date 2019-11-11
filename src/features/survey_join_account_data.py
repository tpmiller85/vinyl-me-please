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
SENSITIVE_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')


class SurveyJoinAccountData(object):
    """
    Vinyl Me, Please survey and customer data processing class.
    
    Loads featurized survey data from .csv, subsets DataFrame to extract
    data to be used for model building, and adds target data (subscription
    status) from production PostgreSQL database by joining on customer email
    address.

    Requires:
        survey data (.csv): Featurized .csv saved to SENSITIVE_DATA_DIRECTORY,
        which must be located outside of any git repo due to Personally
        Identifiable Information (PII).
        production database: Instance of Vinyl Me, Please production database,
        online and reachable by Psycopg2.

    Returns:
        Saves featurized DataFrame to .csv to SENSITIVE_DATA_DIRECTORY. All
        PII should be removed, but still saving outside git repo to be safe.
    """


    def __init__(self, featurized_df_filename='featurized_survey_data.csv'):
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

        self.conn = psycopg2.connect(database="vinyl", user="postgres",
                                     host="localhost", port="5435")
        self.cur = self.conn.cursor()

        # PostgreSQL query to be run on Vinyl Me, Please database. Enclose in
        # triple quotes for psycopg2: ''' '''
        self.db_query = '''SELECT
                              customer_email,
                              total_lifetime_revenue,
                              primary_status
                           FROM
                              mailchimp_list_fields
                           WHERE
                               customer_created_at < '2019-10-01'::date;
                        '''

    def subset_noobs(self):
        self.df_noobs = self.df[(self.df.iloc[:,33] <= 1)]

        # df_noobs = df[(df.iloc[:,33] == 'I just started') 
        #             | (df.iloc[:,33] == '6 - 12 months') 
        #             | (df.iloc[:,33] == '1-3 years')]

        print("Created df_noobs subset.")
        print(f"length total: {len(self.df)}")
        print(f"length noobs: {len(self.df_noobs)}\n")


    def create_model_df(self):
        model_column_list = [5, 9, 12, 13, 16, 33, 34]
        model_column_list = model_column_list + [i for i in range(63, 83)]
        model_column_list = model_column_list + [i for i in range(108, 120)]
        model_column_list = model_column_list + [i for i in range(125, 133)]
        model_column_list = model_column_list + [i for i in range(133, 146)]
        model_column_list = model_column_list + [i for i in range(180, 217)]
        model_column_list = model_column_list + [i for i in range(406, 414)]

        self.df_model = self.df_noobs.iloc[:, model_column_list]


    def create_dummy_cols(self):
        # RUN LAST, WILL FLATTEN INDEX AND SHIFT COLUMN NUMBERS!

        # Col 12 - Where do you live?
        # Col 16 - Do you own/lease a vehicle?

        dummy_df_where_live = pd.get_dummies(self.df_model.iloc[:, 2],
                                      prefix='Where do you live?',
                                      prefix_sep='_')
        dummy_df_house = pd.get_dummies(self.df_model.iloc[:, 3],
                                      prefix='What is your living arrangement?',
                                      prefix_sep='_')
        dummy_df_car = pd.get_dummies(self.df_model.iloc[:, 4],
                                      prefix='Do you own/lease a vehicle?',
                                      prefix_sep='_')

        self.df_model = pd.concat([self.df_model, (dummy_df_where_live * 3)], axis=1)
        self.df_model = pd.concat([self.df_model, (dummy_df_house * 3)], axis=1)
        self.df_model = pd.concat([self.df_model, (dummy_df_car * 3)], axis=1)

        self.df_model.drop(self.df_model.columns[[2, 3, 4]], axis=1, inplace=True)


    # Run PostgreSQL query and return pandas DataFrame
    def query_customer_status(self):
        print("Executing query...")
        self.cur.execute(self.db_query)
        results = self.cur.fetchall()
        self.colnames = [desc[0] for desc in self.cur.description]
        # tuples = zip(colnames, colnames)
        # midx = pd.MultiIndex.from_tuples(tuples)
        self.df_status = pd.DataFrame(results, columns=self.colnames) #columns=colnames
        print("PostgreSQL query complete. Created df_status, colnames.\n")


    def join_survey_status(self):
        self.df_merged = pd.merge(self.df_status,
                        self.df_model,
                        left_on=self.df_status.iloc[:, 0],
                        right_on=self.df_model.iloc[:, 0],
                        how='inner')
        print(f"Joined df_status and df_model on email address.")


    def col_status_encode(self, data_frame, col_idx=3):
        """ 
        Function to binary encode the values in a pandas DataFrame column.

        Parameters: 
            col_idx_list (int): The index of the column to be encoded.
            data_frame: Name of the DataFrame containing the column in question.

        Returns: 
            DataFrame column with values encoded as follows:
                'str(column title)' --> 1
                anything else       --> -1
        """
        data_frame.iloc[:,col_idx] = data_frame.iloc[:,col_idx].apply(lambda x:
                                    1 if str(x) == 'cancelled' else 0)

        print(f"\nEncoded account status - column {col_idx}.\n")


    def remove_pii(self, data_frame, col_idx_lst=[0, 1, 2, 4]):
        """ 
        Function to binary encode the values in a pandas DataFrame column.

        Parameters: 
            col_idx_list (int): The index of the column to be encoded.
            data_frame: Name of the DataFrame containing the column in question.

        Returns: 
            DataFrame column with values encoded as follows:
                'str(column title)' --> 1
                anything else       --> -1
        """
        data_frame.drop(data_frame.columns[col_idx_lst], axis=1, inplace=True)
        print(f"Removed PII data - columns {col_idx_lst}.\n")


    def save_to_csv(self, data_frame, file_name='modeling_data.csv'):
        modeling_data_filepath = os.path.join(SENSITIVE_DATA_DIRECTORY,
                                              file_name)
        data_frame.to_csv(modeling_data_filepath, index=False)
        print(f"Saved cleaned & safe modeling data to {modeling_data_filepath}.")


if __name__ == '__main__':
    survey_join = SurveyJoinAccountData()
    survey_join.subset_noobs()
    survey_join.create_model_df()
    survey_join.create_dummy_cols()
    survey_join.query_customer_status()
    survey_join.join_survey_status()
    survey_join.col_status_encode(survey_join.df_merged)
    survey_join.remove_pii(survey_join.df_merged)
    survey_join.save_to_csv(survey_join.df_merged)
