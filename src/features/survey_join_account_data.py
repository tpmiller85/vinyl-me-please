import os
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from pprint import pprint

import psycopg2

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence 
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import plot_partial_dependence

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLSResults
from statsmodels.tools.tools import add_constant

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set_style("white")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 80)

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]  # Directory this script is in
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]  # The 'src' directory
SRC_PYTHON_DIRECTORY = os.path.join(SRC_DIRECTORY, 'python')  # Directory
PYTHON_DATA_DIRECTORY = os.path.join(SRC_PYTHON_DIRECTORY, 'data')  # Directory

SRC_DATA_DIRECTORY = os.path.join(SRC_DIRECTORY, 'models')  # Directory for pickled models and model info
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]  # The root directory for the project

MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')  # Directory for pickled models and model info
SENSITIVE_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, '../SENSITIVE')  # The data directory

# from src.data.make_survey_dataset import load_data_as_dataframe


class SurveyJoinAccountData(object):

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

        dummy_df_where_live = pd.get_dummies(self.df_model.iloc[:, 2])
        dummy_df_house = pd.get_dummies(self.df_model.iloc[:, 3])
        dummy_df_car = pd.get_dummies(self.df_model.iloc[:, 4])

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
        self.merged_df = pd.merge(self.df_status,
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
        print(f"\nEncoded account status - column {col_idx}.")



if __name__ == '__main__':
    survey_join = SurveyJoinAccountData()
    survey_join.subset_noobs()
    survey_join.create_model_df()
    survey_join.create_dummy_cols()
    survey_join.query_customer_status()
    survey_join.join_survey_status()
    survey_join.col_status_encode(survey_join.df_noobs)

    # Need to save to safe spot, email still there.

