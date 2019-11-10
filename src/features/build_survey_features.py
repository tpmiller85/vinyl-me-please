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

from src.data.make_survey_dataset import load_data_as_dataframe

class BuildSurveyFeatures(object):
    # This class will query a connected PostgreSQL database using psycopg2, and
    # then clean the data for modeling.

    def __init__(self):
        self.df, self.df_col_names = load_data_as_dataframe(
                                  filename='2019 Member Survey - Raw Data.csv')
        print(f"Loaded survey DataFrame of size {self.df.shape}.")

    def col_how_much_use_encode(self, data_frame, col_idx_list):
        """ 
        Function to numerically encode the values in pandas DataFrame columns:
        How often do you use/interact with the following Vinyl Me, Please
            elements? - Survey columns 133-145

        Parameters: 
            col_idx_list (int): The index of the column to be encoded.
            data_frame: Name of the DataFrame containing the column in question.

        Returns: 
            DataFrame column with values encoded as in the 'use' dictionary below.
        """
        use = {"I don't know about it": 0,
               "I don't care about it": 0,
               "Used it once": 1,
               "Sometimes": 2,
               "Frequently": 3}
        
        for col_idx in col_idx_list:
            # Filling NaN values with base case
            data_frame.iloc[:,col_idx].fillna("I don't know about it",
                                              inplace=True)
            data_frame.iloc[:,col_idx] = [use[val] for val in
                                                    data_frame.iloc[:,col_idx]]
        print(f"Encoded \"how much use\" columns: {col_idx_list}")


    def col_how_often_do_encode(self, data_frame, col_idx_list):
        """
        Function to numerically encode the values in pandas DataFrame columns:
        How often you do these things? - Survey columns 415-420

        Parameters: 
            col_idx_list (list of int): List of indicies of the columns to be
                encoded.
            data_frame: DataFrame containing the columns in question.

        Returns: 
            DataFrame with column values encoded as in the 'freq' dictionary
                below.
        """
        
        freq = {'Hardly ever': 0,
                'Every few months': 1,
                'A few times a month': 2,
                'About once a week': 3,
                'Several times per week': 4,
                'Every day': 5
                }
        
        for col_idx in col_idx_list:
            # Filling NaN values with base case
            data_frame.iloc[:,col_idx].fillna("Hardly ever", inplace=True)
            data_frame.iloc[:,col_idx] = [freq[val] for val in
                                                data_frame.iloc[:,col_idx]]
        print(f"Encoded \"how often\" columns: {col_idx_list}")


if __name__ == '__main__':
    build_features = BuildSurveyFeatures()

    how_much_use_encode_list = list(range(133, 146))
    build_features.col_how_much_use_encode(build_features.df,
                                           how_much_use_encode_list)

    often_col_list = list(range(415, 422))
    build_features.col_how_often_do_encode(build_features.df,
                                           often_col_list)