import os
import sys
sys.path.append('.')

import numpy as np
import pandas as pd

import psycopg2

# Set up project directory path names to load and save data
FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
SENSITIVE_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, '../SENSITIVE')

# Import survey data loading script
from src.data.make_survey_dataset import load_data_as_dataframe


class BuildSurveyFeatures(object):
    """
    This class loads survey data from .csv, numerically encodes a number of
    columns of interest, and then saves the featurized DataFrame to .csv in
    a secure directory outside of the git repo.
    """  

    def __init__(self):
        # Load survey data and create main DataFrame and df_col_names. 
        self.df, self.df_col_names = load_data_as_dataframe(
                                  filename='2019 Member Survey - Raw Data.csv')
        print(f"Loaded survey DataFrame of size {self.df.shape}.\n")


    def col_how_much_use_encode(self, data_frame, col_idx_list):
        """ 
        Function to numerically encode the values in pandas DataFrame columns:
        How often do you use/interact with the following Vinyl Me, Please
            elements? - Survey columns 133-145

        Parameters: 
            col_idx_list (list of int): Numerical indicies of the columns to be
                encoded.
            data_frame: DataFrame with columns to be encoded.

        Returns: 
            DataFrame column with values encoded as in the 'use' dictionary
            below.
        """
        use = {"I don't know about it": 0,
               "I don't care about it": 0,
               "Used it once": 1,
               "Sometimes": 2,
               "Frequently": 3}
        
        for col_idx in col_idx_list:
            # Filling NaN values with base case.
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
            col_idx_list (list of int): Numerical indicies of the columns to be
                encoded.
            data_frame: DataFrame with columns to be encoded.

        Returns: 
            DataFrame column with values encoded as in the 'freq' dictionary
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
            # Filling NaN values with base case.
            data_frame.iloc[:,col_idx].fillna("Hardly ever", inplace=True)
            data_frame.iloc[:,col_idx] = [freq[val] for val in
                                          data_frame.iloc[:,col_idx]]
        print(f"Encoded \"how often\" columns: {col_idx_list}")


    def col_how_long_records_encode(self, data_frame, col_idx_list=[33]):
        """ 
        Function to numerically encode the values in pandas DataFrame column:
        How long have you been buying records? - Survey column 33

        Parameters: 
            col_idx_list (int): The index of the column to be encoded.
            data_frame: DataFrame with columns to be encoded.

        Returns: 
            DataFrame with column values encoded as in the 'length' dictionary
                below.
        """
        length = {'I just started': 0,
                '6 - 12 months': 0.5,
                '1-3 years': 1,
                '3-5 years': 1.5,
                '5-10 years': 2,
                '10-15 years': 2.5,
                'More than 15 years': 3}
        
        for col_idx in col_idx_list:
            # Filling NaN values with base case.
            data_frame.iloc[:,col_idx].fillna('I just started', inplace=True)
            data_frame.iloc[:,col_idx] = [length[val] for val in 
                                          data_frame.iloc[:,col_idx]]
        print(f"Encoded \"how long collect records\" column: {col_idx_list}")


    def encode_records_own(self, data_frame, col_idx=34):
        """ 
        Function to bin and normalize the values in pandas DataFrame column:
        About how many records do you own? - Survey column 34

        Parameters: 
            col_idx (int): The index of the column to be encoded.
            data_frame: Name of the DataFrame containing the column in question.

        Returns: 
            DataFrame column with values encoded on a scale of 0-5, binned to
            the nearest whole number, with values at the 90th percentile and up
            set to 5.
        """

        data_frame.iloc[:,col_idx].fillna(0, inplace=True)
        max_cutoff = np.percentile(data_frame.iloc[:,col_idx].to_numpy(), 90)
        data_frame.iloc[:,col_idx] = np.where(
                                      data_frame.iloc[:,col_idx] >= max_cutoff,
                                      max_cutoff,
                                      data_frame.iloc[:,col_idx])
        data_frame.iloc[:,col_idx] = (data_frame.iloc[:,col_idx]
                                      / max_cutoff) * 5
        data_frame.iloc[:,col_idx] = data_frame.iloc[:,col_idx].apply(
                                                              lambda x: int(x))
        print(f"Encoded \"how many records own\" column: {col_idx}")


    def col_binary_encode(self, data_frame, col_idx_list):
        """ 
        Function to binary encode the values in a pandas DataFrame column.

        Parameters:
            col_idx_list (list of int): Indicies of the columns to be encoded.
            data_frame: Name of DataFrame containing the columns in question.

        Returns: 
            DataFrame column with values encoded as follows:
                'str(column title)' --> 1
                anything else       --> -1
        """
        for col_idx in col_idx_list:
            label = data_frame.columns.get_level_values(1)[col_idx]
            data_frame.iloc[:,col_idx] = data_frame.iloc[:,col_idx].apply(
                                          lambda x: 1 if (str(x) == str(label))
                                          | (x == 1) else -1)
        print(f"Binary encoded columns: {col_idx_list}")


    def col_age_encode(self, data_frame, col_idx_list=[9]):
        """ 
        Function to numerically encode the values in pandas DataFrame column:
        How old are you? - Survey column 9

        Parameters: 
            col_idx_list (int): The index of the column to be encoded.
            data_frame: Name of the DataFrame containing the column in question.

        Returns: 
            DataFrame column with values encoded as in the 'age' dictionary below.
        """
        age = {'Under 18': 0,
               '18-20': 0.5,
               '21-24': 1,
               '25-34': 1.5,
               '35-44': 2,
               '45-54': 2.5,
               '55-64': 3,
               '65+': 3.5}
        
        for col_idx in col_idx_list:
            # Filling NaN values with most common age group.
            data_frame.iloc[:,col_idx].fillna('25-34', inplace=True)
            data_frame.iloc[:,col_idx] = [age[val] for val
                                            in data_frame.iloc[:,col_idx]]
        print(f"Encoded \"how old are you\" column: {col_idx}")


    def col_gender_encode(self, data_frame, col_idx_list=[10]):
        """ 
        Function to numerically encode the values in pandas DataFrame column:
        To what gender do you identify? - Survey column 10

        Parameters: 
            col_idx_list (int): The index of the column to be encoded.
            data_frame: Name of the DataFrame containing the column in question.

        Returns: 
            DataFrame column with values encoded as follows:
                Male          --> 1
                Female, Other --> 0
        """
        for col_idx in col_idx_list:
            data_frame.iloc[:,col_idx] = data_frame.iloc[:,col_idx].apply(
                                            lambda x: 1 if (str(x) == 'Male')
                                            else 0)
        print(f"Encoded \"gender identity\" column: {col_idx_list}")


    def col_records_month_encode(self, data_frame, col_idx_list=[36]):
        """ 
        Function to numerically encode the values in pandas DataFrame column:
        About how many records do you buy per month? - Survey column 36

        Parameters: 
            col_idx_list (int): The index of the column to be encoded.
            data_frame: Name of the DataFrame containing the column in question.

        Returns: 
            DataFrame column with values encoded as in the 'record_nums' dictionary below.
                0 - 1         --> 0-1           --> 0
                3-Feb         --> 2-3           --> 1
                5-Apr         --> 4-5           --> 2
                10-Jun        --> 6-10          --> 3
                More than 10  --> More than 10  --> 4
        """
        record_nums = {'0 - 1': 0,
                    '3-Feb': 1,
                    '5-Apr': 2,
                    '10-Jun': 3,
                    'More than 10': 4}
        
        for col_idx in col_idx_list:
            data_frame.iloc[:,col_idx].fillna('0 - 1', inplace=True)
            data_frame.iloc[:,col_idx] = [record_nums[num] for num \
                                        in data_frame.iloc[:,col_idx]]
        print(f"Encoded \"records buy per month\" column: {col_idx_list}")


    def col_satisfy_encode(self, data_frame, col_idx_list):
        """ 
        Function to numerically encode the values in a pandas DataFrame columns:
        How satisfied are you with the following? - Survey columns 125 - 132

        Parameters: 
            col_idx_list (int): The index of the column to be encoded.
            data_frame: Name of the DataFrame containing the column in question.

        Returns: 
            DataFrame column with values encoded as in the 'sentiment' dictionary below.
        """
        sentiment = {'Very Satisfied': 2,
                    'Satisfied': 1,
                    'Neutral': 0,
                    'Dissatisfied': -1,
                    'Very Dissatsified': -2}
        
        for col_idx in col_idx_list:
            data_frame.iloc[:,col_idx].fillna('Neutral', inplace=True)
            data_frame.iloc[:,col_idx] = [sentiment[rating] for rating
                                            in data_frame.iloc[:,col_idx]]
        print(f"Encoded \"how satisfied are you\" columns: {col_idx_list}")


    def col_agree_encode(self, data_frame, col_idx_list):
        """ 
        Function to numerically encode the values in a pandas DataFrame columns:
        How much do you agree with these statements?

        Parameters: 
            col_idx_list (int): The index of the column to be encoded.
            data_frame: Name of the DataFrame containing the column in question.

        Returns: 
            DataFrame column with values encoded as in the 'sentiment' dictionary below.
        """
        sentiment = {'strongly agree': 2,
                    'agree': 1,
                    'neutral': 0,
                    'disagree': -1,
                    'strongly disagree': -2}
        
        for col_idx in col_idx_list:
            data_frame.iloc[:,col_idx].fillna('Neutral', inplace=True)
            data_frame.iloc[:,col_idx] = [sentiment[str(rating).lower()] \
                                        for rating in data_frame.iloc[:,col_idx]]
        print(f"Encoded \"how much agree\" columns: {col_idx_list}")


    def save_to_csv(self, data_frame, file_name='featurized_survey_data.csv'):
        featurized_df_filepath = os.path.join(SENSITIVE_DATA_DIRECTORY,
                                              file_name)
        data_frame.to_csv(featurized_df_filepath, index=False)
        print(f"Saved featurized survey data to {featurized_df_filepath}.")


if __name__ == '__main__':
    build_features = BuildSurveyFeatures()

    build_features.col_age_encode(build_features.df, col_idx_list=[9])
    build_features.col_gender_encode(build_features.df, col_idx_list=[10])
    
    encode_col_list = list(range(17, 33))
    build_features.col_binary_encode(build_features.df, encode_col_list)
    
    build_features.col_how_long_records_encode(build_features.df,
                                               col_idx_list=[33])
    build_features.encode_records_own(build_features.df, col_idx=34)
    build_features.col_records_month_encode(build_features.df,
                                            col_idx_list=[36])

    agree_encode_col_list = list(range(63, 83))
    build_features.col_agree_encode(build_features.df, agree_encode_col_list)

    encode_col_list = list(range(108, 120))
    build_features.col_binary_encode(build_features.df, encode_col_list)

    satisfy_encode_col_list = list(range(125, 133))
    build_features.col_satisfy_encode(build_features.df,
                                      satisfy_encode_col_list)

    how_much_use_encode_list = list(range(133, 146))
    build_features.col_how_much_use_encode(build_features.df,
                                           how_much_use_encode_list)

    agree_encode_col_list = list(range(180, 217))
    build_features.col_agree_encode(build_features.df, agree_encode_col_list)

    encode_col_list = list(range(406, 414))
    build_features.col_binary_encode(build_features.df, encode_col_list)

    often_col_list = list(range(415, 422))
    build_features.col_how_often_do_encode(build_features.df, often_col_list)

    build_features.save_to_csv(build_features.df,
                               file_name='featurized_survey_data.csv')