import os
import sys
sys.path.append('.')

import pandas as pd

# Set up project directory path names to load and save data
FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
SENSITIVE_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, '../SENSITIVE')


def load_data_as_dataframe(filename='2019 Member Survey - Raw Data.csv'):
    """
    Loads survey data from .csv file located in SENSITIVE_DATA_DIRECTORY,
    outside of this git repo, due to customer PII.

    Args:
        filename (str): The name of the survey data file.

    Returns:
        df: survey data as pandas DataFrame, with two-level MultiIndex.
        df_col_names: pandas DataFrame with column names for ease of use.
    """ 

    filepath = os.path.join(SENSITIVE_DATA_DIRECTORY, filename)
    df = pd.read_csv(filepath,
                            header=[0,1],
                            low_memory=False)
    df_col_names = pd.DataFrame(df.columns.to_numpy().reshape([-1, 1]))
    return df, df_col_names