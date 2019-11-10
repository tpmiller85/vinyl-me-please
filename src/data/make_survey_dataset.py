import os
import sys
sys.path.append('.')

import pandas as pd

FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]  # Directory this script is in
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]  # The 'src' directory
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]  # The root directory for the project
SENSITIVE_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, '../SENSITIVE')  # The data directory


def load_data_as_dataframe(filename='2019 Member Survey - Raw Data.csv'):
    #
    filepath = os.path.join(SENSITIVE_DATA_DIRECTORY, filename)
    df = pd.read_csv(filepath,
                            header=[0,1],
                            low_memory=False)
    df_col_names = pd.DataFrame(df.columns.to_numpy().reshape([-1, 1]))
    return df, df_col_names