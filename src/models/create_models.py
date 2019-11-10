import os
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from pprint import pprint

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence 
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import plot_partial_dependence

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')  # annoying warnings



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
SAFE_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')  # Directory

MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')  # Directory for pickled models and model info
SENSITIVE_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, '../SENSITIVE')  # The data directory


class CreateModels(object):

    def __init__(self, modeling_data_filename='modeling_data.csv'):
        modeling_data_filepath = os.path.join(SAFE_DATA_DIRECTORY,
                                              modeling_data_filename)
        if os.path.exists(modeling_data_filepath):
            self.df = pd.read_csv(modeling_data_filepath,
                                  low_memory=False)
            print(f"Loaded survey modeling data from "
                  f"{modeling_data_filepath}.\n")
        else:
            print('Failed to load survey modeling data!\n'
                  'Need to create survey modeling data using '
                  'src/features/survey_join_account_data.py\n')
            sys.exit()
    
    def make_train_test_data(self, data_frame, test_size=0.2):
        self.y = data_frame.iloc[:, 0]
        self.X = data_frame.drop(data_frame.columns[0], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                            self.X,
                                                            self.y,
                                                            test_size=test_size,
                                                            random_state=42)
        print("Created X and y and split into train/test.")


    def grid_search(self, X_train, y_train, algo='adaboost'):
        ''' gets a rough idea where the best parameters lie '''
    
        if algo == 'adaboost':
            classifier = AdaBoostClassifier()
            boosting_grid_rough = {'learning_rate': np.logspace(-3, 0, num = 4),
                                   'n_estimators': [50, 100, 200, 500, 1000],
                                   'random_state': [42]}
        elif algo == 'gbc':
            classifier = GradientBoostingClassifier()
            boosting_grid_rough = {'learning_rate': np.logspace(-3, 0, num = 4),
                                   'n_estimators': [50, 100, 200, 500, 1000],
                                   'random_state': [42]}


'n_estimators':range(20,81,10)
'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8, random_state=10), 
param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,max_features='sqrt', subsample=0.8, random_state=10), 
param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9, min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10),

learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
max_depths = np.linspace(1, 32, 32, endpoint=True)




from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc





        coarse_search = GridSearchCV(classifier,
                                    boosting_grid_rough,
                                    scoring='f1',
                                    n_jobs=-1)
        print(f"Starting grid search - coarse using {classifier}.")
        print("Will take several minutes.")
        coarse_search.fit(self.X_train, self.y_train)
        coarse_params = coarse_search.best_params_
        coarse_score = coarse_search.best_score_
        print("Coarse search best parameters for {classifier}:")
        for param, val in coarse_params.items():
            print("{0:<20s} | {1}".format(param, val))
        print("Coarse search best score: {0:0.3f}".format(coarse_score))

        if algo == 'adaboost':
            boosting_grid_fine = {'learning_rate': [0.05, 0.1, 0.15],
                                  'n_estimators': [150, 200, 250],
                                  'random_state': [42]}
        elif algo == 'gbr':
            classifier = GBR()

                      dict(gbm__n_estimators = [50, 100, 150, 200],
                           gbm__max_depth = [5, 6, 7, 8, 9, 10]),
                      cv = 5,
                      scoring = make_scorer(mean_squared_error),
                      verbose = 100)



        fine_search = GridSearchCV(AdaBoostClassifier(),
                                boosting_grid_fine_noobs_df,
                                scoring='f1',
                                n_jobs=-1)
        print("\nStarting grid search - fine")
        fine_search.fit(X_train, y_train)
        fine_params = fine_search.best_params_
        fine_score = fine_search.best_score_
        print("Fine search best parameters:")
        for param, val in fine_params.items():
            print("{0:<20s} | {1}".format(param, val))
        print("Fine search best score: {0:0.3f}".format(fine_score))
        model_best = fine_search.best_estimator_
        print("Returning best model.")
        return model_best


if __name__ == '__main__':
    create_models = CreateModels()
    create_models.make_train_test_data(create_models.df)
