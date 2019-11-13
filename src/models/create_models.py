import os
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

### ----- Set up project directory path names to load and save data ----- ###
FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')
SENSITIVE_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, '../SENSITIVE')


class CreateModels(object):
    """
    Creates classifier models for Vinyl Me, Please customer survey data.
    
    Loads featurized, model-ready survey data from .csv, subsets DataFrame
    into X and y arrays, makes train/test split, performs grid search for
    GradientBoostingClassifier or AdaBoostClassifier models, and then pickles
    and saves models.

    Requires:
        - modeling data (.csv): Featurized, model-ready survey .csv with account
        status in first column. Must be saved to SENSITIVE_DATA_DIRECTORY, which
        must be located outside of any git repo due to possibility of Personally
         Identifiable Information (PII).

    Returns:
        Saves trained, pickled GradientBoostingClassifier or AdaBoostClassifier
        model to MODELS_DIRECTORY.
    """

    def __init__(self, modeling_data_filename='modeling_data.csv'):
        """
        Loads featurized, model-ready survey data from .csv.

        Returns:
            df - DataFrame for building models.
        """

        modeling_data_filepath = os.path.join(SENSITIVE_DATA_DIRECTORY,
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
        """Separates modeling (X) and target (y) data, makes train/test split.

        Args:
            data_frame = Model DataFrame with account status in first column.
            test_size = Fraction of data to be split off as test data.
        
        Returns:
            X - Full modeling data set.
            y - Target data for full data set.
            X_train, X_test, y_train, y_test - Subsetted data.
        """

        self.y = data_frame.iloc[:, 0]
        self.X = data_frame.drop(data_frame.columns[0], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                            self.X,
                                                            self.y,
                                                            test_size=test_size,
                                                            random_state=42)
        print("Created X and y and split into train/test.")

    def grid_search(self, algo='adaboost'):
        """
        Performs GridSearchCV and returns the best model.

        Args:
            algo (str) - 'adaboost' (AdaBoostClassifier) or 'gbc'
            (GradientBoostingClassifier). Will perform GridSearchCV using
            selected algorithm and appropriate pre-defined boosting grids.
        
        Returns:
            Best model based on F1 score.
        """

        X_train = self.X_train
        y_train = self.y_train

        if algo == 'adaboost':
            self.classifier = AdaBoostClassifier()
            boosting_grid_rough = {'learning_rate': np.logspace(-3, 0, num = 4),
                                   'n_estimators': [50, 200, 500, 1000],
                                   'random_state': [42]}
        elif algo == 'gbc':
            self.classifier = GradientBoostingClassifier()
            boosting_grid_rough = {
                        'learning_rate': np.logspace(-4, -1, num = 4),
                        'max_depth': [1, 2, 5],
                        'n_estimators': [50, 200, 400],
                        'min_samples_split': np.logspace(-3, -1, num = 3),
                        'min_samples_leaf': np.logspace(-3, -1, num = 3),
                        'random_state': [42]}

        #------------ Coarse GridSearchCV Section ------------#
        #------- Un-comment if coarse search is needed -------#

        # coarse_search = GridSearchCV(self.classifier,
        #                             boosting_grid_rough,
        #                             scoring='f1',
        #                             error_score=np.nan,
        #                             n_jobs=-1)
        # print(f"Starting grid search - coarse using "
        #       f"{type(self.classifier).__name__}.")
        # print("Will take several minutes.")
        # 
        # coarse_search.fit(X_train, y_train)
        # coarse_params = coarse_search.best_params_
        # coarse_score = coarse_search.best_score_
        # print(f"Coarse search best parameters for {self.classifier}:")
        # for param, val in coarse_params.items():
        #     print("{0:<20s} | {1}".format(param, val))
        # print("Coarse search best score: {0:0.3f}".format(coarse_score))
        #-----------------------------------------------------#

        if algo == 'adaboost':
            boosting_grid_fine = {'learning_rate': [0.05, 0.1, 0.15],
                                  'n_estimators': [150, 200, 250],
                                  'random_state': [42]}
        elif algo == 'gbc':
            boosting_grid_fine = {'learning_rate': [0.0005, 0.001, 0.005],
                                   'max_depth': [1, 2, 3],
                                   'n_estimators': [100, 200, 300],
                                   'min_samples_split': [0.005, 0.01, 0.05],
                                   'min_samples_leaf': [0.005, 0.01, 0.05],
                                   'random_state': [42]}

        fine_search = GridSearchCV(self.classifier,
                                boosting_grid_fine,
                                scoring='f1',
                                error_score=np.nan,
                                n_jobs=-1)
        print(f"Starting grid search - fine using "
              f"{type(self.classifier).__name__}.")
        print("Will take several minutes.")
        fine_search.fit(X_train, y_train)
        fine_params = fine_search.best_params_
        fine_score = fine_search.best_score_
        print(f"Fine search best parameters for "
              f"{type(self.classifier).__name__}:")
        for param, val in fine_params.items():
            print("{0:<20s} | {1}".format(param, val))
        print("Fine search best score: {0:0.3f}".format(fine_score))
        self.model_best = fine_search.best_estimator_

        y_pred = self.model_best.predict(self.X_test)
        fscore = f1_score(self.y_test, y_pred)
        print(f"F1 score on holdout set using best "
              f"{type(self.classifier).__name__} model: {fscore:.3f}\n")
        print(f"Returning best {type(self.classifier).__name__} model.")
        return self.model_best

    def save_model(self, model):
        """Saves trained, pickled model to MODELS_DIRECTORY.

        Model was previously trained using data split into train/test sets.
        Before saving, model using best hyperparameters is trained on the full
        dataset.

        Args:
            model: Trined model to be saved.

        Returns:
            Saves trained, pickled model to MODELS_DIRECTORY.
        """

        self.model_best.fit(self.X, self.y)
        model_name = type(self.classifier).__name__
        save_model_filepath = os.path.join(MODELS_DIRECTORY, model_name)

        with open(f"{save_model_filepath}.pkl","wb") as f:
            pickle.dump(self.model_best, f)
        print(f"Saving best {type(self.classifier).__name__} model to "
              f"{save_model_filepath}.")


if __name__ == '__main__':
    create_models = CreateModels()
    create_models.make_train_test_data(create_models.df)
    create_models.grid_search(algo='adaboost')
    create_models.save_model(create_models.model_best)
