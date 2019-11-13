import os
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from pprint import pprint
import pickle
from sklearn.inspection import plot_partial_dependence
import matplotlib.pyplot as plt

### ----- Set up project directory path names to load and save data ----- ###
FILE_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
ROOT_IMGS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'images')
MODELS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')
SENSITIVE_DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, '../SENSITIVE')


class MakePlots(object):
    """
    Creates visualizations for models trained on Vinyl Me, Please survey data.
    
    Loads trained model from pickle file and modeling data from .csv, gets most
    important features, and makes plots showing relative feature importances
    as well as partial dependence plots.
    
    Requires:
        - modeling data (.csv): Data used to train models. Must be saved to
        SENSITIVE_DATA_DIRECTORY, which must be located outside of any git repo
        due to possibility of Personally Identifiable Information (PII).
        - trained model (.pkl): Trained, pickled model (either
        AdaBoostClassifier or GradientBoostingClassifier), saved to
        MODELS_DIRECTORY. 

    Returns:
        Saves trained, pickled GradientBoostingClassifier or AdaBoostClassifier
        model to MODELS_DIRECTORY.
    """


    def __init__(self,
                 modeling_data_filename='modeling_data.csv',
                 saved_model_filename='AdaBoostClassifier.pkl'):
        # Loads modeling data, creates X and y.
        modeling_data_filepath = os.path.join(SENSITIVE_DATA_DIRECTORY,
                                              modeling_data_filename)
        if os.path.exists(modeling_data_filepath):
            self.df = pd.read_csv(modeling_data_filepath,
                                  low_memory=False)
            self.y = self.df.iloc[:, 0]
            self.X = self.df.drop(self.df.columns[0], axis=1)

            print(f"Loaded survey modeling data from "
                  f"{modeling_data_filepath}.\n")
        else:
            print('Failed to load survey modeling data!\n'
                  'Need to create survey modeling data using '
                  'src/features/survey_join_account_data.py\n')
            sys.exit()
        
        # Loads saved, pickled model.
        saved_model_filepath = os.path.join(MODELS_DIRECTORY, saved_model_filename)
        if os.path.exists(saved_model_filepath):
            with open(saved_model_filepath, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Loaded pickled model {saved_model_filename} from "
                  f"{saved_model_filepath}.\n")
        else:
            print('Failed to load pickled model!\n'
                  'Need to train and save pickled model using '
                  'src/models/create_models.py\n')
            sys.exit()

    def get_top_features(self, model, num_features=9):
        """Gets 'num_features' top features for classifier model.

        Args:
            model: Previously trained model.
            num_features (int): Number of top features to get, in order of
            feature importance.
        
        Returns:
            top_n_feature_indicies: Numerical indexes for top features.
            top_n_feature_names: Column names for top features.
            imp_nums_std: Standardized relative feature importance numbers.
            feat_imp_argsort: Argsort to get top features from DataFrame.
        """

        self.feat_imp_argsort = np.argsort(list(model.feature_importances_)
                                                                       )[::-1]
        print(f"All sorted indicies: {self.feat_imp_argsort}")

        unordered_feature_names = self.X.columns.to_list()
        ordered_feature_names = [unordered_feature_names[idx]
                                    for idx in self.feat_imp_argsort]
        
        self.top_n_feature_indicies = self.feat_imp_argsort[:num_features]
        self.top_n_feature_names = ordered_feature_names[:num_features]

        imp_nums = model.feature_importances_
        imp_nums_sort = [imp_nums[idx] for idx in self.feat_imp_argsort][
                                                                :num_features]
        self.imp_nums_std = (imp_nums_sort / max(imp_nums_sort)) * 100

        print(f"\nTop {num_features} features:")
        print(f"Indicies: {self.top_n_feature_indicies}\n")
        pprint(self.top_n_feature_names)

    def rename_feature_names(self, data_frame):
        """Creates more readable names for features of interest.
        
        Args:
            data_frame: X DataFrame containing features and data.
        
        Returns:
            pretty_features: DataFrame containing unordered feature names,
            with select features replaced with more readable ones.
        """

        self.pretty_features = data_frame.columns.to_list()
        self.pretty_features[0] = 'Age'
        self.pretty_features[35] = 'Record of the Month satisfaction'
        self.pretty_features[90] = 'Must hear album before purchasing'
        self.pretty_features[49] = 'How often Essentials Record of the Month'
        self.pretty_features[109] = 'I do not own or lease a vehicle'
        self.pretty_features[58] = 'Curated playlists for music discovery'
        self.pretty_features[17] = 'Have dedicated vinyl listening room'
        self.pretty_features[36] = 'Add-on subscriptions satisfaction'
        self.pretty_features[60] = 'Know everything about favorite artists'
        self.pretty_features[55] = 'How often Classics listening notes booklet'
        self.pretty_features[2] = 'How many records do you own?'
        self.pretty_features[15] = 'Listen to vinyl more than stream music'
        self.pretty_features[83] = 'I buy everything my fav artists release'
        print("created custom feature names")

    def make_barplot(self, num_features=9):
        # barplot
        x_bar = [self.pretty_features[idx] for idx in 
                                         self.feat_imp_argsort][:num_features]
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        plt.barh(x_bar, self.imp_nums_std, color='#40FF40')
        plt.xticks(rotation='0', fontsize=18)
        plt.yticks(fontsize=22)
        # ax.legend(['Relative Importance'], fontsize=22, loc='lower right')
        # ax.set_facecolor('whitesmoke')
        plt.xlabel('Relative Importance', fontsize=22)
        ax.invert_yaxis()
        plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
        save_image_path = os.path.join(ROOT_IMGS_DIRECTORY,
                                              'charts/barplot.png')
        plt.savefig(save_image_path, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        print(f"barplot saved to {save_image_path}.")
        plt.show()

    def make_pdp_single(self, feat_idx_lst=[0]):
        print(f"Creating partial_dependence_pair plot.")
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        plt.title("Top Features Partial Dependence Plots", fontsize='large')
        ax.set_facecolor('whitesmoke')
        plot_partial_dependence(self.model,
                                self.X,
                                feat_idx_lst,
                                feature_names=self.pretty_features,
                                fig=fig,
                                line_kw={'c': '#40FF40', 'linewidth': 10},
                                n_jobs=-1)
        plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
        save_image_path = os.path.join(ROOT_IMGS_DIRECTORY,
                                            'charts/partial_dependence_single')
        plt.savefig(save_image_path, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        print(f"barpartial_dependence_single plot saved to {save_image_path}.")
        plt.show()

    def make_pdp_pair(self, feat_idx_lst=[0, 35]):
        print(f"Creating partial_dependence_pair plot.")
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        plt.title("Top Features Partial Dependence Plots", fontsize='large')
        ax.set_facecolor('whitesmoke')
        plot_partial_dependence(self.model,
                                self.X,
                                feat_idx_lst,
                                feature_names=self.pretty_features,
                                fig=fig,
                                line_kw={'c': '#40FF40', 'linewidth': 10},
                                n_jobs=-1)
        plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
        save_image_path = os.path.join(ROOT_IMGS_DIRECTORY,
                                              'charts/partial_dependence_pair')
        plt.savefig(save_image_path, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        print(f"barpartial_dependence_pair plot saved to {save_image_path}.")
        plt.show()


    def make_pdp_all(self):
        # Partial Dependence Plots
        print(f"Creating partial_dependence_all plot. This will take a moment.")
        fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
        plot_partial_dependence(self.model,
                                self.X,
                                self.top_n_feature_indicies,
                                feature_names=self.pretty_features,
                                fig=fig,
                                line_kw={'c': '#40FF40', 'linewidth': 8},
                                n_jobs=-1)
        # plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
        save_image_path = os.path.join(ROOT_IMGS_DIRECTORY,
                                              'charts/partial_dependence_all')
        plt.savefig(save_image_path, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        print(f"barpartial_dependence_all plot saved to {save_image_path}.")
        plt.show()


if __name__ == '__main__':
    make_plots = MakePlots()
    make_plots.get_top_features(make_plots.model, num_features=5)
    make_plots.rename_feature_names(make_plots.X)

    make_plots.make_barplot(num_features=5)
    # make_plots.make_pdp_single(feat_idx_lst=[0])
    make_plots.make_pdp_pair(feat_idx_lst=[0, 35])
    # make_plots.make_pdp_all()