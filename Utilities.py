from operator import index

import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB, GaussianNB
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.svm import SVC
# from imblearn.over_sampling import SMOTE

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import Callback

import math
import os
import time
import sys
import subprocess
import warnings
from itertools import product
from typing import List
import IPython.display as display

class Utilities():

    def __init__(self):
        pass

    def progress_bar(current_message, current, total, bar_length=20):
        fraction = current / total
        arrow = int(fraction * bar_length - 1) * '-' + '>'
        padding = int(bar_length - len(arrow)) * ' '
        ending = '\n' if current == total else '\r'
        print(f'{current_message}: [{arrow}{padding}] {int(fraction * 100)}%', end=ending)

    def check_csv_exists(folder_path, sample_index):
        # read the CSV file into a dataframe and append to the list
        filename = os.path.join(folder_path, f'df_{index}.csv')
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            return False
        return filename

    def load_dataframe(filename):
        # read the CSV file into a dataframe and append to the list
        df = pd.read_csv(filename)
        return df

    def save_dataframe_list(list_of_dfs: List[pd.DataFrame], folder_path: str, file_name: str):
        # create directoy if necessary
        os.makedirs(folder_path, exist_ok=True)
        for i, df in enumerate(list_of_dfs):
            file_path = f"{folder_path}/{file_name}_{i}.csv"
            df.to_csv(file_path, index=False)
            df.to_excel(file_path, index=False)


    def save_dataframe(df: pd.DataFrame, folder_path: str, file_name: str):
        print(f"Saving Dataframe to: {folder_path}/{file_name}.csv...", end='')
        # create directoy if necessary
        os.makedirs(folder_path, exist_ok=True)
        df.to_csv(f'{folder_path}/{file_name}.csv', index=False)
        print("Saved.")