# import pandas as pd
# import os
# import re
# import pandas as pd
# from ML_Utilities import ML_Utilities
# from FeatureExtraction import FeatureExtraction
# from PreProcessing import PreProcessing
# from Traditional_ML import Traditional_ML
# from imblearn.over_sampling import SMOTE
# from Utilities import Utilities
# from FeatureSelection import FeatureSelection
# from FEParameter import FEParameter
# from sklearn.svm import SVC
# from sklearn.feature_selection import SequentialFeatureSelector
# from sklearn import tree
# import ast
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
# import matplotlib.pyplot as plt
from HumanDataExtraction import HumanDataExtraction
from HumanDataPebl import HumanDataPebl
from AnalysisData import AnalysisData

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DataPath = r'D:\Human Bio Signals Analysis'
    Make_Trigger=False
    Make_DataSet=True
    Analysis_DataSet=True

    #________________________________________________________Make Trigger Table_______________________________________________________
    if Make_Trigger:
        hdp = HumanDataPebl(DataPath)
        hdp.Make_Trigger_Table()
    #________________________________________________________SortData_______________________________________________________
    if Make_DataSet:
        hde = HumanDataExtraction(DataPath)
        hde.Make_DataSet()
    #________________________________________________________SortData_______________________________________________________
    if Analysis_DataSet:
        ad = AnalysisData(DataPath)
        ad.Linear_Mixed_Effects_Models()
        ad.Analysis_per_particitenpt()
        # ad.Analysis_all_particitenpts()

    # # Utilities.save_dataframe(hde.sorted_DATA, dataframe_path, f'{Analysis_Type}_Sorted_DATA')






