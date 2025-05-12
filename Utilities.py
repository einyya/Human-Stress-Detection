from operator import index
import pandas as pd
import os
from typing import List


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