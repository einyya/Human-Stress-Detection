from Utilities import Utilities
import pandas as pd
import io

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def openfile(ID,dataframe_path_log):
        with open(dataframe_path_log, 'r') as logfile:
            # Initialize an empty list to hold rows of data
            rows = []

            # Read each line in the log file
            for line in logfile:
                # Split the line by commas
                parts = line.strip().split(',')

                # Check if the line has the expected number of fields (4 fields)
                if len(parts) == 4:
                    rows.append(parts)
                else:
                    print(f"Ignoring line with unexpected format: {line.strip()}")

        # Create a DataFrame from the list of rows
        df = pd.DataFrame(rows, columns=['ID', 'Timestamp', 'Task', 'Status'])

        # Convert 'Timestamp' column to datetime format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Time'] = df['Timestamp'].dt.time


        filtered_df = df[df['ID'] == ID]  # Ensure '142' is treated as a string if IDs are strings
        Utilities.save_dataframe(filtered_df, dataframe_path, f'{ID}_DATA')
        return filtered_df


    ID='42'
    data_path = r'D:\PEBL2_Win_Portable_2.1.1_for test\PEBL2_Win_Portable_2.1.1_for test\PEBL2.1'
    dataframe_path = f'{data_path}\logs'
    dataframe_path_log = f'{dataframe_path}\TestLaunch-log.txt'
    task='corsi'
    data_path_battery = fr'{data_path}\battery'
    corsi_path = f'{data_path_battery}\{task}\data\{ID}\corsi-trial-{ID}.csv'
    task = 'PASAT'
    pasat_path = f'{data_path_battery}\{task}\data\{ID}\PASAT-{ID}.csv'
    task = 'twocoladd'
    twocoladd_path = fr'{data_path_battery}\{task}\data\{ID}\twocol-{ID}.csv'
    task = 'scales'
    scales_path = fr'{data_path_battery}\{task}\data\{ID}\physioscales-{ID}.csv'

    Log_df=openfile(ID,dataframe_path_log)
    corsi_df=Utilities.load_dataframe(corsi_path)
    pasat_df=Utilities.load_dataframe(pasat_path)
    twocoladd_df=Utilities.load_dataframe(twocoladd_path)
    scales_df=Utilities.load_dataframe(scales_path)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
