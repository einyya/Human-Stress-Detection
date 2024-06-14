from Utilities import Utilities
import pandas as pd
import io

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def concatDf(type,df_filtered, Log_df):
        for idx, row in corsi_filtered.iterrows():
            # Find the insertion point
            insert_idx = Log_df[Log_df['Timestamp'] > row['Timestamp']].index.min()

            # If no row found with higher time, append at the end
            if pd.isna(insert_idx):
                insert_idx = len(Log_df)

            # Create a new row to insert
            if type=='scales':
                new_row = {
                    'ID': row['subnum'],
                    'Timestamp': row['Timestamp'],
                    'Time': row['Time'],
                    'Task': row['corsi'],
                    'Status': 'mistake'

                }
            elif type=='corsi':
                new_row = {
                    'ID': row['subnum'],
                    'Timestamp': row['Timestamp'],
                    'Time': row['time'],
                    'Task': 'corsi',
                    'Status': 'mistake'
                }
            elif type=='pasat':
                new_row = {
                    'ID': row['subnum'],
                    'Timestamp': row['Timestamp'],
                    'Time': row['Time'],
                    'Task': row['corsi'],
                    'Status': 'mistake'
                }
            elif type=='twocoladd':
                new_row = {
                    'ID': row['subnum'],
                    'Timestamp': row['Timestamp'],
                    'Time': row['Time'],
                    'Task': row['corsi'],
                    'Status': 'mistake'
                }

            # Insert the new row into Log_df
            Log_df = pd.concat(
                [Log_df.iloc[:insert_idx], pd.DataFrame([new_row]), Log_df.iloc[insert_idx:]]).reset_index(
                drop=True)
        return Log_df
    def openfile(ID,dataframe_path_log):
        with open(dataframe_path_log, 'r') as logfile:
            # Initialize an empty list to hold rows of data
            rows = []

            # Read each line in the log file
            for line in logfile:
                # Split the line by commas
                parts = line.strip().split(',')

                # Check if the line has the expected number of fields (4 fields)
                if len(parts) == 5:
                    rows.append(parts)
                else:
                    print(f"Ignoring line with unexpected format: {line.strip()}")

        # Create a DataFrame from the list of rows
        df = pd.DataFrame(rows, columns=['ID', 'Timestamp', 'Time', 'Task', 'Status'])

        # Convert 'Timestamp' column to datetime format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Extract time component as a string
        df['Timestamp'] = df['Timestamp'].dt.time.apply(lambda x: x.strftime('%H:%M:%S'))


        filtered_df = df[df['ID'] == ID]  # Ensure '142' is treated as a string if IDs are strings
        filtered_df = filtered_df.reset_index(drop=True)
        return filtered_df


    ID='2'
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

    scales_df = pd.read_csv(scales_path, header=None)
    scales_df.columns = ['ID', 'Timestamp', 'Value']
    corsi_df=Utilities.load_dataframe(corsi_path)
    pasat_df=Utilities.load_dataframe(pasat_path)
    twocoladd_df=Utilities.load_dataframe(twocoladd_path)


    corsi_df['Timestamp'] = pd.to_datetime(corsi_df['Timestamp']).dt.time
    scales_df['Timestamp'] = pd.to_datetime(scales_df['Timestamp']).dt.time

    corsi_filtered = corsi_df[corsi_df['allcorr'] == 0]

    # Convert Log_df 'Time' column to datetime.time type
    Log_df['Timestamp'] = pd.to_datetime(Log_df['Timestamp']).dt.time

    # Insert rows from corsi_filtered into Log_df
    Log_df=concatDf('corsi',corsi_filtered,Log_df)
    concatDf('scales',scales_df,Log_df)
    concatDf('pasat',pasat_df,Log_df)
    concatDf('twocoladd',twocoladd_df,Log_df)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
