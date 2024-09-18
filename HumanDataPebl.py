from Utilities import Utilities
import pandas as pd
import os
import bioread
import re
import datetime


# Press the green button in the gutter to run the script.
class HumanDataPebl():
    def __init__(self,Directory):
        self.path = Directory
    def concatDf(self,type,df_filtered, Log_df):
        for idx, row in df_filtered.iterrows():
            # Find the insertion point
            Log_df['Timestamp'] = pd.to_datetime(Log_df['Timestamp'], format='%H:%M:%S').dt.time
            row['Timestamp'] = pd.to_datetime(row['Timestamp'], format='%H:%M:%S').time()
            insert_idx = Log_df[Log_df['Timestamp'] > row['Timestamp']].index.min()

            # If no row found with higher time, append at the end
            if pd.isna(insert_idx):
                insert_idx = len(Log_df)

            # Create a new row to insert
            if type=='scales':
                new_row = {
                    'ID': row['subnum'],
                    'Timestamp': row['Timestamp'],
                    'Task': 'Stress Report',
                    'Status': round(row['value'], 1)

                }
            elif type=='corsi':
                new_row = {
                    'ID': row['subnum'],
                    'Timestamp': row['Timestamp'],
                    'Task': 'corsi',
                    'Status': 'mistake'
                }
            elif type=='pasat':
                new_row = {
                    'ID': row['subnum'],
                    'Timestamp': row['Timestamp'],
                    'Task': 'pasat',
                    'Status': 'mistake'
                }
            elif type=='twocoladd':
                new_row = {
                    'ID': row['sub'],
                    'Timestamp': row['Timestamp'],
                    'Task': 'twocoladd',
                    'Status': 'mistake'
                }

            # Insert the new row into Log_df
            Log_df = pd.concat(
                [Log_df.iloc[:insert_idx], pd.DataFrame([new_row]), Log_df.iloc[insert_idx:]]).reset_index(
                drop=True)
        return Log_df

    def openfile(self, ID, dataframe_path_log):
        with open(dataframe_path_log, 'r') as logfile:
            # Initialize an empty list to hold rows of data
            rows = []

            # Read each line in the log file
            for line in logfile:
                # Split the line by commas
                parts = line.strip().split(',')

                # Ensure the line has the correct number of fields
                if len(parts) == 5:
                    # If there are 5 fields, remove the third one (assumed to be erroneous)
                    parts.pop(2)
                if len(parts) == 4:
                    # Append the cleaned up line to the rows
                    rows.append(parts)

        df = pd.DataFrame(rows, columns=['ID', 'Timestamp', 'Task', 'Status'])

        # Convert 'Timestamp' column to datetime format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Extract time component as a string
        df = df.dropna(subset=['Timestamp'])
        df['Timestamp'] = df['Timestamp'].dt.time.apply(lambda x: x.strftime('%H:%M:%S'))
        df['ID'] = df['ID'].str.strip().str.replace('"', '')
        df['ID'] = df['ID'].astype(int)  # or .astype(str) depending on your needs
        df['Status'] = df['Status'].str.strip().str.replace('"', '')
        filtered_df = df[df['ID'] == ID]
        # filtered_df = df[df['Task'] != 'physioscales.pbl']
        filtered_df = filtered_df.reset_index(drop=True)
        return filtered_df

    def Make_Trigger_Table(self):

        Insert_Mistake=False
        Insert_Grade=False
        # -----------------------------------------------------START-----------------------------------------------------------------
        data_path=f'{self.path}\PEBL2'
        Test_Type_path = f'{self.path}\Participants\participation management.xlsx'
        Type_df = pd.read_excel(Test_Type_path, header=1)
        Type_df = Type_df.dropna(axis=1, how='all')
        Type_df['code'] = pd.to_numeric(Type_df['code'], errors='coerce').astype('Int64')
        for index, row in Type_df.iterrows():
            # ID = row['code']
            # Group = row['Group']
            ID = 8
            Group = 'breath'
            print(ID)
            dataframe_path_log = f'{data_path}\logs\TestLaunch-log.txt'
            task='corsi'
            data_path_battery = fr'{data_path}\battery'
            corsi_path = f'{data_path_battery}\{task}\data\{ID}\corsi-trial-{ID}.csv'
            task = 'PASAT'
            pasat_path = f'{data_path_battery}\{task}\data\{ID}\PASAT-{ID}.csv'
            task = 'twocoladd'
            twocoladd_path = fr'{data_path_battery}\{task}\data\{ID}\twocol-{ID}.csv'
            task = 'scales'
            scales_path = fr'{data_path_battery}\{task}\data\{ID}\physioscales-{ID}.csv'
            Log_df=self.openfile(ID,dataframe_path_log)
            scales_df = pd.read_csv(scales_path, header=0)


            # -----------------------------------------------------Add to Log-Scales-----------------------------------------------------------------
            scales_df['Timestamp'] = pd.to_datetime(scales_df['Timestamp']).dt.time

            if Insert_Grade or Insert_Mistake:
                corsi_df = Utilities.load_dataframe(corsi_path)
                pasat_df = Utilities.load_dataframe(pasat_path)
                try:
                    twocoladd_df = pd.read_csv(twocoladd_path, delimiter=',', on_bad_lines='skip')
                    print("CSV file loaded successfully.")
                except pd.errors.ParserError as e:
                    print(f"ParserError: {e}")
                corsi_df['Timestamp'] = pd.to_datetime(corsi_df['Timestamp']).dt.time
                pasat_df['Timestamp'] = pd.to_datetime(pasat_df['Timestamp']).dt.time
                twocoladd_df['Timestamp'] = pd.to_datetime(twocoladd_df['Timestamp']).dt.time
            if Insert_Grade:
                corsi_filtered = corsi_df[corsi_df['allcorr'] == 0]

            if Insert_Mistake:
                corsi_filtered = corsi_df[corsi_df['allcorr'] == 0]
                pasat_filtered = pasat_df[pasat_df['correct'] == 0]
                twocoladd_filtered = twocoladd_df[twocoladd_df['corr'] == 0]
                Log_df=self.concatDf('corsi',corsi_filtered,Log_df)
                Log_df=self.concatDf('pasat',pasat_filtered,Log_df)
                Log_df=self.concatDf('twocoladd',twocoladd_df,Log_df)

            Log_df=self.concatDf('scales',scales_df,Log_df)

            # -----------------------------------------------------BioPac-Start_time-----------------------------------------------------------------
            participant_path=fr'{self.path}\Participants\{Group}_group\P_{ID}\P_{ID}.acq'
            BioPac = bioread.read_file(participant_path)
            Start_time = BioPac.event_markers[00].text
            Start_time = pd.to_datetime(Start_time)
            Start_time = Start_time.time().strftime('%H:%M:%S')
            new_row = {'ID': ID,
                       'Timestamp': Start_time,
                       'Task': 'Biopac record',
                       'Status': 'STARTED'}

            # Append new row to DataFrame
            Log_df = Log_df._append(new_row, ignore_index=True)
            Log_df['Timestamp'] = Log_df['Timestamp'].astype(str)
            # Sort DataFrame by 'Timestamp' column (as strings)
            Log_df = Log_df.sort_values(by='Timestamp')

            Log_df['Timestamp'] = pd.to_datetime(Log_df['Timestamp'], format='%H:%M:%S')
            Start_time = Log_df.loc[Log_df['Task'] == 'Biopac record']['Timestamp'].iloc[0]
            Start_time = pd.to_datetime(Start_time, format='%H:%M:%S')
            # To remove all rows with 'FINISHED' in the "Status" column except for the last occurrence of 'FINISHED'.
            mask = (Log_df['Status'] != 'FINISHED') | (Log_df['Status'] == 'FINISHED') & (Log_df.index == Log_df[Log_df['Status'] == 'FINISHED'].index[-1])
            Log_df = Log_df[mask]
            # Convert the time difference to 100 Hz units
            Log_df['Time'] = Log_df['Timestamp'].apply(lambda x: (x - Start_time).total_seconds())
            Log_df['Timestamp'] = pd.to_datetime(Log_df['Timestamp'], format='%H:%M:%S').dt.time
            Log_df.rename(columns={'Time': 'Time-sec'}, inplace=True)
            if Log_df.loc[Log_df['Task'] == 'Baseline.pbl','Time-sec'].values[0] < Log_df.loc[Log_df['Task'] == 'Biopac record','Time-sec'].values[0]:
                Log_df.loc[Log_df['Task'] == 'Baseline.pbl', 'Time-sec'] = 0

            # Optionally, reset the index if you want a clean index after dropping the row
            Log_df = Log_df.reset_index(drop=True)
            # Convert 'Status' to a numeric type if it's not already
            Log_df['Status'] = pd.to_numeric(Log_df['Status'], errors='coerce')
            # Sort the DataFrame by 'Time-sec' ascending and 'Status' descending
            Log_df = Log_df.sort_values(by=['Time-sec', 'Status'], ascending=[True, False])
            # Reset the index if you want a clean, consecutive index
            Log_df['Time-sec'] = Log_df['Time-sec'].astype(int)
            Log_df = Log_df.reset_index(drop=True)
            directory = fr'{self.path}\Participants\{Group}_group\P_{ID}'
            Log_path = fr'{directory}\Log_{ID}.csv'
            Triger_path = fr'{directory}\Triger_{ID}.csv'
            os.makedirs(directory, exist_ok=True)
            try:
                Log_df.to_csv(Log_path, index=False)
                print(f"CSV file saved successfully.")
            except Exception as e:
                print(f"Error saving CSV file: {e}")
            # # Drop the row where 'Task' is 'Biopac record'
            Log_df = Log_df[Log_df['Task'] != 'Biopac record']
            # Save the dataframe to the specified path
            # --------------------------------------------make time intervals-------------------------------------
            tasks = []
            time_starts = []
            time_ends = []
            stress_reports = []
            final_df = Log_df
            # Iterate over the final_df to reconstruct the original format
            i = 0
            while i < len(final_df):
                if final_df.iloc[i]['Task'] == 'Stress Report':
                    stress_reports.append(final_df.iloc[i]['Status'])
                    i += 1  # Move to the next task after the stress report
                else:
                    tasks.append(final_df.iloc[i]['Task'])
                    time_starts.append(final_df.iloc[i]['Time-sec'])
                    time_ends.append(final_df.iloc[i + 1]['Time-sec'] if i < len(final_df) else final_df.iloc[i]['Time-sec'])
                    if final_df.iloc[i+1]['Task'] == 'physioscales.pbl' and i < len(final_df):
                        i += 2  # Move to the next task
                    else:
                        i += 1  # Move to the next task
                        if i+2 < len(final_df):
                            stress_reports.append(final_df.iloc[i+2]['Status'])
                        else:
                            try:
                                stress_reports.append(final_df.iloc[i + 1]['Status'])
                            except Exception as e:
                                stress_reports.append('nane')
                            break

            # Create the DataFrame in the original format
            Trigger_df = pd.DataFrame({
                'Task': tasks,
                'Time-start-sec': time_starts,
                'Time-end-sec': time_ends,
                'Stress Report': stress_reports
            })
            try:
                Trigger_df.to_csv(Triger_path, index=False)
                print(f"CSV file saved successfully.")
            except Exception as e:
                print(f"Error saving CSV file: {e}")





