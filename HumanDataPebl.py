<<<<<<< HEAD
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





=======
from Utilities import Utilities
import pandas as pd
import os
import bioread
import re
import datetime


# Press the green button in the gutter to run the script.
class HumanDataPebl():
    def __init__(self,Directory):
        self.path=Directory
        self.parlist_df=pd.DataFrame()
    def structure_data(self,df):
        """
        Function to structure task data from raw dataframe.
        Handles three types of tasks: CS, PASAT, and TWOCOL.
        Returns a new dataframe with columns: Sub, Task, Start, End
        """
        # Convert the Time column to datetime format
        df["Time"] = pd.to_datetime(df["Time"])

        # Initialize the new structured list
        structured_data = []

        # Iterate through the DataFrame to match START and END times
        start_times = {}

        for _, row in df.iterrows():
            part = row["Part"]
            sub = row["Sub"]
            time = row["Time"]

            if "Start_" in part or "STARTED" in part:
                task_name = part.replace("Start_", "").replace("PASAT_STARTED ", "PASAT ").replace("TWOCOL_STARTED ", "TWOCOL ")
                start_times[(sub, task_name)] = time
            elif "End_" in part or "ENDED" in part:
                task_name = part.replace("End_", "").replace("PASAT_ENDED ", "PASAT ").replace("TWOCOL_ENDED ", "TWOCOL ")
                start_time = start_times.get((sub, task_name), None)
                if start_time:
                    structured_data.append([sub, f"{task_name}", start_time, time])
                    del start_times[(sub, task_name)]

        # Convert structured data to DataFrame
        final_df = pd.DataFrame(structured_data, columns=["Sub", "Task", "Start", "End"])
        final_df["Task"]=final_df["Task"].str.strip()
        # Merge CS_B1 and CS_B2 into CB_easy and CB_hard
        final_df.loc[(final_df["Task"] == "CS_B1 3000") | (final_df["Task"] == "CS_B2 3000"), "Task"] = "CB_easy"
        final_df.loc[(final_df["Task"] == "CS_B1 650") | (final_df["Task"] == "CS_B2 650"), "Task"] = "CB_hard"

        # Merge consecutive CB_easy and CB_hard tasks
        merged_rows = []
        previous_row = None

        for _, row in final_df.iterrows():
            if previous_row is not None and previous_row["Task"] == row["Task"] and previous_row["Sub"] == row["Sub"] and previous_row["Task"]!='Break':
                # Update the end time of the previous row
                previous_row["End"] = row["End"]
            else:
                # Add previous row to merged list before starting a new one
                if previous_row is not None:
                    merged_rows.append(previous_row)
                previous_row = row.copy()  # Start a new merging process

        # Append the last merged row
        if previous_row is not None:
            merged_rows.append(previous_row)

        # Convert back to DataFrame
        final_df = pd.DataFrame(merged_rows)
        # Merge PASAT and TWOCOL tasks into simplified names
        final_df.loc[final_df["Task"] == "TWOCOL 12000", "Task"] = "TC_easy"
        final_df.loc[final_df["Task"] == "TWOCOL 7000", "Task"] = "TC_hard"
        final_df.loc[final_df["Task"] == "PASAT 3000", "Task"] = "PA_easy"
        final_df.loc[final_df["Task"] == "PASAT 2400", "Task"] = "PA_medium"
        final_df.loc[final_df["Task"] == "PASAT 1800", "Task"] = "PA_hard"

        # Adjust start and end times accordingly
        # final_df = final_df.groupby(["Sub", "Task"]).agg({"Start": "min", "End": "max"}).reset_index()

        # Keep only the hour and minute part of the time columns
        final_df["Start"] = final_df["Start"].dt.strftime("%H:%M:%S")
        final_df["End"] = final_df["End"].dt.strftime("%H:%M:%S")

        return final_df

    def Splite_vas(self,df):
        """
        Function to structure VAS data from raw dataframe.
        Returns a dataframe with columns: Sub, Task, Start, End, Score.
        """
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Timestamp.1"] = pd.to_datetime(df["Timestamp.1"])

        vas_stress = df[["subnum", "Timestamp", "Stress"]].copy()
        vas_fatigue = df[["subnum", "Timestamp.1", "Fatigue"]].copy()


        # Rename columns
        vas_stress.columns = ["Sub", "Start", "Score"]
        vas_fatigue.columns = ["Sub", "Start", "Score"]

        # Add Task column
        vas_stress["Task"] = "VAS_Stress"
        vas_fatigue["Task"] = "VAS_Fatigue"

        # Assign End as None (VAS has only one timestamp per measurement)
        vas_stress["End"] = None
        vas_fatigue["End"] = None

        # Keep only hour and minute part of time
        vas_stress["Start"] = vas_stress["Start"].dt.strftime("%H:%M:%S")
        vas_fatigue["Start"] = vas_fatigue["Start"].dt.strftime("%H:%M:%S")
        vas_stress = vas_stress[["Sub", "Task", "Start", "End", "Score"]]
        vas_fatigue = vas_fatigue[["Sub", "Task", "Start", "End", "Score"]]
        vas_fatigue["Score"] = vas_fatigue["Score"].round(2)  # Round fatigue scores to 2 decimal places
        vas_stress["Score"] = vas_stress["Score"].round(2)  # Round fatigue scores to 2 decimal places

        return vas_stress, vas_fatigue

    def Make_Trigger_Table(self,ID,Group,rangeID):
        # -----------------------------------------------------START-----------------------------------------------------------------
        pebl_path = f'{self.path}\\PEBL2'
        parlist_path = f'{self.path}\\Participants\\participation management.csv'
        parlist_df = pd.read_csv(parlist_path, header=0)
        parlist_df = parlist_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        parlist_df['code'] = pd.to_numeric(parlist_df['code'], errors='coerce').astype('Int64')
        self.parlist_df=parlist_df
        if ID is not None:
            if rangeID:
                parlist_df = parlist_df[parlist_df['code'] >= ID]
            else:
                parlist_df = parlist_df[parlist_df['code'] == ID]
        CS_path = f'{pebl_path}\\battery\\stroop\\data\\CS-log.csv'
        PA_path = f'{pebl_path}\\battery\\PASAT\\data\\PASAT-log.csv'
        TC_path = f'{pebl_path}\\battery\\twocoladd\\data\\twocoladd-log.csv'
        VAS_path = f'{pebl_path}\\battery\\scales\\data\\VAS-log.csv'
        BR_path = f'{pebl_path}\\battery\\Break-log.csv'

        CS_df=pd.read_csv(CS_path)
        PA_df = pd.read_csv(PA_path)
        TC_df = pd.read_csv(TC_path)
        VAS_df = pd.read_csv(VAS_path)
        BR_df = pd.read_csv(BR_path)

        CS_df=self.structure_data(CS_df)
        PA_df=self.structure_data(PA_df)
        TC_df=self.structure_data(TC_df)
        BR_df=self.structure_data(BR_df)
        VAS_st,VAS_fa=self.Splite_vas(VAS_df)

        for index, row in parlist_df.iterrows():
            ID = row['code']
            print(ID)
            Group = row['Group']
            # Task summary files
            CS_ID=CS_df[CS_df["Sub"]==ID]
            TC_ID=TC_df[TC_df["Sub"]==ID]
            PA_ID=PA_df[PA_df["Sub"]==ID]
            BR_ID=BR_df[BR_df["Sub"]==ID]
            VAS_faID=VAS_fa[VAS_fa["Sub"]==ID]
            VAS_stID=VAS_st[VAS_st["Sub"]==ID]
            combined_ID = pd.concat([CS_ID, TC_ID, PA_ID,VAS_faID,VAS_stID,BR_ID], ignore_index=True)

            # -----------------------------------------------------BioPac-Start_time-----------------------------------------------------------------
            participant_path=fr'{self.path}\Participants\{Group}_group\P_{ID}'
            participant_acq=fr'{participant_path}\P_{ID}.acq'
            participant_Trigger=fr'{participant_path}\Trigger_{ID}.csv'
            BioPac = bioread.read_file(participant_acq)
            Start_time = BioPac.event_markers[00].text
            Start_time = pd.to_datetime(Start_time)
            Start_time = Start_time.time().strftime('%H:%M:%S')
            new_row = {'Sub': ID,
                       'Task': 'Biopac record',
                       'Start': Start_time,
                       'End': None}

            # Append new row to DataFrame
            combined_ID = combined_ID._append(new_row, ignore_index=True)
            combined_ID = combined_ID.sort_values(by='Start')
            biopac_time = combined_ID.loc[combined_ID["Task"] == "Biopac record", "Start"].min()
            combined_ID=combined_ID[combined_ID["Start"]>=biopac_time]
            combined_ID = combined_ID.reset_index(drop=True)
            combined_ID["Start"] = pd.to_datetime(combined_ID["Start"], format="%H:%M:%S", errors="coerce").dt.time
            combined_ID["End"] = pd.to_datetime(combined_ID["End"], format="%H:%M:%S", errors='coerce').dt.time
            combined_ID["Start"] = combined_ID["Start"].apply(lambda x: (pd.to_datetime(str(x), format="%H:%M:%S") - pd.to_datetime(str(biopac_time), format="%H:%M:%S")).total_seconds())
            combined_ID["End"] = combined_ID["End"].apply(lambda x: (pd.to_datetime(str(x), format="%H:%M:%S") - pd.to_datetime(str(biopac_time), format="%H:%M:%S")).total_seconds())
            combined_ID.to_csv(participant_Trigger, index=False)

        else:
            CS_ID = CS_df[CS_df["Sub"] == ID]
            TC_ID = TC_df[TC_df["Sub"] == ID]
            PA_ID = PA_df[PA_df["Sub"] == ID]
            BR_ID = BR_df[BR_df["Sub"] == ID]
            VAS_st = VAS_st[VAS_st["Sub"] == ID]
            VAS_fa = VAS_fa[VAS_fa["Sub"] == ID]

            combined_ID = pd.concat([CS_ID, TC_ID, PA_ID, VAS_st, VAS_fa, BR_ID], ignore_index=True)

            # Process BioPac start time
            participant_path = fr'{self.path}\Participants\{Group}_group\P_{ID}'
            participant_acq = fr'{participant_path}\P_{ID}.acq'
            participant_Trigger = fr'{participant_path}\Trigger_{ID}.csv'

            BioPac = bioread.read_file(participant_acq)
            Start_time = pd.to_datetime(BioPac.event_markers[0].text).time().strftime('%H:%M:%S')

            new_row = {'Sub': ID, 'Task': 'Biopac record', 'Start': Start_time, 'End': None}
            combined_ID = combined_ID._append(new_row, ignore_index=True)

            # Processing timestamps
            combined_ID = combined_ID.sort_values(by='Start')
            biopac_time = combined_ID.loc[combined_ID["Task"] == "Biopac record", "Start"].min()
            combined_ID = combined_ID[combined_ID["Start"] >= biopac_time].reset_index(drop=True)

            combined_ID["Start"] = pd.to_datetime(combined_ID["Start"], format="%H:%M:%S", errors="coerce").dt.time
            combined_ID["End"] = pd.to_datetime(combined_ID["End"], format="%H:%M:%S", errors="coerce").dt.time

            combined_ID["Start"] = combined_ID["Start"].apply(lambda x: (
                        pd.to_datetime(str(x), format="%H:%M:%S") - pd.to_datetime(str(biopac_time),
                                                                                   format="%H:%M:%S")).total_seconds())
            combined_ID["End"] = combined_ID["End"].apply(lambda x: (
                        pd.to_datetime(str(x), format="%H:%M:%S") - pd.to_datetime(str(biopac_time),
                                                                                   format="%H:%M:%S")).total_seconds())

            combined_ID.to_csv(participant_Trigger, index=False)
>>>>>>> ceb54f3 (HumanDataPebl)
