
import pandas as pd
import bioread
import numpy as np


# Press the green button in the gutter to run the script.
class HumanDataPebl():
    def __init__(self,Directory):
        self.path=Directory
        self.parlist_df=pd.DataFrame()

    def CreateDataset_PerformanceScore(self, ID, rangeID):
        Performance_path = f'{self.path}\Participants\Dataset\Performance\performance.csv'
        pebl_path = f'{self.path}\\PEBL2'
        parlist_path = f'{self.path}\\Participants\\participation management.csv'

        # Load and clean participant list
        parlist_df = pd.read_csv(parlist_path, header=0)
        parlist_df = parlist_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        parlist_df['code'] = pd.to_numeric(parlist_df['code'], errors='coerce').astype('Int64')
        self.parlist_df = parlist_df

        # Filter participants based on ID parameter
        if ID is not None:
            if rangeID:
                parlist_df = parlist_df[parlist_df['code'] >= ID]
            else:
                parlist_df = parlist_df[parlist_df['code'] == ID]

        # Initialize list to collect dataframes
        performance_dfs = []

        for _, row in parlist_df.iterrows():
            ID = row['code']
            Group = row['Group']
            print(fr'{ID} {Group}')
            # Define file paths
            participant_acq = fr'{self.path}\Participants\{Group}_group\P_{ID}\P_{ID}.acq'
            BioPac = bioread.read_file(participant_acq)
            Start_time_raw = BioPac.event_markers[0].text

            try:
                # נסה תמיד להמיר ל־datetime דרך pd.to_datetime + to_pydatetime
                Start_time_dt = pd.to_datetime(Start_time_raw, errors='coerce')

                if pd.isnull(Start_time_dt):
                    raise ValueError(f"Start_time לא ניתן להמרה: {Start_time_raw}")

                Start_time_dt = Start_time_dt.to_pydatetime()

            except Exception as e:
                print(f"❌ שגיאה בהמרת Start_time: {Start_time_raw} ({type(Start_time_raw)})")
                raise e

            CS_path = f'{pebl_path}\\battery\\stroop\\data\\{ID}\\CS-{ID}.csv'
            PA_path = f'{pebl_path}\\battery\\PASAT\\data\\{ID}\\PASAT-{ID}.csv'
            TC_path = f'{pebl_path}\\battery\\twocoladd\\data\\{ID}\\twocol-{ID}.csv'
            try:
                CS_df = pd.read_csv(CS_path)
            except:
                CS_path = f'{pebl_path}\\battery\\stroop\\data\\{ID}\\CS--{ID}.csv'
                CS_df = pd.read_csv(CS_path)


            # Read CSV files
            PA_df = pd.read_csv(PA_path)
            TC_df = pd.read_csv(TC_path)

            # Keep only the needed columns and add metadata
            PA_df = PA_df[['rt', 'correct', 'isi', 'respond_Q_ts']].copy()
            PA_df['Task'] = 'PASAT'
            PA_df['ID'] = ID
            PA_df['Group'] = Group

            # Map ISI to difficulty levels
            PA_df['Level'] = PA_df['isi'].map({3000: 'easy', 2400: 'medium', 1800: 'hard'})

            # Convert RT from ms to seconds
            PA_df['RT'] = PA_df['rt'] / 1000

            # Reset index to ensure positional slicing by rows
            PA_df = PA_df.reset_index(drop=True)

            if ID==19:
                PA_df['respond_Q_ts'] = pd.to_datetime(
                    PA_df['respond_Q_ts'],
                    format='%a %b %d %H:%M:%S',
                    errors='coerce'
                )
            else:
                PA_df['respond_Q_ts'] = pd.to_datetime(
                    PA_df['respond_Q_ts'],
                    format='%a %b %d %H:%M:%S %Y',
                    errors='coerce'
                )

            first_level_dt = PA_df['respond_Q_ts'].iloc[0]  # נניח שזה אותו תאריך
            start_time_dt = Start_time_dt.replace(
                year=first_level_dt.year,
                month=first_level_dt.month,
                day=first_level_dt.day
            )

            # חישוב זמן יחסי מתחילת ההקלטה
            PA_df['Time'] = (PA_df['respond_Q_ts'] - start_time_dt).dt.total_seconds()

            # Keep only the necessary columns and initialize metadata
            TC_df = TC_df[['rs', 'corr', 'trial','respond_Q_ts']].copy()
            TC_df['Task'] = 'TwoColAdd'
            TC_df['ID'] = ID
            TC_df['Group'] = Group

            # Assign ISI based on index (first 10 trials are 'easy', rest are 'hard')
            TC_df['isi'] = TC_df['trial'].index.map(lambda x: 12000 if x < 10 else 7000)

            # Map ISI values to difficulty levels
            TC_df['Level'] = TC_df['isi'].map({12000: 'easy', 7000: 'hard'})

            # Convert RT from milliseconds to seconds
            TC_df['RT'] = TC_df['rs'] / 1000

            # Rename 'corr' to 'correct' for consistency
            TC_df = TC_df.rename(columns={'corr': 'correct'})

            if ID==19:
                TC_df['respond_Q_ts'] = pd.to_datetime(
                    TC_df['respond_Q_ts'],
                    format='%a %b %d %H:%M:%S',
                    errors='coerce'
                )
            else:
                TC_df['respond_Q_ts'] = pd.to_datetime(
                    TC_df['respond_Q_ts'],
                    format='%a %b %d %H:%M:%S %Y',
                    errors='coerce'
                )

            first_level_dt = TC_df['respond_Q_ts'].iloc[0]  # נניח שזה אותו תאריך
            start_time_dt = Start_time_dt.replace(
                year=first_level_dt.year,
                month=first_level_dt.month,
                day=first_level_dt.day
            )

            TC_df['Time'] = (TC_df['respond_Q_ts'] - start_time_dt).dt.total_seconds()

            CS_df = CS_df[['rt', 'correct', 'trial', 'respond_Q_ts']].copy()
            CS_df['Task'] = 'Stroop'
            CS_df['ID'] = ID
            CS_df['Group'] = Group
            CS_df['isi'] = CS_df['trial'].index.map(lambda x: 3000 if x < 30 else 650)
            CS_df['Level'] = CS_df['isi'].map({3000: 'easy', 650: 'hard'})
            CS_df['RT'] = CS_df['rt'] / 1000

            CS_df['respond_Q_ts'] = pd.to_datetime(
                CS_df['respond_Q_ts'],
                format='%a %b %d %H:%M:%S %Y',
                errors='coerce'
            )
            first_level_dt = CS_df['respond_Q_ts'].iloc[0]  # נניח שזה אותו תאריך
            start_time_dt = Start_time_dt.replace(
                year=first_level_dt.year,
                month=first_level_dt.month,
                day=first_level_dt.day
            )

            CS_df['Time'] = (CS_df['respond_Q_ts'] - start_time_dt).dt.total_seconds()

            Individual_performance=pd.concat([PA_df, TC_df, CS_df])
            Individual_performance_path = fr'{self.path}\Participants\{Group}_group\P_{ID}\Performance-{ID}.csv'
            desired_columns = ['ID', 'Group', 'Task', 'isi', 'Level', 'RT', 'correct','Time']
            remaining_columns = [col for col in Individual_performance.columns if col not in desired_columns]
            column_order = desired_columns + remaining_columns
            # Select columns that actually exist in the dataframe
            available_columns = [col for col in column_order if col in Individual_performance.columns]
            Individual_performance = Individual_performance[available_columns]
            Individual_performance.drop(columns=['trial'])
            Individual_performance['Task_level'] = (
                    Individual_performance['Task'] + ' | ' + Individual_performance['Level']
            )
            Individual_performance.to_csv(Individual_performance_path)
            RTSummary = Individual_performance.groupby(['ID','Group','Task','Level'])['RT'].agg(
                ['count', 'mean']).reset_index()
            RTSummary.to_csv(fr'{self.path}\Participants\{Group}_group\P_{ID}\RT_Summary_{ID}.csv', index=False)

            # 📊 Summary table for Accuracy by Task_Level and Group
            AccuracySummary = Individual_performance.groupby(['ID','Group','Task','Level'])['correct'].agg(
                ['count', 'mean']).reset_index()
            AccuracySummary.rename(columns={'mean': 'accuracy_mean', 'std': 'accuracy_std'}, inplace=True)
            AccuracySummary.to_csv(fr'{self.path}\Participants\{Group}_group\P_{ID}\Accuracy_Summary_{ID}.csv', index=False)

            # Add to list
            performance_dfs.append(Individual_performance)

        # Combine all dataframes
        if performance_dfs:
            Performance_df = pd.concat(performance_dfs, ignore_index=True)
            Performance_df=Performance_df.drop(columns=['trial'])
            Performance_df = Performance_df[Performance_df['Level'].notna()]

            # Reorder columns: ID, Group, test_type, isi, RT, correct, then any remaining columns
            desired_columns = ['ID', 'Group', 'Task', 'isi','Level','RT', 'correct','Time']
            remaining_columns = [col for col in Performance_df.columns if col not in desired_columns]
            column_order = desired_columns + remaining_columns

            # Select columns that actually exist in the dataframe
            available_columns = [col for col in column_order if col in Performance_df.columns]
            Performance_df = Performance_df[available_columns]
            Performance_df.to_csv(Performance_path, index=False)

            # 📊 Summary table for RT by Task_Level and Group
            RTSummary = Performance_df.groupby(['Level', 'Group'])['RT'].agg(
                ['count', 'mean', 'std']).reset_index()
            RTSummary.to_csv(fr'{self.path}\Participants\Dataset\Performance\\RT_Summary.csv', index=False)

            # 📊 Summary table for Accuracy by Task_Level and Group
            AccuracySummary = Performance_df.groupby(['Level', 'Group'])['correct'].agg(
                ['count', 'mean', 'std']).reset_index()
            AccuracySummary.rename(columns={'mean': 'accuracy_mean', 'std': 'accuracy_std'}, inplace=True)
            AccuracySummary.to_csv(fr'{self.path}\Participants\Dataset\Performance\Accuracy_Summary.csv', index=False)

            print(f"Performance data saved to: {Performance_path}")
            return Performance_df
        else:
            print("No data was processed successfully.")

    def CreateDataset_StressScore(self,ID,rangeID):
        SubjectiveDataset_path=fr'{self.path}\Participants\Dataset\Subjective\SubjectiveDataset.csv'
        participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df=pd.read_csv(participants_path)
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')
        if ID is not None:
            if rangeID:
                Participants_df = Participants_df[Participants_df['code'] >= ID]
            else:
                Participants_df = Participants_df[Participants_df['code'] == ID]
        Total_Trigger=pd.DataFrame()
        for _,row in Participants_df.iterrows():
            ID    = row['code']
            Group = row['Group']
            print(fr'{ID} {Group}')
            directory = fr'{self.path}\Participants\{Group}_group\P_{ID}'
            Trigger_df = pd.read_csv(fr'{directory}\Trigger_{ID}.csv')
            # ── Ratings table: keep only rows that actually carry stress / fatigue values ──
            # create two new columns, NaN everywhere except the correct VAS rows

            Trigger_df["Stress"] = np.where(Trigger_df["Task"] == "VAS_Stress",
                                            Trigger_df["Score"],
                                            np.nan)

            Trigger_df["Fatigue"] = np.where(Trigger_df["Task"] == "VAS_Fatigue",
                                             Trigger_df["Score"],
                                             np.nan)

            Trigger_F=Trigger_df["Fatigue"].dropna().reset_index()
            Trigger_S=Trigger_df["Stress"].dropna().reset_index()
            Trigger_df = Trigger_df[Trigger_df['Task'] != 'VAS_Stress']
            Trigger_df = Trigger_df[Trigger_df['Task'] != 'VAS_Fatigue']
            Trigger_df = Trigger_df.drop(columns=['Score','End','Start']).reset_index(drop=True)
            Trigger_df['Stress']=Trigger_S['Stress']
            start_stress=Trigger_S.iloc[0]
            normalized_stress = (Trigger_df['Stress'] - start_stress['Stress'])
            Trigger_df['Stress_S'] = normalized_stress
            normalized_stress_std = (Trigger_df['Stress'] - start_stress['Stress']) / Trigger_df['Stress'].std()
            Trigger_df['Stress_S_std'] = normalized_stress_std
            Trigger_df['Fatigue']=Trigger_F['Fatigue']
            start_fatigue=Trigger_F.iloc[0]
            normalized_fatigue = (Trigger_F['Fatigue'] - start_fatigue['Fatigue'])
            Trigger_df['Fatigue_S'] = normalized_fatigue
            normalized_fatigue_std = (Trigger_df['Fatigue'] - start_fatigue['Fatigue']) / Trigger_df['Fatigue'].std()
            Trigger_df['Fatigue_S_std'] = normalized_fatigue_std
            Trigger_df['Group']=Group
            # Define task mapping for known prefixes
            task_map = {
                'PA': 'PASAT',
                'TC': 'TwoColAdd',
                'CS': 'Stroop',
                'Break': 'Break',
                'Start': 'Start',
                'VAS': 'VAS'
            }

            # Extract prefix and level
            Trigger_df[['Prefix', 'Level']] = Trigger_df['Task'].str.extract(r'([A-Za-z]+)_?(easy|medium|hard)?')
            Trigger_df.rename(columns={'Task': 'Task_Level'}, inplace=True)
            # Map prefix to full task name
            Trigger_df['Task'] = Trigger_df['Prefix'].map(task_map).fillna(Trigger_df['Prefix'])

            # Clean up
            Trigger_df = Trigger_df.drop(columns=['Prefix'])

            Trigger_df.at[0,'Task']='Start'
            # Replace all 'Task' values that contain 'Break' with the corresponding 'Task_Level' value from the same row
            Trigger_df.loc[Trigger_df['Task'].str.contains('Break', na=False), 'Task'] = (
                Trigger_df.loc[Trigger_df['Task'].str.contains('Break', na=False), 'Task_Level']
            )
            Trigger_df['Task'] = Trigger_df['Task'].replace({
                'Music1': 'Break1',
                'Music2': 'Break2',
                'Music3': 'Break3',
                'Music4': 'Break4',
                'Breath1': 'Break1',
                'Breath2': 'Break2',
                'Breath3': 'Break3',
                'Breath4': 'Break4',
                'Baseline': 'Break1'
            })
            # --- Task Phase Simplification ---
            Trigger_df['Task_phase1'] = Trigger_df['Task'].replace(
                ['Break', 'Break1','Break2','Break3','Break4', 'Music', 'Breath'], 'Break')
            # Define the replacement dictionary for specific Task_Level values
            replace_dict = {
                'Baseline': 'Break1',
                'Biopac record': 'start',
                'Music1': 'Break1',
                'Breath1': 'Break1',
                'Music2': 'Break2',
                'Breath2': 'Break2',
                'Music3': 'Break3',
                'Breath3': 'Break3',
                'Music4': 'Break4',
                'Breath4': 'Break4',
                'Break1': 'Break1',
                'Break2': 'Break2',
                'Break3': 'Break3',
                'Break4': 'Break4',
                'start': 'start'
            }

            # Replace Task_Level values according to the dictionary
            Trigger_df['Task_phase2'] = Trigger_df['Task_Level'].replace(replace_dict)

            # Identify rows where Task_Level was not found in the replacement dictionary
            mask_missing = ~Trigger_df['Task_Level'].isin(replace_dict.keys())

            # For those rows, construct Task_phase2 by combining Task and Level
            Trigger_df.loc[mask_missing, 'Task_phase2'] = (
                    Trigger_df.loc[mask_missing, 'Task'] + ' | ' + Trigger_df.loc[mask_missing, 'Level']
            )
            Total_Trigger=pd.concat([Trigger_df,Total_Trigger])
        Total_Trigger.to_csv(SubjectiveDataset_path)
        StressSummary = Total_Trigger.groupby(['Task', 'Group'])['Stress'].agg(['count', 'mean', 'std']).reset_index()
        StressSummary.to_csv(fr'{self.path}\Participants\Dataset\Subjective\Stress_Summary.csv')
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
            Participants_path = f'{self.path}\\Participants\\participation management.csv'
            Participants_df = (
                pd.read_csv(Participants_path)
                .dropna(axis=1, how='all')
                .dropna(subset=['participant', 'Date', 'departmant'], how='all')
            )
            Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')
            Participants_df = Participants_df[Participants_df['code'] == sub]
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
        # Merge CS_B1 and CS_B2 into CS_easy and CS_hard
        final_df.loc[(final_df["Task"] == "CS_B1 3000") | (final_df["Task"] == "CS_B2 3000"), "Task"] = "CS_easy"
        final_df.loc[(final_df["Task"] == "CS_B1 650") | (final_df["Task"] == "CS_B2 650"), "Task"] = "CS_hard"

        # Merge consecutive CS_easy and CS_hard tasks
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

    def Make_Trigger_Table(self,ID,rangeID):
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
            if ID == 42:
                VAS_faID.loc[VAS_faID['Score'] == 'z', 'Score'] = 2.14
            combined_ID = pd.concat([CS_ID, TC_ID, PA_ID,VAS_faID,VAS_stID,BR_ID], ignore_index=True)

            # -----------------------------------------------------BioPac-Start_time-----------------------------------------------------------------
            participant_path=fr'{self.path}\Participants\{Group}_group\P_{ID}'
            performance_Accuracy_path=fr'{participant_path}\Accuracy_Summary_{ID}.csv'
            performance_RT_path=fr'{participant_path}\RT_Summary_{ID}.csv'
            performance_Accuracy=pd.read_csv(performance_Accuracy_path)
            performance_RT=pd.read_csv(performance_RT_path)
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
            if Group == 'natural':
                if ID==47:
                    combined_ID.loc[2, 'Task'] = 'Break1'
                    combined_ID.loc[12, 'Task'] = 'Break2'
                    combined_ID.loc[23, 'Task'] = 'Break3'
                    combined_ID.loc[33, 'Task'] = 'Break4'
                elif ID == 32:
                    combined_ID.loc[3, 'Task'] = 'Break1'
                    combined_ID.loc[12, 'Task'] = 'Break2'
                    combined_ID.loc[23, 'Task'] = 'Break3'
                    combined_ID.loc[33, 'Task'] = 'Break4'
                elif ID == 34:
                    combined_ID.loc[3, 'Task'] = 'Break1'
                    combined_ID.loc[12, 'Task'] = 'Break2'
                    combined_ID.loc[23, 'Task'] = 'Break3'
                    combined_ID.loc[32, 'Task'] = 'Break4'
                else:
                    combined_ID.loc[3, 'Task'] = 'Break1'
                    combined_ID.loc[12, 'Task'] = 'Break2'
                    combined_ID.loc[24, 'Task'] = 'Break3'
                    combined_ID.loc[33, 'Task'] = 'Break4'
            # Corrected version - assign Score values based on conditions
            combined_ID.loc[combined_ID['Task'] == 'CS_easy', 'Accuracy_Mean'] = \
                performance_Accuracy.loc[(performance_Accuracy['Task'] == 'Stroop') &
                                         (performance_Accuracy['Level'] == 'easy'), 'accuracy_mean'].values
            combined_ID.loc[combined_ID['Task'] == 'CS_easy', 'RT_Mean'] = \
                performance_RT.loc[(performance_RT['Task'] == 'Stroop') &
                                         (performance_RT['Level'] == 'easy'), 'mean'].values
            combined_ID.loc[combined_ID['Task'] == 'CS_hard', 'Accuracy_Mean'] = \
                performance_Accuracy.loc[(performance_Accuracy['Task'] == 'Stroop') &
                                         (performance_Accuracy['Level'] == 'hard'), 'accuracy_mean'].values
            combined_ID.loc[combined_ID['Task'] == 'CS_hard', 'RT_Mean'] = \
                performance_RT.loc[(performance_RT['Task'] == 'Stroop') &
                                         (performance_RT['Level'] == 'hard'), 'mean'].values
            combined_ID.loc[combined_ID['Task'] == 'PA_easy', 'Accuracy_Mean'] = \
                performance_Accuracy.loc[(performance_Accuracy['Task'] == 'PASAT') &
                                         (performance_Accuracy['Level'] == 'easy'), 'accuracy_mean'].values
            combined_ID.loc[combined_ID['Task'] == 'PA_easy', 'RT_Mean'] = \
                performance_RT.loc[(performance_RT['Task'] == 'PASAT') &
                                         (performance_RT['Level'] == 'easy'), 'mean'].values
            combined_ID.loc[combined_ID['Task'] == 'PA_medium', 'Accuracy_Mean'] = \
                performance_Accuracy.loc[(performance_Accuracy['Task'] == 'PASAT') &
                                         (performance_Accuracy['Level'] == 'medium'), 'accuracy_mean'].values
            combined_ID.loc[combined_ID['Task'] == 'PA_medium', 'RT_Mean'] = \
                performance_RT.loc[(performance_RT['Task'] == 'PASAT') &
                                         (performance_RT['Level'] == 'medium'), 'mean'].values
            combined_ID.loc[combined_ID['Task'] == 'PA_hard', 'Accuracy_Mean'] = \
                performance_Accuracy.loc[(performance_Accuracy['Task'] == 'PASAT') &
                                         (performance_Accuracy['Level'] == 'hard'), 'accuracy_mean'].values
            combined_ID.loc[combined_ID['Task'] == 'PA_hard', 'RT_Mean'] = \
                performance_RT.loc[(performance_RT['Task'] == 'PASAT') &
                                         (performance_RT['Level'] == 'hard'), 'mean'].values
            combined_ID.loc[combined_ID['Task'] == 'TC_easy', 'Accuracy_Mean'] = \
                performance_Accuracy.loc[(performance_Accuracy['Task'] == 'TwoColAdd') &
                                         (performance_Accuracy['Level'] == 'easy'), 'accuracy_mean'].values
            combined_ID.loc[combined_ID['Task'] == 'TC_easy', 'RT_Mean'] = \
                performance_RT.loc[(performance_RT['Task'] == 'TwoColAdd') &
                                         (performance_RT['Level'] == 'easy'), 'mean'].values
            combined_ID.loc[combined_ID['Task'] == 'TC_hard', 'Accuracy_Mean'] = \
                performance_Accuracy.loc[(performance_Accuracy['Task'] == 'TwoColAdd') &
                                         (performance_Accuracy['Level'] == 'hard'), 'accuracy_mean'].values
            combined_ID.loc[combined_ID['Task'] == 'TC_hard', 'RT_Mean'] = \
                performance_RT.loc[(performance_RT['Task'] == 'TwoColAdd') &
                                         (performance_RT['Level'] == 'hard'), 'mean'].values
            combined_ID['Score'] = pd.to_numeric(combined_ID['Score'], errors='coerce')
            combined_ID['Score_Norm'] = np.nan
            combined_ID['Score_Delta'] = np.nan

            for vas_task in ['VAS_Stress', 'VAS_Fatigue']:
                m = combined_ID['Task'] == vas_task
                s = combined_ID.loc[m, 'Score']

                first = s.dropna().iloc[0] if not s.dropna().empty else np.nan
                if pd.notna(first) and first != 0:
                    combined_ID.loc[m, 'Score_Norm'] = (s - first).round(2)
                else:
                    combined_ID.loc[m, 'Score_Norm'] = np.nan

                combined_ID.loc[m, 'Score_Delta'] = s.diff().round(2)
            combined_ID.to_csv(participant_Trigger, index=False)

        # else:
        #     CS_ID = CS_df[CS_df["Sub"] == ID]
        #     TC_ID = TC_df[TC_df["Sub"] == ID]
        #     PA_ID = PA_df[PA_df["Sub"] == ID]
        #     BR_ID = BR_df[BR_df["Sub"] == ID]
        #     VAS_st = VAS_st[VAS_st["Sub"] == ID]
        #     VAS_fa = VAS_fa[VAS_fa["Sub"] == ID]
        #
        #     combined_ID = pd.concat([CS_ID, TC_ID, PA_ID, VAS_st, VAS_fa, BR_ID], ignore_index=True)
        #
        #     # Process BioPac start time
        #     participant_path = fr'{self.path}\Participants\{Group}_group\P_{ID}'
        #     participant_acq = fr'{participant_path}\P_{ID}.acq'
        #     participant_Trigger = fr'{participant_path}\Trigger_{ID}.csv'
        #
        #     BioPac = bioread.read_file(participant_acq)
        #     Start_time = pd.to_datetime(BioPac.event_markers[0].text).time().strftime('%H:%M:%S')
        #
        #     new_row = {'Sub': ID, 'Task': 'Biopac record', 'Start': Start_time, 'End': None}
        #     combined_ID = combined_ID._append(new_row, ignore_index=True)
        #
        #     # Processing timestamps
        #     combined_ID = combined_ID.sort_values(by='Start')
        #     biopac_time = combined_ID.loc[combined_ID["Task"] == "Biopac record", "Start"].min()
        #     combined_ID = combined_ID[combined_ID["Start"] >= biopac_time].reset_index(drop=True)
        #
        #     combined_ID["Start"] = pd.to_datetime(combined_ID["Start"], format="%H:%M:%S", errors="coerce").dt.time
        #     combined_ID["End"] = pd.to_datetime(combined_ID["End"], format="%H:%M:%S", errors="coerce").dt.time
        #
        #     combined_ID["Start"] = combined_ID["Start"].apply(lambda x: (
        #                 pd.to_datetime(str(x), format="%H:%M:%S") - pd.to_datetime(str(biopac_time),
        #                                                                            format="%H:%M:%S")).total_seconds())
        #     combined_ID["End"] = combined_ID["End"].apply(lambda x: (
        #                 pd.to_datetime(str(x), format="%H:%M:%S") - pd.to_datetime(str(biopac_time),
        #                                                                            format="%H:%M:%S")).total_seconds())
        #     combined_ID.to_csv(participant_Trigger, index=False)

