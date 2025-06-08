
from Utilities import Utilities
import pandas as pd
import os
import bioread
import re
import datetime
import numpy as np

# Press the green button in the gutter to run the script.
class HumanDataPebl():
    def __init__(self,Directory):
        self.path=Directory
        self.parlist_df=pd.DataFrame()
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
            Trigger_df['Stress'] = (Trigger_S['Stress'] - Trigger_S['Stress'].mean()) / Trigger_S['Stress'].std()
            Trigger_df['Fatigue']=Trigger_F['Fatigue']
            Trigger_df['Stress'] = (Trigger_S['Stress'] - Trigger_S['Stress'].mean()) / Trigger_S['Stress'].std()
            Trigger_df['Group']=Group
            Trigger_df.at[0,'Task']='Start'
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
            Total_Trigger=pd.concat([Trigger_df,Total_Trigger])
        Total_Trigger.to_csv(SubjectiveDataset_path)
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
            combined_ID = pd.concat([CS_ID, TC_ID, PA_ID,VAS_faID,VAS_stID,BR_ID], ignore_index=True)
            if ID == 42:
                VAS_faID.loc[VAS_faID['Score'] == 'z', 'Score'] = 2.14
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
            if Group == 'control':
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

