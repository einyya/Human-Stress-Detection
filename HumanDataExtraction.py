import matplotlib.pyplot as plt
import bioread
import neurokit2 as nk
from scipy.signal import spectrogram
import matplotlib.cm as cm
import numpy as np
from scipy.signal import medfilt
import seaborn as sns
from tqdm import tqdm
import os
from scipy.stats import linregress
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
import logging
import sys
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
# Press the green button in the gutter to run the script.
class HumanDataExtraction():

    def __init__(self,Directory):
        self.path = Directory
        self.sorted_DATA = pd.DataFrame()

    def Check_MinMaxVlaue(self, ID, rangeID):
        Participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')

        if ID is not None:
            if rangeID:
                Participants_df = Participants_df[Participants_df['code'] >= ID]
            else:
                Participants_df = Participants_df[Participants_df['code'] == ID]

        eda_min = np.inf
        eda_max = -np.inf
        min_info = {}
        max_info = {}

        for _, row in Participants_df.iterrows():
            ID=row['code']
            group = row['group'] if 'group' in row else row['Group']  # Adjust if needed
            eda_path = os.path.join(self.path, "Participants", group + '_group', f"P_{ID}", "EDA.csv")

            if os.path.exists(eda_path):
                try:
                    eda_data = pd.read_csv(eda_path)

                    if 'EDA' in eda_data.columns:
                        eda_values = eda_data['EDA'].values
                        current_min = np.nanmin(eda_values)
                        current_max = np.nanmax(eda_values)
                    else:
                        print(f"'EDA' column not found in {eda_path}")
                        continue

                    if current_min < eda_min:
                        eda_min = current_min
                        min_info = {
                            "value": eda_min,
                            "participant": ID,
                            "group": group,
                            "file": eda_path
                        }

                    if current_max > eda_max:
                        eda_max = current_max
                        max_info = {
                            "value": eda_max,
                            "participant": ID,
                            "group": group,
                            "file": eda_path
                        }

                except Exception as e:
                    print(f"Error reading {eda_path}: {e}")
            else:
                print(f"Missing EDA file for P_{ID} in group {group}_group")

        print(f"\nGlobal EDA Min Value: {min_info.get('value', 'N/A')}")
        print(
            f"Found in: Group = {min_info.get('group', 'N/A')}_group, Participant = {min_info.get('participant', 'N/A')}")
        print(f"File: {min_info.get('file', 'N/A')}")

        print(f"\nGlobal EDA Max Value: {max_info.get('value', 'N/A')}")
        print(
            f"Found in: Group = {max_info.get('group', 'N/A')}_group, Participant = {max_info.get('participant', 'N/A')}")
        print(f"File: {max_info.get('file', 'N/A')}")

    def Check_MedianFilter(self,ID,rangeID):
        Participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')
        if ID is not None:
            if rangeID:
                Participants_df = Participants_df[Participants_df['code'] >= ID]
            else:
                Participants_df = Participants_df[Participants_df['code'] == ID]
        num_participants = len(Participants_df)

        medfilt_window = [3,5,11,101]
        for window in medfilt_window:
            fig, ax = plt.subplots(nrows=num_participants, figsize=(15, 4 * num_participants), squeeze=False)
            ax = ax.flatten()  # להפוך לרשימה חד־ממדית גם אם רק משתתף אחד
            for idx, (i, row) in enumerate(Participants_df.iterrows()):
                ID = row['code']
                Group = row['Group']
                directory = fr'{self.path}\Participants\{Group}_group\P_{ID}'
                BioPac_path = fr'{directory}\P_{ID}.acq'
                BioPac = bioread.read_file(BioPac_path)

                eda_data = BioPac.named_channels['EDA'].data
                eda_time = BioPac.named_channels['EDA'].time_index
                EDA_Clean = medfilt(eda_data, kernel_size=window)

                ax[i].plot(eda_time, eda_data, label=f'Participant {ID} Raw')
                ax[i].plot(eda_time, EDA_Clean, label=f'Participant {ID} Filtered')
                ax[i].set_title(f'Participant {ID} - EDA Signal')
                ax[i].legend()
                ax[i].set_xlabel('Time (s)')
                ax[i].set_ylabel('EDA (µS)')

            save_path = fr'D:\Human Bio Signals Analysis\Preprocessing\MedianFilter\EDA_data_{window}.png'
            plt.savefig(save_path)
    def CleanData(self,ID,rangeID,plot=False):
        Participants_path = f'{self.path}\Participants\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')
        medfilt_window=101
        if ID is not None:
            if rangeID:
                Participants_df = Participants_df[Participants_df['code'] >= ID]
            else:
                Participants_df = Participants_df[Participants_df['code'] == ID]

        for j, row in Participants_df.iterrows():
            ID = row['code']
            Group = row['Group']
            print(ID)
            directory = fr'{self.path}\Participants\{Group}_group\P_{ID}'
            BioPac_path = fr'{directory}\P_{ID}.acq'
            BioPac = bioread.read_file(BioPac_path)
            sampling_rate = BioPac.named_channels['ECG'].samples_per_second
            # __________________________________________EDA_______________________________________________
            eda_data= BioPac.named_channels['EDA'].data
            eda_time = BioPac.named_channels['EDA'].time_index
            eda_data = medfilt(eda_data, kernel_size=medfilt_window)
            eda_signals, info_eda = nk.eda_process(eda_data, sampling_rate=sampling_rate)
            EDA_data = pd.DataFrame({
                'Time': eda_time,
                'EDA_Clean': eda_signals['EDA_Clean'],
                'EDA_Phasic': eda_signals['EDA_Phasic'],
                'EDA_Tonic': eda_signals['EDA_Tonic'],
                'EDA_SCR_Amplitude': eda_signals['SCR_Amplitude'],
                'EDA_SCR_Height': eda_signals['SCR_Height'],
                'EDA_SCR_Onsets': eda_signals['SCR_Onsets'],
                'EDA_SCR_Peaks': eda_signals['SCR_Peaks'],
                'EDA_SCR_Recovery': eda_signals['SCR_Recovery'],
                'EDA_SCR_RecoveryTime': eda_signals['SCR_RecoveryTime'],
                'EDA_SCR_RiseTime': eda_signals['SCR_RiseTime']
            })
            EDA_data.to_csv(fr'{directory}\EDA.csv')

            # __________________________________________ECG_______________________________________________
            ecg_data = BioPac.named_channels['ECG'].data
            ecg_time_s = BioPac.named_channels['ECG'].time_index
            ecg_cleaned = nk.ecg_clean(ecg_data, sampling_rate=sampling_rate)
            r_peaks, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
            ecg_signals, info_ecg = nk.ecg_process(ecg_cleaned, sampling_rate)
            hrv_rri = np.diff(info_ecg["ECG_R_Peaks"]) / sampling_rate * 1000
            hrv_rri = np.insert(hrv_rri, 0, 0)
            time_rri = info_ecg["ECG_R_Peaks"] / sampling_rate
            HRV_F = pd.DataFrame({
                'Time': ecg_time_s,
                'ECG':ecg_cleaned,
                'ECG_R_Peaks':r_peaks['ECG_R_Peaks']
            })
            HRV_F.to_csv(fr'{directory}\HRV_F.csv')
            HRV_RR = pd.DataFrame({
                'Time': time_rri,
                'HRV_RR': hrv_rri,
                'HRV_Peaks':info_ecg["ECG_R_Peaks"]
            })
            first_row = HRV_RR.iloc[[0]]
            RR_rest = HRV_RR.iloc[1:]
            RR_rest = RR_rest[~((RR_rest['HRV_RR'] < 300) | (RR_rest['HRV_RR'] > 1500))]
            HRV_RR = pd.concat([first_row, RR_rest], ignore_index=True)
            HRV_RR.to_csv(fr'{directory}\HRV_RR.csv')
            # __________________________________________RSP_C_______________________________________________
            rsp_c_data = BioPac.named_channels['Chest Respiration'].data
            rsp_c_time = BioPac.named_channels['Chest Respiration'].time_index
            rsp_c_data_cleaned = nk.rsp_clean(rsp_c_data, sampling_rate=sampling_rate)
            rsp_c_signal, info_rsp_c = nk.rsp_process(rsp_c_data_cleaned, sampling_rate=sampling_rate)
            rsp_c_rate=nk.signal_rate(peaks=info_rsp_c["RSP_Peaks"])
            rsp_c_peaks=np.diff(info_rsp_c["RSP_Peaks"]) / sampling_rate * 1000
            rsp_c_peaks = np.insert(rsp_c_peaks, 0, 0)
            time_rsp_c = info_rsp_c["RSP_Peaks"]/ sampling_rate
            hrv_c_rsa=nk.hrv_rsa(ecg_signals,rsp_c_signal, info_ecg, sampling_rate=sampling_rate, continuous=True)
            full_rsp_rate_c = nk.signal_rate(info_rsp_c["RSP_Peaks"], sampling_rate=sampling_rate)
            rsp_c_rvt = nk.rsp_rvt(rsp_c_signal['RSP_Clean'], method="birn2006", show=False)

            RSP_c_cleaned = pd.DataFrame({
                'Time': time_rsp_c,
                'RSP_c_RR': rsp_c_peaks,
                'RSP_c_Peaks': info_rsp_c["RSP_Peaks"],
                'RSP_c_Troughs': info_rsp_c["RSP_Troughs"],
                'RSP_c_rsp_rate': full_rsp_rate_c

            })
            RSP_c_cleaned.to_csv(fr'{directory}\RSP_c_RR.csv')
            RSP_c_cleaned = pd.DataFrame({
                'Time': rsp_c_time,
                'RSP_c': rsp_c_signal['RSP_Clean'],
                'RSP_c_Rate': rsp_c_signal["RSP_Rate"],
                'RSP_c_rvt': rsp_c_rvt,
                'RSP_c_Troughs': rsp_c_signal["RSP_Troughs"],
                'RSP_c_RVT': rsp_c_signal["RSP_RVT"],
                'RSP_c_Phase_Completion': rsp_c_signal["RSP_Phase_Completion"],
                'RSP_c_Symmetry_PeakTrough': rsp_c_signal["RSP_Symmetry_PeakTrough"],
                'RSP_c_Symmetry_RiseDecay': rsp_c_signal["RSP_Symmetry_RiseDecay"],
                'RSP_c_RSA_Gates': hrv_c_rsa["RSA_Gates"],
                'RSP_c_RSA_P2T': hrv_c_rsa["RSA_P2T"],
                'RSP_c_Amplitude': rsp_c_signal["RSP_Amplitude"]

            })
            RSP_c_cleaned.to_csv(fr'{directory}\RSP_c_F.csv')

            # __________________________________________RSP_D_______________________________________________
            rsp_d_data = BioPac.named_channels['Diaphragmatic Respiration'].data
            rsp_d_time = BioPac.named_channels['Diaphragmatic Respiration'].time_index
            rsp_d_data_cleaned = nk.rsp_clean(rsp_d_data, sampling_rate=sampling_rate)
            rsp_d_signal, info_rsp_d = nk.rsp_process(rsp_d_data_cleaned, sampling_rate=sampling_rate)
            rsp_d_peaks=np.diff(info_rsp_d["RSP_Peaks"]) / sampling_rate * 1000
            rsp_d_peaks = np.insert(rsp_d_peaks, 0, 0)
            time_rsp_d = info_rsp_d["RSP_Peaks"]/ sampling_rate
            hrv_d_rsa=nk.hrv_rsa(ecg_signals,rsp_d_signal, info_ecg, sampling_rate=sampling_rate, continuous=True)
            full_rsp_rate_d = nk.signal_rate(info_rsp_d["RSP_Peaks"], sampling_rate=sampling_rate)
            rsp_d_rvt = nk.rsp_rvt(rsp_d_signal['RSP_Clean'], method="birn2006", show=False)

            RSP_d_cleaned = pd.DataFrame({
                'Time': time_rsp_d,
                'RSP_d_RR': rsp_d_peaks,
                'RSP_d_Peaks': info_rsp_d["RSP_Peaks"],
                'RSP_d_Troughs': info_rsp_d["RSP_Troughs"],
                'RSP_d_rsp_rate': full_rsp_rate_d
            })
            RSP_d_cleaned.to_csv(fr'{directory}\RSP_d_RR.csv')
            RSP_d_cleaned = pd.DataFrame({
                'Time': rsp_d_time,
                'RSP_d': rsp_d_signal['RSP_Clean'],
                'RSP_d_Rate': rsp_d_signal["RSP_Rate"],
                'RSP_d_rvt': rsp_d_rvt,
                'RSP_d_Troughs': rsp_d_signal["RSP_Troughs"],
                'RSP_d_RVT': rsp_d_signal["RSP_RVT"],
                'RSP_d_Phase_Completion': rsp_d_signal["RSP_Phase_Completion"],
                'RSP_d_Symmetry_PeakTrough': rsp_d_signal["RSP_Symmetry_PeakTrough"],
                'RSP_d_Symmetry_RiseDecay': rsp_d_signal["RSP_Symmetry_RiseDecay"],
                'RSP_d_RSA_Gates': hrv_d_rsa["RSA_Gates"],
                'RSP_d_RSA_P2T': hrv_d_rsa["RSA_P2T"],
                'RSP_d_Amplitude': rsp_d_signal["RSP_Amplitude"]
            })
            RSP_d_cleaned.to_csv(fr'{directory}\RSP_d_F.csv')

    def BoxPlot(self, ID, Group):
        Participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df = pd.read_csv(Participants_path)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')

        if ID is not None:
            Participants_df = Participants_df[Participants_df['code'] == ID]

        for _, row in Participants_df.iterrows():
            ID = row['code']
            Group = row['Group']
            directory = fr'{self.path}\\Participants\\{Group}_group\\P_{ID}'

            # Load data
            Trigger_df = pd.read_csv(fr'{directory}\\Trigger_{ID}.csv')
            EDA_df = pd.read_csv(fr'{directory}\\EDA.csv')
            RR_df = pd.read_csv(fr'{directory}\\RR.csv')

            eda_time = EDA_df['Time']
            eda_signal = EDA_df['EDA']

            rr_time = RR_df['Time']
            rr_signal = RR_df['RR']

            # Ensure plot directory exists
            plot_dir = fr'{directory}\\plots'
            os.makedirs(plot_dir, exist_ok=True)

            # --- EDA Boxplot ---
            eda_by_task = []

            for _, trigger in Trigger_df.iterrows():
                task = trigger['Task']
                start = trigger['Start']
                end = trigger['End']

                segment = eda_signal[(eda_time >= start) & (eda_time <= end)]
                eda_by_task.extend([{'Task': task, 'EDA': val} for val in segment])

            df_eda = pd.DataFrame(eda_by_task)

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df_eda, x='Task', y='EDA')
            plt.title(f'Participant {ID} - EDA by Task Phase')
            plt.xlabel('Task Phase')
            plt.ylabel('EDA Amplitude')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(fr'{plot_dir}\\boxplot_EDA.png')
            plt.close()

            # --- RR Boxplot ---
            rr_by_task = []

            for _, trigger in Trigger_df.iterrows():
                task = trigger['Task']
                start = trigger['Start']
                end = trigger['End']

                segment = rr_signal[(rr_time >= start) & (rr_time <= end)]
                rr_by_task.extend([{'Task': task, 'RR': val} for val in segment])

            df_rr = pd.DataFrame(rr_by_task)

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df_rr, x='Task', y='RR')
            plt.title(f'Participant {ID} - RR by Task Phase')
            plt.xlabel('Task Phase')
            plt.ylabel('R-R Interval')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(fr'{plot_dir}\\boxplot_RR.png')
            plt.close()
    def plot_with_annotations(self, data, time_index, Trigger_df, feature, Channel, ID, Group, Insert_Mistake):
        plt.figure(figsize=(16, 6))  # Wider plot with adjusted dimensions
        plt.plot(time_index, data, label=f'{feature}')

        max_rate = max(data)
        plt.ylim(0, max_rate * 1.5)  # Adjust y-axis limit for better label placement

        # Ensure 'Stress Report' column has no NaN values
        Trigger_df['Stress Report'] = Trigger_df['Stress Report'].fillna('')

        for i, row in Trigger_df.iterrows():
            start = row['Time-start-sec']
            end = row['Time-end-sec']
            task = row['Task']
            stress_report = row['Stress Report']
            mid_point = (start + end) / 2
            y_pos = max_rate * (1.1 + 0.1 * (i % 2))  # Alternates text height for readability

            if task == 'mistake':
                line_color = 'red'
                plt.axvline(x=start, color=line_color, linestyle='--', alpha=0.7)
            else:
                if task in ['Baseline', 'wait_timer', 'breathing_part_1', 'breathing_part_2',
                            'breathing_part_3', 'breathing_part_4', 'music_part_1',
                            'music_part_2', 'music_part_3', 'music_part_4']:
                    # plt.axvspan(start, end, color='yellow', alpha=0.3, label='rest')
                    label_text = f"{task}"  # Consistent label structure
                    plt.text(mid_point, y_pos, label_text, ha='center', va='bottom', fontsize=8, color='black',
                             rotation=0)
                else:
                    # Use distinct colors for higher levels
                    plt.axvspan(start, end, color='orange', alpha=0.8, label=f'{task}')
                    label_text = f"{task}\nStress:{stress_report}"  # Consistent label structure
                    plt.text(mid_point, y_pos, label_text, ha='center', va='bottom', fontsize=8, color='black',
                             rotation=0)

        # Add labels, title, and legend
        plt.xlabel("Start Time (s)")
        plt.ylabel(f'{feature}')
        plt.title(f'{feature} For Participant {ID} Group {Group}')
        plt.grid(True)
        plt.tight_layout()  # Ensures everything fits nicely

        # Save the plot
        if Insert_Mistake:
            plt.savefig(
                f"D:\\Human Bio Signals Analysis\\Analysis\\{Channel}\\{feature}\\{Group}\\Participant_{ID}_{feature}_include_mistake.png")
        else:
            plt.savefig(
                f"D:\\Human Bio Signals Analysis\\Analysis\\{Channel}\\{feature}\\{Group}\\Participant_{ID}_{feature}.png")

        plt.show()
    def AX_Spectrogram(self,ID,Group):
        Participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')

        if ID is not None:
            Participants_df = Participants_df[Participants_df['code'] == ID]
        for _, row in Participants_df.iterrows():
            ID = row['code']
            Group = row['Group']
            directory = fr'{self.path}\\Participants\\{Group}_group\\P_{ID}'
            Trigger_path = fr'{directory}\\Trigger_{ID}.csv'
            Trigger_df = pd.read_csv(Trigger_path, header=0)
            BioPac_path = fr'{directory}\\P_{ID}.acq'
            BioPac = bioread.read_file(BioPac_path)

            fig, ax = plt.subplots(2, figsize=(15, 12), sharex=True)

            # Load EDA data
            eda_data = BioPac.named_channels['EDA'].data
            eda_time = BioPac.named_channels['EDA'].time_index
            eda_sampling_rate = BioPac.named_channels['EDA'].samples_per_second

            # Load ECG data
            ecg_data = BioPac.named_channels['ECG'].data
            ecg_time = BioPac.named_channels['ECG'].time_index
            ecg_sampling_rate = BioPac.named_channels['ECG'].samples_per_second

            # Process ECG to extract HRV (RR intervals)
            ecg_cleaned = nk.ecg_clean(ecg_data, sampling_rate=ecg_sampling_rate)
            r_peaks, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=ecg_sampling_rate)
            ecg_signals, info = nk.ecg_process(ecg_cleaned, ecg_sampling_rate)
            hrv_rri = np.diff(info["ECG_R_Peaks"]) / ecg_sampling_rate * 1000  # Convert to ms
            time_rri = info["ECG_R_Peaks"][1:] / ecg_sampling_rate  # Time axis for HRV

            # Process EDA
            eda_signals, info = nk.eda_process(eda_data, sampling_rate=eda_sampling_rate)
            eda_Phasic = eda_signals["EDA_Phasic"]

            # Compute spectrogram for HRV
            f_hrv, t_hrv, Sxx_hrv = spectrogram(hrv_rri, fs=1 / np.mean(np.diff(time_rri)), nperseg=20)

            # Compute spectrogram for EDA
            f_eda, t_eda, Sxx_eda = spectrogram(eda_data, fs=eda_sampling_rate, nperseg=256)

            # Plot HRV spectrogram
            hrv_img = ax[0].pcolormesh(t_hrv, f_hrv, np.log(Sxx_hrv), shading='gouraud')  # Avoid log(0)
            ax[0].set_ylabel("HRV Frequency (Hz)")
            ax[0].set_title(f"HRV Spectrogram - Participant {ID}")
            cbar_hrv = plt.colorbar(hrv_img, ax=ax[0])
            cbar_hrv.set_label('Log Power')
            y_min_ax0, y_max_ax0 = ax[0].get_ylim()

            # Plot EDA spectrogram
            eda_img = ax[1].pcolormesh(t_eda, f_eda, np.log(Sxx_eda), shading='gouraud')  # Avoid log(0)
            ax[1].set_ylabel("EDA Frequency (Hz)")
            ax[1].set_xlabel("Time (s)")
            ax[1].set_title(f"EDA Spectrogram - Participant {ID}")
            cbar_eda = plt.colorbar(eda_img, ax=ax[1])
            cbar_eda.set_label('Log Power')
            y_min_ax1, y_max_ax1 = ax[1].get_ylim()


            # Add Task Labels & Event Markers
            for _, trigger in Trigger_df.iterrows():
                start_time = trigger["Start"]
                end_time = trigger["End"]
                task_name = trigger["Task"]
                if pd.notna(end_time):
                    if task_name=='CB_easy' or task_name=='CB_hard':
                        ax[0].axvline(start_time, color='white', alpha=0.3)
                        ax[0].axvline(end_time, color='white', alpha=0.3)
                        ax[1].axvline(start_time, color='white', alpha=0.3)
                        ax[1].axvline(end_time, color='white', alpha=0.3)
                        ax[0].text((start_time+end_time)/2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name,
                                   ha='center', fontsize=10, color='black')
                    elif task_name == 'PA_easy' or task_name == 'PA_medium'or task_name == 'PA_hard':
                        ax[0].axvline(start_time, color='white', alpha=0.3)
                        ax[0].axvline(end_time, color='white', alpha=0.3)
                        ax[1].axvline(start_time, color='white', alpha=0.3)
                        ax[1].axvline(end_time, color='white', alpha=0.3)
                        ax[0].text((start_time + end_time) / 2,y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name,
                                   ha='center', fontsize=8, color='black')
                    elif task_name == 'TC_easy' or task_name == 'TC_hard':
                        ax[0].axvline(start_time, color='white', alpha=0.3)
                        ax[0].axvline(end_time, color='white', alpha=0.3)
                        ax[1].axvline(start_time, color='white', alpha=0.3)
                        ax[1].axvline(end_time, color='white', alpha=0.3)
                        ax[0].text((start_time + end_time) / 2,y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name,
                                   ha='center', fontsize=8, color='black')
                    else:
                        ax[0].axvline(start_time, color='white', alpha=0.3)
                        ax[0].axvline(end_time, color='white', alpha=0.3)
                        ax[1].axvline(start_time, color='white', alpha=0.3)
                        ax[1].axvline(end_time, color='white', alpha=0.3)
                        ax[0].text((start_time + end_time) / 2,y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name,
                                   ha='center', fontsize=10, color='black')

            plt.tight_layout()
            plot_path = fr'{directory}\\plots\\Spectrogram_Plot_{ID}.png'
            plt.savefig(plot_path, dpi=300)
            plt.show()
    def AX_Spectrogram_4_signals(self,ID,Group):
        Participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')

        if ID is not None:
            Participants_df = Participants_df[Participants_df['code'] == ID]
        for _, row in Participants_df.iterrows():
            ID = row['code']
            Group = row['Group']
            directory = fr'{self.path}\\Participants\\{Group}_group\\P_{ID}'
            Trigger_path = fr'{directory}\\Trigger_{ID}.csv'
            Trigger_df = pd.read_csv(Trigger_path, header=0)
            BioPac_path = fr'{directory}\\P_{ID}.acq'
            BioPac = bioread.read_file(BioPac_path)

            fig, ax = plt.subplots(4, figsize=(15, 12), sharex=True)
            # Load EDA data
            eda_data = BioPac.named_channels['EDA'].data
            eda_time = BioPac.named_channels['EDA'].time_index
            eda_sampling_rate = BioPac.named_channels['EDA'].samples_per_second
            # eda_signals, info = nk.eda_process(eda_data, sampling_rate=eda_sampling_rate)
            # eda_data = eda_signals['EDA_Clean']

            f_eda, t_eda, Sxx_eda = spectrogram(eda_data, fs=eda_sampling_rate, nperseg=256)

            # Plot EDA spectrogram
            eda_img = ax[0].pcolormesh(t_eda, f_eda, np.log(Sxx_eda), shading='gouraud')
            ax[0].set_ylabel("EDA Frequency (Hz)")
            ax[0].set_title(f"EDA Spectrogram - Participant {ID}")
            # cbar_eda = plt.colorbar(eda_img, ax=ax[0])
            # cbar_eda.set_label('Log Power')
            y_min_ax0, y_max_ax0 = ax[0].get_ylim()

            ax[1].plot(eda_time, eda_data, label=f"Participant {ID}")

            # Load ECG data
            ecg_data = BioPac.named_channels['ECG'].data
            ecg_time = BioPac.named_channels['ECG'].time_index
            ecg_sampling_rate = BioPac.named_channels['ECG'].samples_per_second

            # Process ECG to extract HRV (RR intervals)
            ecg_cleaned = nk.ecg_clean(ecg_data, sampling_rate=ecg_sampling_rate)
            r_peaks, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=ecg_sampling_rate)
            ecg_signals, info = nk.ecg_process(ecg_cleaned, ecg_sampling_rate)
            hrv_rri = np.diff(info["ECG_R_Peaks"]) / ecg_sampling_rate * 1000  # Convert to ms
            time_rri = info["ECG_R_Peaks"][1:] / ecg_sampling_rate  # Time axis for HRV

            f_hrv, t_hrv, Sxx_hrv = spectrogram(hrv_rri, fs=1 / np.mean(np.diff(time_rri)), nperseg=20)

            hrv_img = ax[2].pcolormesh(t_hrv, f_hrv, np.log(Sxx_hrv), shading='gouraud')  # Avoid log(0)
            ax[2].set_ylabel("HRV Frequency (Hz)")
            ax[2].set_title(f"HRV Spectrogram - Participant {ID}")
            # cbar_hrv = plt.colorbar(hrv_img, ax=ax[0])
            # cbar_hrv.set_label('Log Power')

            ax[3].plot(time_rri, hrv_rri, label=f"Participant {ID}")

            # Add Task Labels & Event Markers
            for _, trigger in Trigger_df.iterrows():
                start_time = trigger["Start"]
                end_time = trigger["End"]
                task_name = trigger["Task"]
                if pd.notna(end_time):
                    if task_name=='CB_easy' or task_name=='CB_hard':
                        ax[0].axvline(start_time, color='white', alpha=0.3)
                        ax[0].axvline(end_time, color='white', alpha=0.3)
                        ax[1].axvline(start_time, color='purple', alpha=0.3)
                        ax[1].axvline(end_time, color='purple', alpha=0.3)
                        ax[2].axvline(start_time, color='white', alpha=0.3)
                        ax[2].axvline(end_time, color='white', alpha=0.3)
                        ax[3].axvline(start_time, color='purple', alpha=0.3)
                        ax[3].axvline(end_time, color='purple', alpha=0.3)
                        ax[0].text((start_time+end_time)/2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name, ha='center', fontsize=10, color='black')
                    elif task_name == 'PA_easy' or task_name == 'PA_medium'or task_name == 'PA_hard':
                        ax[0].axvline(start_time, color='white', alpha=0.3)
                        ax[0].axvline(end_time, color='white', alpha=0.3)
                        ax[1].axvline(start_time, color='purple', alpha=0.3)
                        ax[1].axvline(end_time, color='purple', alpha=0.3)
                        ax[2].axvline(start_time, color='white', alpha=0.3)
                        ax[2].axvline(end_time, color='white', alpha=0.3)
                        ax[3].axvline(start_time, color='purple', alpha=0.3)
                        ax[3].axvline(end_time, color='purple', alpha=0.3)
                        ax[0].text((start_time + end_time) / 2,y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name, ha='center', fontsize=8, color='black')
                    elif task_name == 'TC_easy' or task_name == 'TC_hard':
                        ax[0].axvline(start_time, color='white', alpha=0.3)
                        ax[0].axvline(end_time, color='white', alpha=0.3)
                        ax[1].axvline(start_time, color='purple', alpha=0.3)
                        ax[1].axvline(end_time, color='purple', alpha=0.3)
                        ax[2].axvline(start_time, color='white', alpha=0.3)
                        ax[2].axvline(end_time, color='white', alpha=0.3)
                        ax[3].axvline(start_time, color='purple', alpha=0.3)
                        ax[3].axvline(end_time, color='purple', alpha=0.3)
                        ax[0].text((start_time + end_time) / 2,y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name, ha='center', fontsize=8, color='black')
                    else:
                        ax[0].axvline(start_time, color='white', alpha=0.3)
                        ax[0].axvline(end_time, color='white', alpha=0.3)
                        ax[1].axvline(start_time, color='purple', alpha=0.3)
                        ax[1].axvline(end_time, color='purple', alpha=0.3)
                        ax[2].axvline(start_time, color='white', alpha=0.3)
                        ax[2].axvline(end_time, color='white', alpha=0.3)
                        ax[3].axvline(start_time, color='purple', alpha=0.3)
                        ax[3].axvline(end_time, color='purple', alpha=0.3)
                        ax[0].text((start_time + end_time) / 2,y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name, ha='center', fontsize=10, color='black')

            plt.tight_layout()
            plot_path = fr'{directory}\\plots\\Spectrogram_4_signals_Plot_{ID}.png'
            plt.savefig(plot_path, dpi=300)
            plt.show()
    def AX_Spectrogram_signals(self,ID,Group):
        Participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')

        if ID is not None:
            Participants_df = Participants_df[Participants_df['code'] == ID]
        for _, row in Participants_df.iterrows():
            ID = row['code']
            Group = row['Group']
            directory = fr'{self.path}\\Participants\\{Group}_group\\P_{ID}'
            Trigger_path = fr'{directory}\\Trigger_{ID}.csv'
            Trigger_df = pd.read_csv(Trigger_path, header=0)
            BioPac_path = fr'{directory}\\P_{ID}.acq'
            BioPac = bioread.read_file(BioPac_path)

            fig, ax = plt.subplots(2, figsize=(15, 12), sharex=True)
            Channels = ['EDA', 'HRV']
            for channel in Channels:
                if channel=='EDA':
                    # Load EDA data
                    eda_data = BioPac.named_channels['EDA'].data
                    eda_time = BioPac.named_channels['EDA'].time_index
                    eda_sampling_rate = BioPac.named_channels['EDA'].samples_per_second
                    eda_signals, info = nk.eda_process(eda_data, sampling_rate=eda_sampling_rate)
                    eda_data = eda_signals['EDA_Clean']

                    f_eda, t_eda, Sxx_eda = spectrogram(eda_data, fs=eda_sampling_rate, nperseg=256)
                    # Plot EDA spectrogram
                    eda_img = ax[0].pcolormesh(t_eda, f_eda, np.log(Sxx_eda), shading='gouraud')
                    ax[0].set_ylabel("EDA Frequency (Hz)")
                    ax[0].set_xlabel("Time (s)")
                    ax[0].set_title(f"EDA Spectrogram - Participant {ID}")
                    # cbar_eda = plt.colorbar(eda_img, ax=ax[0])
                    # cbar_eda.set_label('Log Power')
                    y_min_ax0, y_max_ax0 = ax[0].get_ylim()

                    ax[1].plot(eda_time, eda_data, label=f"Participant {ID}")

                if channel=='HRV':
                    # Load ECG data
                    ecg_data = BioPac.named_channels['ECG'].data
                    ecg_time = BioPac.named_channels['ECG'].time_index
                    ecg_sampling_rate = BioPac.named_channels['ECG'].samples_per_second

                    # Process ECG to extract HRV (RR intervals)
                    ecg_cleaned = nk.ecg_clean(ecg_data, sampling_rate=ecg_sampling_rate)
                    r_peaks, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=ecg_sampling_rate)
                    ecg_signals, info = nk.ecg_process(ecg_cleaned, ecg_sampling_rate)
                    hrv_rri = np.diff(info["ECG_R_Peaks"]) / ecg_sampling_rate * 1000  # Convert to ms
                    time_rri = info["ECG_R_Peaks"][1:] / ecg_sampling_rate  # Time axis for HRV
                    # Compute spectrogram for HRV
                    f_hrv, t_hrv, Sxx_hrv = spectrogram(hrv_rri, fs=1 / np.mean(np.diff(time_rri)), nperseg=20)

                    # Plot HRV spectrogram
                    hrv_img = ax[0].pcolormesh(t_hrv, f_hrv, np.log(Sxx_hrv), shading='gouraud')  # Avoid log(0)
                    ax[0].set_ylabel("HRV Frequency (Hz)")
                    ax[0].set_title(f"HRV Spectrogram - Participant {ID}")
                    cbar_hrv = plt.colorbar(hrv_img, ax=ax[0])
                    cbar_hrv.set_label('Log Power')
                    y_min_ax0, y_max_ax0 = ax[0].get_ylim()

                    ax[1].plot(time_rri, hrv_rri, label=f"Participant {ID}")

                # Add Task Labels & Event Markers
                for _, trigger in Trigger_df.iterrows():
                    start_time = trigger["Start"]
                    end_time = trigger["End"]
                    task_name = trigger["Task"]
                    if pd.notna(end_time):
                        if task_name=='CB_easy' or task_name=='CB_hard':
                            # ax[0].axvline(start_time, color='white', alpha=0.3)
                            # ax[0].axvline(end_time, color='white', alpha=0.3)
                            ax[1].axvline(start_time, color='purple', alpha=0.3)
                            ax[1].axvline(end_time, color='purple', alpha=0.3)
                            ax[0].text((start_time+end_time)/2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name,
                                       ha='center', fontsize=10, color='black')
                        elif task_name == 'PA_easy' or task_name == 'PA_medium'or task_name == 'PA_hard':
                            # ax[0].axvline(start_time, color='white', alpha=0.3)
                            # ax[0].axvline(end_time, color='white', alpha=0.3)
                            ax[1].axvline(start_time, color='purple', alpha=0.3)
                            ax[1].axvline(end_time, color='purple', alpha=0.3)
                            ax[0].text((start_time + end_time) / 2,y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name,
                                       ha='center', fontsize=8, color='black')
                        elif task_name == 'TC_easy' or task_name == 'TC_hard':
                            # ax[0].axvline(start_time, color='white', alpha=0.3)
                            # ax[0].axvline(end_time, color='white', alpha=0.3)
                            ax[1].axvline(start_time, color='purple', alpha=0.3)
                            ax[1].axvline(end_time, color='purple', alpha=0.3)
                            ax[0].text((start_time + end_time) / 2,y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name,
                                       ha='center', fontsize=8, color='black')
                        else:
                            # ax[0].axvline(start_time, color='white', alpha=0.3)
                            # ax[0].axvline(end_time, color='white', alpha=0.3)
                            ax[1].axvline(start_time, color='purple', alpha=0.3)
                            ax[1].axvline(end_time, color='purple', alpha=0.3)
                            ax[0].text((start_time + end_time) / 2,y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name,
                                       ha='center', fontsize=10, color='black')

                plt.tight_layout()
                plot_path = fr'{directory}\\plots\\Spectrogram_{channel}_Plot_{ID}.png'
                plt.savefig(plot_path, dpi=300)
                plt.show()
    def AX_plot_3_part_HRV(self,ID):
        # Load participant management data
        Participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')

        if ID is not None:
            if isinstance(ID, list):  # Handle multiple IDs
                Participants_df = Participants_df[Participants_df['code'].isin(ID)]
            else:  # Handle single ID
                Participants_df = Participants_df[Participants_df['code'] == ID]

        fig, ax = plt.subplots(3, figsize=(15, 12), sharex=True)
        fig.suptitle(f'HRV for particapents {ID}')

        # Loop through participants
        for i, row in Participants_df.iterrows():
            participant_id = row['code']
            participant_group = row['Group']

            # Define paths
            directory = fr'{self.path}\\Participants\\{participant_group}_group\\P_{participant_id}'
            Trigger_path = fr'{directory}\\Trigger_{participant_id}.csv'
            BioPac_path = fr'{directory}\\P_{participant_id}.acq'


            # Load data
            Trigger_df = pd.read_csv(Trigger_path, header=0)
            BioPac = bioread.read_file(BioPac_path)

            ecg_data = BioPac.named_channels['ECG'].data
            ecg_time = BioPac.named_channels['ECG'].time_index
            ecg_sampling_rate = BioPac.named_channels['ECG'].samples_per_second

            # Process ECG
            ecg_cleaned = nk.ecg_clean(ecg_data, sampling_rate=ecg_sampling_rate)
            _, r_peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=ecg_sampling_rate)
            ecg_signals, info = nk.ecg_process(ecg_cleaned, ecg_sampling_rate)
            hrv_rri = np.diff(info["ECG_R_Peaks"]) / ecg_sampling_rate * 1000
            time_rri = info["ECG_R_Peaks"][1:] / ecg_sampling_rate

            # Plot EDA and HRV
            ax[i].plot(time_rri, hrv_rri, label=f"Participant {participant_id}")
            ax[i].legend(loc='upper right')

            y_min_ax0, y_max_ax0 = ax[0].get_ylim()
            # Annotate tasks
            for _, trigger in Trigger_df.iterrows():
                start_time, end_time, task_name, task_score = trigger["Start"], trigger["End"], trigger["Task"], \
                trigger["Score"]
                if pd.notna(end_time):
                    if task_name == 'CB_easy' or task_name == 'CB_hard':
                        ax[i].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                        if i==0:
                            ax[0].text((start_time + end_time) / 2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name, ha='center', fontsize=8, color='black')
                    elif task_name == 'PA_easy' or task_name == 'PA_medium' or task_name == 'PA_hard':
                        ax[i].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                        if i==0:
                            ax[0].text((start_time + end_time) / 2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name, ha='center', fontsize=8, color='black')
                    elif task_name == 'TC_easy' or task_name == 'TC_hard':
                        ax[i].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                        if i==0:
                            ax[0].text((start_time + end_time) / 2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name, ha='center', fontsize=8, color='black')
                    else:
                        ax[i].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                        if i==0:
                            ax[0].text((start_time + end_time) / 2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name, ha='center', fontsize=8, color='black')
                else:
                    ax[i].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
            plt.tight_layout()
        plot_path = fr'{self.path}\Participants\plots_data\HRV_comparision.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
    def AX_plot_3in1norm_EDA(self, ID):
        # Load participant management data
        Participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0).dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')

        if ID is not None:
            Participants_df = Participants_df[Participants_df['code'].isin(ID)] if isinstance(ID, list) else \
            Participants_df[Participants_df['code'] == ID]

        plt.figure(figsize=(12, 6))
        plt.title(f'EDA for participants {ID}')

        colors = cm.viridis(np.linspace(0, 1, len(Participants_df)))  # Dynamic color map

        for i, row in Participants_df.iterrows():
            participant_id, participant_group = row['code'], row['Group']
            directory = fr'{self.path}\\Participants\\{participant_group}_group\\P_{participant_id}'
            Trigger_path, BioPac_path = fr'{directory}\\Trigger_{participant_id}.csv', fr'{directory}\\P_{participant_id}.acq'

            # Load Data
            Trigger_df = pd.read_csv(Trigger_path, header=0)
            BioPac = bioread.read_file(BioPac_path)

            # Extract EDA signals
            eda_data = BioPac.named_channels['EDA'].data
            eda_sampling_rate = BioPac.named_channels['EDA'].samples_per_second
            eda_signals, _ = nk.eda_process(eda_data, sampling_rate=eda_sampling_rate)
            eda_cleaned = eda_signals['EDA_Clean']
            # eda_cleaned = (eda_cleaned - np.min(eda_cleaned)) / (np.max(eda_cleaned) - np.min(eda_cleaned))
            eda_cleaned = (eda_cleaned - np.mean(eda_cleaned)) / np.std(eda_cleaned)
            eda_time = np.arange(len(eda_cleaned)) / eda_sampling_rate  # Ensure correct time scaling

            # Plot EDA
            plt.plot(eda_time, eda_cleaned, label=f"Participant {participant_id}", color=colors[i])

            y_min_ax0, y_max_ax0 = plt.gca().get_ylim()

            # Annotate tasks
            for _, trigger in Trigger_df.iterrows():
                start_time, end_time, task_name = trigger["Start"], trigger["End"], trigger["Task"]
                if pd.notna(end_time):
                    plt.axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                    if i == 0:  # Only label once
                        text_y = max(y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), y_min_ax0 * 0.95)
                        plt.text((start_time + end_time) / 2, text_y, task_name, ha='center', fontsize=8, color='black')

        plt.legend(loc='upper right')
        plt.tight_layout()
        plot_path = fr'{self.path}\Participants\plots_data\EDA_3in1norm_comparision.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
    def AX_plot_3in1_EDA(self, ID):
        # Load participant management data
        Participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0).dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')

        if ID is not None:
            Participants_df = Participants_df[Participants_df['code'].isin(ID)] if isinstance(ID, list) else \
            Participants_df[Participants_df['code'] == ID]

        plt.figure(figsize=(12, 6))
        plt.title(f'EDA for participants {ID}')

        colors = cm.viridis(np.linspace(0, 1, len(Participants_df)))  # Dynamic color map

        for i, row in Participants_df.iterrows():
            participant_id, participant_group = row['code'], row['Group']
            directory = fr'{self.path}\\Participants\\{participant_group}_group\\P_{participant_id}'
            Trigger_path, BioPac_path = fr'{directory}\\Trigger_{participant_id}.csv', fr'{directory}\\P_{participant_id}.acq'

            # Load Data
            Trigger_df = pd.read_csv(Trigger_path, header=0)
            BioPac = bioread.read_file(BioPac_path)

            # Extract EDA signals
            eda_data = BioPac.named_channels['EDA'].data
            eda_sampling_rate = BioPac.named_channels['EDA'].samples_per_second
            eda_signals, _ = nk.eda_process(eda_data, sampling_rate=eda_sampling_rate)
            eda_cleaned = eda_signals['EDA_Clean']
            eda_time = np.arange(len(eda_cleaned)) / eda_sampling_rate  # Ensure correct time scaling

            # Plot EDA
            plt.plot(eda_time, eda_cleaned, label=f"Participant {participant_id}", color=colors[i])

            y_min_ax0, y_max_ax0 = plt.gca().get_ylim()

            # Annotate tasks
            for _, trigger in Trigger_df.iterrows():
                start_time, end_time, task_name = trigger["Start"], trigger["End"], trigger["Task"]
                if pd.notna(end_time):
                    plt.axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                    if i == 0:  # Only label once
                        text_y = max(y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), y_min_ax0 * 0.95)
                        plt.text((start_time + end_time) / 2, text_y, task_name, ha='center', fontsize=8, color='black')

        plt.legend(loc='upper right')
        plt.tight_layout()
        plot_path = fr'{self.path}\Participants\plots_data\EDA_3in1_comparision.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
    def AX_plot_3_part_EDA(self,ID):
        # Load participant management data
        Participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')

        if ID is not None:
            if isinstance(ID, list):  # Handle multiple IDs
                Participants_df = Participants_df[Participants_df['code'].isin(ID)]
            else:  # Handle single ID
                Participants_df = Participants_df[Participants_df['code'] == ID]

        fig, ax = plt.subplots(3, figsize=(15, 12), sharex=True)
        fig.suptitle(f'EDA for particapents {ID}')

        # Loop through participants
        for i, row in Participants_df.iterrows():
            participant_id = row['code']
            participant_group = row['Group']

            # Define paths
            directory = fr'{self.path}\\Participants\\{participant_group}_group\\P_{participant_id}'
            Trigger_path = fr'{directory}\\Trigger_{participant_id}.csv'
            BioPac_path = fr'{directory}\\P_{participant_id}.acq'


            # Load data
            Trigger_df = pd.read_csv(Trigger_path, header=0)
            BioPac = bioread.read_file(BioPac_path)

            # Create plots
            eda_data = BioPac.named_channels['EDA'].data
            print(len(eda_data))
            eda_sampling_rate = BioPac.named_channels['EDA'].samples_per_second
            eda_signals, info = nk.eda_process(eda_data, sampling_rate=eda_sampling_rate)
            eda_data=eda_signals['EDA_Clean']
            eda_time = BioPac.named_channels['EDA'].time_index
            # Plot EDA and HRV
            ax[i].plot(eda_time, eda_data, label=f"Participant {participant_id}")
            ax[i].legend(loc='upper right')

            y_min_ax0, y_max_ax0 = ax[0].get_ylim()
            # Annotate tasks
            for _, trigger in Trigger_df.iterrows():
                start_time, end_time, task_name, task_score = trigger["Start"], trigger["End"], trigger["Task"], \
                trigger["Score"]
                if pd.notna(end_time):
                    if task_name == 'CB_easy' or task_name == 'CB_hard':
                        ax[i].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                        if i==0:
                            ax[0].text((start_time + end_time) / 2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name, ha='center', fontsize=8, color='black')
                    elif task_name == 'PA_easy' or task_name == 'PA_medium' or task_name == 'PA_hard':
                        ax[i].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                        if i==0:
                            ax[0].text((start_time + end_time) / 2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name, ha='center', fontsize=8, color='black')
                    elif task_name == 'TC_easy' or task_name == 'TC_hard':
                        ax[i].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                        if i==0:
                            ax[0].text((start_time + end_time) / 2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name, ha='center', fontsize=8, color='black')
                    else:
                        ax[i].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                        if i==0:
                            ax[0].text((start_time + end_time) / 2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name, ha='center', fontsize=8, color='black')
                else:
                    ax[i].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
            plt.tight_layout()
        plot_path = fr'{self.path}\Participants\plots_data\EDA_intervals_comparision.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
    def AX_plot_signals(self,ID,rangeID):
        # Load participant management data
        Participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')

        if ID is not None:
            if rangeID:
                Participants_df = Participants_df[Participants_df['code'] >= ID]
            else:
                Participants_df = Participants_df[Participants_df['code'] == ID]

        # Loop through participants
        for _, row in Participants_df.iterrows():
            participant_id = row['code']
            participant_group = row['Group']

            # Define paths
            directory = fr'{self.path}\\Participants\\{participant_group}_group\\P_{participant_id}'
            Trigger_path = fr'{directory}\\Trigger_{participant_id}.csv'
            BioPac_path = fr'{directory}\\P_{participant_id}.acq'

            try:
                # Load data
                Trigger_df = pd.read_csv(Trigger_path, header=0)
                BioPac = bioread.read_file(BioPac_path)

                # Create plots
                fig, ax = plt.subplots(2, figsize=(15, 12), sharex=True)
                eda_data = BioPac.named_channels['EDA'].data
                eda_sampling_rate = BioPac.named_channels['EDA'].samples_per_second
                eda_signals, info = nk.eda_process(eda_data, sampling_rate=eda_sampling_rate)
                eda_data=eda_signals['EDA_Clean']
                eda_time = BioPac.named_channels['EDA'].time_index
                ecg_data = BioPac.named_channels['ECG'].data
                ecg_time = BioPac.named_channels['ECG'].time_index
                ecg_sampling_rate = BioPac.named_channels['ECG'].samples_per_second

                # Process ECG
                ecg_cleaned = nk.ecg_clean(ecg_data, sampling_rate=ecg_sampling_rate)
                _, r_peaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=ecg_sampling_rate)
                ecg_signals, info = nk.ecg_process(ecg_cleaned, ecg_sampling_rate)
                hrv_rri = np.diff(info["ECG_R_Peaks"]) / ecg_sampling_rate * 1000
                time_rri = info["ECG_R_Peaks"][1:] / ecg_sampling_rate

                # Plot EDA and HRV
                ax[0].plot(eda_time, eda_data, label=f"Participant {participant_id}")
                ax[1].plot(time_rri, hrv_rri, label=f"Participant {participant_id}")
                y_min_ax0, y_max_ax0 = ax[0].get_ylim()

                # Annotate tasks
                for _, trigger in Trigger_df.iterrows():
                    start_time, end_time, task_name, task_score = trigger["Start"], trigger["End"], trigger["Task"], \
                    trigger["Score"]
                    if pd.notna(end_time):
                        if task_name == 'CB_easy' or task_name == 'CB_hard':
                            ax[0].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                            ax[1].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                            ax[0].text((start_time + end_time) / 2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name,
                                       ha='center', fontsize=10, color='black')
                        elif task_name == 'PA_easy' or task_name == 'PA_medium' or task_name == 'PA_hard':
                            ax[0].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                            ax[1].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                            ax[0].text((start_time + end_time) / 2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name,
                                       ha='center', fontsize=8, color='black')
                        elif task_name == 'TC_easy' or task_name == 'TC_hard':
                            ax[0].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                            ax[1].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                            ax[0].text((start_time + end_time) / 2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name,
                                       ha='center', fontsize=8, color='black')
                        else:
                            ax[0].axvspan(start_time, end_time, color='#C3E6CB', alpha=0.3)
                            ax[1].axvspan(start_time, end_time, color='#C3E6CB', alpha=0.3)
                            ax[0].text((start_time + end_time) / 2, y_min_ax0 - 0.05 * (y_max_ax0 - y_min_ax0), task_name,
                                       ha='center', fontsize=10, color='black')
                    else:
                        ax[0].axvline(start_time, color='green', linestyle='--')
                        ax[1].axvline(start_time, color='green', linestyle='--')

                ax[0].set_ylabel("EDA Signal")
                ax[0].legend(loc="upper right", fontsize=8)
                ax[1].set_ylabel("ECG RR intervals")
                ax[1].legend(loc="upper right", fontsize=8)
                plt.tight_layout()
                plot_path = fr'{directory}\plots\EDA_RR_intervals_{ID}.png'
                plt.savefig(plot_path, dpi=300)
                plt.show()

            except Exception as e:
                print(f"Error processing participant {participant_id}: {e}")
    # def Cor_plot(self,Window,Overlap,ID, Group)
    #     Participants_path = f'{self.path}\\Participants\\participation management.csv'
    #     Participants_df = pd.read_csv(Participants_path, header=0)
    #     Participants_df = Participants_df.dropna(axis=1, how='all')
    #     Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
    #     Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')
    #     if ID is not None:
    #         Participants_df = Participants_df[Participants_df['code'] == ID]
    #
    #     for j, row in Participants_df.iterrows():
    #         ID = row['code']
    #         print(ID)
    #         Group = row['Group']
    #         directory = fr'{self.path}\\Participants\\{Group}_group\\P_{ID}'
    #         Trigger_path = fr'{directory}\\Trigger_{ID}.csv'
    #         Trigger_df = pd.read_csv(Trigger_path, header=0)
    #         BioPac_path = fr'{directory}\\P_{ID}.acq'
    #         BioPac = bioread.read_file(BioPac_path)
    #         EDA = pd.read_csv(fr'{directory}\\EDA.csv')
    #         RR = pd.read_csv(fr'{directory}\\RR.csv')
    #         RSP_D = BioPac.named_channels['Diaphragmatic Respiration'].data
    #         RSP_D_T = BioPac.named_channels['Diaphragmatic Respiration'].time_index
    #         RSP_C = BioPac.named_channels['Chest Respiration'].data
    #         RSP_C_T = BioPac.named_channels['Chest Respiration'].time_index
    #         eda_data = EDA['EDA']
    def AX_plot_signals_VAS(self, ID, rangeID,Signals_plot,Cor_plot):
        VAS_plot=False
        EDA_plot=True
        RR_plot=False
        RSP_D_plot=False
        RSP_C_plot=False
        Participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')
        if ID is not None:
            if rangeID:
                Participants_df = Participants_df[Participants_df['code'] >= ID]
            else:
                Participants_df = Participants_df[Participants_df['code'] == ID]
        stress_all = pd.DataFrame(columns=['ID', 'Group', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_CVNN', 'HRV_pNN20', 'HRV_pNN50','Stress'])
        fatigue_all= pd.DataFrame(columns=['ID', 'Group', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_CVNN', 'HRV_pNN20', 'HRV_pNN50','Fatigue'])
        all_corr_records = []  # every (ID, Group, Target, Feature, r, p)
        for j, row in Participants_df.iterrows():
            ID = row['code']
            Group = row['Group']
            directory = fr'{self.path}\\Participants\\{Group}_group\\P_{ID}'
            Trigger_path = fr'{directory}\\Trigger_{ID}.csv'
            Trigger_df = pd.read_csv(Trigger_path, header=0)
            BioPac_path = fr'{directory}\\P_{ID}.acq'
            BioPac = bioread.read_file(BioPac_path)
            EDA = pd.read_csv(fr'{directory}\\EDA.csv')
            RR = pd.read_csv(fr'{directory}\\RR.csv')
            RSP_D = BioPac.named_channels['Diaphragmatic Respiration'].data
            RSP_D_T = BioPac.named_channels['Diaphragmatic Respiration'].time_index
            RSP_C = BioPac.named_channels['Chest Respiration'].data
            RSP_C_T = BioPac.named_channels['Chest Respiration'].time_index
            eda_data = EDA['EDA']
            if Signals_plot:
                if VAS_plot:
                    # Original Combined Plot
                    fig, ax = plt.subplots(6, figsize=(15, 18), sharex=True)

                    ax[0].plot(EDA['Time'], EDA['EDA'], label=f"Participant {ID}")
                    ax[1].scatter(RSP_D_T, RSP_D, label=f"Participant {ID}")
                    ax[2].scatter(RSP_C_T, RSP_C, label=f"Participant {ID}")
                    ax[3].scatter(RR['Time'], RR['RR'], label=f"Participant {ID}")
                    # Set y-axis limits to 500-1500 ms for consistent RR interval scaling
                    ax[3].set_ylim([400, 1500])
                    ax[0].set_ylim([-0.005, 6.4])  # Set y-axis limits to 500-1200 ms for consistent RR interval scaling

                    stress_data = Trigger_df[Trigger_df["Task"] == "VAS_Stress"]
                    fatigue_data = Trigger_df[Trigger_df["Task"] == "VAS_Fatigue"]

                    ax[4].scatter(stress_data["Start"], stress_data["Score"], color='blue', label="VAS Stress")
                    ax[5].scatter(fatigue_data["Start"], fatigue_data["Score"], color='red', label="VAS Fatigue")

                    for _, trigger in Trigger_df.iterrows():
                        start_time = trigger["Start"]
                        end_time = trigger["End"]
                        task_name = trigger["Task"]
                        if pd.notna(end_time):
                            if task_name == 'CB_easy' or task_name == 'CB_hard':
                                ax[0].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[1].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[2].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[3].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[4].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[5].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[0].text((start_time + end_time) / 2, min(eda_data) - 0.10 * max(eda_data), task_name,
                                           ha='center', fontsize=10, color='black')
                            elif task_name == 'PA_easy' or task_name == 'PA_medium' or task_name == 'PA_hard':
                                ax[0].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[1].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[2].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[3].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[4].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[5].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[0].text((start_time + end_time) / 2, min(eda_data) - 0.10 * max(eda_data), task_name,
                                           ha='center', fontsize=10, color='black')
                            elif task_name == 'TC_easy' or task_name == 'TC_hard':
                                ax[0].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[1].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[2].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[3].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[4].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[5].axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                ax[0].text((start_time + end_time) / 2, min(eda_data) - 0.10 * max(eda_data), task_name,
                                           ha='center', fontsize=10, color='black')
                            else:
                                ax[0].axvspan(start_time, end_time, color='#C3E6CB', alpha=0.3)
                                ax[1].axvspan(start_time, end_time, color='#C3E6CB', alpha=0.3)
                                ax[2].axvspan(start_time, end_time, color='#C3E6CB', alpha=0.3)
                                ax[3].axvspan(start_time, end_time, color='#C3E6CB', alpha=0.3)
                                ax[4].axvspan(start_time, end_time, color='#C3E6CB', alpha=0.3)
                                ax[5].axvspan(start_time, end_time, color='#C3E6CB', alpha=0.3)
                                ax[0].text((start_time + end_time) / 2, min(eda_data) - 0.10 * max(eda_data), task_name,
                                           ha='center', fontsize=10, color='black')
                        else:
                            ax[0].axvline(start_time, color='green', linestyle='--')
                            ax[1].axvline(start_time, color='green', linestyle='--')
                            ax[2].axvline(start_time, color='green', linestyle='--')
                            ax[3].axvline(start_time, color='green', linestyle='--')
                            ax[4].axvline(start_time, color='green', linestyle='--')
                            ax[5].axvline(start_time, color='green', linestyle='--')

                    ax[0].set_ylabel("EDA Signal")
                    ax[0].legend(loc="upper right", fontsize=8)
                    ax[1].set_ylabel("RSP_D")
                    ax[1].legend(loc="upper right", fontsize=8)
                    ax[2].set_ylabel("RSP_C")
                    ax[2].legend(loc="upper right", fontsize=8)
                    ax[3].set_ylabel("ECG RR intervals")
                    ax[3].legend(loc="upper right", fontsize=8)
                    ax[4].set_ylabel("VAS Stress Score")
                    ax[4].legend(loc="upper right", fontsize=8)
                    ax[5].set_ylabel("VAS Fatigue Score")
                    ax[5].legend(loc="upper right", fontsize=8)

                    plt.tight_layout()
                    plot_path = fr'{directory}\\plots\\Trigger_Plot_{ID}_vas.png'
                    plt.savefig(plot_path, dpi=300)
                    plt.show()

                # Create separate directories for each signal type if they don't exist
                analysis_base_path = "C:\\Users\\e3bom\\Desktop\\Human Bio Signals Analysis\\Analysis"
                eda_path = f"{analysis_base_path}\\EDA"
                rsp_d_path = f"{analysis_base_path}\\RSP_D"
                rsp_c_path = f"{analysis_base_path}\\RSP_C"
                rr_path = f"{analysis_base_path}\\RR"

                for path in [eda_path, rsp_d_path, rsp_c_path, rr_path]:
                    os.makedirs(path, exist_ok=True)

                # 1. Separate EDA Plot
                if EDA_plot:
                    fig_eda = plt.figure(figsize=(12, 6))
                    ax_eda = fig_eda.add_subplot(111)
                    ax_eda.plot(EDA['Time'], EDA['EDA'], label=f"Participant {ID}")
                    ax_eda.set_title(f"EDA Signal - Participant {ID}  Group:{Group}")
                    ax_eda.set_xlabel("Time (s)")
                    ax_eda.set_ylabel("EDA Signal")
                    ax_eda.set_ylim([-0.005, 6.4])  # Set y-axis limits to 500-1200 ms for consistent RR interval scaling

                    # Add task regions to the EDA plot
                    for _, trigger in Trigger_df.iterrows():
                        start_time = trigger["Start"]
                        end_time = trigger["End"]
                        task_name = trigger["Task"]

                        if pd.notna(end_time):
                            if task_name in ['CB_easy', 'CB_hard', 'PA_easy', 'PA_medium', 'PA_hard', 'TC_easy', 'TC_hard']:
                                ax_eda.axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                                # Use transform for text positioning to avoid blended transform warning
                                # y_pos = min(eda_data) - 0.10 * max(eda_data)
                                # ax_eda.text(0.5 * (start_time + end_time), y_pos,
                                #             task_name, ha='center', fontsize=10, color='black')
                            else:
                                ax_eda.axvspan(start_time, end_time, color='#C3E6CB', alpha=0.3)
                                # y_pos = min(eda_data) - 0.10 * max(eda_data)
                                # ax_eda.text(0.5 * (start_time + end_time), y_pos,
                                #             task_name, ha='center', fontsize=10, color='black')
                        else:
                            ax_eda.axvline(start_time, color='green', linestyle='--')

                    plt.legend(loc='upper right')  # Specify legend location instead of 'best'
                    plt.tight_layout()
                    plt.savefig(f"{eda_path}\\EDA_P{ID}.png", dpi=300)
                    print(fr"EDA plot {ID}")
                    plt.close()

                # 2. Separate RSP_D Plot
                if RSP_D_plot:
                    fig_rsp_d = plt.figure(figsize=(12, 6))
                    ax_rsp_d = fig_rsp_d.add_subplot(111)
                    RSP_D = RSP_D - RSP_D.mean()
                    ax_rsp_d.scatter(RSP_D_T, RSP_D, label=f"Participant {ID}", s=1)
                    ax_rsp_d.set_title(f"Diaphragmatic Respiration - Participant {ID} Group:{Group}")
                    ax_rsp_d.set_xlabel("Time (s)")
                    ax_rsp_d.set_ylabel("RSP_D")
                    ax_rsp_d.set_ylim([-8, 8])  # Set y-axis limits to 500-1200 ms for consistent RR interval scaling

                    # Add task regions to the RSP_D plot
                    for _, trigger in Trigger_df.iterrows():
                        start_time = trigger["Start"]
                        end_time = trigger["End"]
                        task_name = trigger["Task"]

                        if pd.notna(end_time):
                            if task_name in ['CB_easy', 'CB_hard', 'PA_easy', 'PA_medium', 'PA_hard', 'TC_easy', 'TC_hard']:
                                ax_rsp_d.axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                            else:
                                ax_rsp_d.axvspan(start_time, end_time, color='#C3E6CB', alpha=0.3)
                        else:
                            ax_rsp_d.axvline(start_time, color='green', linestyle='--')

                    ax_rsp_d.legend(loc='upper right')  # Specify legend location instead of 'best'
                    plt.tight_layout()
                    plt.savefig(f"{rsp_d_path}\\RSP_D_P{ID}.png", dpi=300)
                    print(fr"RSP_D plot {ID}")
                    plt.close()

                # 3. Separate RSP_C Plot
                if RSP_C_plot:
                    fig_rsp_c = plt.figure(figsize=(12, 6))
                    ax_rsp_c = fig_rsp_c.add_subplot(111)
                    RSP_C=RSP_C-RSP_C.mean()
                    ax_rsp_c.scatter(RSP_C_T, RSP_C, label=f"Participant {ID}", s=1)
                    ax_rsp_c.set_title(f"Chest Respiration - Participant {ID} Group:{Group}")
                    ax_rsp_c.set_xlabel("Time (s)")
                    ax_rsp_c.set_ylabel("RSP_C")
                    ax_rsp_c.set_ylim([-8, 8])  # Set y-axis limits to 500-1200 ms for consistent RR interval scaling

                    # Add task regions to the RSP_C plot
                    for _, trigger in Trigger_df.iterrows():
                        start_time = trigger["Start"]
                        end_time = trigger["End"]
                        task_name = trigger["Task"]

                        if pd.notna(end_time):
                            if task_name in ['CB_easy', 'CB_hard', 'PA_easy', 'PA_medium', 'PA_hard', 'TC_easy', 'TC_hard']:
                                ax_rsp_c.axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                            else:
                                ax_rsp_c.axvspan(start_time, end_time, color='#C3E6CB', alpha=0.3)
                        else:
                            ax_rsp_c.axvline(start_time, color='green', linestyle='--')

                    ax_rsp_c.legend(loc='upper right')  # Specify legend location instead of 'best'
                    plt.tight_layout()
                    plt.savefig(f"{rsp_c_path}\\RSP_C_P{ID}.png", dpi=300)
                    print(fr"RSP_C plot {ID}")
                    plt.close()

                # 4. Separate RR Plot
                if RR_plot:
                    fig_rr = plt.figure(figsize=(15, 3))  # width 15″, height 3″
                    ax_rr = fig_rr.add_subplot(111)
                    ax_rr.scatter(RR['Time'], RR['RR'], label=f"Participant {ID}",
                                  s=10)  # Increased point size to 25 for better visibility
                    # Set y-axis limits to 500-1200 ms for consistent RR interval scaling
                    ax_rr.set_title(f"RR Intervals - Participant {ID} Group:{Group}" )
                    ax_rr.set_xlabel("Time (s)")
                    ax_rr.set_ylabel("RR Interval (ms)")
                    ax_rr.set_ylim([400, 1500])  # Set y-axis limits to 500-1200 ms for consistent RR interval scaling

                    # Add task regions to the RR plot
                    for _, trigger in Trigger_df.iterrows():
                        start_time = trigger["Start"]
                        end_time = trigger["End"]
                        task_name = trigger["Task"]

                        if pd.notna(end_time):
                            if task_name in ['CB_easy', 'CB_hard', 'PA_easy', 'PA_medium', 'PA_hard', 'TC_easy', 'TC_hard']:
                                ax_rr.axvspan(start_time, end_time, color='#AEC6CF', alpha=0.3)
                            else:
                                ax_rr.axvspan(start_time, end_time, color='#C3E6CB', alpha=0.3)
                        else:
                            ax_rr.axvline(start_time, color='green', linestyle='--')

                    ax_rr.legend(loc='upper right')
                    plt.tight_layout()
                    plt.savefig(f"{rr_path}\\RR_P{ID}.png", dpi=300)
                    print(fr"RR plot {ID}")
                    plt.close()

            if Cor_plot:
                # ──────────────────────────── 1.  BEFORE the participant loop ─────────────────────────

                # ── constants for this analysis -------------------------------------------------------
                window, overlap = 30, 0.0
                base_dir = fr"{self.path}\Participants\{Group}_group\P_{ID}"
                hrv_file = fr"{base_dir}\Features\HRV\HRV_Time_{window}_{overlap}.csv"

                if not os.path.exists(hrv_file):
                    print(f"[WARN] HRV file not found for P_{ID}: {hrv_file}")
                    continue

                # 0) read + drop unneeded columns ------------------------------------------------------
                hrv_df = (pd.read_csv(hrv_file)
                          .drop(columns=["Time", "Class"], errors="ignore"))

                # 1) aggregates ------------------------------------------------------------------------
                stress_df = (hrv_df
                             .dropna(subset=["Stress"])
                             .groupby(["ID", "Group", "Stress"], as_index=False)
                             .mean()
                             .drop(columns=["Fatigue"], errors="ignore"))

                fatigue_df = (hrv_df.dropna(subset=["Fatigue"])
                              .groupby(["ID", "Group", "Stress"], as_index=False)
                              .mean()
                              .drop(columns=["Stress"], errors="ignore"))
                stress_all = pd.concat([stress_all, stress_df], ignore_index=True)
                fatigue_all = pd.concat([fatigue_all, fatigue_df], ignore_index=True)
                fatigue_df=fatigue_df.drop(columns=['ID','Group'])
                stress_df=stress_df.drop(columns=['ID','Group'])
                # keep Stress rows for the big scatter later
                tmp_stress = stress_df.copy()
                tmp_stress["ID"] = ID
                tmp_stress["Group"] = Group

                # ── iterate over Stress + Fatigue -----------------------------------------------------
                targets = [("Stress", stress_df), ("Fatigue", fatigue_df)]
                plots_dir_stress = os.path.join(base_dir, r"Features\HRV\Cor\Plot\Stress")
                plots_dir_fatigue = os.path.join(base_dir, r"Features\HRV\Cor\Plot\Fatighe")
                plots_dir_stress_analysis = r"C:\Users\e3bom\Desktop\Human Bio Signals Analysis\Analysis\RR\Plot\cor\Stress"
                plots_dir_fatigue_analysis = r"C:\Users\e3bom\Desktop\Human Bio Signals Analysis\Analysis\RR\Plot\cor\Fatigue"

                os.makedirs(plots_dir_stress, exist_ok=True)
                os.makedirs(plots_dir_fatigue, exist_ok=True)
                os.makedirs(plots_dir_stress_analysis, exist_ok=True)
                os.makedirs(plots_dir_fatigue_analysis, exist_ok=True)

                corr_rows_participant = []  # for the per‑participant Excel sheet

                for y_label, tbl in targets:
                    if tbl.empty:
                        print(f"[WARN] {y_label} table is empty for P_{ID}; skipping plot")
                        continue

                    feature_cols = [c for c in tbl.columns if c != y_label]
                    n_feats = len(feature_cols)

                    fig, axes = plt.subplots(1, n_feats,
                                             figsize=(3 * n_feats, 4),
                                             sharey=True)
                    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

                    for ax, feat in zip(axes, feature_cols):
                        x = tbl[feat].values
                        y = tbl[y_label].values

                        # scatter
                        ax.scatter(x, y, alpha=0.8)

                        # regression line & correlation
                        slope, intercept, r, p, _ = linregress(x, y)
                        x_line = np.linspace(x.min(), x.max(), 100)
                        ax.plot(x_line, intercept + slope * x_line, linestyle="--")

                        ax.set_title(f"{feat}  (r = {r:.2f})")
                        ax.set_xlabel("mean value")
                        ax.grid(True)

                        # save correlation record
                        corr_rows_participant.append(
                            {"ID": ID, "Group": Group, "Target": y_label,
                             "Feature": feat, "r": r, "p": p}
                        )
                    plot_path = fr'{directory}\\plots\\Trigger_Plot_{ID}_vas.png'
                    axes[0].set_ylabel(y_label)
                    fig.suptitle(f"P{ID} {Group}– {y_label} vs HRV features (window {window}s)",
                                 fontsize=14)
                    fig.tight_layout(rect=[0, 0, 1, 0.96])
                    if y_label=='Fatigue':
                        plot_path1 = os.path.join(plots_dir_fatigue, f"{y_label}_vs_HRV_P{ID}_{window}s.png")
                        plot_path2= os.path.join(plots_dir_fatigue_analysis, f"{y_label}_vs_HRV_P{ID}_{window}s.png")
                    else:
                        plot_path1 = os.path.join(plots_dir_stress, f"{y_label}_vs_HRV_P{ID}_{window}s.png")
                        plot_path2= os.path.join(plots_dir_stress_analysis, f"{y_label}_vs_HRV_P{ID}_{window}s.png")

                    fig.savefig(plot_path1, dpi=300, bbox_inches="tight")
                    fig.savefig(plot_path2, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    print(f"{ID}")

                # ── write participant‑level correlation Excel sheet ------------------------------
                corr_dir = os.path.join(base_dir, r"Features\HRV\Cor")
                os.makedirs(corr_dir, exist_ok=True)
                pd.DataFrame(corr_rows_participant).to_excel(
                    os.path.join(corr_dir, f"corr_P{ID}_{window}s.xlsx"), index=False
                )
                # also extend the global master list
                all_corr_records.extend(corr_rows_participant)

                # ─────────────────────────── END participant block ──────────────────────────────

                # ──────────────────────────── 3.  AFTER the participant loop ─────────────────────────
                # (Place this once, at the very bottom of AX_plot_signals_VAS)

        # ── 3a. master correlation workbook ──────────────────────────────────────────────
        master_corr_df = pd.DataFrame(all_corr_records)
        master_excel = os.path.join(
            self.path, r"Participants\All_HRV_Correlations_30s.xlsx"
        )
        # ── 1. pick the numeric column you want on the y-axis ───────────────
        Y = "r"  # or "p" if you prefer p-values
        # ── 2. basic box-plot, grouped by Feature and coloured by Target ────
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(data=master_corr_df, x="Feature", y=Y, hue="Target",
                         palette="Set2", width=0.6, fliersize=2)
        # cosmetic touches
        ax.axhline(0, color="gray", lw=0.8, ls="--")  # reference line for r=0
        ax.set_xlabel("")  # feature names already below
        ax.set_ylabel("Pearson r" if Y == "r" else "p-value")
        ax.set_title("Distribution of correlations per HRV feature")
        ax.legend(title="Target", loc="upper right")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.show()
        master_corr_df.to_excel(master_excel, index=False)
        print(f"[INFO] master correlation workbook saved → {master_excel}")

        # ── 3b. build Stress scatter matrices ────────────────────────────────────────────
        master_excel_stress = os.path.join(
            self.path, r"Participants\All_HRV_stress_30s.xlsx"
        )
        stress_all.to_excel(master_excel_stress, index=False)
        print(f"[INFO] master correlation workbook saved → {master_excel}")

        # ── 3b. build Fatigue scatter matrices ────────────────────────────────────────────
        master_excel_fatigue = os.path.join(
            self.path, r"Participants\All_HRV_fatigue_30s.xlsx"
        )
        fatigue_all.to_excel(master_excel_fatigue, index=False)
        print(f"[INFO] master correlation workbook saved → {master_excel}")

    def HRV_Window_Feature(self, ID,rangeID):

        # ── 1. Load participant table ────────────────────────────────────
        part_csv = fr"{self.path}\Participants\participation management.csv"
        part_df = (pd.read_csv(part_csv)
                   .dropna(axis=1, how="all")
                   .dropna(subset=["participant", "Date", "departmant"], how="all"))
        part_df["code"] = pd.to_numeric(part_df["code"], errors="coerce").astype("Int64")

        if ID is not None:
            if rangeID:
                part_df = part_df[part_df['code'] >= ID]
            else:
                part_df = part_df[part_df['code'] == ID]

        # ── 2. Master record holder ──────────────────────────────────────
        big_records: list[dict] = []

        # ── 3. Iterate over participants ────────────────────────────────
        for _, row in part_df.iterrows():
            ID = row["code"]
            Group = row["Group"]

            # 3.1  Paths
            base_path = fr"{self.path}\Participants\{Group}_group\P_{ID}"
            feature_dir = os.path.join(base_path, "Features", "HRV")
            time_plot_dir = os.path.join(base_path, "plots", "HRV_TIME_WINDOW")
            hist_plot_dir = os.path.join(base_path, "plots", "HRV_TIME_HIST")
            os.makedirs(time_plot_dir, exist_ok=True)
            os.makedirs(hist_plot_dir, exist_ok=True)

            fscore_records: list[dict] = []

            # 3.2  Loop over *every* HRV window file
            for fname in os.listdir(feature_dir):
                if not (fname.startswith("HRV_Time_") and fname.endswith(".csv")):
                    continue

                csv_path = os.path.join(feature_dir, fname)
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue

                # Parse window / overlap from file name: HRV_Time_<win>s_<ovlp>percent.csv
                parts = fname.replace(".csv", "").split("_")
                window = int(parts[2].rstrip("s"))
                overlap = parts[3].rstrip("percent")

                # ── 3.2.1  Class label normalisation ───────────────────
                if "Class" not in df.columns:
                    continue
                classified = df.dropna(subset=["Class"]).copy()
                rest_mask = classified["Class"].str.contains(
                    r"^(breath|natural|music)", case=False, na=False
                )
                classified.loc[rest_mask, "Class"] = "rest"
                classified = classified[classified["Class"].isin(["rest", "test"])]
                if classified.empty:
                    continue

                # ── 3.2.2  Histogram plot ──────────────────────────────
                hrv_feats = [
                    "HRV_CVNN", "HRV_RMSSD", "HRV_SDNN",
                    "HRV_MeanNN", "HRV_pNN20", "HRV_pNN50"
                ]
                n_feats = len(hrv_feats)
                n_cols = 2
                n_rows = int(np.ceil(n_feats / n_cols))

                hist_subdir = os.path.join(hist_plot_dir, f"{window}s_{overlap}")
                os.makedirs(hist_subdir, exist_ok=True)

                fig, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(12, 4 * n_rows),
                                         constrained_layout=True)
                axes = axes.flatten()

                for ax, feat in zip(axes, hrv_feats):
                    rest_vals = classified.loc[classified["Class"] == "rest", feat].dropna()
                    test_vals = classified.loc[classified["Class"] == "test", feat].dropna()

                    if rest_vals.empty and test_vals.empty:
                        ax.axis("off")
                        continue

                    ax.hist(rest_vals, bins=30, alpha=0.6, density=True, label="rest")
                    ax.hist(test_vals, bins=30, alpha=0.6, density=True, label="test")
                    ax.set_title(feat)
                    ax.legend()

                fig.suptitle(
                    f"Participant P_{ID}\nWindow: {window}s   Overlap: {overlap}",
                    fontsize=14,
                    ha="center"
                )

                out_png = os.path.join(
                    hist_subdir,
                    f"P_{ID}_{window}s_{overlap}_hist.png"
                )
                fig.savefig(out_png, dpi=300)
                plt.close(fig)

                # ── 3.2.3  F-score calculation ─────────────────────────
                for feat in hrv_feats:
                    rest = classified.loc[classified["Class"] == "rest", feat].dropna()
                    test = classified.loc[classified["Class"] == "test", feat].dropna()
                    if rest.empty or test.empty:
                        continue

                    diff2 = (rest.mean() - test.mean()) ** 2
                    var_sum = rest.var(ddof=1) + test.var(ddof=1)
                    f_val = diff2 / var_sum if var_sum else np.nan

                    fscore_records.append({
                        "Participant": ID,
                        "Group": Group,
                        "Window_s": window,
                        "Overlap_pc": overlap,
                        "Feature": feat,
                        "F_score": f_val
                    })

            # 3.3  Per-participant Excel --------------------------------
            if fscore_records:
                p_df = pd.DataFrame(fscore_records).sort_values("F_score", ascending=False)
                p_xlsx = os.path.join(base_path, f"F_scores_P{ID}.xlsx")
                p_df.to_excel(p_xlsx, index=False)
                print(f"✅  F-scores saved → {p_xlsx}")

                big_records.extend(fscore_records)

        # ── 4. Master workbook -------------------------------------------
        if big_records:
            big_df = pd.DataFrame(big_records).sort_values(
                ["Window_s", "Overlap_pc", "F_score"],
                ascending=[True, True, False])

            summary = (big_df
                       .groupby(["Window_s", "Overlap_pc"], as_index=False)["F_score"]
                       .mean()
                       .rename(columns={"F_score": "F_score_mean"})
                       .sort_values(["Window_s", "Overlap_pc"]))

            master_xlsx = os.path.join(self.path, "Participants", "F_scores_All.xlsx")
            with pd.ExcelWriter(master_xlsx, engine="openpyxl") as writer:
                big_df.to_excel(writer, sheet_name="Raw", index=False)
                summary.to_excel(writer, sheet_name="Summary", index=False)

            print(f"✅  Master workbook saved → {master_xlsx}")
        else:
            print("⚠️  No F-scores produced – check your input files.")

    # -------------------------------------------------------------------
    def RSP_Parts(self,ID,Group):
        Participants_path = f'{self.path}\Participants\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')
        if ID is not None:
            Participants_df = Participants_df[Participants_df['code'] == ID]
        for _, row in Participants_df.iterrows():
            ID = row['code']
            Group = row['Group']
            directory = fr'{self.path}\Participants\{Group}_group\P_{ID}'
            directory_Breath = fr'{self.path}\Participants\{Group}_group\P_{ID}\plot\breath'

            Trigger_path = fr'{directory}\Trigger_{ID}.csv'
            Trigger_df = pd.read_csv(Trigger_path, header=0)

            BioPac_path = fr'{directory}\P_{ID}.acq'
            BioPac = bioread.read_file(BioPac_path)
            #RSP!!!!!!!!!!!!!!
            #New


    def HRV_Window_Feature_all(self):
            window_sizes = [5, 10, 30, 60]  # seconds
            overlaps = [0.0, 0.5]  # no overlap and 50%

            time_window_hist_dir = fr'D:\Human Bio Signals Analysis\Participants\Dataset\HRV_TIME_HIST'
            os.makedirs(time_window_hist_dir, exist_ok=True)

            for window_size in window_sizes:
                for overlap in overlaps:
                    try:
                        dataset_path = fr'D:\Human Bio Signals Analysis\Participants\Dataset\HRV_AllParticipants_{window_size}s_{int(overlap * 100)}percent.csv'
                        results = pd.read_csv(dataset_path)

                        plot_base = fr'D:\Human Bio Signals Analysis\Participants\Dataset\Plots\{window_size}s_{int(overlap * 100)}percent'
                        hist_dir = os.path.join(plot_base, "HRV_TIME_HIST")
                        os.makedirs(hist_dir, exist_ok=True)

                        hrv_features = ['HRV_CVNN', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_MeanNN', 'HRV_pNN20', 'HRV_pNN50']

                        fig, ax = plt.subplots(len(hrv_features), figsize=(12, 20))
                        for i, feature in enumerate(hrv_features):
                            if feature not in results.columns:
                                continue

                            rest_data = results[results["Class"].isin(["breath", "music", "natural"])][feature].dropna()
                            test_data = results[results["Class"] == "test"][feature].dropna()

                            ax[i].hist(rest_data, bins=30, alpha=0.6, label="Rest", density=True)
                            ax[i].hist(test_data, bins=30, alpha=0.6, label="Test", density=True)
                            ax[i].set_title(
                                f"{feature} Histogram\nWindow={window_size}s, Overlap={int(overlap * 100)}%")
                            ax[i].set_xlabel(feature)
                            ax[i].set_ylabel("Density")
                            ax[i].legend()

                        plt.tight_layout()
                        hist_path = os.path.join(hist_dir,
                                                 f"HRV_Histogram_win{window_size}s_overlap{int(overlap * 100)}.png")
                        plt.savefig(hist_path, dpi=300)
                        plt.close()

                        print(f"✅ Saved histogram for {window_size}s / {int(overlap * 100)}%: {hist_path}")

                    except Exception as e:
                        print(f"❌ Error for {window_size}s, {overlap * 100:.0f}%: {e}")

    def HRV_Window_2Features(self, ID, Group):
        Participants_path = f'{self.path}\Participants\participation management.csv'
        Participants_df = pd.read_csv(Participants_path, header=0)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df = Participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')

        if ID is not None:
            Participants_df = Participants_df[Participants_df['code'] == ID]

        for _, row in Participants_df.iterrows():
            ID = row['code']
            Group = row['Group']
            directory = fr'{self.path}\Participants\{Group}_group\P_{ID}'

            Trigger_path = fr'{directory}\Trigger_{ID}.csv'
            Trigger_df = pd.read_csv(Trigger_path, header=0)

            BioPac_path = fr'{directory}\P_{ID}.acq'
            BioPac = bioread.read_file(BioPac_path)

            EDA = pd.read_csv(fr'{directory}\EDA.csv')
            eda_data = EDA['EDA']

            sample_rate = BioPac.named_channels['ECG'].samples_per_second
            part_data = BioPac.named_channels['ECG'].data

            window_sizes = [5, 10, 30, 60]  # seconds
            overlaps = [0.0, 0.5]  # no overlap and 50%
            # window_sizes = [10]  # seconds
            # overlaps = [0.5]  # no overlap and 50%
            time_window_plot_dir = fr'{directory}\plots\HRV_TIME_scatter'
            os.makedirs(time_window_plot_dir, exist_ok=True)

            def label_window(time, trigger_df):
                for _, row in trigger_df.iterrows():
                    if pd.notna(row["End"]) and row["Task"].startswith("Breath"):
                        if row["Start"] <= time <= row["End"]:
                            return "breath"
                    elif pd.notna(row["End"]) and not row["Task"].startswith("VAS") and not row["Task"].startswith(
                            "Breath"):
                        if row["Start"] <= time <= row["End"]:
                            return "test"
                return None

            for window_size in window_sizes:
                for overlap in overlaps:
                    results = pd.DataFrame()
                    window_samples = int(window_size * sample_rate)
                    step = int(window_samples * (1 - overlap))

                    for j, i in enumerate(range(0, len(part_data) - window_samples + 1, step)):
                        segment_data = part_data[i:i + window_samples]
                        try:
                            ecg_cleaned = nk.ecg_clean(segment_data, sampling_rate=sample_rate)
                            _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sample_rate)
                            if len(rpeaks["ECG_R_Peaks"]) > 2:
                                hrv = nk.hrv_time(rpeaks, sampling_rate=sample_rate, show=False)
                                center_time = (i + window_samples / 2) / sample_rate
                                new_row = {
                                    'Time': center_time,
                                    'HRV_CVNN': hrv.get('HRV_CVNN', [None])[0],
                                    'HRV_RMSSD': hrv.get('HRV_RMSSD', [None])[0],
                                    'HRV_SDNN': hrv.get('HRV_SDNN', [None])[0],
                                    'HRV_MeanNN': hrv.get('HRV_MeanNN', [None])[0],
                                    'HRV_pNN20': hrv.get('HRV_pNN20', [None])[0],
                                    'HRV_pNN50': hrv.get('HRV_pNN50', [None])[0]
                                }
                                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
                        except Exception as e:
                            print(f"Error in window {j}: {e}")


                    # Plot HRV time series
                    hrv_features = ['HRV_CVNN', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_MeanNN', 'HRV_pNN20', 'HRV_pNN50']

                    # Label and create histogram for this setting
                    results['Class'] = results['Time'].apply(lambda t: label_window(t, Trigger_df))
                    results.to_csv(fr'{directory}\Features\HRV_Time_{window_size}_{overlap}.csv')
                    classified = results.dropna(subset=['Class'])
                    # Define color mapping
                    color_map = {'breath': 'blue', 'test': 'orange'}

                    # Drop rows with missing class values
                    filtered_results = results.dropna(subset=['Class'])

                    # Map colors
                    colors = filtered_results['Class'].map(color_map)

                    # Create scatter plot
                    plt.scatter(filtered_results['HRV_CVNN'], filtered_results['HRV_RMSSD'], c=colors)

                    # Add labels and show plot
                    plt.xlabel('HRV_CVNN')
                    plt.ylabel('HRV_RMSSD')
                    plt.title('HRV_CVNN vs HRV_RMSSD by Class')
                    plt.savefig(fr'{time_window_plot_dir}\Scatter_{window_size}_{overlap}.png')
                    plt.show()

    def MissingData(self, ID, rangeID):
        data_path = fr'{self.path}\Participants\Dataset\Dataset_By_Window\Raw_Data\Dataset_total.csv'
        summary_0_1_path = fr'{self.path}\Participants\Dataset\Missing_Data\Missing_Data_0_1.csv'
        summary_path = fr'{self.path}\Participants\Dataset\Missing_Data\Missing_Data_no_0_1.csv'
        cleaned_output_folder = fr'{self.path}\Participants\Dataset\Dataset_By_Window\Clean_Data'
        cols_to_drop_precent=0.15
        NaN_Path = fr'{self.path}\Participants\Dataset\Missing_Data\NaN_prec{cols_to_drop_precent}.csv'
        nan_summary_rows = []  # initialize list outside the loop
        os.makedirs(fr'{self.path}\Participants\Dataset\Missing_Data', exist_ok=True)
        os.makedirs(cleaned_output_folder, exist_ok=True)

        data_df = pd.read_csv(data_path)
        data_df_group=data_df.copy()
        # Columns to exclude from missing frequency calculation
        exclude_cols = ['Class', 'Test_Type', 'Level', 'Accuracy', 'RT', 'Stress', 'Fatigue',
                        'ID', 'Group', 'Time']
        rows = []

        # Step 1: Calculate missing frequencies by window and overlap
        for (window, overlap), group in data_df_group.groupby(["Window", "Overlap"]):
            group_clean = group.drop(columns=[col for col in exclude_cols if col in group.columns])
            nan_frequencies = group_clean.isna().sum() / len(group_clean)
            nan_frequencies["Window"] = window
            nan_frequencies["Overlap"] = overlap
            rows.append(nan_frequencies)

        result_df = pd.DataFrame(rows)
        columns = ['Window', 'Overlap'] + [col for col in result_df.columns if col not in ['Window', 'Overlap']]
        result_df = result_df[columns]
        result_df.to_csv(fr'{self.path}\Participants\Dataset\Missing_Data\Missing_Data.csv', index=False)

        # Step 2: Drop constant binary columns (all 1s or all 0s)
        df = result_df.copy()
        summary_rows = []
        for col in df.columns:
            col_data = df[col].dropna()
            unique_vals = col_data.unique()
            if len(unique_vals) == 1 and unique_vals[0] in [0, 1]:
                summary_rows.append({
                    "Column": col,
                    "Reason": f"All {int(unique_vals[0])}"
                })

        binary_constant_cols = [row["Column"] for row in summary_rows]
        df_clean = df.drop(columns=binary_constant_cols)
        df_clean.to_csv(summary_path, index=False)

        summary_df = pd.DataFrame(summary_rows).sort_values(by="Reason", ascending=True).reset_index(drop=True)
        summary_df.to_csv(summary_0_1_path, index=False)

        # Step 3: Drop high-missing columns (>15%) per window-overlap and save cleaned files
        for _, row in result_df.iterrows():
            window = row['Window']
            overlap = row['Overlap']
            subset = data_df[(data_df['Window'] == window) & (data_df['Overlap'] == overlap)]

            # Find columns with more than 15% missing
            cols_to_drop = row.drop(labels=['Window', 'Overlap'])
            cols_to_drop = cols_to_drop[cols_to_drop > cols_to_drop_precent].index.tolist()
            cleaned_subset = subset.drop(columns=cols_to_drop)
            # Count rows before dropping NaNs
            Rows_Before_NaN = len(cleaned_subset)

            # Drop rows with any NaNs
            # Drop rows with NaNs in feature columns only (not in excluded meta columns)
            feature_cols = [col for col in cleaned_subset.columns if col not in exclude_cols]
            cleaned_subset = cleaned_subset.dropna(subset=feature_cols)
            # Count rows after dropping NaNs
            Rows_After_NaN = len(cleaned_subset)

            # Calculate percentage of rows removed
            percent_NaN = (Rows_Before_NaN - Rows_After_NaN) / Rows_Before_NaN if Rows_Before_NaN > 0 else 0

            nan_summary_rows.append({
                "Window": window,
                "Overlap": overlap,
                "Rows_Before_NaN": Rows_Before_NaN,
                "Rows_After_NaN": Rows_After_NaN,
                "percent_NaN (%)": round(percent_NaN, 2),
            })

            filename = f"Dataset_{int(window)}s_{int(overlap * 100)}.csv"
            filepath = os.path.join(cleaned_output_folder, filename)

            cleaned_subset.to_csv(filepath, index=False)
        nan_df = pd.DataFrame(nan_summary_rows)
        nan_df.to_csv(NaN_Path)
        print("Finished generating cleaned datasets with missing-value thresholding.")
        return result_df

    def unite_datasets(self, input_folder, output_filename="Dataset_total.csv"):
        all_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
        all_dfs = []

        for file in all_files:
            file_path = os.path.join(input_folder, file)
            try:
                df = pd.read_csv(file_path, low_memory=False)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        Dataset_total = pd.concat(all_dfs, ignore_index=True)
        Dataset_total=Dataset_total.sort_values(by=['Window','Overlap','ID'])
        output_path = os.path.join(input_folder, output_filename)
        Dataset_total.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path}")

    def CreateDataset(self, ID=None, rangeID=False):
        def label_window(t):
            for _, tr in Trigger_df.iterrows():
                task = str(tr["Task"]).lower()
                if pd.notna(tr["End"]) and tr["Start"] <= t <= tr["End"]:
                    if "breath" in task:
                        return "breath", None, None
                    if "music" in task:
                        return "music", None, None
                    if "baseline" in task or "break" in task:
                        return "natural", None, None
                    if not task.startswith("vas"):
                        parts = task.split("_")
                        test_type = parts[0].upper() if len(parts) > 0 else None
                        level = parts[1].lower() if len(parts) > 1 else None
                        return "test", level, test_type
            return None, None, None
        def impute_features(df, ws, ol, missing_threshold=0.15, verbose=True, model_type='linear'):
            delete_missing_values = True
            df = df.copy()
            excluded_columns = ['ID', 'Group', 'Time', 'Stress', 'Fatigue', 'Class', 'Accuracy', 'RT',
                                'Test_Type', 'Level']
            numeric_cols = [col for col in df.columns if col not in excluded_columns]

            # Step 1: Drop columns with too many missing values
            # שלב 1: בדיקת חוסרים וצבירת עמודות שיש להן יותר מהסף
            cols_to_drop = []
            missing_ratios = {}

            for col in numeric_cols:
                missing_ratio = df[col].isna().mean()
                if missing_ratio > missing_threshold:
                    cols_to_drop.append(col)
                    if verbose:
                        print(f"❌ Dropping column '{col}' – {missing_ratio:.1%} missing values")
                missing_ratios[col] = missing_ratio  # נרשום בכל מקרה

            # שלב 2: יצירת מילון עם missing_ratio אם העמודה הושמטה, אחרת 0
            drop_status = {
                col: missing_ratios[col] if col in cols_to_drop else 0
                for col in numeric_cols
            }

            # שלב 3: יצירת DataFrame שורה אחת
            drop_status_df = pd.DataFrame([drop_status])
            drop_status_df.insert(0, 'ID', df['ID'])
            drop_status_df.insert(1, 'Group', df['Group'])
            drop_status_df.insert(1, 'Window', ws)
            drop_status_df.insert(2, 'Overlap', ol)

            df.drop(columns=cols_to_drop, inplace=True)
            numeric_cols = [col for col in numeric_cols if col not in cols_to_drop]

            if len(numeric_cols) == 0:
                if verbose:
                    print("⚠️ No columns left to impute")
                    df_nan = df[df[numeric_cols].isna().any(axis=1)]
                    df_nan.insert(1, 'Window', ws)
                    df_nan.insert(2, 'Overlap', ol)
                    df_nan.insert(3, 'Frec', len(df_nan) / len(df))
                    df_nan = df_nan.drop(
                        columns=['Class', 'Accuracy', 'RT', 'Stress', 'Fatigue', 'Test_Type', 'Level'])
                return df, drop_status_df, df_nan

            # Step 2: Row-wise imputation
            model_cls = RandomForestRegressor if model_type == 'rf' else LinearRegression
            df_nan = df[df[numeric_cols].isna().any(axis=1)]
            df_nan.insert(1, 'Window', ws)
            df_nan.insert(2, 'Overlap', ol)
            df_nan.insert(3, 'Frec', len(df_nan) / len(df))
            df_nan = df_nan.drop(
                columns=['Class', 'Accuracy', 'RT', 'Stress', 'Fatigue', 'Test_Type', 'Level'])
            predicted_count = 0
            rows_to_drop = []

            for idx, row in df_nan.iterrows():
                target_cols = row[numeric_cols][row[numeric_cols].isna()].index.tolist()
                available_features = row[numeric_cols][row[numeric_cols].notna()].index.tolist()
                for target_col in target_cols:
                    if verbose:
                        print(f"\n🔧 Row {idx}: Predicting '{target_col}' using {available_features}")

                    if not available_features:
                        if delete_missing_values:
                            rows_to_drop.append(idx)
                        else:
                            df.loc[idx, target_col] = \
                                df[target_col].interpolate(method='linear', limit_direction='both').loc[idx]
                        df_nan.loc[idx, target_col] = 'interpolate'
                        continue  # No features to use

                    # Build training set: rows where both target and available features are not NaN
                    train_mask = df[target_col].notna()
                    for feat in available_features:
                        train_mask &= df[feat].notna()

                    if train_mask.sum() < 3:
                        if verbose:
                            print(f"   ⚠️ Not enough data to predict '{target_col}' in row {idx}")
                        if delete_missing_values:
                            rows_to_drop.append(idx)
                        else:
                            df.loc[idx, target_col] = \
                                df[target_col].interpolate(method='linear', limit_direction='both').loc[idx]
                        df_nan.loc[idx, target_col] = 'interpolate'
                        continue

                    X_train = df.loc[train_mask, available_features]
                    y_train = df.loc[train_mask, target_col]

                    model = model_cls()
                    try:
                        model.fit(X_train, y_train)
                        X_test = row[available_features].values.reshape(1, -1)
                        prediction = model.predict(X_test)[0]
                        if delete_missing_values:
                            rows_to_drop.append(idx)
                        else:
                            df.loc[idx, target_col] = prediction
                        interpolation = \
                        df[target_col].interpolate(method='linear', limit_direction='both').loc[idx]
                        diff = prediction - interpolation
                        df_nan.loc[idx, target_col] = f'prediction_{diff:.3f}'
                        predicted_count += 1
                        if verbose:
                            print(f"   ✅ Predicted: {prediction:.4f}")
                    except Exception as e:
                        if verbose:
                            print(f"   ⚠️ Failed to predict '{target_col}' in row {idx}: {str(e)}")
                        if delete_missing_values:
                            rows_to_drop.append(idx)
                        else:
                            df.loc[idx, target_col] = \
                                df[target_col].interpolate(method='linear', limit_direction='both').loc[idx]
                        df_nan.loc[idx, target_col] = 'interpolate'
                        continue
            df = df.drop(index=rows_to_drop, errors='ignore')
            if verbose:
                print(f"\n🎉 Row-wise imputation completed. {predicted_count} values predicted.")

            return df, drop_status_df, df_nan
        def trigger_attrs(t):
            future_rows = rating_df[rating_df["Start"] >= t]
            if len(future_rows) < 2:
                future_rows = rating_df
            tr_s = future_rows.iloc[0] if not pd.isna(future_rows.iloc[0]["Stress"]) else future_rows.iloc[
                1]
            tr_f = future_rows.iloc[1] if not pd.isna(future_rows.iloc[0]["Stress"]) else future_rows.iloc[
                0]
            stress = tr_s.get("Stress", np.nan)
            fatigue = tr_f.get("Fatigue", np.nan)
            match = Performance_df[(Performance_df["Start"] <= t) & (Performance_df["End"] >= t)]
            accuracy = match["Accuracy_Mean"].values[0] if not match.empty else np.nan
            rt = match["RT_Mean"].values[0] if not match.empty else np.nan
            return stress, fatigue, accuracy, rt
        def compute_slope(signal):
            if len(signal) < 2 or np.all(np.isnan(signal)):
                return np.nan
            x = np.arange(len(signal))
            slope, _, _, _, _ = stats.linregress(x, signal)
            return slope
        window_sizes = [5,10,30,60]
        overlaps = [0,0.5]

        total_dataset_dir = fr'{self.path}\Participants\Dataset\Dataset_By_Window\Raw_Data'
        os.makedirs(total_dataset_dir, exist_ok=True)

        # ── Load participant table ─────────────
        participants_path = f'{self.path}\\Participants\\participation management.csv'
        participants_df = pd.read_csv(participants_path).dropna(axis=1, how='all')
        participants_df = participants_df.dropna(subset=['participant', 'Date', 'departmant'], how='all')
        participants_df['code'] = pd.to_numeric(participants_df['code'], errors='coerce').astype('Int64')

        if ID is not None:
            if rangeID:
                participants_df = participants_df[participants_df['code'] >= ID]
            else:
                participants_df = participants_df[participants_df['code'] == ID]

        Data_Total_df = pd.DataFrame()
        # ── Iterate window configs ──────────────────────
        for window_size in tqdm(window_sizes, desc=f" Window Sizes", leave=False):
            for overlap in tqdm(overlaps, desc=f"Overlap in {window_size}s ", leave=False):
                # Iterate over participants
                Data_df = pd.DataFrame()
                for _, row in tqdm(participants_df.iterrows(), desc=f"Participants", leave=False):
                    ID = row['code']
                    print(ID)
                    Group = row['Group']
                    directory = fr'{self.path}\Participants\{Group}_group\P_{ID}'


                    # ── Load data ─────────────────────────────────────
                    Trigger_df = pd.read_csv(fr'{directory}\Trigger_{ID}.csv')
                    Trigger_df["Stress"] = np.where(Trigger_df["Task"] == "VAS_Stress", Trigger_df["Score"],
                                                    np.nan)
                    Trigger_df["Fatigue"] = np.where(Trigger_df["Task"] == "VAS_Fatigue",
                                                     Trigger_df["Score"], np.nan)
                    rating_df = Trigger_df[
                        (~Trigger_df["Stress"].isna()) | (~Trigger_df["Fatigue"].isna())].sort_values(
                        "Start").reset_index(drop=True)
                    rating_df = rating_df.drop(columns=['Score', 'End'])

                    relevant_tasks = ['Task', 'CB_easy', 'CB_hard', 'PA_easy', 'PA_medium', 'PA_hard',
                                      'TC_easy',
                                      'TC_hard']
                    Performance_df = Trigger_df[Trigger_df['Task'].isin(relevant_tasks)]
                    Performance_df = Performance_df.drop(columns=['Score'], errors='ignore')

                    BioPac = bioread.read_file(fr'{directory}\P_{ID}.acq')
                    sample_rate = BioPac.named_channels['ECG'].samples_per_second
                    HRV_RR_data = pd.read_csv(fr'{directory}\HRV_RR.csv')
                    HRV_F_data = pd.read_csv(fr'{directory}\HRV_F.csv')
                    EDA_data = pd.read_csv(fr'{directory}\EDA.csv')
                    RSP_c_F = pd.read_csv(fr'{directory}\RSP_c_F.csv')
                    RSP_d_F = pd.read_csv(fr'{directory}\RSP_d_F.csv')
                    RSP_c_RR = pd.read_csv(fr'{directory}\RSP_c_RR.csv')
                    RSP_d_RR = pd.read_csv(fr'{directory}\RSP_d_RR.csv')
                    window_samples = int(window_size)
                    step = window_samples * (1 - overlap)
                    RSP_c_max_time = BioPac.named_channels['Chest Respiration'].time_index[-1]
                    for j, i in enumerate(tqdm(np.arange(0, RSP_c_max_time - window_samples + 1, step), desc=f"{ID} Windows {window_size}s/{overlap}")):
                        HRV_RR_window = HRV_RR_data.loc[
                            (HRV_RR_data['Time'] >= i) &
                            (HRV_RR_data['Time'] < (i + window_samples)),'HRV_RR'].values
                        HRV_F_window = HRV_F_data.loc[
                            (HRV_F_data['Time'] >= i) &
                            (HRV_F_data['Time'] < (i + window_samples))]
                        RSP_c_RR_window = RSP_c_RR.loc[
                            (RSP_c_RR['Time'] >= i ) &
                            (RSP_c_RR['Time'] < (i + window_samples))]
                        RSP_d_RR_window  = RSP_d_RR.loc[
                            (RSP_d_RR['Time'] >= i) &
                            (RSP_d_RR['Time'] < (i + window_samples))]
                        EDA_window = EDA_data.loc[
                            (EDA_data['Time'] >= i) &
                            (EDA_data['Time'] < (i + window_samples))]
                        RSP_c_F_window = RSP_c_F.loc[
                            (RSP_c_F['Time'] >= i) &
                            (RSP_c_F['Time'] < (i + window_samples))]
                        RSP_d_F_window = RSP_d_F.loc[
                            (RSP_d_F['Time'] >= i ) &
                            (RSP_d_F['Time'] < (i + window_samples))]

                        try:
                            if i == 0:
                                HRV_RR_window = HRV_RR_window[1:]
                                HRV_RR_window = np.array(HRV_RR_window, dtype=float)
                            else:
                                HRV_RR_window = np.array(HRV_RR_window, dtype=float)
                            rr_diff = np.diff(HRV_RR_window) if len(HRV_RR_window) > 1 else np.array([np.nan])
                            # Time-domain HRV features
                            hrv_features = {
                                'HRV_MeanNN': np.nanmean(HRV_RR_window) if len(HRV_RR_window) > 0 else np.nan,
                                'HRV_SDNN': np.nanstd(HRV_RR_window, ddof=1) if len(HRV_RR_window) > 1 else np.nan,
                                'HRV_RMSSD': np.sqrt(np.nanmean(rr_diff ** 2)) if len(rr_diff) > 0 else np.nan,
                                'HRV_CVNN': (
                                    np.nanstd(HRV_RR_window, ddof=1) / np.nanmean(HRV_RR_window) * 100
                                    if np.nanmean(HRV_RR_window) > 0 else np.nan
                                ),
                                'HRV_pNN20': (
                                    np.sum(np.abs(rr_diff) > 20) / len(rr_diff) * 100
                                    if len(rr_diff) > 0 else np.nan
                                ),
                                'HRV_pNN50': (
                                    np.sum(np.abs(rr_diff) > 50) / len(rr_diff) * 100
                                    if len(rr_diff) > 0 else np.nan
                                ),
                                'HRV_MedianNN': np.nanmedian(HRV_RR_window) if len(HRV_RR_window) > 0 else np.nan,
                                'HRV_IQRNN': (
                                    np.nanpercentile(HRV_RR_window, 75) - np.nanpercentile(HRV_RR_window, 25)
                                    if len(HRV_RR_window) > 0 else np.nan
                                )
                            }

                        except Exception as e:
                            rr_diff = None
                            hrv_features = {
                                'HRV_MeanNN': np.nan,
                                'HRV_SDNN': np.nan,
                                'HRV_RMSSD': np.nan,
                                'HRV_CVNN': np.nan,
                                'HRV_pNN20': np.nan,
                                'HRV_pNN50': np.nan,
                                'HRV_MedianNN': np.nan,
                                'HRV_IQRNN': np.nan
                            }
                            print(f"⚠️ HRV feature computation failed: {e}")

                        try:
                            hrv_welch = nk.hrv_frequency(HRV_F_window["ECG_R_Peaks"], sampling_rate=sample_rate, show=True, psd_method="welch")
                            hrv_features.update({
                                'HRV_LF': hrv_welch["HRV_LF"][0],
                                'HRV_HF': hrv_welch["HRV_HF"][0],
                                'HRV_HFn': hrv_welch["HRV_HFn"][0],
                                'HRV_LFn': hrv_welch["HRV_LFn"][0],
                                'HRV_LFHF': hrv_welch["HRV_LFHF"][0],
                                'HRV_LnHF': hrv_welch["HRV_LnHF"][0],
                                'HRV_TP': hrv_welch["HRV_TP"][0],
                                'HRV_VHF': hrv_welch["HRV_VHF"][0]
                            })
                        except:
                            hrv_features.update({
                                'HRV_VLF': np.nan,
                                'HRV_LF': np.nan,
                                'HRV_HF': np.nan,
                                'HRV_LF_HF': np.nan,
                                'HRV_TotalPower': np.nan,
                                'HRV_LF_Norm': np.nan,
                                'HRV_HF_Norm': np.nan
                            })

                        try:
                            tonic = EDA_window["EDA_Tonic"].values
                            phasic = EDA_window["EDA_Phasic"].values
                            eda_clean = EDA_window["EDA_Clean"].values
                            scr_peaks = np.where(EDA_window["EDA_SCR_Peaks"] == 1)[0]
                            scr_amplitude = np.nanmean(EDA_window["EDA_SCR_Amplitude"])

                            eda_features = {
                                'EDA_Tonic_Mean': np.nanmean(tonic),
                                'EDA_Tonic_Std': np.nanstd(tonic),
                                'EDA_Tonic_Slope': compute_slope(tonic),
                                'EDA_Phasic_Mean': np.nanmean(phasic),
                                'EDA_Phasic_Std': np.nanstd(phasic),
                                'EDA_Phasic_Slope': compute_slope(phasic),
                                'EDA_Clean_Median': np.nanmedian(eda_clean),
                                'EDA_Clean_Slope': compute_slope(eda_clean),
                                'EDA_SCR_Peaks_Count': len(scr_peaks),
                                'EDA_SCR_Amplitude_Mean': scr_amplitude
                            }

                        except Exception as e:
                            print(f"[ERROR] failed to extract from EDA_window: {e}")
                            eda_features = {key: np.nan for key in [
                                'EDA_Tonic_Mean', 'EDA_Tonic_Std', 'EDA_Tonic_Slope',
                                'EDA_Phasic_Mean', 'EDA_Phasic_Std', 'EDA_Phasic_Slope',
                                'EDA_Clean_Median', 'EDA_Clean_Slope',
                                'EDA_SCR_Peaks_Count', 'EDA_SCR_Amplitude_Mean'
                            ]}
                        try:
                            eda_sympathetic = nk.eda_sympathetic(EDA_window["EDA_Clean"], sampling_rate=sample_rate, method='ghiasi')
                            eda_features.update({
                                'EDA_Sympathetic': eda_sympathetic['EDA_Sympathetic']})

                        except:
                            eda_features.update({
                                'EDA_Sympathetic': np.nan})
                        try:
                            rsp_c_rrv = nk.rsp_rrv(RSP_c_RR_window['RSP_c_rsp_rate'], troughs=RSP_c_RR_window['RSP_c_Troughs'], sampling_rate=sample_rate)
                            rrv_c_features = {
                                'RSP_C_RRV_RMSSD': rsp_c_rrv.get("RRV_RMSSD", [np.nan])[0],
                                'RSP_C_RRV_MeanBB': rsp_c_rrv.get("RRV_MeanBB", [np.nan])[0],
                                'RSP_C_RRV_MedianBB': rsp_c_rrv.get("RRV_MedianBB", [np.nan])[0],
                                'RSP_C_RRV_MadBB': rsp_c_rrv.get("RRV_MadBB", [np.nan])[0],
                                'RSP_C_RRV_SD2': rsp_c_rrv.get("RRV_SD2", [np.nan])[0],
                                'RSP_C_RRV_SD1': rsp_c_rrv.get("RRV_SD1", [np.nan])[0],
                                'RSP_C_RRV_SD2SD1': rsp_c_rrv.get("RRV_SD2SD1", [np.nan])[0],
                                'RSP_C_RRV_ApEn': rsp_c_rrv.get("RRV_ApEn", [np.nan])[0],
                                'RSP_C_RRV_CVSD': rsp_c_rrv.get("RRV_CVSD", [np.nan])[0],
                                'RSP_C_RRV_CVBB': rsp_c_rrv.get("RRV_CVBB", [np.nan])[0],
                                'RSP_C_RRV_MCVBB': rsp_c_rrv.get("RRV_MCVBB", [np.nan])[0],
                                'RSP_C_RRV_LF': rsp_c_rrv.get("RRV_LF", [np.nan])[0],
                                'RSP_C_RRV_HF': rsp_c_rrv.get("RRV_HF", [np.nan])[0],
                                'RSP_C_RRV_LFHF': rsp_c_rrv.get("RRV_LFHF", [np.nan])[0],
                                'RSP_C_RRV_VLF': rsp_c_rrv.get("RRV_VLF", [np.nan])[0],
                                'RSP_C_RRV_HFn': rsp_c_rrv.get("RRV_HFn", [np.nan])[0],
                                'RSP_C_RRV_LFn': rsp_c_rrv.get("RRV_LFn", [np.nan])[0]
                            }

                        except Exception as e:
                            print(f"RRV_C Chest feature extraction failed: {e}")
                            rrv_c_features = {
                                'RSP_C_RRV_RMSSD': np.nan,
                                'RSP_C_RRV_MeanBB': np.nan,
                                'RSP_C_RRV_MedianBB': np.nan,
                                'RSP_C_RRV_MadBB': np.nan,
                                'RSP_C_RRV_SD2': np.nan,
                                'RSP_C_RRV_SD1': np.nan,
                                'RSP_C_RRV_SD2SD1': np.nan,
                                'RSP_C_RRV_ApEn': np.nan,
                                'RSP_C_RRV_CVSD': np.nan,
                                'RSP_C_RRV_CVBB': np.nan,
                                'RSP_C_RRV_MCVBB': np.nan,
                                'RSP_C_RRV_LF': np.nan,
                                'RSP_C_RRV_HF': np.nan,
                                'RSP_C_RRV_LFHF': np.nan,
                                'RSP_C_RRV_VLF': np.nan,
                                'RSP_C_RRV_HFn': np.nan,
                                'RSP_C_RRV_LFn': np.nan
                            }
                        try:
                            # --- מאפייני RVT ---
                            rvt_c_features = {
                                'RSP_C_RVT_Mean_BIRN': np.nanmean(RSP_c_F_window['RSP_c_rvt']),
                                'RSP_C_RVT_Median_BIRN': np.nanmedian(RSP_c_F_window['RSP_c_rvt']),
                                'RSP_C_RVT_Std_BIRN': np.nanstd(RSP_c_F_window['RSP_c_rvt']),
                            }
                        except Exception as e:
                            print(f"RVT error: {e}")
                            rvt_c_features = {
                                'RSP_C_RVT_Mean_BIRN': np.nan,
                                'RSP_C_RVT_Median_BIRN': np.nan,
                                'RSP_C_RVT_Std_BIRN': np.nan
                            }
                        try:
                            rsp_c_features = {
                                'RSP_C_Rate_Mean': RSP_c_F_window['RSP_c_Rate'].mean(),
                                'RSP_C_Rate_Median': RSP_c_F_window['RSP_c_Rate'].median(),
                                'RSP_C_Rate_Std': RSP_c_F_window['RSP_c_Rate'].std(),

                                'RSP_C_RVT_Mean': RSP_c_F_window['RSP_c_RVT'].mean(),
                                'RSP_C_RVT_Median': RSP_c_F_window['RSP_c_RVT'].median(),
                                'RSP_C_RVT_Std': RSP_c_F_window['RSP_c_RVT'].std(),

                                'RSP_C_PhaseCompletion_Mean': RSP_c_F_window['RSP_c_Phase_Completion'].mean(),
                                'RSP_C_PhaseCompletion_Median': RSP_c_F_window['RSP_c_Phase_Completion'].median(),

                                'RSP_C_Sym_PeakTrough_Mean': RSP_c_F_window['RSP_c_Symmetry_PeakTrough'].mean(),
                                'RSP_C_Sym_PeakTrough_Median': RSP_c_F_window['RSP_c_Symmetry_PeakTrough'].median(),

                                'RSP_C_Sym_RiseDecay_Mean': RSP_c_F_window['RSP_c_Symmetry_RiseDecay'].mean(),
                                'RSP_C_Sym_RiseDecay_Median': RSP_c_F_window['RSP_c_Symmetry_RiseDecay'].median(),

                                'RSP_C_Amplitude_Mean': RSP_c_F_window['RSP_c_Amplitude'].mean(),
                                'RSP_C_Amplitude_Median': RSP_c_F_window['RSP_c_Amplitude'].median(),
                                'RSP_C_Amplitude_Std': RSP_c_F_window['RSP_c_Amplitude'].std(),

                                'RSP_C_RSA_P2T_Mean': RSP_c_F_window['RSP_c_RSA_P2T'].mean(),
                                'RSP_C_RSA_P2T_Median': RSP_c_F_window['RSP_c_RSA_P2T'].median(),

                                'RSP_C_RSA_Gates_Mean': RSP_c_F_window['RSP_c_RSA_Gates'].mean(),
                                'RSP_C_RSA_Gates_Median': RSP_c_F_window['RSP_c_RSA_Gates'].median(),
                            }

                        except Exception as e:
                            print(f"RSP_c_features error: {e}")
                            rsp_c_features = {}

                        # חיבור כל המאפיינים של chest respiration
                        rsp_c_features.update(rrv_c_features)
                        rsp_c_features.update(rvt_c_features)

                        try:
                            rsp_d_rrv = nk.rsp_rrv(RSP_d_RR_window['RSP_d_rsp_rate'], troughs=RSP_d_RR_window['RSP_d_Troughs'], sampling_rate=sample_rate)
                            rrv_d_features = {
                                'RSP_D_RRV_RMSSD': rsp_d_rrv.get("RRV_RMSSD", [np.nan])[0],
                                'RSP_D_RRV_MeanBB': rsp_d_rrv.get("RRV_MeanBB", [np.nan])[0],
                                'RSP_D_RRV_MedianBB': rsp_d_rrv.get("RRV_MedianBB", [np.nan])[0],
                                'RSP_D_RRV_MadBB': rsp_d_rrv.get("RRV_MadBB", [np.nan])[0],
                                'RSP_D_RRV_SD2': rsp_d_rrv.get("RRV_SD2", [np.nan])[0],
                                'RSP_D_RRV_SD1': rsp_d_rrv.get("RRV_SD1", [np.nan])[0],
                                'RSP_D_RRV_SD2SD1': rsp_d_rrv.get("RRV_SD2SD1", [np.nan])[0],
                                'RSP_D_RRV_ApEn': rsp_d_rrv.get("RRV_ApEn", [np.nan])[0],
                                'RSP_D_RRV_CVSD': rsp_d_rrv.get("RRV_CVSD", [np.nan])[0],
                                'RSP_D_RRV_CVBB': rsp_d_rrv.get("RRV_CVBB", [np.nan])[0],
                                'RSP_D_RRV_MCVBB': rsp_d_rrv.get("RRV_MCVBB", [np.nan])[0],
                                'RSP_D_RRV_LF': rsp_d_rrv.get("RRV_LF", [np.nan])[0],
                                'RSP_D_RRV_HF': rsp_d_rrv.get("RRV_HF", [np.nan])[0],
                                'RSP_D_RRV_LFHF': rsp_d_rrv.get("RRV_LFHF", [np.nan])[0],
                                'RSP_D_RRV_VLF': rsp_d_rrv.get("RRV_VLF", [np.nan])[0],
                                'RSP_D_RRV_HFn': rsp_d_rrv.get("RRV_HFn", [np.nan])[0],
                                'RSP_D_RRV_LFn': rsp_d_rrv.get("RRV_LFn", [np.nan])[0]
                            }

                        except Exception as e:
                            print(f"RRV_D feature extraction failed: {e}")
                            rrv_d_features = {key: np.nan for key in [
                                'RSP_D_RRV_RMSSD', 'RSP_D_RRV_MeanBB',
                                'RSP_D_RRV_MedianBB',
                                'RSP_D_RRV_MadBB', 'RSP_D_RRV_SD2', 'RSP_D_RRV_SD1', 'RSP_D_RRV_SD2SD1',
                                'RSP_D_RRV_ApEn', 'RSP_D_RRV_CVSD', 'RSP_D_RRV_CVBB', 'RSP_D_RRV_MCVBB',
                                'RSP_D_RRV_LF',
                                'RSP_D_RRV_HF', 'RSP_D_RRV_LFHF', 'RSP_D_RRV_VLF', 'RSP_D_RRV_HFn', 'RSP_D_RRV_LFn'
                            ]}

                        try:
                            rvt_d_features = {
                                'RSP_D_RVT_Mean_BIRN': np.nanmean(RSP_d_F_window['RSP_d_rvt']),
                                'RSP_D_RVT_Median_BIRN': np.nanmedian(RSP_d_F_window['RSP_d_rvt']),
                                'RSP_D_RVT_Std_BIRN': np.nanstd(RSP_d_F_window['RSP_d_rvt']),
                            }
                        except Exception as e:
                            print(f"RVT error: {e}")
                            rvt_d_features = {
                                'RSP_D_RVT_Mean_BIRN': np.nan,
                                'RSP_D_RVT_Median_BIRN': np.nan,
                                'RSP_D_RVT_Std_BIRN': np.nan
                            }
                        try:
                            rsp_d_features = {
                                'RSP_D_Rate_Mean': RSP_d_F_window['RSP_d_Rate'].mean(),
                                'RSP_D_Rate_Median': RSP_d_F_window['RSP_d_Rate'].median(),
                                'RSP_D_Rate_Std': RSP_d_F_window['RSP_d_Rate'].std(),

                                'RSP_D_RVT_Mean': RSP_d_F_window['RSP_d_RVT'].mean(),
                                'RSP_D_RVT_Median': RSP_d_F_window['RSP_d_RVT'].median(),
                                'RSP_D_RVT_Std': RSP_d_F_window['RSP_d_RVT'].std(),

                                'RSP_D_PhaseCompletion_Mean': RSP_d_F_window['RSP_d_Phase_Completion'].mean(),
                                'RSP_D_PhaseCompletion_Median': RSP_d_F_window['RSP_d_Phase_Completion'].median(),

                                'RSP_D_Sym_PeakTrough_Mean': RSP_d_F_window['RSP_d_Symmetry_PeakTrough'].mean(),
                                'RSP_D_Sym_PeakTrough_Median': RSP_d_F_window['RSP_d_Symmetry_PeakTrough'].median(),

                                'RSP_D_Sym_RiseDecay_Mean': RSP_d_F_window['RSP_d_Symmetry_RiseDecay'].mean(),
                                'RSP_D_Sym_RiseDecay_Median': RSP_d_F_window['RSP_d_Symmetry_RiseDecay'].median(),

                                'RSP_D_Amplitude_Mean': RSP_d_F_window['RSP_d_Amplitude'].mean(),
                                'RSP_D_Amplitude_Median': RSP_d_F_window['RSP_d_Amplitude'].median(),
                                'RSP_D_Amplitude_Std': RSP_d_F_window['RSP_d_Amplitude'].std(),

                                'RSP_D_RSA_P2T_Mean': RSP_d_F_window['RSP_d_RSA_P2T'].mean(),
                                'RSP_D_RSA_P2T_Median': RSP_d_F_window['RSP_d_RSA_P2T'].median(),

                                'RSP_D_RSA_Gates_Mean': RSP_d_F_window['RSP_d_RSA_Gates'].mean(),
                                'RSP_D_RSA_Gates_Median': RSP_d_F_window['RSP_d_RSA_Gates'].median(),

                            }

                        except Exception as e:
                            print(f"RSP_d_features error: {e}")
                            rsp_d_features = {}

                        # חיבור הכל
                        rsp_d_features.update(rrv_d_features)
                        rsp_d_features.update(rvt_d_features)

                        center_time = (i + window_samples / 2)
                        stress, fatigue, accuracy, rt = trigger_attrs(center_time)
                        cls, level, test_type = label_window(center_time)

                        row_data = {
                            'ID': ID,
                            'Group': Group,
                            'Time': center_time,
                            **hrv_features,
                            **eda_features,
                            **rsp_c_features,
                            **rsp_d_features,
                            'Class': cls,
                            'Test_Type': test_type,
                            'Level': level,
                            'Accuracy': accuracy,
                            'RT': rt,
                            'Stress': stress,
                            'Fatigue': fatigue
                        }

                        Data_df = pd.concat([Data_df, pd.DataFrame([row_data])], ignore_index=True)
                    print(fr"finished {ID}")
                Data_df.insert(0, 'Window', window_size)
                Data_df.insert(1, 'Overlap', overlap)
                Data_Total_df = pd.concat([Data_Total_df, Data_df], ignore_index=True)
                save_name = f'Dataset_{window_size}s_{int(overlap * 100)}.csv'
                save_path = os.path.join(total_dataset_dir, save_name)
                Data_df.to_csv(save_path, index=False)
                print(f"✅ Saved: {save_name}")

        save_name = f'Dataset_total.csv'
        save_path = os.path.join(total_dataset_dir, save_name)
        Data_Total_df=Data_Total_df.sort_values(by=['Window','Overlap','ID'])
        Data_Total_df.to_csv(save_path, index=False)
        print(f"✅ Saved: {save_name}")


