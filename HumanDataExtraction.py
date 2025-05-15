import matplotlib.pyplot as plt
import bioread
import neurokit2 as nk
from scipy.signal import spectrogram
import matplotlib.cm as cm
import numpy as np
from scipy.signal import medfilt
import seaborn as sns
import os
from scipy.stats import linregress
import pandas as pd


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
            Trigger_path = fr'{directory}\Trigger_{ID}.csv'
            Trigger_df = pd.read_csv(Trigger_path, header=0)
            BioPac_path = fr'{directory}\P_{ID}.acq'
            BioPac = bioread.read_file(BioPac_path)

            eda_data = BioPac.named_channels['EDA'].data
            eda_time = BioPac.named_channels['EDA'].time_index
            ecg_data = BioPac.named_channels['ECG'].data
            ecg_time = BioPac.named_channels['ECG'].time_index
            ecg_sampling_rate = BioPac.named_channels['ECG'].samples_per_second

            ecg_cleaned = nk.ecg_clean(ecg_data, sampling_rate=ecg_sampling_rate)
            r_peaks, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=ecg_sampling_rate)
            ecg_signals, info = nk.ecg_process(ecg_cleaned, ecg_sampling_rate)
            hrv_rri = np.diff(info["ECG_R_Peaks"]) / ecg_sampling_rate * 1000
            time_rri = info["ECG_R_Peaks"][1:] / ecg_sampling_rate
            eda_signals, info = nk.eda_process(eda_data, sampling_rate=BioPac.named_channels['EDA'].samples_per_second)
            EDA_Clean = medfilt(eda_signals['EDA_Clean'], kernel_size=medfilt_window)
            EDA_Clean = pd.DataFrame({
                'Time': eda_time,
                'EDA': EDA_Clean  # adjust column index as needed
            })
            EDA_Clean.to_csv(fr'{directory}\EDA.csv')
            RR_cleaned = pd.DataFrame({
                'Time': time_rri,
                'RR': hrv_rri  # adjust column index as needed
            })
            RR = pd.DataFrame({
                'Time': time_rri,
                'RR': hrv_rri  # adjust column index as needed
            })
            invalid_rr = (RR_cleaned['RR'] < 300) | (RR_cleaned['RR'] > 1500)
            # הפיכת הערכים החריגים ל-NaN
            RR_cleaned.loc[invalid_rr, 'RR'] = None

            # אינטרפולציה ליניארית לפי עמודת הזמן
            # RR_cleaned['RR'] = RR_cleaned['RR'].interpolate(method='linear', limit_direction='both')
            if plot:
                fig,ax=plt.subplots(2)
                ax[0].plot(RR['Time'],RR['RR'])
                ax[1].plot(RR_cleaned['Time'],RR_cleaned['RR'])
                plt.show()
            RR_cleaned.to_csv(fr'{directory}\RR.csv')

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
            sample_rate = BioPac.named_channels['Chest Respiration'].samples_per_second
            if Group.lower() == 'control':
                task_pattern = r'break2|PA_easy|PA_medium|PA_hard|break3'
            elif Group.lower() == 'music':
                task_pattern = r'music2|PA_easy|PA_medium|PA_hard|music3'
            elif Group.lower() == 'breath':
                task_pattern = r'breath2|PA_easy|PA_medium|PA_hard|breath3'
            else:
                task_pattern = r''  # fallback if needed

            mask = Trigger_df['Task'].str.contains(task_pattern, case=False, na=False)
            filtered_df = Trigger_df[mask].reset_index(drop=True)            # mask = Trigger_df['Task'].str.contains(r'breath ?[1-4]|star', case=False, na=False)
            Itterations=[0,1,2,3,4]
            # fig, ax = plt.subplots(3, figsize=(10, 14))
            for i in Itterations:
                start=int(filtered_df.iloc[i]['Start']*sample_rate)
                end=int(filtered_df.iloc[i]['End']*sample_rate)
                part_data = BioPac.named_channels['Chest Respiration'].data[start:end]
                part_time=BioPac.named_channels['Chest Respiration'].time_index[start:end]
                plt.plot(part_time,part_data)
                # ax[i].plot(part_time,part_data)
            plt.savefig(fr'{directory_Breath}\breath')
            plt.show()
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
    def CreateDataset(self, ID,rangeID):
        """Build time‑windowed feature datasets and add Stress, Fatigue,
        and FromTrigger columns drawn from the participant's Trigger CSV."""

        # ── Config ──────────────────────────────────────────
        interpolate_rr = False                  # RR interpolation toggle
        window_sizes = [5, 10, 30, 60]  # sec
        overlaps = [0.0, 0.5]
        # fraction
        total_dataset_dir = r'C:\Users\e3bom\Desktop\Human Bio Signals Analysis\Participants\Dataset'
        os.makedirs(total_dataset_dir, exist_ok=True)

        # ── Participant table ───────────────────────────────
        Participants_path = f'{self.path}\\Participants\\participation management.csv'
        Participants_df   = (
            pd.read_csv(Participants_path)
              .dropna(axis=1, how='all')
              .dropna(subset=['participant', 'Date', 'departmant'], how='all')
        )
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')
        if ID is not None:
            if rangeID:
                Participants_df = Participants_df[Participants_df['code'] >= ID]
            else:
                Participants_df = Participants_df[Participants_df['code'] == ID]

        # ── Accumulators for group datasets ─────────────────
        total_HRV_dict        = {}
        total_EDA_dict        = {}
        total_RSP_chest_dict  = {}
        total_RSP_diaph_dict  = {}

        # ── Iterate participants ────────────────────────────
        for _, row in Participants_df.iterrows():
            ID    = row['code']
            Group = row['Group']
            directory = fr'{self.path}\Participants\{Group}_group\P_{ID}'

            try:
                Trigger_df = pd.read_csv(fr'{directory}\Trigger_{ID}.csv')
                # ── Ratings table: keep only rows that actually carry stress / fatigue values ──
                # create two new columns, NaN everywhere except the correct VAS rows

                Trigger_df["Stress"] = np.where(Trigger_df["Task"] == "VAS_Stress",
                                                Trigger_df["Score"],
                                                np.nan)

                Trigger_df["Fatigue"] = np.where(Trigger_df["Task"] == "VAS_Fatigue",
                                                 Trigger_df["Score"],
                                                 np.nan)

                # now your original line works
                rating_df = (Trigger_df[(~Trigger_df["Stress"].isna()) | (~Trigger_df["Fatigue"].isna())]
                             .sort_values("Start")
                             .reset_index(drop=True))
                rating_df = rating_df.drop(columns=['Score', 'End'])  # ← preferred, explicit
                BioPac= bioread.read_file(fr'{directory}\P_{ID}.acq')
                sample_rate = BioPac.named_channels['ECG'].samples_per_second

                # ── Helper: label per timepoint ─────────────
                def label_window(t):
                    for _, tr in Trigger_df.iterrows():
                        task = str(tr["Task"]).lower()
                        if pd.notna(tr["End"]) and tr["Start"] <= t <= tr["End"]:
                            if "breath"   in task: return "breath"
                            if "music"    in task: return "music"
                            if "baseline" in task or "break" in task: return "natural"
                            if not task.startswith("vas"): return "test"
                    return None

                                # ── Helper: stress & fatigue  ───────────

                first_iteration = True
                def trigger_attrs(t):
                    """Return Stress, Fatigue and forward gap (Δt) to the **next**
                    VAS rating starting at or after *t*.  Because the experiment
                    begins and ends with a rating there is always such a future
                    score, so no NaNs will appear.

                    Δt = rating_start − t   (always ≥ 0).
                    """
                    # Check if first or second rating is before or at time t

                    if rating_df["Start"].iloc[0] >= t or rating_df["Start"].iloc[1] >= t:
                        future_rows = rating_df  # Use all rows
                    else:
                        future_rows = rating_df[rating_df["Start"] >= t]  # Filter for rows after t
                        # Not enough rows?  Use the full table as a safe fallback
                        if len(future_rows) < 2:  # handles 0 or 1 row
                            future_rows = rating_df
                    if future_rows.empty:
                        # Should nt raot happen, but fall back to the very lasting
                        tr_s = future_rows.iloc[-2]          # first future rating (closest)
                        tr_f = future_rows.iloc[-1]          # first future rating (closest)
                    else:
                        val = future_rows.iloc[0]["Stress"]
                        if pd.isna(val):                        # first future rating (closest)
                            tr_s = future_rows.iloc[1]
                            tr_f = future_rows.iloc[0]          # first future rating (closest)
                        else:
                            tr_s = future_rows.iloc[0]          # first future rating (closest)
                            tr_f = future_rows.iloc[1]          # first future rating (closest)
                    stress  = tr_s.get("Stress",  np.nan)
                    fatigue = tr_f.get("Fatigue", np.nan)
                    return stress, fatigue

                # ── Helper: HRV calculation ────────────────
                def calculate_hrv_features(rr_values):
                    RR = pd.DataFrame({"RR": rr_values})
                    RR.loc[(RR['RR'] < 300) | (RR['RR'] > 1500), 'RR'] = np.nan
                    if interpolate_rr:
                        RR['RR'] = RR['RR'].interpolate(limit_direction='both')
                    rr = RR['RR'].dropna().values
                    if len(rr) < 3:
                        return {k: None for k in [
                            'HRV_MeanNN','HRV_SDNN','HRV_RMSSD',
                            'HRV_CVNN','HRV_pNN20','HRV_pNN50']}
                    mean_nn = np.mean(rr)
                    sdnn    = np.std(rr, ddof=1)
                    diff_rr = np.diff(rr)
                    return {
                        'HRV_MeanNN': mean_nn,
                        'HRV_SDNN'  : sdnn,
                        'HRV_RMSSD' : np.sqrt(np.mean(diff_rr ** 2)),
                        'HRV_CVNN'  : sdnn / mean_nn * 100,
                        'HRV_pNN20' : np.sum(np.abs(diff_rr) > 20) / len(diff_rr) * 100,
                        'HRV_pNN50' : np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100,
                    }

                # ── Window sweep ────────────────────────────
                for window_size in window_sizes:
                    for overlap in overlaps:
                        window_samples = int(window_size * sample_rate)
                        step           = int(window_samples * (1 - overlap))
                        suffix         = f'Time_{window_size}_{overlap}.csv'

                        HRV_df          = pd.DataFrame()
                        EDA_df          = pd.DataFrame()
                        RSP_df_chest    = pd.DataFrame()
                        RSP_df_diaph    = pd.DataFrame()

                        for signal_type in BioPac.named_channels:
                            # signal_type='EDA'
                            part_data = BioPac.named_channels[signal_type].data
                            for j, i in enumerate(range(0, len(part_data) - window_samples + 1, step)):
                                segment= part_data[i:i + window_samples]
                                center_time  = (i + window_samples / 2) / sample_rate
                                stress, fatigue = trigger_attrs(center_time)
                                cls = label_window(center_time)

                                try:
                                    # ── ECG → HRV ────────────────────────
                                    if signal_type == 'ECG':
                                        ecg_cleaned = nk.ecg_clean(segment, sampling_rate=sample_rate)
                                        _, rpeaks   = nk.ecg_peaks(ecg_cleaned, sampling_rate=sample_rate)
                                        if len(rpeaks["ECG_R_Peaks"]) > 2:
                                            RR = np.diff(rpeaks["ECG_R_Peaks"]) / sample_rate * 1000
                                            hrv_features = calculate_hrv_features(RR)
                                            if hrv_features['HRV_MeanNN'] is not None:
                                                HRV_df = pd.concat([
                                                    HRV_df,
                                                    pd.DataFrame([{**{
                                                        'ID': ID,
                                                        'Group': Group,
                                                        'Time': center_time,
                                                        **hrv_features,
                                                        'Class': cls,
                                                        'Stress': stress,
                                                        'Fatigue': fatigue
                                                    }}])
                                                ], ignore_index=True)

                                    # ── EDA ──────────────────────────────
                                    elif signal_type == 'EDA':
                                        eda_signals, info = nk.eda_process(segment, sampling_rate=sample_rate)
                                        eda_cleaned = medfilt(eda_signals['EDA_Clean'], kernel_size=101)
                                        EDA_df = pd.concat([
                                            EDA_df,
                                            pd.DataFrame([{**{
                                                'ID': ID,
                                                'Group': Group,
                                                'Time': center_time,
                                                'EDA_Tonic_Mean'     : eda_signals['EDA_Tonic'].mean(),
                                                'EDA_Phasic_Mean'    : eda_signals['EDA_Phasic'].mean(),
                                                'SCR_Peaks_Count'    : len(info['SCR_Peaks']),
                                                'SCR_Amplitude_Mean' : eda_signals['EDA_Phasic'].max() - eda_signals['EDA_Phasic'].min(),
                                                'EDA_Clean_Median'   : eda_cleaned.mean(),
                                                'Class': cls,
                                                'Stress': stress,
                                                'Fatigue': fatigue
                                            }}])
                                        ], ignore_index=True)

                                    # ── Chest Respiration ────────────────
                                    elif signal_type == 'Chest Respiration':
                                        rsp_signals, _ = nk.rsp_process(segment, sampling_rate=sample_rate)
                                        RSP_df_chest = pd.concat([
                                            RSP_df_chest,
                                            pd.DataFrame([{**{
                                                'ID': ID,
                                                'Group': Group,
                                                'Time': center_time,
                                                'RSP_Rate'                 : rsp_signals.get('RSP_Rate', [None])[0],
                                                'RSP_Amplitude'            : rsp_signals.get('RSP_Amplitude', [None])[0],
                                                'RSP_Symmetry_PeakTrough'  : rsp_signals.get('RSP_Symmetry_PeakTrough', [None])[0],
                                                'Class': cls,
                                                'Stress': stress,
                                                'Fatigue': fatigue
                                            }}])
                                        ], ignore_index=True)

                                    # ── Diaphragmatic Respiration ────────
                                    elif signal_type == 'Diaphragmatic Respiration':
                                        rsp_signals, _ = nk.rsp_process(segment, sampling_rate=sample_rate)
                                        RSP_df_diaph = pd.concat([
                                            RSP_df_diaph,
                                            pd.DataFrame([{**{
                                                'ID': ID,
                                                'Group': Group,
                                                'Time': center_time,
                                                'RSP_Rate'                 : rsp_signals.get('RSP_Rate', [None])[0],
                                                'RSP_Amplitude'            : rsp_signals.get('RSP_Amplitude', [None])[0],
                                                'RSP_Symmetry_PeakTrough'  : rsp_signals.get('RSP_Symmetry_PeakTrough', [None])[0],
                                                'Class': cls,
                                                'Stress': stress,
                                                'Fatigue': fatigue
                                            }}])
                                        ], ignore_index=True)
                                except Exception as e:
                                    print(f"⛔ Error in window {j} for {signal_type} - ID {ID}: {e}")

                        # ── Save participant‑level CSVs ───────────────
                        base_path = fr'{directory}\Features'
                        os.makedirs(base_path, exist_ok=True)

                        if not HRV_df.empty:
                            p = os.path.join(base_path, 'HRV', f'HRV_{suffix}')
                            os.makedirs(os.path.dirname(p), exist_ok=True)
                            HRV_df.to_csv(p, index=False)
                            total_HRV_dict.setdefault((window_size, overlap), pd.DataFrame())
                            total_HRV_dict[(window_size, overlap)] = pd.concat([
                                total_HRV_dict[(window_size, overlap)], HRV_df])

                        if not EDA_df.empty:
                            p = os.path.join(base_path, 'EDA', f'EDA_{suffix}')
                            os.makedirs(os.path.dirname(p), exist_ok=True)
                            EDA_df.to_csv(p, index=False)
                            total_EDA_dict.setdefault((window_size, overlap), pd.DataFrame())
                            total_EDA_dict[(window_size, overlap)] = pd.concat([
                                total_EDA_dict[(window_size, overlap)], EDA_df])

                        if not RSP_df_chest.empty:
                            p = os.path.join(base_path, 'RSP_chest', f'RSP_Chest_{suffix}')
                            os.makedirs(os.path.dirname(p), exist_ok=True)
                            RSP_df_chest.to_csv(p, index=False)
                            total_RSP_chest_dict.setdefault((window_size, overlap), pd.DataFrame())
                            total_RSP_chest_dict[(window_size, overlap)] = pd.concat([
                                total_RSP_chest_dict[(window_size, overlap)], RSP_df_chest])

                        if not RSP_df_diaph.empty:
                            p = os.path.join(base_path, 'RSP_diaph', f'RSP_Diaph_{suffix}')
                            os.makedirs(os.path.dirname(p), exist_ok=True)
                            RSP_df_diaph.to_csv(p, index=False)
                            total_RSP_diaph_dict.setdefault((window_size, overlap), pd.DataFrame())
                            total_RSP_diaph_dict[(window_size, overlap)] = pd.concat([
                                total_RSP_diaph_dict[(window_size, overlap)], RSP_df_diaph])

                        print(f"✅ Saved P_{ID} | W={window_size}s | O={overlap * 100:.0f}%")

            except Exception as e:
                print(f"❌ Error processing P_{ID}: {e}")

        # ── Save combined datasets ─────────────────────────
        for (window_size, overlap) in total_HRV_dict.keys():
            hrv_df = total_HRV_dict.get((window_size, overlap), pd.DataFrame())
            eda_df = total_EDA_dict.get((window_size, overlap), pd.DataFrame())
            rsp_chest_df = total_RSP_chest_dict.get((window_size, overlap), pd.DataFrame())
            rsp_diaph_df = total_RSP_diaph_dict.get((window_size, overlap), pd.DataFrame())

            # Merge all on Time, ID, Group
            merged_df = hrv_df.merge(eda_df, on=["Time", "ID", "Group"], how="outer")
            merged_df = merged_df.merge(rsp_chest_df, on=["Time", "ID", "Group"], how="outer")
            merged_df = merged_df.merge(rsp_diaph_df, on=["Time", "ID", "Group"], how="outer")

            # Optional: drop duplicated columns or reorder if needed
            merged_df.to_csv(os.path.join(
                total_dataset_dir,
                f'Dataset_{window_size}s_{int(overlap * 100)}.csv'
            ), index=False)

