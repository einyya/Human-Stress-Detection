import matplotlib.pyplot as plt
import pandas as pd
import bioread
import neurokit2 as nk
import os
import numpy as np
from Utilities import Utilities
# Press the green button in the gutter to run the script.
class HumanDataExtraction():

    def __init__(self,Directory):
        self.path = Directory
        self.sorted_DATA = pd.DataFrame()
    def CleanData(self):
        # Load the dataset
        data = pd.read_csv(r'D:\Human Bio Signals Analysis\Participants\Dataset\Dataset_1_EDA.csv')

        data_imputed = data.copy()  # Make a copy of the original data
        numeric_columns = data_imputed.columns.difference(['Part'])  # Exclude 'Part' column

        # Apply mean imputation only on numeric columns
        data_imputed[numeric_columns] = data_imputed[numeric_columns].fillna(data_imputed[numeric_columns].mean())

        # Check if there are any missing values left
        print(data_imputed.isna().sum())
        data_imputed.to_csv(r'D:\Human Bio Signals Analysis\Participants\Dataset\Dataset_1_EDA.csv', index=False)

    def Make_DataSet(self,):
        dataset_path = f'{self.path}\Participants\Dataset\Dataset.csv'
        Participants_path = f'{self.path}\Participants\participation management.xlsx'
        Participants_df = pd.read_excel(Participants_path, header=1)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')
        for j, row in Participants_df.iterrows():
            ID = row['code']
            Group = row['Group']
            # ID = 16
            # Group = 'breath'
            print(ID)
            print(Group)
            Plots= False
            directory = fr'{self.path}\Participants\{Group}_group\P_{ID}'
            Triger_path = fr'{directory}\Triger_{ID}.csv'
            Triger_df = pd.read_csv(Triger_path, header=0)
            BioPac_path = fr'{self.path}\Participants\{Group}_group\P_{ID}\P_{ID}.acq'
            BioPac = bioread.read_file(BioPac_path)
            self.segment(ID,Group,Triger_df,BioPac,window_length_s=60,overlap=0)

            # Initialize a list to collect rows
            dataParticipent_path = fr'{directory}\data_{ID}.csv'
            Experimant_channels = ['ECG','Diaphragmatic Respiration','Chest Respiration','EDA']
            combined_per_participent = pd.DataFrame()

            for i, (index, row) in enumerate(Triger_df.iterrows(), start=1):
                # Extract values from the row
                stress_report = row['Stress Report']
                part = row['Task']
                break_outer_loop = False  # Set flag to true
                # Calculate start_index and end_index
                part_start_time = row['Time-start-sec']
                part_end_time = row['Time-end-sec']
                features_df = pd.DataFrame()
                for i, (channel) in enumerate(Experimant_channels, start=1):
                    try:
                        sample_rate = BioPac.named_channels[channel].samples_per_second
                    except KeyError:
                        features_EDA = pd.DataFrame({
                            'EDA_Tonic_Mean': [np.nan],
                            'EDA_Phasic_Mean': [np.nan],
                            'SCR_Peaks_Count': [np.nan],
                            'SCR_Amplitude_Mean': [np.nan]
                        })
                        features_df = pd.concat([features_df, features_EDA], axis=1)
                        print(f'no {channel} ')
                        break
                    start_index = int(part_start_time * sample_rate)
                    end_index = int(part_end_time * sample_rate)
                    if start_index > len(BioPac.named_channels[channel].data):
                        break
                    if end_index > len(BioPac.named_channels[channel].data):
                        end_index = len(BioPac.named_channels[channel].data)
                    part_data = BioPac.named_channels[channel].data[start_index:end_index]

                    try:
                        if channel=='ECG':
                            ecg_signals, info = nk.ecg_process(part_data,sample_rate)
                            HRV = nk.ecg_intervalrelated(ecg_signals)
                            rpeaks = info['ECG_R_Peaks']
                            hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=sample_rate, show=True)
                            SE = nk.entropy_shannon(part_data, symbolize='A')[0]
                            # Collect HRV metrics
                            HRmean = HRV[['ECG_Rate_Mean']].copy().astype(float).round(1)
                            avNN = HRV[['HRV_MeanNN']].copy().astype(float).round(1)
                            sdNN = HRV[['HRV_SDNN']].copy().astype(float).round(1)
                            rMSSD = HRV[['HRV_RMSSD']].copy().astype(float).round(1)
                            PHRNN50 = HRV[['HRV_pNN50']].copy().astype(float).round(1)
                            PHRNN20 = HRV[['HRV_pNN20']].copy().astype(float).round(1)
                            HRV_VHF = hrv_freq[['HRV_VHF']].copy().astype(float).round(4)
                            HRV_VLF = hrv_freq[['HRV_VLF']].copy().astype(float).round(4)
                            HRV_LF = hrv_freq[['HRV_LF']].copy().astype(float).round(4)
                            HRV_HF = hrv_freq[['HRV_HF']].copy().astype(float).round(4)
                            HRV_LFHF = hrv_freq[['HRV_LFHF']].copy().astype(float).round(2)
                            HRV_LFn = hrv_freq[['HRV_LFn']].copy().astype(float).round(3)
                            HRV_HFn = hrv_freq[['HRV_HFn']].copy().astype(float).round(3)
                            HRV_LnHF = hrv_freq[['HRV_LnHF']].copy().astype(float).round(3)
                            HRV_TP = hrv_freq[['HRV_TP']].copy().astype(float).round(3)
                            HRV_ULF = hrv_freq[['HRV_ULF']].copy().astype(float).round(3)
                            HRV_ShanEn = pd.DataFrame([SE], columns=['HRV_ShanEn'])
                            features_ECG=pd.concat([HRmean, avNN, sdNN, rMSSD,PHRNN50,PHRNN20,HRV_VHF,HRV_ULF, HRV_VLF, HRV_LF, HRV_HF, HRV_LFHF, HRV_LFn, HRV_HFn, HRV_LnHF,HRV_ULF,HRV_TP,HRV_ShanEn],axis=1)
                            features_df = pd.concat([features_df, features_ECG], axis=1)

                        if channel=='Chest Respiration' or channel=='Diaphragmatic Respiration':
                            rsp_signals, info = nk.rsp_process(part_data, sampling_rate=sample_rate)
                            BRV_DATA = nk.rsp_intervalrelated(rsp_signals)
                            BRV_DATA2 = nk.rsp_rrv(rsp_signals, show=True)
                            # Collect RSP metrics
                            if channel=='Chest Respiration':
                                RSP_Rate_Mean = pd.DataFrame({f'Chest_RSP_Rate_Mean': [BRV_DATA['RSP_Rate_Mean'][0]]})
                                BRV = pd.DataFrame({'Chest_BRV': [BRV_DATA2['RRV_MeanBB'][0]]})
                                BRavNN = pd.DataFrame({'Chest_BRavNN': [BRV_DATA['RAV_Mean'][0]]})
                                BRsdNN = pd.DataFrame({'Chest_BRavNN': [BRV_DATA['RRV_SD1'][0]]})
                                RSP_Phase_Duration_Expiration = pd.DataFrame({f'Chest_RSP_Phase_Duration_Expiration': [BRV_DATA['RSP_Phase_Duration_Expiration'][0]]})
                                RSP_Phase_Duration_Inspiration = pd.DataFrame({f'Chest_RSP_Phase_Duration_Inspiration': [BRV_DATA['RSP_Phase_Duration_Inspiration'][0]]})
                                RSP_Phase_Duration_Ratio = pd.DataFrame({f'Chest_RSP_Phase_Duration_Ratio': [BRV_DATA['RSP_Phase_Duration_Ratio'][0]]})
                                RSP_RVT = pd.DataFrame({f'Chest_RSP_RVT': [BRV_DATA['RSP_RVT'][0]]})
                                RSP_Symmetry_PeakTrough = pd.DataFrame({f'Chest_RSP_Symmetry_PeakTrough': [BRV_DATA['RSP_Symmetry_PeakTrough'][0]]})
                                RSP_Symmetry_RiseDecay = pd.DataFrame({f'Chest_RSP_Symmetry_RiseDecay': [BRV_DATA['RSP_Symmetry_RiseDecay'][0]]})

                            else:
                                RSP_Rate_Mean = pd.DataFrame({f'Diaph_RSP_Rate_Mean': [BRV_DATA['RSP_Rate_Mean'][0]]})
                                BRV = pd.DataFrame({'Diaph_BRV': [BRV_DATA2['RRV_MeanBB'][0]]})
                                BRavNN = pd.DataFrame({'Diaph_BRavNN': [BRV_DATA['RAV_Mean'][0]]})
                                BRsdNN = pd.DataFrame({'Diaph_BRavNN': [BRV_DATA['RRV_SD1'][0]]})
                                RSP_Phase_Duration_Expiration = pd.DataFrame({f'Diaph_RSP_Phase_Duration_Expiration': [BRV_DATA['RSP_Phase_Duration_Expiration'][0]]})
                                RSP_Phase_Duration_Inspiration = pd.DataFrame({f'Diaph_RSP_Phase_Duration_Inspiration': [BRV_DATA['RSP_Phase_Duration_Inspiration'][0]]})
                                RSP_Phase_Duration_Ratio = pd.DataFrame({f'Diaph_RSP_Phase_Duration_Ratio': [BRV_DATA['RSP_Phase_Duration_Ratio'][0]]})
                                RSP_RVT = pd.DataFrame({f'Diaph_RSP_RVT': [BRV_DATA['RSP_RVT'][0]]})
                                RSP_Symmetry_PeakTrough = pd.DataFrame({f'Diaph_RSP_Symmetry_PeakTrough': [BRV_DATA['RSP_Symmetry_PeakTrough'][0]]})
                                RSP_Symmetry_RiseDecay = pd.DataFrame({f'Diaph_RSP_Symmetry_RiseDecay': [BRV_DATA['RSP_Symmetry_RiseDecay'][0]]})

                            features_RSP = pd.concat([RSP_Rate_Mean, BRV, BRavNN, BRsdNN,
                                                      RSP_Phase_Duration_Expiration, RSP_Phase_Duration_Inspiration,
                                                      RSP_Phase_Duration_Ratio, RSP_RVT,
                                                      RSP_Symmetry_PeakTrough, RSP_Symmetry_RiseDecay], axis=1)
                            features_df = pd.concat([features_df, features_RSP], axis=1)

                        if channel == 'EDA':
                            eda_signals, info = nk.eda_process(part_data, sampling_rate=sample_rate)
                            EDA_Tonic_Mean = pd.DataFrame({'EDA_Tonic_Mean': [eda_signals['EDA_Tonic'].mean()]})
                            EDA_Phasic_Mean = pd.DataFrame({'EDA_Phasic_Mean': [eda_signals['EDA_Phasic'].mean()]})
                            SCR_Peaks_Count = pd.DataFrame({'SCR_Peaks_Count': [len(info['SCR_Peaks'])]})
                            SCR_Amplitude_Mean = pd.DataFrame({'SCR_Amplitude_Mean': [eda_signals['EDA_Phasic'].max() - eda_signals['EDA_Phasic'].min()]})
                            features_EDA = pd.concat([EDA_Tonic_Mean, EDA_Phasic_Mean, SCR_Peaks_Count, SCR_Amplitude_Mean], axis=1)
                            features_df = pd.concat([features_df, features_EDA], axis=1)

                        print(f"CSV file saved {part} in {channel} successfully.")
                        print(ID)
                        print(Group)
                        if Plots and channel == 'ECG':
                            plots_path = fr'{self.path}\Participants\{Group}_group\P_{ID}\plots'
                            PSD_path = os.path.join(plots_path, f'PSD for participant {ID} in {part}.png')
                            plt.savefig(PSD_path, dpi=300, bbox_inches='tight')
                            plt.show()

                            # Plots-ECG:
                            ecg_signals.ECG_Clean.plot()
                            # ecg_signals.ECG_Clean[0:2500].plot()
                            plt.xlabel('Time-250HZ')
                            plt.ylabel('ECG Signal')
                            plt.title(f'ECG Signal for participant {ID} in {part}')
                            file_path = os.path.join(plots_path, f'ECG Signal for participant {ID} in {part}.png')
                            plt.savefig(file_path, dpi=300, bbox_inches='tight')
                            plt.show()

                            # Plot the ECG signal with detected peaks
                            plt.figure(figsize=(14, 6))
                            plt.plot(ecg_signals['ECG_Raw'][0:2500], label='ECG Signal', color='blue')
                            plt.plot(ecg_signals['ECG_R_Peaks'][0:2500], 'ro', label='Detected Peaks')
                            plt.title(
                                f'ECG Signal with Peak Detection for Channel: {channel} for participant {ID} in {part}')
                            plt.xlabel('Time')
                            plt.ylabel('ECG Amplitude')
                            file_path = os.path.join(plots_path,
                                                     f'ECG Signal with Peak Detection for participant {ID} in {part}.png')
                            plt.savefig(file_path, dpi=300, bbox_inches='tight')
                            plt.show()

                            # Plots-Features:
                            ecg_signals.ECG_Rate.plot()
                            # ecg_signals.ECG_Rate[0:2500].plot()
                            plt.xlabel('Time-250HZ')
                            plt.ylabel('ECG Rate')
                            plt.title(f'ECG Rate for participant {ID} in {part}')
                            file_path = os.path.join(plots_path, f'ECG Rate for participant {ID} in {part}.png')
                            plt.savefig(file_path, dpi=300, bbox_inches='tight')
                            plt.show()

                            ecg_signals.ECG_Rate.hist(bins=30, edgecolor='black')
                            # Add labels and title
                            plt.xlabel('ECG Rate')
                            plt.ylabel('Frequency')
                            plt.title(f'Histogram of ECG Rate for participant {ID} in {part}')
                            file_path = os.path.join(plots_path,
                                                     f'ECG Rate Histogram for participant {ID} in {part}.png')
                            plt.savefig(file_path, dpi=300, bbox_inches='tight')
                            plt.show()
                            # Plots-ECG:
                            ecg_signals.ECG_Clean.plot()
                            # ecg_signals.ECG_Clean[0:2500].plot()
                            plt.xlabel('Time-250HZ')
                            plt.ylabel('ECG Signal')
                            plt.title(f'ECG Signal for participant {ID} in {part}')
                            file_path = os.path.join(plots_path, f'ECG Signal for participant {ID} in {part}.png')
                            plt.savefig(file_path, dpi=300, bbox_inches='tight')
                            plt.show()

                            # Plot the ECG signal with detected peaks
                            plt.figure(figsize=(14, 6))
                            plt.plot(ecg_signals['ECG_Raw'][0:2500], label='ECG Signal', color='blue')
                            plt.plot(ecg_signals['ECG_R_Peaks'][0:2500], 'ro', label='Detected Peaks')
                            plt.title(
                                f'ECG Signal with Peak Detection for Channel: {channel} for participant {ID} in {part}')
                            plt.xlabel('Time')
                            plt.ylabel('ECG Amplitude')
                            file_path = os.path.join(plots_path,
                                                     f'ECG Signal with Peak Detection for participant {ID} in {part}.png')
                            plt.savefig(file_path, dpi=300, bbox_inches='tight')
                            plt.show()

                            # Plots-Features:
                            ecg_signals.ECG_Rate.plot()
                            # ecg_signals.ECG_Rate[0:2500].plot()
                            plt.xlabel('Time-250HZ')
                            plt.ylabel('ECG Rate')
                            plt.title(f'ECG Rate for participant {ID} in {part}')
                            file_path = os.path.join(plots_path, f'ECG Rate for participant {ID} in {part}.png')
                            plt.savefig(file_path, dpi=300, bbox_inches='tight')
                            plt.show()

                            ecg_signals.ECG_Rate.hist(bins=30, edgecolor='black')
                            # Add labels and title
                            plt.xlabel('ECG Rate')
                            plt.ylabel('Frequency')
                            plt.title(f'Histogram of ECG Rate for participant {ID} in {part}')
                            file_path = os.path.join(plots_path,
                                                     f'ECG Rate Histogram for participant {ID} in {part}.png')
                            plt.savefig(file_path, dpi=300, bbox_inches='tight')
                            plt.show()
                    except Exception as e:
                        print(f"Error saving CSV {ID} in {part} in {channel} file: {e}")
                        break
                if not features_df.empty:
                    os.makedirs(directory, exist_ok=True)
                    rows = []
                    # Collect row data in a list
                    rows.append({
                        'participant': ID,
                        'Part': part,
                        'Stress Report': stress_report,
                    })
                    # Convert list of rows to a DataFrame
                    part_df = pd.DataFrame(rows)
                    combined_df = pd.concat([part_df,features_df], axis=1)
                    print(ID)
                    print(part)
                    combined_per_participent = pd.concat([combined_per_participent,combined_df], axis=0)
                # Concatenate part_df and features_df to self.sorted_DATA
            combined_per_participent.to_csv(dataParticipent_path, index=False)
            self.sorted_DATA = pd.concat([self.sorted_DATA, combined_per_participent], ignore_index=True)
            # combined_df = pd.DataFrame()
        try:
            self.sorted_DATA.to_csv(dataset_path, index=False)
            print(f"CSV file saved successfully.")
        except Exception as e:
            print(f"Error saving CSV file: {e}")

    # segments data with overlap using rolling window, if semgnet_heartbeats true, then segment is centralised around the heartbeat (R peak)
    def segment(self,ID,Group,Triger_df,BioPac, window_length_s: float, overlap: float,segment_hearbeats=False):
        preprocessed_data_df = pd.DataFrame()
        # Initialize a dictionary to hold DataFrames for each channel
        preprocessed = {}
        Experimant_channels = ['ECG', 'Diaphragmatic Respiration', 'Chest Respiration', 'EDA']
        for i, (channel) in enumerate(Experimant_channels, start=1):
            preprocessed[channel] = pd.DataFrame(
                columns=['ID', 'Group', 'Part',f'{channel}', 'Start Index', 'Start Stress Report', 'End Index', 'End Stress Report'])
            sample_rate = BioPac.named_channels[channel].samples_per_second
            # convert window_length in seconds to samples
            self.window_samples = int(window_length_s * sample_rate)
            # Calculate the step_size as the fraction of the total window samples
            step_size = int(self.window_samples * (1 - overlap))
            Triger_df['Time-start-sec'] = Triger_df['Time-start-sec'] * sample_rate
            Triger_df['Time-end-sec'] = Triger_df['Time-end-sec'] * sample_rate
            # Initialize starting variables
            current_index = 0
            row_current = Triger_df[(Triger_df['Time-start-sec'] <= current_index) & (Triger_df['Time-end-sec'] >= current_index)]
            # If a match is found, extract the Stress Report
            if not row_current.empty:
                current_stressed = row_current['Stress Report'].values[0]
                print(f"Stress Report for current index {current_index}: {current_stressed}")
            else:
                print(f"No matching Stress Report found for current index {current_index}.")

            # faster to concatenate at the end
            preprocessed_DATA_list = []

            # get all R peaks index if required
            if segment_hearbeats:
                r_peaks = nk.ecg_peaks(self.sorted_DATA['ECG'], sampling_rate=self.sampling_frequency)

            # Loop through the entire dataframe
            while current_index < len(BioPac.named_channels[channel].data):
                Utilities.progress_bar('Segmenting data', current_index, total=len(BioPac.named_channels[channel].data))
                # calculate end index in window and exit if out of bounds
                end_index = current_index + self.window_samples
                if (end_index > len(BioPac.named_channels[channel].data)):
                    break

                row_end = Triger_df[(Triger_df['Time-start-sec'] <= end_index) & (Triger_df['Time-end-sec'] >= end_index)]
                # If a match is found, extract the Stress Report
                if not row_end.empty:
                    end_stressed = row_end['Stress Report'].values[0]
                    print(f"Stress Report for current index {end_index}: {end_stressed}")
                else:
                    print(f"No matching Stress Report found for current index {end_index}.")


                # If the next window has a different label, skip to next start of next label
                if end_stressed != current_stressed:
                    while (current_stressed == row_current['Stress Report'].values[0]):
                        current_index += 1
                        row_current = Triger_df[(Triger_df['Time-start-sec'] <= current_index) & (Triger_df['Time-end-sec'] >= current_index)]
                    current_stressed = end_stressed

                # otherwise, add segment to list of pre-processed ECG
                else:
                    if segment_hearbeats:
                        # get index of next r peak
                        while not bool(r_peaks[0]['ECG_R_Peaks'][current_index]):
                            current_index += 1
                        # append segment centred on r-peak to dataframe
                        preprocessed_DATA_list.append(self.sorted_ECG.iloc[(current_index - (self.window_samples // 2)):(
                                    current_index + (self.window_samples // 2))].astype('Float64'))
                        # shift the window to next non r-peak index
                        current_index += 1
                    else:
                        # append segment to dataframe
                        segment=BioPac.named_channels[channel].data[current_index:current_index + self.window_samples]
                        preprocessed_data_df = preprocessed_data_df.append({
                            'ID': ID,
                            'Group': Group,
                            'Part':Triger_df['Task'][end_index],
                            f'{channel}': segment.tolist(),  # Store full segment data as a list
                            'Start Index': current_index,
                            'End Index': end_index,
                            'Start Stress Report': current_stressed,
                            'End Stress Report': end_stressed,
                        }, ignore_index=True)
                        current_index += step_size
            preprocessed[channel] = pd.DataFrame(columns=[f'{channel}', 'Start Index', 'Start Stress Report', 'End Index', 'End Stress Report'])

            self.segment_DATA = pd.concat(preprocessed_DATA_list, axis=0, ignore_index=True).astype('Float64')
            Utilities.progress_bar('Segmenting data', current_index, current_index)

    def create_2d(self):
        # convert the pandas DataFrame into a 2D pandas where each row has the size of window and the corresponding label (stress level)

        # Calculate the number of rows required
        num_rows = len(self.preprocessed_ECG['ECG']) // self.window_samples

        # Create an empty dataframe to hold the reshaped data
        df_reshaped = pd.DataFrame(index=range(num_rows), columns=[f"ECG {i}" for i in range(self.window_samples)])

        # Reshape the data
        for i in range(num_rows):
            start_idx = i * self.window_samples
            end_idx = (i + 1) * self.window_samples
            values = self.preprocessed_ECG['ECG'].iloc[start_idx:end_idx].values
            df_reshaped.iloc[i, :] = values

        self.preprocessed_ECG_2d = df_reshaped
        self.preprocessed_ECG_2d['Stress Level'] = self.preprocessed_ECG['Stress Level'][::self.window_samples].reset_index(drop=True)