import matplotlib.pyplot as plt
import pandas as pd
import bioread
import neurokit2 as nk
import os
import numpy as np

# Press the green button in the gutter to run the script.
class HumanDataExtraction():

    def __init__(self,Directory):
        self.path = Directory
        self.sorted_DATA = pd.DataFrame()
    def Make_DataSet(self):
        dataset_path = f'{self.path}\Participants\Dataset\Dataset.csv'
        Participants_path = f'{self.path}\Participants\participation management.xlsx'
        Participants_df = pd.read_excel(Participants_path, header=1)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')
        for j, row in Participants_df.iterrows():
            ID = row['code']
            Group = row['Group']
            # ID = 12
            # Group = 'breath'
            print(ID)
            print(Group)
            Plots= False
            directory = fr'{self.path}\Participants\{Group}_group\P_{ID}'
            Triger_path = fr'{directory}\Triger_{ID}.csv'
            Triger_df = pd.read_csv(Triger_path, header=0)
            BioPac_path = fr'{self.path}\Participants\{Group}_group\P_{ID}\P_{ID}.acq'
            BioPac = bioread.read_file(BioPac_path)
            # Initialize a list to collect rows
            dataParticipent_path = fr'{directory}\data_{ID}.csv'

            Experimant_channels = ['ECG']
            features_list = []
            combined_df_Participent = pd.DataFrame()

            for i, (index, row) in enumerate(Triger_df.iterrows(), start=1):
                # Extract values from the row
                stress_report = row['Stress Report']
                part = row['Task']
                break_outer_loop = False  # Set flag to true

                # Calculate start_index and end_index
                part_start_time = row['Time-start-sec']
                part_end_time = row['Time-end-sec']
                sample_rate = BioPac.named_channels[Experimant_channels[0]].samples_per_second

                start_index = int(part_start_time * sample_rate)
                end_index = int(part_end_time * sample_rate)
                for channel_name in Experimant_channels:
                    if end_index > len(BioPac.named_channels[channel_name].data) or part=='finish.pbl' or part=='music_part_4.pbl' or part=='breathing_part_4.pbl':
                        break_outer_loop = True  # Set flag to true
                        combined_df_Participent.to_csv(dataParticipent_path, index=False)
                        combined_df_Participent=pd.DataFrame()
                        break  # Break inner loop

                    part_data = BioPac.named_channels[channel_name].data[start_index:end_index]
                    try:
                        ecg_signals, info = nk.ecg_process(part_data, sample_rate)
                        HRV = nk.ecg_intervalrelated(ecg_signals)
                        print(f"CSV file saved {part}  successfully.")
                        print(ID)
                        print(Group)
                    except Exception as e:
                        print(f"Error saving CSV {ID} IN {part} file: {e}")
                        if Plots:
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
                                f'ECG Signal with Peak Detection for Channel: {channel_name} for participant {ID} in {part}')
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
                        break


                    plots_path=fr'{self.path}\Participants\{Group}_group\P_{ID}\plots'
                    os.makedirs(directory, exist_ok=True)
                    if Plots:
                        #Plots-ECG:
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
                        plt.title(f'ECG Signal with Peak Detection for Channel: {channel_name} for participant {ID} in {part}')
                        plt.xlabel('Time')
                        plt.ylabel('ECG Amplitude')
                        file_path = os.path.join(plots_path, f'ECG Signal with Peak Detection for participant {ID} in {part}.png')
                        plt.savefig(file_path, dpi=300, bbox_inches='tight')
                        plt.show()

                        #Plots-Features:
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
                        file_path = os.path.join(plots_path, f'ECG Rate Histogram for participant {ID} in {part}.png')
                        plt.savefig(file_path, dpi=300, bbox_inches='tight')
                        plt.show()

                    # Collect HRV metrics
                    HRNmean = HRV[['ECG_Rate_Mean']].copy().astype(float).round(1)
                    avNN = HRV[['HRV_MeanNN']].copy().astype(float).round(1)
                    sdNN = HRV[['HRV_SDNN']].copy().astype(float).round(1)
                    rMSSD = HRV[['HRV_RMSSD']].copy().astype(float).round(1)
                    PHRNN50 = HRV[['HRV_pNN50']].copy().astype(float).round(1)
                    PHRNN20 = HRV[['HRV_pNN20']].copy().astype(float).round(1)
                    rows = []

                    # Collect row data in a list
                    rows.append({
                        'participant': ID,
                        'Part': part,
                        'Stress Report': stress_report,
                    })
                    features_df=pd.concat([HRNmean, avNN, sdNN, rMSSD,PHRNN50,PHRNN20], axis=1)

                    # Convert list of rows to a DataFrame
                    part_df = pd.DataFrame(rows)

                    combined_df = pd.concat([part_df,features_df], axis=1)
                    combined_df_Participent=pd.concat([combined_df,combined_df_Participent],ignore_index=True)
                    # Concatenate part_df and features_df to self.sorted_DATA
                    self.sorted_DATA = pd.concat([self.sorted_DATA, combined_df], ignore_index=True)

                if break_outer_loop:
                    break

        try:
            self.sorted_DATA.to_csv(dataset_path, index=False)
            print(f"CSV file saved successfully.")
        except Exception as e:
            print(f"Error saving CSV file: {e}")

