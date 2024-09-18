def concatDf(type, df_filtered, Log_df):
    for idx, row in df_filtered.iterrows():
        # Find the insertion point
        insert_idx = Log_df[Log_df['Timestamp'] > row['Timestamp']].index.min()

        # If no row found with higher time, append at the end
        if pd.isna(insert_idx):
            insert_idx = len(Log_df)

        # Create a new row to insert
        if type == 'scales':
            new_row = {
                'ID': row['subnum'],
                'Timestamp': row['Timestamp'],
                'Time': row['time'],
                'Task': 'Stress Report',
                'Status': row['value']

            }
        elif type == 'corsi':
            new_row = {
                'ID': row['subnum'],
                'Timestamp': row['Timestamp'],
                'Time': row['time'],
                'Task': 'corsi',
                'Status': 'mistake'
            }
        elif type == 'pasat':
            new_row = {
                'ID': row['subnum'],
                'Timestamp': row['Timestamp'],
                'Time': row['time'],
                'Task': 'pasat',
                'Status': 'mistake'
            }
        elif type == 'twocoladd':
            new_row = {
                'ID': row['sub'],
                'Timestamp': row['Timestamp'],
                'Time': row['time'],
                'Task': 'twocoladd',
                'Status': 'mistake'
            }

        # Insert the new row into Log_df
        Log_df = pd.concat(
            [Log_df.iloc[:insert_idx], pd.DataFrame([new_row]), Log_df.iloc[insert_idx:]]).reset_index(
            drop=True)
    return Log_df


def openfile(ID, dataframe_path_log):
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


ID = '4'
Group = 'control_group'
# Group='music_group'
# Group='breath_group '

data_path = r'D:\PEBL2_Win_Portable_2.1.1_for test\PEBL2_Win_Portable_2.1.1_for test\PEBL2.1'
dataframe_path = f'{data_path}\logs'
dataframe_path_log = f'{dataframe_path}\TestLaunch-log.txt'
Test_Type_path = r'D:\Participants\participation management.xlsx'
task = 'corsi'
data_path_battery = fr'{data_path}\battery'
corsi_path = f'{data_path_battery}\{task}\data\{ID}\corsi-trial-{ID}.csv'
task = 'PASAT'
pasat_path = f'{data_path_battery}\{task}\data\{ID}\PASAT-{ID}.csv'
task = 'twocoladd'
twocoladd_path = fr'{data_path_battery}\{task}\data\{ID}\twocol-{ID}.csv'
task = 'scales'
scales_path = fr'{data_path_battery}\{task}\data\{ID}\physioscales-{ID}.csv'

Log_df = openfile(ID, dataframe_path_log)
Type_df = pd.read_excel(Test_Type_path, header=1)
scales_df = pd.read_csv(scales_path, header=0)
corsi_df = Utilities.load_dataframe(corsi_path)
pasat_df = Utilities.load_dataframe(pasat_path)
try:
    twocoladd_df = pd.read_csv(twocoladd_path, delimiter=',', on_bad_lines='skip')
    print("CSV file loaded successfully.")
except pd.errors.ParserError as e:
    print(f"ParserError: {e}")

print(f"File path: {twocoladd_path}")
test_type = Type_df.loc[Type_df['code'].astype(str) == ID, 'Group'].values[0]
participant_path = fr'D:\Participants\{test_type}_group\P_{ID}\P_04.jcq'
with open(participant_path, 'r') as file:
    content = file.read()
time_pattern = re.compile(r'\b\d{2}:\d{2}:\d{2}\b')
Start_time = time_pattern.findall(content)[0]

corsi_df['Timestamp'] = pd.to_datetime(corsi_df['Timestamp']).dt.time
scales_df['Timestamp'] = pd.to_datetime(scales_df['Timestamp']).dt.time
Log_df['Timestamp'] = pd.to_datetime(Log_df['Timestamp']).dt.time
pasat_df['Timestamp'] = pd.to_datetime(pasat_df['Timestamp']).dt.time
twocoladd_df['Timestamp'] = pd.to_datetime(twocoladd_df['Timestamp']).dt.time

corsi_filtered = corsi_df[corsi_df['allcorr'] == 0]
pasat_filtered = pasat_df[pasat_df['correct'] == 0]
twocoladd_filtered = twocoladd_df[twocoladd_df['corr'] == 0]

# Convert Log_df 'Time' column to datetime.time type

# Insert rows from corsi_filtered into Log_df
Log_df = concatDf('corsi', corsi_filtered, Log_df)
Log_df = concatDf('scales', scales_df, Log_df)
Log_df = concatDf('pasat', pasat_filtered, Log_df)
Log_df = concatDf('twocoladd', twocoladd_df, Log_df)
new_row = {'ID': ID,
           'Timestamp': Start_time,
           'Time': '0',
           'Task': 'Biopac record',
           'Status': 'STARTED'}

# Append new row to DataFrame
Log_df = Log_df._append(new_row, ignore_index=True)
Log_df['Timestamp'] = Log_df['Timestamp'].astype(str)
# Sort DataFrame by 'Timestamp' column (as strings)
Log_df = Log_df.sort_values(by='Timestamp')
Log_df['Timestamp'] = pd.to_datetime(Log_df['Timestamp'], format='%H:%M:%S')

# Define the start time as the first timestamp
start_time = Log_df['Timestamp'].iloc[0]

# Calculate the time difference in seconds for each row

# Convert the time difference to 100 Hz units
Log_df['Time'] = Log_df['Timestamp'].apply(lambda x: (x - start_time).total_seconds()) * 100
Log_df['Timestamp'] = pd.to_datetime(Log_df['Timestamp'], format='%H:%M:%S').dt.time
directory = fr'D:\Participants\{Group}\P_{ID}'
file_path = fr'{directory}\Triger_{ID}.csv'
os.makedirs(directory, exist_ok=True)
# Save the dataframe to the specified path
try:
    Log_df.to_csv(file_path, index=False)
    print(f"CSV file saved successfully at {file_path}.")
except Exception as e:
    print(f"Error saving CSV file: {e}")
