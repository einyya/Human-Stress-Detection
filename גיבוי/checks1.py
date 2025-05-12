import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'D:\Human Bio Signals Analysis\Participants\Dataset\Task and report corr.csv'
data = pd.read_csv(file_path)

# Filter rows where Task is 'PASAT.pbl' or 'twocol.pbl'
filtered_data = data[(data['Task'] == 'PASAT.pbl') | (data['Task'] == 'twocol.pbl')]

# Convert 'Stress Report' and 'Score' columns to numeric (handle any non-numeric values)
filtered_data['Stress Report'] = pd.to_numeric(filtered_data['Stress Report'], errors='coerce')
filtered_data['Score'] = pd.to_numeric(filtered_data['Score'], errors='coerce')

# Calculate the correlation
correlation = filtered_data['Stress Report'].corr(filtered_data['Score'])

# Generate a scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(filtered_data['Score'],filtered_data['Stress Report'], alpha=0.7)
plt.title('Scatterplot of Stress Report vs Score')
plt.xlabel('Score')
plt.ylabel('Stress Report')
plt.grid(True)
plt.show()

print(f"Correlation: {correlation}")
