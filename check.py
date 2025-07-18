import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load your data
df = pd.read_csv(r'C:\Users\e3bom\Desktop\Human Bio Signals Analysis\Analysis\Breath Analysis\Apena\df_cluster_D.csv')

# Check the first rows to verify the column names
print(df.head())

# Assuming your columns are named exactly:
# 'Bin Center', 'Count', 'STD'
# If needed, adjust names accordingly.

# Prepare data for clustering
X = df[['Bin Center', 'Count', 'STD']].values

# Create KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Print cluster assignments
print(df[['Bin Center', 'Count', 'STD', 'Cluster']].head())

# 3D scatter plot
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df['Bin Center'],
    df['Count'],
    df['STD'],
    c=df['Cluster'],
    cmap='viridis',
    s=50
)

ax.set_xlabel('Bin Center')
ax.set_ylabel('Count')
ax.set_zlabel('STD')
ax.set_title('3D Clusters (KMeans)')

plt.colorbar(scatter, label='Cluster')
plt.show()
