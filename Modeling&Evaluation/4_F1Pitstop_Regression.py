import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load preprocessed dataset
df = pd.read_csv('dataset/Pitstop_Data_Preprocessed.csv')

# Load original dataset (for decoding constructor names)
df_orig = pd.read_csv('dataset/Formula1_Pitstop_Data_2011-2024_all_rounds.csv')

# Recreate label encoder to map constructor codes to names
le = LabelEncoder()
le.fit(df_orig['Constructor'])

# Compute mean statistics per constructor
features = [
    'TotalPitStops',
    'AvgPitStopTime',
    'FastestStop',
    'SlowestStop',
    'FirstPitLap',
    'LastPitLap'
]
constructor_df = df.groupby('Constructor')[features].mean().reset_index()
X = constructor_df[features]

# Generate Elbow Plot to determine optimal number of clusters
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, 'o-', linewidth=2)
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(K_range)
plt.tight_layout()
plt.show()

# Compute silhouette scores for each k
sil_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)

plt.figure(figsize=(8, 4))
plt.plot(K_range, sil_scores, 'o-', linewidth=2)
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Various k')
plt.xticks(K_range)
plt.tight_layout()
plt.show()

# Perform final clustering using the best k (e.g., based on highest silhouette score)
best_k = K_range[sil_scores.index(max(sil_scores))]
kmeans = KMeans(n_clusters=4, random_state=42)
constructor_df['Cluster'] = kmeans.fit_predict(X)
constructor_df['ConstructorName'] = le.inverse_transform(constructor_df['Constructor'])

# Apply PCA to reduce dimensionality for visualization
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X)
constructor_df['PC1'], constructor_df['PC2'] = pcs[:, 0], pcs[:, 1]

# Visualize the final clusters using PCA components
plt.figure(figsize=(10, 8))
for c in sorted(constructor_df['Cluster'].unique()):
    subset = constructor_df[constructor_df['Cluster'] == c]
    plt.scatter(subset['PC1'], subset['PC2'], s=100, label=f'Cluster {c}')
for _, row in constructor_df.iterrows():
    plt.annotate(
        row['ConstructorName'],
        (row['PC1'], row['PC2']),
        textcoords="offset points",
        xytext=(5, 5),
        ha='left',
        fontsize=9
    )
plt.title(f'Constructor Strategy Clusters (k=4)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.show()

# Display constructor names by cluster
for cluster_id in sorted(constructor_df['Cluster'].unique()):
    names = constructor_df.loc[constructor_df['Cluster'] == cluster_id, 'ConstructorName'].tolist()
    print(f"Cluster {cluster_id}: {names}")

# Display mean feature values per cluster
cluster_summary = constructor_df.groupby('Cluster')[features].mean()
print("\nCluster Summary (Mean values):")
print(cluster_summary)
