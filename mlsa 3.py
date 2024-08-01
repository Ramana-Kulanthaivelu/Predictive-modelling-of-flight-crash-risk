# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 22:08:30 2024

@author: Ramana_
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/reduced_dataset.csv")

# Define the features to use for clustering
features = ['crew_age', 'latitude', 'longitude', 'acft_model', 'wx_temp', 'wind_dir_deg', 'wind_vel_kts', 'altimeter']

# Standardize the features
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data[features])

# Perform hierarchical clustering using complete linkage
Z = linkage(data_standardized, method='complete')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z, labels=data.index, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Determine the optimal number of clusters using the silhouette score
sil_scores = []
for n_clusters in range(2, 11):
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    sil_score = silhouette_score(data_standardized, clusters)
    sil_scores.append(sil_score)
    print(f'Number of clusters: {n_clusters}, Silhouette Score: {sil_score:.3f}')

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), sil_scores, 'o-', markersize=8)
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid()
plt.show()

# Choose the optimal number of clusters based on the silhouette score
optimal_clusters = sil_scores.index(max(sil_scores)) + 2
print(f'Optimal number of clusters: {optimal_clusters}')

# Perform clustering with the optimal number of clusters
final_clusters = fcluster(Z, optimal_clusters, criterion='maxclust')

# Add the cluster labels to the original dataset
data['Cluster'] = final_clusters

# Plot the clusters in a pairplot
sns.pairplot(data, vars=features, hue='Cluster', palette='tab10', diag_kind='kde')
plt.suptitle('Pairplot of Clusters', y=1.02)
plt.show()

# Plot heatmap of cluster centroids
cluster_centroids = pd.DataFrame(data.groupby('Cluster')[features].mean())
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_centroids.T, annot=True, cmap='coolwarm')
plt.title('Cluster Centroids')
plt.show()

# Print the cluster centroids
print(cluster_centroids)
