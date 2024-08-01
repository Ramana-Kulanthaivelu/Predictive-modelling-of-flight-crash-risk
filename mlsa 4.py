# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 22:28:37 2024

@author: Ramana_
"""

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
from scipy.spatial.distance import pdist

# Load the dataset
data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/reduced_dataset.csv")

# Define the features to use for clustering
features = ['crew_age', 'latitude', 'longitude', 'acft_model', 'wx_temp', 'wind_dir_deg', 'wind_vel_kts', 'altimeter']

# Standardize the features
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data[features])

# Define linkage methods and distance metrics
linkage_methods = ['single', 'complete', 'average', 'ward']
distance_metrics = ['euclidean', 'cityblock']

# Initialize results storage
silhouette_scores = pd.DataFrame(columns=linkage_methods, index=distance_metrics)

# Perform hierarchical clustering and calculate silhouette scores for each method and metric
for metric in distance_metrics:
    distance_matrix = pdist(data_standardized, metric=metric)
    for method in linkage_methods:
        Z = linkage(distance_matrix, method=method)
        
        # Determine the optimal number of clusters using the silhouette score
        sil_scores = []
        for n_clusters in range(2, 11):
            clusters = fcluster(Z, n_clusters, criterion='maxclust')
            sil_score = silhouette_score(data_standardized, clusters)
            sil_scores.append(sil_score)
            print(f'Distance Metric: {metric}, Linkage Method: {method}, Number of clusters: {n_clusters}, Silhouette Score: {sil_score:.3f}')
        
        # Store the maximum silhouette score for the current method and metric
        silhouette_scores.loc[metric, method] = max(sil_scores)

        # Plot dendrogram
        plt.figure(figsize=(12, 8))
        dendrogram(Z, labels=data.index, leaf_rotation=90, leaf_font_size=10)
        plt.title(f'Dendrogram ({metric} distance, {method} linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()

# Plot the silhouette scores for each distance metric and linkage method
plt.figure(figsize=(12, 8))
for metric in distance_metrics:
    plt.plot(linkage_methods, silhouette_scores.loc[metric], marker='o', label=f'{metric.capitalize()} Distance')

plt.title('Silhouette Scores for Different Linkage Methods and Distance Metrics')
plt.xlabel('Linkage Method')
plt.ylabel('Max Silhouette Score')
plt.legend()
plt.grid()
plt.show()

# Print the silhouette scores for review
print('Silhouette Scores for Different Linkage Methods and Distance Metrics:')
print(silhouette_scores)


from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

# Define distance metrics and linkage methods
metrics = ['euclidean', 'Manhattan']
linkage_methods = ['single', 'complete', 'average', 'ward']

# Store results
results = []

for metric in metrics:
    for method in linkage_methods:
        # Perform hierarchical clustering
        Z = linkage(data_standardized, method=method, metric=metric)
        
        # Determine the optimal number of clusters using silhouette score
        sil_scores = []
        for n_clusters in range(2, 11):
            clusters = fcluster(Z, n_clusters, criterion='maxclust')
            sil_score = silhouette_score(data_standardized, clusters)
            sil_scores.append(sil_score)
        
        # Find the optimal number of clusters
        optimal_clusters = sil_scores.index(max(sil_scores)) + 2
        
        # Save results
        results.append({
            'Distance Metric': metric,
            'Linkage Method': method,
            'Optimal Number of Clusters': optimal_clusters,
            'Best Silhouette Score': max(sil_scores)
        })

# Print results
results_df = pd.DataFrame(results)
print(results_df)

# Plot silhouette scores
for metric in metrics:
    plt.figure(figsize=(12, 8))
    for method in linkage_methods:
        sil_scores = []
        for n_clusters in range(2, 11):
            Z = linkage(data_standardized, method=method, metric=metric)
            clusters = fcluster(Z, n_clusters, criterion='maxclust')
            sil_score = silhouette_score(data_standardized, clusters)
            sil_scores.append(sil_score)
        
        plt.plot(range(2, 11), sil_scores, 'o-', label=f'{method} ({metric})')

    plt.title(f'Silhouette Scores for Different Linkage Methods and Distance Metrics ({metric})')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.grid()
    plt.show()
