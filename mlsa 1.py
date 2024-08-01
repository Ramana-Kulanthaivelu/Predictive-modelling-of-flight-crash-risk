# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:35:43 2024

@author: Ramana_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/reduced_dataset.csv")

# Define features for analysis
features = ['crew_age', 'latitude', 'longitude', 'acft_model', 'wx_temp', 'wind_dir_deg', 'wind_vel_kts', 'altimeter']
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=6)  # Specify the number of principal components
X_pca = pca.fit_transform(X_scaled)

# Get the loadings (components)
loadings = pca.components_.T

# Plot the loadings for the first 6 principal components
num_components = 6
num_features = len(features)

plt.figure(figsize=(14, 8))
for i in range(num_components):
    plt.plot(range(num_features), loadings[:, i], marker='o', linestyle='-', label=f'PC{i+1}')

plt.xticks(range(num_features), features, rotation=45)
plt.xlabel('Features')
plt.ylabel('Loadings')
plt.title('Loadings of the First 6 Principal Components')
plt.legend()
plt.grid()
plt.show()
