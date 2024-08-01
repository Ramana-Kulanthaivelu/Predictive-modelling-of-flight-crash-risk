# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:50:03 2024

@author: Ramana_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis

# Load the dataset
data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/reduced_dataset.csv")

# Define the features to use for factor analysis
features = ['crew_age', 'latitude', 'longitude', 'acft_model', 'wx_temp', 'wind_dir_deg', 'wind_vel_kts', 'altimeter']

# Standardize the features
data_standardized = (data[features] - data[features].mean()) / data[features].std()

# Calculate the correlation matrix
correlation_matrix = np.corrcoef(data_standardized, rowvar=False)

# Perform Eigen decomposition on the correlation matrix
eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

# Plot the scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', markersize=8)
plt.axhline(y=1, color='r', linestyle='-')
plt.title('Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# Number of factors to retain (eigenvalues > 1)
num_factors = np.sum(eigenvalues > 1)
print(f'Number of factors to retain: {num_factors}')

# Perform factor analysis
fa = FactorAnalysis(n_components=num_factors)
fa.fit(data_standardized)
factor_loadings = fa.components_.T

# Create a DataFrame to hold the factor loadings
factor_loadings_df = pd.DataFrame(factor_loadings, index=features, columns=[f'Factor {i+1}' for i in range(num_factors)])

# Plot the factor loadings
plt.figure(figsize=(12, 8))
sns.heatmap(factor_loadings_df, annot=True, cmap='coolwarm')
plt.title('Factor Loadings')
plt.show()

# Print the factor loadings
print(factor_loadings_df)
