# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:16:50 2024

@author: Ramana_
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/dataset_outlier_removed1.csv")

# Define a function to calculate the Z-scores
def calculate_z_scores(data):
    return (data - data.mean()) / data.std()

# Plot Z-scores for each feature
def plot_z_scores(data, features):
    z_scores = calculate_z_scores(data[features])
    
    # Cap Z-scores at -1 and 10 for better visualization
    z_scores = z_scores.applymap(lambda x: min(max(x, -1), 10))
    
    plt.figure(figsize=(15, 10))
    
    for feature in features:
        plt.scatter(data.index, z_scores[feature], label=feature, alpha=0.5)
    
    plt.axhline(y=1, color='r', linestyle='-')
    plt.axhline(y=-1, color='r', linestyle='-')
    
    plt.xlabel('Index')
    plt.ylabel('Z-score')
    plt.title('Z-scores of Features')
    plt.legend(loc='upper right')
    plt.show()

# Example usage
categorical_columns = ['latitude', 'longitude', 'code', 'Occurrence_Code', 'acft_model', 'light_cond', 'eng_type']
features = ['vis_sm', 'crew_age', 'num_eng', 'wx_temp', 'wind_dir_deg', 'wind_vel_kts', 'altimeter'] + categorical_columns

# Plot Z-scores for each feature
plot_z_scores(data, features)
