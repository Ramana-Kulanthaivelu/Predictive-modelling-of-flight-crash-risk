# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:28:55 2024

@author: Ramana_
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/dataset_outlier_removed1.csv")

# Ensure the 'crew_age' column is numeric
data['crew_age'] = pd.to_numeric(data['crew_age'], errors='coerce')

# Plotting
plt.figure(figsize=(12, 8))

# Create a box plot
plt.boxplot(
    [data[data['crew_category'] == category]['crew_age'].dropna() for category in data['crew_category'].unique()],
    labels=data['crew_category'].unique()
)

# Customize plot
plt.title('Age Distribution by Pilot Category')
plt.xlabel('Pilot Category')
plt.ylabel('Age')
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
