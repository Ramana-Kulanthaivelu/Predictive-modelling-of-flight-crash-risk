# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 02:32:51 2024

@author: Ramana_
"""
import pandas as pd

# Load the dataset
aircraft_data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/dataset_stage5.csv")

# Drop rows with any NaN values
cleaned_data = aircraft_data.dropna()

# Print the shape of the cleaned dataset
print(cleaned_data.shape)

# Optionally, save the cleaned dataset back to a file
cleaned_data.to_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/final_dataset_missingvalues.csv", index=False)

# Show a preview of the cleaned data
print(cleaned_data.head())
