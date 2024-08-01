# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:20:09 2024

@author: Ramana_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/dataset_outlier_removed1.csv")
sns.boxplot(x='Occurrence_Code', y='wind_vel_kts', data=data)
plt.title('Wind Speed by Occurrence Code')
plt.xlabel('Occurrence Code')
plt.ylabel('Wind Speed (kts)')
plt.show()


sns.violinplot(x='light_cond', y='vis_sm', data=data)
plt.title('Visibility by Light Conditions')
plt.xlabel('Light Conditions')
plt.ylabel('Visibility (sm)')
plt.show()


sns.pairplot(data[['vis_sm', 'crew_age', 'num_eng', 'wx_temp', 'wind_dir_deg', 'wind_vel_kts']])
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data1 = pd.read_excel("C:/Users/Admin/Desktop/Dissertation/avall (1)/Features/events.xlsx")

# Convert 'ev_date' to datetime
data1['ev_date'] = pd.to_datetime(data1['ev_date'])

# Filter data up to the year 2023
data1 = data1[data1['ev_date'] <= '2023-12-31']

# Set 'ev_date' as the index and resample by year
data1.set_index('ev_date')['ev_id'].resample('Y').count().plot()

# Add titles and labels
plt.title('Yearly Incidents Up to 2023')
plt.xlabel('Date')
plt.ylabel('Number of Incidents')
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset (assuming it contains only anomalies)
data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/dataset_outlier_removed1.csv")

categorical_columns = ['latitude', 'longitude', 'code', 'Occurrence_Code', 'acft_model', 'light_cond', 'eng_type']

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Select features for anomaly detection
features = ['vis_sm', 'crew_age', 'num_eng', 'wx_temp', 'wind_dir_deg', 'wind_vel_kts', 'altimeter'] + categorical_columns

X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data['Occurrence_Code'], cmap='viridis')
plt.title('t-SNE Visualization of Feature Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
