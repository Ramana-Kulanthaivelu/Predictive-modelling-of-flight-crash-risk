import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
aircraft_data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/dataset_stage3.csv")

# Define columns to impute and features to use
numerical_column = 'vis_sm'
features_for_imputation = ['wx_temp', 'latitude', 'longitude']

# Encode categorical variables if needed
label_encoders = {}
for col in ['light_cond', 'latitude', 'longitude']:
    le = LabelEncoder()
    aircraft_data[col] = le.fit_transform(aircraft_data[col].astype(str))  # Convert to string for safe encoding
    label_encoders[col] = le

# Extract features and target for KNN imputation
imputation_data = aircraft_data[features_for_imputation + [numerical_column]].copy()

# Initialize KNN imputer
imputer = KNNImputer(n_neighbors=5)  # Adjust the number of neighbors if necessary

# Impute missing values in 'vis_sm' based on the other features
imputation_data_imputed = imputer.fit_transform(imputation_data)

# Update the original DataFrame with imputed values
aircraft_data[numerical_column] = imputation_data_imputed[:, -1]  # Last column contains imputed 'vis_sm'

# Decode categorical variables back to original labels
for col in ['light_cond', 'latitude', 'longitude']:
    aircraft_data[col] = label_encoders[col].inverse_transform(aircraft_data[col].astype(int))

# Print the shape of the cleaned dataset to confirm changes
print(aircraft_data.shape)

# Optionally, save the cleaned dataset back to a file
aircraft_data.to_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/dataset_stage4.csv", index=False)

# Show a preview of the cleaned data
print(aircraft_data.head())
