import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
aircraft_data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/dataset_stage4.csv")

# Replace 0 values with NaN for 'wind_dir_deg' and 'altimeter'
aircraft_data['wind_dir_deg'].replace(0, np.nan, inplace=True)
aircraft_data['altimeter'].replace(0, np.nan, inplace=True)
aircraft_data['wind_vel_kts'].replace(0, np.nan, inplace=True)

# Define function to impute missing values using RandomForestRegressor
def impute_missing_values(df, target, predictors):
    # Drop rows where predictors have missing values
    train_data = df.dropna(subset=predictors + [target])
    
    # If there are no rows left after dropping, return the original DataFrame
    if train_data.empty:
        print(f"Not enough data to train the model for {target}")
        return df

    # Separate features and target
    X_train = train_data[predictors]
    y_train = train_data[target]
    
    # Handle missing values in predictors by filling them with mean (simple strategy)
    X_train.fillna(X_train.mean(), inplace=True)
    
    # Initialize and train the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Identify rows with missing target values
    missing_data = df[df[target].isna()]
    
    # Make predictions for missing values
    if not missing_data.empty:
        # Ensure predictors do not have missing values
        X_missing = missing_data[predictors]
        X_missing.fillna(X_missing.mean(), inplace=True)
        predicted_values = model.predict(X_missing)
        df.loc[df[target].isna(), target] = predicted_values

    return df

# Define predictors for imputation
predictors_for_wind_vel = ['wind_dir_deg', 'altimeter']  # Adjust if necessary
predictors_for_altimeter = ['wind_vel_kts', 'wind_dir_deg']  # Adjust if necessary
predictors_for_wind_deg  = ['wind_vel_kts','altimeter']
# Impute missing values for 'wind_vel_kts' and 'altimeter'
aircraft_data = impute_missing_values(aircraft_data, 'wind_vel_kts', predictors_for_wind_vel)
aircraft_data = impute_missing_values(aircraft_data, 'altimeter', predictors_for_altimeter)
aircraft_data = impute_missing_values(aircraft_data, 'wind_dir_deg', predictors_for_wind_deg)
# Print the shape of the cleaned dataset to confirm changes
print(aircraft_data.shape)

# Optionally, save the cleaned dataset back to a file
aircraft_data.to_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/dataset_stage5.csv", index=False)

# Show a preview of the cleaned data
print(aircraft_data.head())
