import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
aircraft_data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/dataset_stage2.csv")

# Drop rows with missing values in specific columns
# Resetting index after dropping rows

# Encode categorical variables
label_encoders = {}
for col in ['crew_category', 'acft_model', 'eng_type']:
    le = LabelEncoder()
    aircraft_data[col] = le.fit_transform(aircraft_data[col].astype(str))  # Convert to string for safe encoding
    label_encoders[col] = le

# Function to check if there are enough samples for training
def has_enough_samples(df, predictors):
    return not df[predictors].isnull().any(axis=1).all()

# Select relevant features for filling missing values
predictor_columns = ['acft_model', 'eng_type']

def impute_missing_values(df, feature, predictors):
    if not has_enough_samples(df, predictors):
        print(f"Not enough samples to train the model for {feature}")
        return

    not_null_data = df[df[feature].notnull()]
    null_data = df[df[feature].isnull()]

    # Ensure predictors do not have missing values in not_null_data
    not_null_data = not_null_data.dropna(subset=predictors)

    model = DecisionTreeClassifier()
    model.fit(not_null_data[predictors], not_null_data[feature])

    # Filter null_data to ensure predictors do not have missing values
    valid_null_data = null_data.dropna(subset=predictors)
    
    if not valid_null_data.empty:
        predicted_values = model.predict(valid_null_data[predictors])
        df.loc[valid_null_data.index, feature] = predicted_values

# Impute missing values for crew_category
impute_missing_values(aircraft_data, 'crew_category', predictor_columns)

# Impute missing values for crew_age
def impute_missing_crew_age(df):
    predictors = ['acft_model', 'eng_type', 'crew_category']
    if not has_enough_samples(df, predictors):
        print(f"Not enough samples to train the model for crew_age")
        return

    not_null_data = df[df['crew_age'].notnull()]
    null_data = df[df['crew_age'].isnull()]

    # Ensure predictors do not have missing values in not_null_data
    not_null_data = not_null_data.dropna(subset=predictors)

    model = DecisionTreeClassifier()
    model.fit(not_null_data[predictors], not_null_data['crew_age'])

    # Filter null_data to ensure predictors do not have missing values
    valid_null_data = null_data.dropna(subset=predictors)
    
    if not valid_null_data.empty:
        predicted_values = model.predict(valid_null_data[predictors])
        df.loc[valid_null_data.index, 'crew_age'] = predicted_values

impute_missing_crew_age(aircraft_data)

# Encode crew_category and acft_model and eng_type back
for col in ['crew_category', 'acft_model', 'eng_type']:
    aircraft_data[col] = label_encoders[col].inverse_transform(aircraft_data[col])

# Print the shape of the cleaned dataset to confirm changes
print(aircraft_data.shape)

# Optionally, save the cleaned dataset back to a file
aircraft_data.to_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/dataset_stage3.csv", index=False)

# Show a preview of the cleaned data
print(aircraft_data.head())
