import pandas as pd

# Load the dataset
aircraft_data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/dataset_stage1.csv")

# Drop rows with missing values in specific columns
aircraft_data.dropna(subset=['acft_model','latitude','longitude'], inplace=True)

# Step 1: Identify unique aircraft models and their corresponding engine types and number of engines
unique_models = aircraft_data.groupby('acft_model')[['eng_type', 'num_eng']].first().reset_index()

# Create a dictionary to map aircraft models to their engine types and number of engines
model_to_eng_info = unique_models.set_index('acft_model').T.to_dict('list')

# Function to fill missing values based on aircraft model
def fill_missing_eng_info(row):
    if pd.isna(row['eng_type']) or pd.isna(row['num_eng']):
        model = row['acft_model']
        if model in model_to_eng_info:
            if pd.isna(row['eng_type']):
                row['eng_type'] = model_to_eng_info[model][0]
            if pd.isna(row['num_eng']):
                row['num_eng'] = model_to_eng_info[model][1]
    return row

# Step 2: Apply the function to fill missing values
aircraft_data = aircraft_data.apply(fill_missing_eng_info, axis=1)

# Print the shape of the cleaned dataset to confirm changes
print(aircraft_data.shape)

# Optionally, save the cleaned dataset back to a file
aircraft_data.to_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/dataset_stage2.csv", index=False)

# Show a preview of the cleaned data
print(aircraft_data.head())

