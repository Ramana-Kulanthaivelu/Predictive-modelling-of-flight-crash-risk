# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:48:50 2024

@author: Ramana_
"""

import pandas as pd
#load the data
aircraft = pd.read_excel("C:/Users/Admin/Desktop/Dissertation/avall (1)/Features/aircraft.xlsx")
engines = pd.read_excel("C:/Users/Admin/Desktop/Dissertation/avall (1)/Features/engines.xlsx")
events = pd.read_excel("C:/Users/Admin/Desktop/Dissertation/avall (1)/Features/events.xlsx")
evt_sq = pd.read_excel("C:/Users/Admin/Desktop/Dissertation/avall (1)/Features/Events_Sequence.xlsx")
crew = pd.read_excel("C:/Users/Admin/Desktop/Dissertation/avall (1)/Features/Flight_Crew.xlsx")
weather = pd.read_excel("C:/Users/Admin/Desktop/Dissertation/avall (1)/Features/dt_events.xlsx")

print(aircraft)
print(engines)
print(events)
print(evt_sq)
print(crew)

#filter the required columns from each dataset
req_c1 = ['ev_id','Aircraft_Key','cert_max_gr_wt','acft_category','num_eng']
fd_aircraft = aircraft[req_c1]
print(fd_aircraft)

req_c2 = ['ev_id','Aircraft_Key','eng_no','eng_type']
fd_engines = engines[req_c2]
print(fd_engines)

req_c3 = ['ev_id','latitude','longitude','light_cond','sky_cond_nonceil','sky_cond_ceil','vis_sm','wx_temp',
          'wind_dir_deg','wind_vel_kts','altimeter']
fd_events = events[req_c3]
print(fd_events)

req_c4 = ['ev_id','Aircraft_Key','Occurrence_No','Occurrence_Code']
fd_evtsq = evt_sq[req_c4]
print(fd_evtsq)

req_c5 = ['ev_id','Aircraft_Key','crew_no','crew_category','crew_age','med_certf']
fd_crew = crew[req_c5]
print(fd_crew)

req_c6 = ['ev_id','col_name','code']
fd_weather = weather[req_c6]
print(fd_weather)

#merging the tables

merge_1 = pd.merge(
    pd.merge(
        pd.merge(fd_aircraft, fd_engines, on=['ev_id', 'Aircraft_Key']),
        fd_evtsq, on=['ev_id', 'Aircraft_Key']
    ),
    fd_crew, on=['ev_id', 'Aircraft_Key']
)

# Then, merge the resulting table with the fifth table on 'ev_id' only
merged_data = pd.merge(pd.merge(merge_1, fd_events, on='ev_id'),fd_weather,on= ['ev_id'])

# Print the result
print(merged_data)


# Filter rows based on the given conditions

# Filter for acft_category == 'AIR'
filtered_merged_data = merged_data[merged_data['acft_category'] == 'AIR']

# Further filter for eng_no == 1
filtered_merged_data = filtered_merged_data[filtered_merged_data['eng_no'] == 1]

# Further filter where col_name is either 'wx_brief_src' or 'wx_brief_src0'
filtered_merged_data = filtered_merged_data[
    (filtered_merged_data['col_name'] == 'wx_brief_src') | 
    (filtered_merged_data['col_name'] == 'wx_brief_src0')
]

# Display the filtered DataFrame
print(filtered_merged_data)


# List of unwanted columns
unwanted_clmns = ['acft_category', 'eng_no', 'col_name', 'Occurrence_No', 'ev_id', 'Aircraft_Key','crew_no']

# Dropping the unwanted columns from the filtered DataFrame
final_dataset = filtered_merged_data.drop(columns=unwanted_clmns)

print(final_dataset.columns)

import pandas as pd

# Assuming final_dataset is your DataFrame

# Define the desired column order
desired_order = [
    'Occurrence_Code', 'cert_max_gr_wt', 'num_eng', 'eng_type',
    'crew_category', 'crew_age', 'med_certf', 'latitude', 'longitude',
    'light_cond', 'sky_cond_nonceil', 'sky_cond_ceil', 'vis_sm', 'wx_temp',
    'wind_dir_deg', 'wind_vel_kts', 'altimeter', 'code'
]

# Reorder the DataFrame columns
final_dataset = final_dataset[desired_order]

# Display the reordered DataFrame
print(final_dataset)

# Save the reordered DataFrame to a CSV file
final_dataset.to_csv('final_dataset_reordered.csv', index=False)


import pandas as pd


# Calculate the number of NaN values in each column
nan_values = final_dataset.isna().sum()

# Calculate the number of missing values (same as NaN values)
missing_values = final_dataset.isnull().sum()

# Create a DataFrame to display the results
missing_data_info = pd.DataFrame({
    'Column': final_dataset.columns,
    'NaN Values': nan_values,
    'Missing Values': missing_values
})

# Display the result
print(missing_data_info)






