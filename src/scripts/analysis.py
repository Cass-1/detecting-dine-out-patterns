# %% [markdown]
# # Get Data

# %% [markdown]
# ## Import and Read In

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from haversine import haversine_vector, Unit
import seaborn as sns
from IPython.display import display
import folium
from folium.plugins import HeatMap, FastMarkerCluster, MarkerCluster
from scipy.stats import kstest

# %%
def clean_restaurant_data(df):
    df.columns = df.columns.str.lower()
    df['longitude'] = df['longitude'] / 1000000 * -1
    df['latitude'] = df['latitude'] / 1000000
    return df

# Load the data
file_path = '../../data/real_movements.csv'
movements_data = pd.read_csv(file_path, parse_dates=['datetime'])
restaurant_data = pd.read_csv("../../data/real_restaurants.csv")  # Replace with your file path
visits_data = pd.read_csv("../../data/visits.csv")

# cleaning
restaurant_data = clean_restaurant_data(restaurant_data)
movements_data = movements_data.drop_duplicates(subset=['datetime', 'id'])

# %% [markdown]
# ## Distance

# %%

# Define functions
def get_distance_between_rows(df):
    """
    Calculate the Haversine distance between consecutive rows in a DataFrame (in kilometers)
    Parameters:
    df (pandas.DataFrame): DataFrame containing 'latitude' and 'longitude' columns.
    Returns:
    numpy.ndarray: Array of distances between consecutive rows.
    """
    
    coords1 = df[['latitude', 'longitude']].values[:-1]
    coords2 = df[['latitude', 'longitude']].values[1:]
    distances = haversine_vector(coords1, coords2, Unit.KILOMETERS)
    return distances

# check
def get_time_between_rows(df):
    """
    Calculate the time difference between consecutive rows in a DataFrame (in hours)
    Parameters:
    df (pandas.DataFrame): DataFrame containing a 'datetime' column with date and time information.
    Returns: numpy.ndarray: Array of time differences between consecutive rows.
    """

    date = pd.to_datetime(df['datetime'])
    datetime1 = date.values[:-1]
    datetime2 = date.values[1:]
    difference = (datetime2 - datetime1).astype('timedelta64[s]')
    getHours = np.vectorize(lambda x: x / np.timedelta64(1, 'h'))
    return getHours(difference)


# get distance and time between each measurement
grouped = movements_data.groupby('id')
distance_dict = {}
time_dict = {}
midpoints_latitude_dict = {}
midpoints_longitude_dict = {}

for name, group in grouped:
    distances = get_distance_between_rows(group)
    times = get_time_between_rows(group)
    distance_dict[name] = distances
    time_dict[name] = times
    midpoints_latitude = (group['latitude'].values[:-1] + group['latitude'].values[1:]) / 2
    midpoints_longitude = (group['longitude'].values[:-1] + group['longitude'].values[1:]) / 2
    midpoints_latitude_dict[name] = midpoints_latitude
    midpoints_longitude_dict[name] = midpoints_longitude

# replaces missing items w/ NaN
distance_results = pd.DataFrame({ key:pd.Series(value) for key, value in distance_dict.items() })
time_results = pd.DataFrame({ key:pd.Series(value) for key, value in time_dict.items() })

# %% [markdown]
# ## Velocity

# %%
# get velocity
velocity_results = distance_results / time_results
velocity_results.columns = [f'Velocity_{col}' for col in velocity_results.columns]

# add midpoints data to velocity_results
for name in midpoints_latitude_dict.keys():
    velocity_results[f'Midpoint_Latitude_{name}'] = pd.Series(midpoints_latitude_dict[name])
    velocity_results[f'Midpoint_Longitude_{name}'] = pd.Series(midpoints_longitude_dict[name])

velocity_results['Velocity_121'] = velocity_results['Velocity_121'].abs()
velocity_results['Velocity_125'] = velocity_results['Velocity_125'].abs()
velocity_results.head()

# %%
velocity_results_121 = velocity_results[['Velocity_121', 'Midpoint_Latitude_121', 'Midpoint_Longitude_121']]
velocity_results_125 = velocity_results[['Velocity_125', 'Midpoint_Latitude_125', 'Midpoint_Longitude_125']]

velocity_results_125.dropna(how='all')

# %% [markdown]
# ## Removing outliers

# %% [markdown]
# ### Finding Distributions

# %%
distance_results.columns = distance_results.columns.astype(str)
#  boxplot
plt.boxplot(x=distance_results['121'], vert=False, showfliers=True,)
plt.title("Boxplot of Distance for ID 121 with Outliers")
plt.xlabel("Distance (km)")
plt.show()

# histogram
plt.hist(distance_results['121'], log=True)
plt.title("Histogram of Distance Traveled Every Measurement for ID 121")
plt.xlabel("Distance (km)")
plt.ylabel("Frequency")
plt.show()

# histogram
sns.histplot(distance_results['121'].abs(), bins=100, log_scale=(True, False))
plt.xlabel(f'Distance (km)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Distance for ID 121')

# summary stats
display(distance_results['121'].describe()) 

# %%
plt.boxplot(distance_results['125'].dropna(), vert=False, showfliers=False)
plt.title("Boxplot of Distance for ID 125 without Outliers")
plt.xlabel("Distance (km)")
plt.show()
plt.hist(distance_results['125'], log=True)
plt.show()
display(distance_results['125'].describe()) 

# histogram
sns.histplot(distance_results['125'].abs(), bins=100, log_scale=(True, False))
plt.xlabel(f'Distance (km)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Distance for ID 125')

# %%
velocity_results_121_clean = velocity_results_121[velocity_results_121['Velocity_121'].abs() <= 1200]
velocity_results_125_clean = velocity_results_125[velocity_results_125['Velocity_125'].abs() <= 1200]

# %%
plt.boxplot(x=velocity_results_121_clean['Velocity_121'], vert=False, showfliers=False)
plt.show()
plt.hist(velocity_results_121_clean['Velocity_121'], log=True)
plt.show()
display(velocity_results_121_clean['Velocity_121'].describe()) 

# %%
plt.boxplot(x=velocity_results_125_clean['Velocity_125'].dropna(), vert=False, showfliers=False)
plt.show()
plt.hist(velocity_results_125_clean['Velocity_125'], log=True)
plt.show()
display(velocity_results_125_clean['Velocity_125'].describe()) 


# %%
print((len(velocity_results_121_clean) - len(velocity_results_121)) / len(velocity_results_121))
display(velocity_results_121_clean.describe())
display(velocity_results_121.describe())

# %%
print((len(velocity_results_125_clean) - len(velocity_results_125.dropna())) / len(velocity_results_125.dropna()))
display(velocity_results_125_clean.describe())
display(velocity_results_125.describe())


# %%
# mean = distance_results['121'].mean()
# std = distance_results['121'].std()
# threshold = 3
# outlier = []

# z_scores_121 = pd.DataFrame((distance_results['121'] - mean) / std)
# z_scores_121 = z_scores_121.rename(columns={'121': 'Z_Scores_121'})
# # z_scores_121 = z_scores_121[z_scores_121['Z_Scores_121'].abs() > threshold]

# display(z_scores_121.describe())
# z_scores_121_abovebound = z_scores_121[z_scores_121['Z_Scores_121'].abs() > threshold]
# display(z_scores_121_abovebound.describe())

# testing normal distribution

kstest(velocity_results['Velocity_121'], 'norm') # not normal
kstest(velocity_results['Velocity_125'].dropna(), 'norm') # not normal


# %%
(velocity_results_121_clean[velocity_results_121_clean['Velocity_121'] == 0]).count()

# %%
DRIVING_SPEED_MAX = 130
DRIVING_SPEED_MIN = 7

velocity_results_125_clean['travel_type'] = pd.Series()
velocity_results_125_clean.loc[velocity_results_125_clean['Velocity_125'] > DRIVING_SPEED_MAX, 'travel_type'] = 'Plane'
velocity_results_125_clean.loc[velocity_results_125_clean['Velocity_125'] <= DRIVING_SPEED_MAX, 'travel_type'] = 'Driving'
velocity_results_125_clean.loc[velocity_results_125_clean['Velocity_125'] < DRIVING_SPEED_MIN , 'travel_type'] = 'Walking'
velocity_results_125_clean.loc[velocity_results_125_clean['Velocity_125'] == 0 , 'travel_type'] = 'Not Moving'


plt.figure(figsize=(12, 8))
plt.barh(velocity_results_125_clean['travel_type'].value_counts().index, velocity_results_125_clean['travel_type'].value_counts())
# Add percentages to the end of each bar
for index, value in enumerate(velocity_results_125_clean['travel_type'].value_counts()):
    plt.text(value, index, f'{value / len(velocity_results_125_clean) * 100:.2f}%', va='center')

plt.xlabel('Count')
plt.ylabel('Travel Type')
plt.title('Distribution of Travel Types for Person 125')
plt.show()

# %%
velocity_results_121_clean['travel_type'] = pd.Series()
velocity_results_121_clean.loc[velocity_results_121_clean['Velocity_121'] > DRIVING_SPEED_MAX, 'travel_type'] = 'Plane'
velocity_results_121_clean.loc[velocity_results_121_clean['Velocity_121'] <= DRIVING_SPEED_MAX, 'travel_type'] = 'Driving'
velocity_results_121_clean.loc[velocity_results_121_clean['Velocity_121'] < DRIVING_SPEED_MIN , 'travel_type'] = 'Walking'
velocity_results_121_clean.loc[velocity_results_121_clean['Velocity_121'] == 0 , 'travel_type'] = 'Not Moving'


plt.figure(figsize=(12, 8))
plt.barh(velocity_results_121_clean['travel_type'].value_counts().index, velocity_results_121_clean['travel_type'].value_counts())
# Add percentages to the end of each bar
for index, value in enumerate(velocity_results_121_clean['travel_type'].value_counts()):
    plt.text(value, index, f'{value / len(velocity_results_121_clean) * 100:.2f}%', va='center')

plt.xlabel('Count')
plt.ylabel('Travel Type')
plt.title('Distribution of Travel Types for Person 121')
plt.show()

# %%


# %% [markdown]
# # GeoMaps

# %% [markdown]
# ## Restaurants

# %%
marker_cluster = MarkerCluster()
restaurant_data.drop_duplicates().apply(lambda x: marker_cluster.add_child(folium.Marker(
    location=[x['latitude'], x['longitude']],
    popup=f"{x['name']}\n{x['category']}",
    )), axis=1)

restaurant_map = folium.Map(location=[restaurant_data['latitude'].mean(), restaurant_data['longitude'].mean()], zoom_start=5)
restaurant_map.add_child(marker_cluster)

# %% [markdown]
# ## Visits

# %% [markdown]
# ### Prep Visits Data

# %%
visits_data = visits_data.merge(restaurant_data[['restaurant id', 'latitude', 'longitude','name', 'category']], on='restaurant id', how='left')

marker_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred','lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
# if (len(visits_data['category'].unique()) > marker_colors):
#     raise Exception("not enough colors")

color_index = 0

for item in visits_data['category'].unique():
    visits_data.loc[visits_data['category'] == item, 'color'] = marker_colors[color_index]
    color_index += 1

# %% [markdown]
# ### Person 121

# %%
person_121 = movements_data[movements_data['id'] == 121]
visits_121 = visits_data[visits_data['id'] == 121]

# Create a base map
base_map = folium.Map(location=[person_121['latitude'].mean(), person_121['longitude'].mean()], zoom_start=5)

# Extract the latitude and longitude data
heat_data = person_121[['latitude', 'longitude']].dropna().values.tolist()

# Add the heatmap layer
HeatMap(heat_data).add_to(base_map)

# Add clustered places as pins on the map
for _, row in visits_121.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Time: {row['start_time']} - {row['end_time']}\nName: {row['name']}\nType: {row['category']}",
        icon=folium.Icon(color=row['color'])
    ).add_to(base_map)

# Display the map
base_map


# %% [markdown]
# ### Person 125

# %%
visits_125 = visits_data[visits_data['id'] == 125]
person_125 = movements_data[movements_data['id'] == 125]


# Create a base map
base_map = folium.Map(location=[person_125['latitude'].mean(), person_125['longitude'].mean()], zoom_start=5)

# Extract the latitude and longitude data
heat_data = person_125[['latitude', 'longitude']].dropna().values.tolist()

# Add the heatmap layer
HeatMap(heat_data).add_to(base_map)

# Add clustered places as pins on the map
visits_125.apply(lambda row: folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Time: {row['start_time']} - {row['end_time']}\nName: {row['name']}\nType: {row['category']}",
        icon=folium.Icon(color=row['color'])
    ).add_to(base_map), axis=1)
    

# Display the map
base_map

# %% [markdown]
# ## Velocity

# %% [markdown]
# ### Person 121

# %%
velocity_results_121 = pd.DataFrame({
    'latitude': velocity_results_121_clean['Midpoint_Latitude_121'],
    'longitude': velocity_results_121_clean['Midpoint_Longitude_121'],
    'velocity': velocity_results_121_clean['Velocity_121']
})
velocity_results_121 = velocity_results_121.dropna(subset=['latitude', 'longitude', 'velocity'])
# Create a base map centered at the mean latitude and longitude
base_map = folium.Map(location=[velocity_results_121['latitude'].mean(), velocity_results_121['longitude'].mean()], zoom_start=5)

# Prepare the data for the heatmap
heat_data = velocity_results_121[['latitude', 'longitude', 'velocity']].values.tolist()

# Add the heatmap layer
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(base_map)

# Display the map
base_map

# %% [markdown]
# ### Person 125

# %%
velocity_results_125 = pd.DataFrame({
    'latitude': velocity_results_125_clean['Midpoint_Latitude_125'],
    'longitude': velocity_results_125_clean['Midpoint_Longitude_125'],
    'velocity': velocity_results_125_clean['Velocity_125']
})
velocity_results_125 = velocity_results_125.dropna(subset=['latitude', 'longitude', 'velocity'])
# Create a base map centered at the mean latitude and longitude
base_map = folium.Map(location=[velocity_results_125['latitude'].mean(), velocity_results_125['longitude'].mean()], zoom_start=5)

# Prepare the data for the heatmap
heat_data = velocity_results_125[['latitude', 'longitude', 'velocity']].values.tolist()

# Add the heatmap layer
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(base_map)

# Display the map
base_map

# %% [markdown]
# ## Restaurants and Movement

# %%
restaurant_map_121 = restaurant_map
restaurant_map_121.add_child(HeatMap(person_121[['latitude', 'longitude']].dropna().values.tolist()))

# %%
restaurant_map_125 = restaurant_map
restaurant_map_125.add_child(HeatMap(person_125[['latitude', 'longitude']].dropna().values.tolist()))

# %% [markdown]
# # Velocity Distribution

# %%
# Plot the histogram with logarithmic scaling
sns.histplot(velocity_results_121_clean['Velocity_121'], bins=100, log_scale=(True, False))

# Add labels and title
plt.xlabel(f' Speed (km/h)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Speed for ID 121')

# Show the plot
plt.show()

    # Plot the histogram with logarithmic scaling
sns.histplot(velocity_results_125_clean['Velocity_125'], bins=100, log_scale=(True, False))

# Add labels and title
plt.xlabel(f' Speed (km/h)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Speed for ID 125')

# Show the plot
plt.show()

# %% [markdown]
# # Rough Maps

# %% [markdown]
# ## Person 121

# %%
plt.figure(figsize=(10, 6))

# Plot the restaurant locations

plt.scatter(restaurant_data['longitude'], restaurant_data['latitude'], label='restaurants', color='red', marker='s', s=10, alpha=0.7)
# Plot the person's path
person_data = person_121.dropna(subset=['longitude', 'latitude'])
plt.plot(person_data['longitude'], person_data['latitude'], label=f'Path of {121}', color='blue', alpha=0.6)

# Add title and labels
plt.title(f'{121}\'s Path with Restaurant Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
# Count the number of visits to each location
visit_counts = person_data.groupby(['longitude', 'latitude']).size().reset_index(name='counts')

# Plot the restaurant locations with visit weights
plt.figure(figsize=(10, 6))
plt.scatter(restaurant_data['longitude'], restaurant_data['latitude'], label='restaurants', color='red', marker='s',s=10, alpha=0.7)
plt.scatter(visit_counts['longitude'], visit_counts['latitude'], s=visit_counts['counts'], label=f'Path of {121}', color='blue', alpha=0.6)

# Add title and labels
plt.title(f'Restaurant Locations and {121}\'s Path with Visit Weights')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

display(visit_counts.sort_values(by='counts', ascending=False).head(10))

# %% [markdown]
# ## Person 125

# %%
plt.figure(figsize=(10, 6))

# FULL GRAPH
plt.scatter(restaurant_data['longitude'], restaurant_data['latitude'], label='restaurants', color='red', marker='s', s=10, alpha=0.7)
# Plot the person's path
person_data = person_125.dropna(subset=['longitude', 'latitude'])

# plot person's movement
plt.plot(person_data['longitude'], person_data['latitude'], label=f'Path of {125}', color='blue', alpha=0.6)

# add title and labels
plt.title(f'{125}\'s Path and R')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




visit_counts = person_data.groupby(['longitude', 'latitude']).size().reset_index(name='counts')

# Plot the restaurant locations with visit weights
plt.figure(figsize=(10, 6))
plt.scatter(restaurant_data['longitude'], restaurant_data['latitude'], label='restaurants', color='red', marker='s', alpha=0.7)
plt.scatter(visit_counts['longitude'], visit_counts['latitude'], s=visit_counts['counts'], label=f'Path of {125}', color='blue', alpha=0.6)

# Add title and labels
plt.title(f'Restaurant Locations and {125}\'s Path with Visit Weights')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

display(visit_counts.sort_values(by='counts', ascending=False).head(10))

# %% [markdown]
# # Sleep

# %%
def longest_stretch_no_movement(df, threshold, movements_data):
    """input in seconds and km"""
    longest_stretch = {}
    stretch_timestamps = {}
    stretch_durations = {}
    
    for col in df.columns:
        small_movement = df[col] <= threshold
        max_stretch = 0
        current_stretch = 0
        start_index = 0
        end_index = 0
        
        for i, movement in enumerate(small_movement):
            if movement:
                if current_stretch == 0:
                    current_start_index = i
                current_stretch += 1
            else:
                if current_stretch > max_stretch:
                    max_stretch = current_stretch
                    start_index = current_start_index
                    end_index = i - 1
                current_stretch = 0
        
        # Check the last stretch
        if current_stretch > max_stretch:
            max_stretch = current_stretch
            start_index = current_start_index
            end_index = len(small_movement) - 1
        
        longest_stretch[col] = max_stretch
        start_time = movements_data.iloc[start_index]['datetime']
        end_time = movements_data.iloc[end_index]['datetime']
        stretch_timestamps[col] = (start_time, end_time)
        stretch_durations[col] = (end_time - start_time).total_seconds() / 3600  # duration in hours
    
    return longest_stretch, stretch_timestamps, stretch_durations

LOW_MOVEMENT_BOUND = 0
# Find the longest stretch of no movement for each individual, their timestamps, and durations
longest_stretch_no_movement_results, stretch_timestamps, stretch_durations = longest_stretch_no_movement(velocity_results, LOW_MOVEMENT_BOUND, movements_data)

# Turn the results into a DataFrame
longest_stretch_df = pd.DataFrame({
    'Longest Stretch (no movement)': longest_stretch_no_movement_results,
    'Start Time': {k: v[0] for k, v in stretch_timestamps.items()},
    'End Time': {k: v[1] for k, v in stretch_timestamps.items()},
    'Duration (hours)': stretch_durations
})

longest_stretch_df.head(10)

# %% [markdown]
# # Summary Stats

# %%
distance_results.apply(lambda x: x.describe())

# %%
velocity_results['Velocity_121'].describe()


