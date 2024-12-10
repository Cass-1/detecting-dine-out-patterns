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
# Load the data
file_path = '../../data/movements.csv'
movements_data = pd.read_csv(file_path, parse_dates=['datetime'])
restaurant_data = pd.read_csv("../../data/restaurants.csv")  # Replace with your file path
visits_data = pd.read_csv("../../data/visits.csv")

# cleaning
restaurant_data.columns = restaurant_data.columns.str.lower()
movements_data = movements_data.drop_duplicates(subset=['datetime', 'id'])

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

# %%
# get velocity
velocity_results = (distance_results / time_results).abs()
velocity_results.columns = [f'Velocity_{col}' for col in velocity_results.columns]

# add midpoints data to velocity_results
for name in midpoints_latitude_dict.keys():
    velocity_results[f'Midpoint_Latitude_{name}'] = pd.Series(midpoints_latitude_dict[name])
    velocity_results[f'Midpoint_Longitude_{name}'] = pd.Series(midpoints_longitude_dict[name])

# %%
velocity_results_I000 = velocity_results[['Velocity_I000', 'Midpoint_Latitude_I000', 'Midpoint_Longitude_I000']].dropna(how='all')
velocity_results_I001 = velocity_results[['Velocity_I001', 'Midpoint_Latitude_I001', 'Midpoint_Longitude_I001']].dropna(how='all')
velocity_results_I002 = velocity_results[['Velocity_I002', 'Midpoint_Latitude_I002', 'Midpoint_Longitude_I002']].dropna(how='all')
velocity_results_I003 = velocity_results[['Velocity_I003', 'Midpoint_Latitude_I003', 'Midpoint_Longitude_I003']].dropna(how='all')
velocity_results_I004 = velocity_results[['Velocity_I004', 'Midpoint_Latitude_I004', 'Midpoint_Longitude_I004']].dropna(how='all')
velocity_results_I005 = velocity_results[['Velocity_I005', 'Midpoint_Latitude_I005', 'Midpoint_Longitude_I005']].dropna(how='all')
velocity_results_I006 = velocity_results[['Velocity_I006', 'Midpoint_Latitude_I006', 'Midpoint_Longitude_I006']].dropna(how='all')
velocity_results_I007 = velocity_results[['Velocity_I007', 'Midpoint_Latitude_I007', 'Midpoint_Longitude_I007']].dropna(how='all')
velocity_results_I008 = velocity_results[['Velocity_I008', 'Midpoint_Latitude_I008', 'Midpoint_Longitude_I008']].dropna(how='all')
velocity_results_I009 = velocity_results[['Velocity_I009', 'Midpoint_Latitude_I009', 'Midpoint_Longitude_I009']].dropna(how='all')

# %%
# histogram
sns.histplot(distance_results['I000'].abs(), bins=100, log_scale=(True, False))
plt.xlabel(f'Distance (km)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Distance for ID I000')
plt.show()

# histogram
sns.histplot(distance_results['I001'].abs(), bins=100, log_scale=(True, False))
plt.xlabel(f'Distance (km)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Distance for ID I001')
plt.show()

# histogram
sns.histplot(distance_results['I002'].abs(), bins=100, log_scale=(True, False))
plt.xlabel(f'Distance (km)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Distance for ID I002')
plt.show()

# histogram
sns.histplot(distance_results['I003'].abs(), bins=100, log_scale=(True, False))
plt.xlabel(f'Distance (km)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Distance for ID I003')
plt.show()

# histogram
sns.histplot(distance_results['I004'].abs(), bins=100, log_scale=(True, False))
plt.xlabel(f'Distance (km)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Distance for ID I004')
plt.show()

# histogram
sns.histplot(distance_results['I005'].abs(), bins=100, log_scale=(True, False))
plt.xlabel(f'Distance (km)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Distance for ID I005')
plt.show()

# histogram
sns.histplot(distance_results['I006'].abs(), bins=100, log_scale=(True, False))
plt.xlabel(f'Distance (km)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Distance for ID I006')
plt.show()

# histogram
sns.histplot(distance_results['I007'].abs(), bins=100, log_scale=(True, False))
plt.xlabel(f'Distance (km)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Distance for ID I007')
plt.show()

# histogram
sns.histplot(distance_results['I008'].abs(), bins=100, log_scale=(True, False))
plt.xlabel(f'Distance (km)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Distance for ID I008')
plt.show()

# histogram
sns.histplot(distance_results['I009'].abs(), bins=100, log_scale=(True, False))
plt.xlabel(f'Distance (km)')
plt.ylabel('Frequency')
plt.title(f'Logrithmic Histogram of Distance for ID I009')
plt.show()

# %%
velocity_results_I000 = velocity_results_I000[velocity_results_I000.abs() <= 1200]
velocity_results_I001 = velocity_results_I001[velocity_results_I001.abs() <= 1200]
velocity_results_I002 = velocity_results_I002[velocity_results_I002.abs() <= 1200]
velocity_results_I003 = velocity_results_I003[velocity_results_I003.abs() <= 1200]
velocity_results_I004 = velocity_results_I004[velocity_results_I004.abs() <= 1200]
velocity_results_I005 = velocity_results_I005[velocity_results_I005.abs() <= 1200]
velocity_results_I006 = velocity_results_I006[velocity_results_I006.abs() <= 1200]
velocity_results_I007 = velocity_results_I007[velocity_results_I007.abs() <= 1200]
velocity_results_I008 = velocity_results_I008[velocity_results_I008.abs() <= 1200]
velocity_results_I009 = velocity_results_I009[velocity_results_I009.abs() <= 1200]

# %%
# histogram for velocity I000
sns.histplot(velocity_results_I000['Velocity_I000'], bins=100, log_scale=(True, False))
plt.xlabel(f'Speed (km/h)')
plt.ylabel('Frequency')
plt.title(f'Logarithmic Histogram of Speed for ID I000')
plt.show()

# histogram for velocity I001
sns.histplot(velocity_results_I001['Velocity_I001'], bins=100, log_scale=(True, False))
plt.xlabel(f'Speed (km/h)')
plt.ylabel('Frequency')
plt.title(f'Logarithmic Histogram of Speed for ID I001')
plt.show()

# histogram for velocity I002
sns.histplot(velocity_results_I002['Velocity_I002'], bins=100, log_scale=(True, False))
plt.xlabel(f'Speed (km/h)')
plt.ylabel('Frequency')
plt.title(f'Logarithmic Histogram of Speed for ID I002')
plt.show()

# histogram for velocity I003
sns.histplot(velocity_results_I003['Velocity_I003'], bins=100, log_scale=(True, False))
plt.xlabel(f'Speed (km/h)')
plt.ylabel('Frequency')
plt.title(f'Logarithmic Histogram of Speed for ID I003')
plt.show()

# histogram for velocity I004
sns.histplot(velocity_results_I004['Velocity_I004'], bins=100, log_scale=(True, False))
plt.xlabel(f'Speed (km/h)')
plt.ylabel('Frequency')
plt.title(f'Logarithmic Histogram of Speed for ID I004')
plt.show()

# histogram for velocity I005
sns.histplot(velocity_results_I005['Velocity_I005'], bins=100, log_scale=(True, False))
plt.xlabel(f'Speed (km/h)')
plt.ylabel('Frequency')
plt.title(f'Logarithmic Histogram of Speed for ID I005')
plt.show()

# histogram for velocity I006
sns.histplot(velocity_results_I006['Velocity_I006'], bins=100, log_scale=(True, False))
plt.xlabel(f'Speed (km/h)')
plt.ylabel('Frequency')
plt.title(f'Logarithmic Histogram of Speed for ID I006')
plt.show()

# histogram for velocity I007
sns.histplot(velocity_results_I007['Velocity_I007'], bins=100, log_scale=(True, False))
plt.xlabel(f'Speed (km/h)')
plt.ylabel('Frequency')
plt.title(f'Logarithmic Histogram of Speed for ID I007')
plt.show()

# histogram for velocity I008
sns.histplot(velocity_results_I008['Velocity_I008'], bins=100, log_scale=(True, False))
plt.xlabel(f'Speed (km/h)')
plt.ylabel('Frequency')
plt.title(f'Logarithmic Histogram of Speed for ID I008')
plt.show()

# histogram for velocity I009
sns.histplot(velocity_results_I009['Velocity_I009'], bins=100, log_scale=(True, False))
plt.xlabel(f'Speed (km/h)')
plt.ylabel('Frequency')
plt.title(f'Logarithmic Histogram of Speed for ID I009')
plt.show()

# %%
velocity_results_I000_clean = velocity_results_I000.dropna(subset=['Midpoint_Latitude_I000', 'Midpoint_Longitude_I000', 'Velocity_I000'])
velocity_results_I000_df = pd.DataFrame({
    'latitude': velocity_results_I000_clean['Midpoint_Latitude_I000'],
    'longitude': velocity_results_I000_clean['Midpoint_Longitude_I000'],
    'velocity': velocity_results_I000_clean['Velocity_I000']
})
base_map = folium.Map(location=[velocity_results_I000_df['latitude'].mean(), velocity_results_I000_df['longitude'].mean()], zoom_start=5)
heat_data = velocity_results_I000_df[['latitude', 'longitude', 'velocity']].values.tolist()
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(base_map)
base_map

# %%
velocity_results_I001_clean = velocity_results_I001.dropna(subset=['Midpoint_Latitude_I001', 'Midpoint_Longitude_I001', 'Velocity_I001'])
velocity_results_I001_df = pd.DataFrame({
    'latitude': velocity_results_I001_clean['Midpoint_Latitude_I001'],
    'longitude': velocity_results_I001_clean['Midpoint_Longitude_I001'],
    'velocity': velocity_results_I001_clean['Velocity_I001']
})
base_map = folium.Map(location=[velocity_results_I001_df['latitude'].mean(), velocity_results_I001_df['longitude'].mean()], zoom_start=5)
heat_data = velocity_results_I001_df[['latitude', 'longitude', 'velocity']].values.tolist()
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(base_map)
base_map

# %%
velocity_results_I002_clean = velocity_results_I002.dropna(subset=['Midpoint_Latitude_I002', 'Midpoint_Longitude_I002', 'Velocity_I002'])
velocity_results_I002_df = pd.DataFrame({
    'latitude': velocity_results_I002_clean['Midpoint_Latitude_I002'],
    'longitude': velocity_results_I002_clean['Midpoint_Longitude_I002'],
    'velocity': velocity_results_I002_clean['Velocity_I002']
})
base_map = folium.Map(location=[velocity_results_I002_df['latitude'].mean(), velocity_results_I002_df['longitude'].mean()], zoom_start=5)
heat_data = velocity_results_I002_df[['latitude', 'longitude', 'velocity']].values.tolist()
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(base_map)
base_map

# %%
velocity_results_I003_clean = velocity_results_I003.dropna(subset=['Midpoint_Latitude_I003', 'Midpoint_Longitude_I003', 'Velocity_I003'])
velocity_results_I003_df = pd.DataFrame({
    'latitude': velocity_results_I003_clean['Midpoint_Latitude_I003'],
    'longitude': velocity_results_I003_clean['Midpoint_Longitude_I003'],
    'velocity': velocity_results_I003_clean['Velocity_I003']
})
base_map = folium.Map(location=[velocity_results_I003_df['latitude'].mean(), velocity_results_I003_df['longitude'].mean()], zoom_start=5)
heat_data = velocity_results_I003_df[['latitude', 'longitude', 'velocity']].values.tolist()
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(base_map)
base_map

# %%
velocity_results_I003_clean = velocity_results_I003.dropna(subset=['Midpoint_Latitude_I003', 'Midpoint_Longitude_I003', 'Velocity_I003'])
velocity_results_I003_df = pd.DataFrame({
    'latitude': velocity_results_I003_clean['Midpoint_Latitude_I003'],
    'longitude': velocity_results_I003_clean['Midpoint_Longitude_I003'],
    'velocity': velocity_results_I003_clean['Velocity_I003']
})
base_map = folium.Map(location=[velocity_results_I003_df['latitude'].mean(), velocity_results_I003_df['longitude'].mean()], zoom_start=5)
heat_data = velocity_results_I003_df[['latitude', 'longitude', 'velocity']].values.tolist()
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(base_map)
base_map

# %%
velocity_results_I004_clean = velocity_results_I004.dropna(subset=['Midpoint_Latitude_I004', 'Midpoint_Longitude_I004', 'Velocity_I004'])
velocity_results_I004_df = pd.DataFrame({
    'latitude': velocity_results_I004_clean['Midpoint_Latitude_I004'],
    'longitude': velocity_results_I004_clean['Midpoint_Longitude_I004'],
    'velocity': velocity_results_I004_clean['Velocity_I004']
})
base_map = folium.Map(location=[velocity_results_I004_df['latitude'].mean(), velocity_results_I004_df['longitude'].mean()], zoom_start=5)
heat_data = velocity_results_I004_df[['latitude', 'longitude', 'velocity']].values.tolist()
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(base_map)
base_map

# %%
velocity_results_I005_clean = velocity_results_I005.dropna(subset=['Midpoint_Latitude_I005', 'Midpoint_Longitude_I005', 'Velocity_I005'])
velocity_results_I005_df = pd.DataFrame({
    'latitude': velocity_results_I005_clean['Midpoint_Latitude_I005'],
    'longitude': velocity_results_I005_clean['Midpoint_Longitude_I005'],
    'velocity': velocity_results_I005_clean['Velocity_I005']
})
base_map = folium.Map(location=[velocity_results_I005_df['latitude'].mean(), velocity_results_I005_df['longitude'].mean()], zoom_start=5)
heat_data = velocity_results_I005_df[['latitude', 'longitude', 'velocity']].values.tolist()
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(base_map)
base_map

# %%
velocity_results_I006_clean = velocity_results_I006.dropna(subset=['Midpoint_Latitude_I006', 'Midpoint_Longitude_I006', 'Velocity_I006'])
velocity_results_I006_df = pd.DataFrame({
    'latitude': velocity_results_I006_clean['Midpoint_Latitude_I006'],
    'longitude': velocity_results_I006_clean['Midpoint_Longitude_I006'],
    'velocity': velocity_results_I006_clean['Velocity_I006']
})
base_map = folium.Map(location=[velocity_results_I006_df['latitude'].mean(), velocity_results_I006_df['longitude'].mean()], zoom_start=5)
heat_data = velocity_results_I006_df[['latitude', 'longitude', 'velocity']].values.tolist()
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(base_map)
base_map

# %%
velocity_results_I007_clean = velocity_results_I007.dropna(subset=['Midpoint_Latitude_I007', 'Midpoint_Longitude_I007', 'Velocity_I007'])
velocity_results_I007_df = pd.DataFrame({
    'latitude': velocity_results_I007_clean['Midpoint_Latitude_I007'],
    'longitude': velocity_results_I007_clean['Midpoint_Longitude_I007'],
    'velocity': velocity_results_I007_clean['Velocity_I007']
})
base_map = folium.Map(location=[velocity_results_I007_df['latitude'].mean(), velocity_results_I007_df['longitude'].mean()], zoom_start=5)
heat_data = velocity_results_I007_df[['latitude', 'longitude', 'velocity']].values.tolist()
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(base_map)
base_map

# %%
velocity_results_I008_clean = velocity_results_I008.dropna(subset=['Midpoint_Latitude_I008', 'Midpoint_Longitude_I008', 'Velocity_I008'])
velocity_results_I008_df = pd.DataFrame({
    'latitude': velocity_results_I008_clean['Midpoint_Latitude_I008'],
    'longitude': velocity_results_I008_clean['Midpoint_Longitude_I008'],
    'velocity': velocity_results_I008_clean['Velocity_I008']
})
base_map = folium.Map(location=[velocity_results_I008_df['latitude'].mean(), velocity_results_I008_df['longitude'].mean()], zoom_start=5)
heat_data = velocity_results_I008_df[['latitude', 'longitude', 'velocity']].values.tolist()
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(base_map)
base_map

# %%
velocity_results_I009_clean = velocity_results_I009.dropna(subset=['Midpoint_Latitude_I009', 'Midpoint_Longitude_I009', 'Velocity_I009'])
velocity_results_I009_df = pd.DataFrame({
    'latitude': velocity_results_I009_clean['Midpoint_Latitude_I009'],
    'longitude': velocity_results_I009_clean['Midpoint_Longitude_I009'],
    'velocity': velocity_results_I009_clean['Velocity_I009']
})
base_map = folium.Map(location=[velocity_results_I009_df['latitude'].mean(), velocity_results_I009_df['longitude'].mean()], zoom_start=5)
heat_data = velocity_results_I009_df[['latitude', 'longitude', 'velocity']].values.tolist()
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(base_map)
base_map

# %%
restaurant_map = folium.Map(location=[restaurant_data['latitude'].mean(), restaurant_data['longitude'].mean()], zoom_start=5)

restaurant_data.drop_duplicates().apply(lambda x: folium.Marker(
    location=[x['latitude'], x['longitude']],
    popup=f"{x['name']}\n{x['category']}"
).add_to(restaurant_map), axis=1)

restaurant_map


