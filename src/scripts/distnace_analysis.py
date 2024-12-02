# %% [markdown]
# # Distance Analysis

# %% [markdown]
# #### Get Data

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from haversine import haversine_vector
import seaborn as sns

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
    distances = haversine_vector(coords1, coords2)
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

# %%
# Load the data
file_path = '../../data/movements.csv'
movements_data = pd.read_csv(file_path, parse_dates=['datetime'])

# %%
# get distance and time between each measurement
grouped = movements_data.groupby('id')
distance_dict = {}
time_dict = {}

for name, group in grouped:
    distances = get_distance_between_rows(group)
    times = get_time_between_rows(group)
    distance_dict[name] = distances
    time_dict[name] = times

# replaces missing items w/ NaN
distance_results = pd.DataFrame({ key:pd.Series(value) for key, value in distance_dict.items() })
time_results = pd.DataFrame({ key:pd.Series(value) for key, value in time_dict.items() })

# %%
# get velocity
velocity_results = distance_results / time_results
velocity_results.columns = [f'Velocity_{col}' for col in velocity_results.columns]

# %%


# %% [markdown]
# #### Missing Values

# %%
missing_movement_data = pd.DataFrame()
missing_movement_data['latitude'] = grouped['latitude'].apply(lambda x: x.isna().sum())
missing_movement_data['longitude'] = grouped['latitude'].apply(lambda x: x.isna().sum())

missing_movement_data.head(10)

# %%
# verify all missing values are due to lengths
length = distance_results.notna().sum().reset_index()
length.columns = ['id', 'length']
length['distance_from_max'] = length['length'] - length['length'].max()

length['distance_results_na'] = distance_results.isna().sum().values
length['time_results_na'] = time_results.isna().sum().values

length.head(10)


# %%
print(distance_results.apply(lambda x: x.isna().sum()))
print(time_results.apply(lambda x: x.isna().sum()))

# %% [markdown]
# #### Summary Stats

# %%
distance_results.apply(lambda x: x.describe())

# %%
velocity_results.apply(lambda x: x.describe())

# %% [markdown]
# #### Graphs

# %% [markdown]
# ##### Velocity Over Time

# %%
n_1 = 1
n_2 = 1
# Plot the velocity results for each id in separate graphs with downsampling and rolling averages
for col in velocity_results.columns:
    plt.figure(figsize=(10, 3))
    
    # first 10,000 points
    downsampled_data = velocity_results[col].iloc[:10000:]
    
    # Calculate rolling average with a window of 1000 points
    # rolling_avg = velocity_results[col].rolling(window=n_2).mean()
    
    plt.plot(downsampled_data.index, downsampled_data, linestyle='-', label=f'{col}')
    # plt.plot(rolling_avg.index, rolling_avg, linestyle='-', color='red', label=f'{col} (Rolling Avg)')
    
    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Velocity (km/hr)')
    plt.title(f'Velocity Over Time for {col}')
    plt.legend()
    
    # Show the plot
    plt.show()

# %% [markdown]
# ##### Movement Distributions

# %%
for col in distance_results.columns:
    # Plot the histogram with logarithmic scaling
    sns.histplot(distance_results[col], bins=100, log_scale=(True, False))

    # Add labels and title
    plt.xlabel(f'Distance Traveled by {col}(km)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Distances for {col}')

    # Show the plot
    plt.show()

# %%
for col in distance_results.columns:
    # Plot the histogram with logarithmic scaling
    sns.histplot([distance for distance in distance_results[col] if distance > 10], bins=100, log_scale=(True, False))
    

    # Add labels and title
    plt.xlabel(f'Distance Traveled by {col} > 10(km)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Distances for {col}')

    # Show the plot
    plt.show()

# %%
for col in distance_results.columns:
    # Plot the histogram with logarithmic scaling
    sns.histplot([distance for distance in distance_results[col] if distance < 3], bins=100, log_scale=(True, False))
    

    # Add labels and title
    plt.xlabel(f'Distance Traveled by {col} > 10(km)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Distances for {col}')

    # Show the plot
    plt.show()

# %% [markdown]
# ##### Velocity Distributions

# %%
for col in velocity_results.columns:
    # Plot the histogram with logarithmic scaling
    sns.histplot(velocity_results[col], bins=100, log_scale=(True, False))

    # Add labels and title
    plt.xlabel(f'Distance Traveled by {col}(km)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Distances for {col}')

    # Show the plot
    plt.show()


