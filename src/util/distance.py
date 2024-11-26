import numpy as np
from haversine import haversine_vector

def get_distance_between_rows(df):
    coords1 = df[['latitude', 'longitude']].values[:-1]
    coords2 = df[['latitude', 'longitude']].values[1:]
    distances = haversine_vector(coords1, coords2)
    return distances

def average_distance_between_rows(df):
    distances = get_distance_between_rows(df)
    return np.mean(distances)

def median_distance_between_rows(df):
    distances = get_distance_between_rows(df)
    return np.median(distances)

def max_distance_between_rows(df):
    distances = get_distance_between_rows(df)
    return np.max(distances)

def timestamps_of_max_distance(df):
    distances = get_distance_between_rows(df)
    max_index = np.argmax(distances)
    return df.iloc[max_index]['datetime'], df.iloc[max_index + 1]['datetime']

def longest_stretch_not_moving(df, max_distance):
    distances = get_distance_between_rows(df)
    longest_stretch = 0
    current_stretch = 0
    start_index = 0
    longest_start_index = 0
    longest_end_index = 0
    
    for i, distance in enumerate(distances):
        if distance <= max_distance:
            if current_stretch == 0:
                start_index = i
            current_stretch += 1
        else:
            if current_stretch > longest_stretch:
                longest_stretch = current_stretch
                longest_start_index = start_index
                longest_end_index = i
            current_stretch = 0
    
    if current_stretch > longest_stretch:
        longest_stretch = current_stretch
        longest_start_index = start_index
        longest_end_index = len(distances)
    
    return df.iloc[longest_start_index]['datetime'], df.iloc[longest_end_index]['datetime']