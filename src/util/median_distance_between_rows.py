import numpy as np
from haversine import haversine_vector
def median_distance_between_rows(df):
    coords1 = df[['latitude', 'longitude']].values[:-1]
    coords2 = df[['latitude', 'longitude']].values[1:]
    distances = haversine_vector(coords1, coords2)
    return np.median(distances)
