import unittest
import sys
import numpy as np
import pandas as pd
from io import StringIO
from haversine import haversine_vector, haversine

# Add the src directory to the Python path
# sys.path.insert(0, '/workspaces/detecting-dine-out-patterns/src')
# from util.average_distance_between_rows import average_distance_between_rows

import util

class TestAverageDistance(unittest.TestCase):
    def setUp(self):
        # Create a mock CSV file
        self.csv_data = StringIO("""datetime,longitude,latitude
2020-01-01 00:00:00,2,1
2020-01-01 00:01:00,3,1
2020-01-01 00:02:00,4,1
2020-01-01 00:02:00,5,1
2020-01-01 00:02:00,6,1
2020-01-01 00:02:00,7,1
2020-01-01 00:02:00,8,1
""")
        self.df = pd.read_csv(self.csv_data, parse_dates=['datetime'])

    def test_haversine(self):
        # Test the haversine function with known values
        lyon = self.df.sort_values(by='datetime')[['latitude', 'longitude']].values[1]
        paris = self.df.sort_values(by='datetime')[['latitude', 'longitude']].values[2]
        distance = haversine(lyon, paris)
        self.assertAlmostEqual(distance, 111.178, places=3)

    def test_average_distance(self):
        average_dist = util.average_distance_between_rows(self.df)
        self.assertAlmostEqual(average_dist, 111.178, places=3)

if __name__ == '__main__':
    unittest.main()