import unittest

from data_encoder import DataEncoder


class TestEncoder(unittest.TestCase):

    def setUp(self):
        config = { 
            "figure_size": 300,
            "feature_map_sizes": [38, 19, 10, 5, 3, 1],
            "steps": [8, 16, 32, 64, 100, 300],
            "scales": [21, 45, 99, 153, 207, 261, 315],
            "aspect_ratios": [(2,), (2, 3), (2, 3), (2, 3), (2,), (2,)],
            "feature_maps": [38, 19, 10, 5, 3, 1]
        }

        self.data_encoder = DataEncoder(config)

    def test_default_box_shape(self):

        print(len(self.data_encoder.default_boxes))

if __name__ == '__main__':
    unittest.main()
