import unittest

from data_encoder import DataEncoder


class TestEncoder(unittest.TestCase):

    def setUp(self):
        self.config = { 
            "figure_size": 300,
            "feature_map_sizes": [38, 19, 10, 5, 3, 1],
            "steps": [8, 16, 32, 64, 100, 300],
            "scales": [21, 45, 99, 153, 207, 261, 315],
            "aspect_ratios": [(2,), (2, 3), (2, 3), (2, 3), (2,), (2,)],
            "feature_maps": [38, 19, 10, 5, 3, 1]
        }

        self.data_encoder = DataEncoder(self.config)

    def test_default_box_shape(self):

        total_num_boxes = 0
        for idx, feature_map in enumerate(self.config['feature_maps']):
            num_aspect_ratios = 2 + len(self.config['aspect_ratios'][idx]) * 2
            total_num_boxes += (feature_map * feature_map) * num_aspect_ratios

        self.assertEqual(total_num_boxes, len(self.data_encoder.default_boxes))
        

if __name__ == '__main__':
    unittest.main()
