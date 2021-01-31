import yaml
import unittest

from data_encoder import DataEncoder

class TestEncoder(unittest.TestCase):

    def setUp(self):

        with open("tests/test_config.yaml") as f:
            config = yaml.safe_load(f)

        self.data_config = config["data_configuration"]

        self.data_encoder = DataEncoder(self.data_config)

    def test_default_box_shape(self):

        total_num_boxes = 0
        for idx, feature_map in enumerate(self.data_config['feature_maps']):
            num_aspect_ratios = 2 + \
                len(self.data_config['aspect_ratios'][idx]) * 2
            total_num_boxes += (feature_map * feature_map) * num_aspect_ratios

        self.assertEqual(total_num_boxes, len(self.data_encoder.default_boxes))
        

if __name__ == '__main__':
    unittest.main()
