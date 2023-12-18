import unittest

import torch
import yaml

from src.utils.data_encoder import DataEncoder
from torchvision.ops import box_iou


class TestEncoder(unittest.TestCase):
    def setUp(self):
        with open("src/tests/test_config.yaml") as f:
            config = yaml.safe_load(f)

        self.data_config = config["data_configuration"]
        self.model_config = config["model_configuration"]

        self.data_encoder = DataEncoder(self.model_config)

        self.total_num_boxes = 0
        for idx, feature_map in enumerate(self.model_config["feature_map_sizes"]):
            num_aspect_ratios = 2 + len(self.model_config["aspect_ratios"][idx]) * 2
            self.total_num_boxes += (feature_map * feature_map) * num_aspect_ratios

    def test_default_box_shape(self):
        self.assertEqual(self.total_num_boxes, len(self.data_encoder.default_boxes))

    def test_shape_of_encoder_output(self):
        temp_tensors = torch.Tensor(
            [
                [30.6000, 64.0000, 73.2000, 212.8000, 14.0000],
                [220.2000, 73.6000, 276.6000, 252.0000, 14.0000],
            ]
        )

        result = self.data_encoder.encode(temp_tensors)

        self.assertListEqual(list(result.shape), [self.total_num_boxes, 5])

    def test_encoder_output(self):
        temp_tensors = torch.tensor(
            [
                [47.1040, 40.7447, 150.5280, 238.3183, 15.0000],
                [86.0160, 119.1592, 154.1120, 241.3934, 15.0000],
                [144.8960, 110.7027, 224.7680, 256.0000, 15.0000],
                [134.6560, 38.4384, 173.5680, 62.2703, 7.0000],
                [13.8240, 59.9640, 61.9520, 87.6396, 7.0000],
                [39.4240, 49.9700, 76.2880, 76.1081, 7.0000],
                [101.8880, 45.3574, 122.8800, 69.1892, 7.0000],
            ]
        )

        ious = box_iou(self.data_encoder.default_boxes, temp_tensors[:, 0:4])

        print(ious[ious.max(dim=1).values > 0.5])

        # result = self.data_encoder.encode(temp_tensors, 0.5)

        # [print(item) for item in result]


if __name__ == "__main__":
    unittest.main()
