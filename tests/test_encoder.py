import unittest

import torch
import yaml

from utils.data_encoder import DataEncoder
from utils.iou import calculate_iou


class TestEncoder(unittest.TestCase):
    def setUp(self):
        with open("tests/test_config.yaml") as f:
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
        temp_tensors = torch.Tensor(
            [
                [30.6000, 64.0000, 73.2000, 212.8000, 14.0000],
                [220.2000, 73.6000, 276.6000, 252.0000, 14.0000],
            ]
        )

        ious = calculate_iou(temp_tensors[:, 0:4], self.data_encoder.default_boxes)

        print(ious)


if __name__ == "__main__":
    unittest.main()
