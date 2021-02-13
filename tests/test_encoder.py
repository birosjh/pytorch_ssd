import yaml
import unittest

import torch

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
        
    def test_single_iou_output_is_correct(self):

        box1 = torch.tensor([[200.0, 300.0, 350.0, 400.0]])
        box2 = torch.tensor([[250.0, 250.0, 400.0, 350.0]])

        ious = self.data_encoder.calculate_iou(box1, box2)

        # Compares each value, then gives a single boolean for whether they all match or not
        equivalence_check = torch.all(torch.eq(ious, torch.tensor([[0.2]])))

        self.assertTrue(equivalence_check.numpy())
        

    def test_multiple_iou_output_is_correct(self):

        box1 = torch.tensor([
            [200.0, 300.0, 350.0, 400.0],
            [250.0, 250.0, 400.0, 350.0]
        ])
        box2 = torch.tensor([
            [250.0, 250.0, 400.0, 350.0],
            [350.0, 300.0, 550.0, 400.0]
        ])

        ious = self.data_encoder.calculate_iou(box1, box2)

        # Compares each value, then gives a single boolean for whether they all match or not
        # One box exists in both box1 and box2 so its value is 1
        # One set of boxes don't overlap at all so they come out to zero
        equivalence_check = torch.all(
            torch.eq(ious, torch.tensor([
                [0.2000, 0.0000], # box1 row 1 ious for box2
                [1.0000, 0.07692307692] # box1 row 2 ious for box 2
            ]))
        )

        self.assertTrue(equivalence_check.numpy())


if __name__ == '__main__':
    unittest.main()
