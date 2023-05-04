import unittest
import yaml
import torch

from utils.nms import non_maximum_supression

class TestNonMaximumSupression(unittest.TestCase):
    def setUp(self):

        self.confidences = torch.tensor([[
            [0.9, 0, 0, 0, 0],
            [0, 0.6, 0, 0, 0],
            [0, 0, 0.7, 0, 0],
            [0, 0, 0, 0.1, 0],
            [0, 0, 0, 0, 0.2]
        ]])

        self.boxes = torch.tensor([[
            [10, 10, 90, 90],
            [15, 15, 95, 95],
            [100, 100, 120, 120],
            [5, 5, 7, 7],
            [17, 17, 80, 80],
        ]])

        self.iou_threshold = 0.5

    def test_that_nms_has_correct_output(self):

        output = non_maximum_supression(
            self.confidences,
            self.boxes,
            self.iou_threshold,
            "cpu"
        )

        print(output)