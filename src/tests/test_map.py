import unittest

import torch
import yaml

from src.models.metrics.map import MeanAveragePrecision
from src.models.ssd import SSD
from src.utils.data_encoder import DataEncoder

torch.set_printoptions(precision=10)


class TestMeanAveragePrecision(unittest.TestCase):
    def setUp(self):
        with open("src/tests/test_config.yaml") as f:
            config = yaml.safe_load(f)

        self.model_config = config["model_configuration"]
        self.data_config = config["data_configuration"]

        self.map = MeanAveragePrecision(self.data_config["classes"], "cpu", "coco")

    def test_average_precision_output_is_correct(self):
        confidences = torch.tensor([0.32, 0.84, 0.11, 0.51, 0.62, 0.78])
        is_true_positive = torch.tensor([0, 1, 0, 1, 0, 1])
        is_true_positive = is_true_positive > 0

        ap = self.map.average_precision(confidences, is_true_positive)

        print(ap)

    def test_coco_map_output_is_correct(self):

        ground_truths = torch.tensor([
            [174, 101, 349, 351, 1],
            [169, 104, 209, 146, 2],
            [278, 210, 297, 233, 3],
            [273, 333, 297, 354, 4],
            [319, 307, 340, 326, 5]
        ]).unsqueeze(dim=0).type(torch.float)

        ground_truths[:,:, 0:4] /= 500

        confidences = torch.tensor([
            [0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.2, 0.3],
            [0.1, 0.1, 0.1, 0.2, 0.4, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.0, 0.6],
        ]).unsqueeze(dim=0)

        localizations = torch.tensor([
            [175, 111, 349, 351],
            [159, 94, 209, 146],
            [278, 210, 267, 253],
            [273, 333, 297, 354],
            [339, 317, 370, 346]
        ]).unsqueeze(dim=0) / 500

        ap = self.map.coco_mean_average_precision(
            confidences, localizations, ground_truths
        )

        print(ap)
