import unittest

import torch
import yaml

from models.metrics.map import MeanAveragePrecision
from models.ssd import SSD
from utils.data_encoder import DataEncoder

torch.set_printoptions(precision=10)


class TestMeanAveragePrecision(unittest.TestCase):
    def setUp(self):

        with open("tests/test_config.yaml") as f:
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

        fake_batch = torch.zeros(1, 3, 256, 256)
        ground_truths = torch.zeros(1, 6132, 21)

        num_classes = len(self.data_config["classes"]) + 1

        data_encoder = DataEncoder(self.model_config)
        model = SSD(self.model_config, num_classes, data_encoder, "cpu")

        confidences, localizations = model(fake_batch)

        ap = self.map.coco_mean_average_precision(
            confidences, localizations, ground_truths
        )

        print(ap)
