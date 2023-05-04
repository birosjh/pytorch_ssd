import unittest
import yaml
import torch

from utils.data_encoder import DataEncoder
from models.metrics.map import mean_average_precision

torch.set_printoptions(precision=10)


class TestMeanAveragePrecision(unittest.TestCase):
    def setUp(self):

        with open("tests/test_config.yaml") as f:
            config = yaml.safe_load(f)

        self.model_config = config["model_configuration"]

    def test_mean_average_precision_output_is_correct(self):

        data_encoder = DataEncoder(self.model_config)

        predictions = 

