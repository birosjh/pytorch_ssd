import unittest

import yaml

from datasets.image_dataset import ImageDataset
from utils.data_encoder import DataEncoder


class TestLoss(unittest.TestCase):
    def setUp(self):

        with open("tests/test_config.yaml") as f:
            config = yaml.safe_load(f)

        model_config = config["model_configuration"]
        # training_config = config["training_configuration"]
        data_config = config["data_configuration"]

        data_encoder = DataEncoder(model_config)

        dataset = ImageDataset(
            data_config=data_config,
            data_encoder=data_encoder,
            transform=True,
            mode="train",
        )

    def test_localization_loss(self):

        pass


if __name__ == "__main__":
    unittest.main()
