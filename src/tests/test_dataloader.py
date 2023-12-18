import unittest

import torch
import yaml
from torch.utils.data import DataLoader

from src.datamodules.datasets.image_dataset import ImageDataset
from src.utils.data_encoder import DataEncoder
from src.utils.default_box import (
    number_of_default_boxes_per_cell,
    total_number_of_default_boxes,
)


class TestDataloader(unittest.TestCase):
    def setUp(self):
        with open("src/tests/test_config.yaml") as f:
            config = yaml.safe_load(f)

        self.data_config = config["data_configuration"]
        self.model_config = config["model_configuration"]

        self.batch_size = 10

        data_encoder = DataEncoder(self.model_config)

        self.train_dataset = ImageDataset(
            data_config=self.data_config,
            data_encoder=data_encoder,
            mode="train",
        )

        self.val_dataset = ImageDataset(
            data_config=self.data_config,
            data_encoder=data_encoder,
            mode="val",
        )

        num_default_boxes_per_cell = number_of_default_boxes_per_cell(
            self.model_config["aspect_ratios"]
        )

        self.total_num_of_default_boxes = total_number_of_default_boxes(
            num_default_boxes_per_cell, self.model_config["feature_map_sizes"]
        )

    def test_the_shape_a_the_batch(self):
        dataloader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=0
        )

        images, labels = next(iter(dataloader))

        self.assertEqual(
            images.shape,
            torch.Size(
                [
                    self.batch_size,
                    3,  # Three channels
                    self.model_config["figure_size"],
                    self.model_config["figure_size"],
                ]
            ),
        )
        self.assertEqual(
            labels.shape,
            torch.Size(
                [
                    self.batch_size,
                    self.total_num_of_default_boxes,
                    5,  # Four box coords and 1 class
                ]
            ),
        )

    def test_output(self):
        dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=0
        )

        images, labels = next(iter(dataloader))


if __name__ == "__main__":
    unittest.main()
