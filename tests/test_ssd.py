import unittest

import yaml
from torch.utils.data import DataLoader

from datasets.image_dataset import ImageDataset
from models.backbone.simple import SimpleBackbone
from models.ssd import SSD


class TestSSD(unittest.TestCase):
    def setUp(self):

        with open("tests/test_config.yaml") as f:
            config = yaml.safe_load(f)

        model_config = config["model_configuration"]
        training_config = config["training_configuration"]
        data_config = config["data_configuration"]

        dataset = ImageDataset(data_config=data_config, transform=True, mode="train")

        self.dataloader = DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            num_workers=0,
            shuffle=True,
        )

        num_classes = len(data_config["classes"])

        aspect_ratios = data_config["aspect_ratios"]

        layers = model_config["layers"]
        backbone = SimpleBackbone(layers)

        self.model = SSD(backbone, aspect_ratios, num_classes).to("cpu")

    def test_explicit_model_outputs_properly(self):

        images, labels = next(iter(self.dataloader))

        loc, conf = self.model(images)

        print(loc.shape)
        print(conf.shape)


if __name__ == "__main__":
    unittest.main()
