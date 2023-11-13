import unittest

import yaml

from models.backbone.backbone_loader import backbone_loader
from models.backbone.vgg16 import Vgg16


class TestBackbones(unittest.TestCase):
    def setUp(self):
        with open("tests/test_config.yaml") as f:
            config = yaml.safe_load(f)

        self.model_config = config["model_configuration"]

    def test_vgg16_can_be_loaded_with_backbone_loader(self):
        config = {"backbone": "vgg16", "pretrained": False}

        model = backbone_loader(config)

        self.assertTrue(isinstance(model, Vgg16))


if __name__ == "__main__":
    unittest.main()
