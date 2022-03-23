import unittest

import torch

from models.backbone.simple import SimpleBackbone


class TestBackbones(unittest.TestCase):
    def setUp(self):

        self.backbone = SimpleBackbone([3, 16, 32, 64])

    def test_explicit_model_outputs_properly(self):

        x = torch.ones(32, 3, 256, 256)

        output = self.backbone(x)

        print(output.shape)


if __name__ == "__main__":
    unittest.main()
