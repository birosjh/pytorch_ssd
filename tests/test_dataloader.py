import yaml
import unittest

from image_dataset import ImageDataset
from torch.utils.data import DataLoader


class TestDataloader(unittest.TestCase):

    def setUp(self):

        with open("tests/test_config.yaml") as f:
            config = yaml.safe_load(f)

        self.data_config = config["data_configuration"]

        self.batch_size = 10

        self.dataset = ImageDataset(
            data_config=self.data_config,
            transform=True,
            mode="val"
        )


    def test_the_shape_a_the_batch(self):

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=0
        )

        images, labels = next(iter(dataloader))

        print(images.shape)
        print(labels.shape)




if __name__ == '__main__':
    unittest.main()
