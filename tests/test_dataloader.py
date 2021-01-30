import unittest

from image_dataset import ImageDataset
from torch.utils.data import DataLoader


class TestDataloader(unittest.TestCase):

    def setUp(self):
        self.data_config = {
            "annotations": "tests/test_annotations.json"
        }

        self.batch_size = 10

        self.dataset = ImageDataset(
            data_config=self.data_config
        )


    def test_the_number_of_items_in_the_batch(self):

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=0
        )

        images, labels = next(dataloader)

        print(images.shape)




if __name__ == '__main__':
    unittest.main()
