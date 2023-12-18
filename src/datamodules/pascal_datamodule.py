import lightning as L
from torch.utils.data import DataLoader

from src.datamodules.datasets.image_dataset import ImageDataset


class PascalDataModule(L.LightningDataModule):
    def __init__(self, data_config, training_config, data_encoder):
        super().__init__()

        self.data_config = data_config
        self.training_config = training_config
        self.data_encoder = data_encoder

    def setup(self, stage=None):
        self.train_dataset = ImageDataset(
            data_config=self.data_config, data_encoder=self.data_encoder, mode="train"
        )

        self.val_dataset = ImageDataset(
            data_config=self.data_config, data_encoder=self.data_encoder, mode="val"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_config["batch_size"],
            num_workers=self.training_config["num_workers"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.training_config["batch_size"],
            num_workers=self.training_config["num_workers"],
        )
