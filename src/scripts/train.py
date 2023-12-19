import torch
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.lightning_model import LightningSSD
from src.utils.data_encoder import DataEncoder
from src.datamodules.pascal_datamodule import PascalDataModule

from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import lightning as L


def train_model(config : DictConfig) -> None:
    """
    A function to train the SSD model

    Args:
        config (DictConfig): The configurations object
    """

    print(config)

    # Use GPU if available
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Using {} device".format(device))

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["dataset"]
    transform_config = config["transforms"]

    num_classes = len(data_config["classes"]) + 1

    data_encoder = DataEncoder(config["encoder"])

    datamodule = PascalDataModule(
        data_config,
        training_config,
        transform_config,
        data_encoder
    )

    model = LightningSSD(
        model_config,
        training_config,
        data_encoder,
        num_classes
    )

    trainer = L.Trainer(
        max_epochs=training_config["epochs"],
        accelerator="auto",
        devices=1 if device != "cpu" else None,
        logger=[
            CSVLogger(save_dir="logs/"),
            WandbLogger(project="SSD")
        ],
        callbacks=[
            LearningRateMonitor(logging_interval="epoch")
        ],
    )

    trainer.fit(model, datamodule)
