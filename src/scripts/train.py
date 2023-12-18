import torch
import yaml

from src.models.lightning_model import LightningSSD
from src.utils.data_encoder import DataEncoder
from src.datamodules.pascal_datamodule import PascalDataModule

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import lightning as L


def train_model(config_path: str) -> None:
    """
    A function to train the SSD model

    Args:
        config_path (str): Path to desired config file
    """

    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Use GPU if available
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Using {} device".format(device))

    model_config = config["model_configuration"]
    training_config = config["training_configuration"]
    data_config = config["data_configuration"]

    num_classes = len(data_config["classes"]) + 1

    data_encoder = DataEncoder(model_config)

    datamodule = PascalDataModule(
        data_config,
        training_config,
        data_encoder
    )

    model = LightningSSD(
        model_config,
        training_config,
        data_encoder,
        num_classes
    )

    trainer = L.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1 if device != "cpu" else None,
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[
            LearningRateMonitor(logging_interval="step")
        ],
    )

    trainer.fit(model, datamodule)
