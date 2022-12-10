import torch
import yaml

from datasets.image_dataset import ImageDataset
from models.ssd import SSD
from trainer.trainer import Trainer
from utils.data_encoder import DataEncoder


def train_model(config_path: str) -> None:
    """
    A function to train the SSD model

    Args:
        config_path (str): Path to desired config file
    """

    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Use GPU if available
    try:

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except AttributeError:

        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using {} device".format(device))

    model_config = config["model_configuration"]
    training_config = config["training_configuration"]
    data_config = config["data_configuration"]

    data_encoder = DataEncoder(model_config)

    train_dataset = ImageDataset(
        data_config=data_config, data_encoder=data_encoder, mode="train", device=device
    )

    val_dataset = ImageDataset(
        data_config=data_config, data_encoder=data_encoder, mode="val"
    )

    num_classes = len(data_config["classes"]) + 1

    model = SSD(model_config, num_classes, device)

    trainer = Trainer(model, train_dataset, val_dataset, training_config, device)
    trainer.train()
