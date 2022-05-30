import argparse

import torch
import yaml

from datasets.image_dataset import ImageDataset
from models.ssd import SSD
from trainer.trainer import Trainer
from utils.data_encoder import DataEncoder


def load_configurations():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", action="store", required=True)
    arguments = parser.parse_args()

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    return config


def main():

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    config = load_configurations()

    model_config = config["model_configuration"]
    training_config = config["training_configuration"]
    data_config = config["data_configuration"]

    data_encoder = DataEncoder(model_config)

    train_dataset = ImageDataset(
        data_config=data_config,
        data_encoder=data_encoder,
        transform=True,
    )

    val_dataset = ImageDataset(
        data_config=data_config, data_encoder=data_encoder, transform=False, mode="val"
    )

    num_classes = len(data_config["classes"]) + 1

    model = SSD(model_config, num_classes).to(device)

    trainer = Trainer(model, train_dataset, val_dataset, training_config)
    trainer.train()


if __name__ == "__main__":

    main()
