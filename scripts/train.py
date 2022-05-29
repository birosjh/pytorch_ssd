import argparse

import torch
import yaml
from explicit_ssd import SSD
from trainer import Trainer

from datasets.image_dataset import ImageDataset
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

    dataset = ImageDataset(
        data_config=data_config,
        data_encoder=data_encoder,
        transform=True,
    )

    num_classes = len(data_config["classes"]) + 1

    model = SSD(model_config, num_classes).to(device)

    default_boxes = self.data_encoder.default_boxes

    trainer = Trainer(model, dataset, default_boxes, training_config)
    trainer.train()


if __name__ == "__main__":

    main()
