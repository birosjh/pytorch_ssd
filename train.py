import argparse
import torch
import yaml

from explicit_ssd import SSD
from trainer import Trainer
from image_dataset import ImageDataset


def load_configurations():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    arguments = parser.parse_args()

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    return config

def main():

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    config = load_configurations()

    model_config = config["model_configuration"]
    training_config = config["training_configuration"]
    data_config = config["data_configuration"]

    print(config)

    dataset = ImageDataset(
        data_config=data_config,
        transform=True,
    )

    num_classes = len(data_config['classes'])

    model = SSD(num_classes).to(device)

    trainer = Trainer(
        model, 
        dataset,
        loss_function, 
        training_config
    )






if __name__ == "__main__":

    main()
