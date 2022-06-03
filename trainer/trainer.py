"""
trainer.py
---------------------------------------------

Contains a trainer to train an SSD model with the specified dataset.

"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loggers.log_handler import LogHandler
from models.loss.ssd import SSDLoss


class Trainer:
    """
    The trainer takes a model and datasets as an argument
    and trains the model according to the training configurations
    """

    def __init__(self, model, train_dataset, val_dataset, training_config):

        self.model = model

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_config["batch_size"],
            num_workers=training_config["num_workers"],
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=training_config["batch_size"],
            num_workers=training_config["num_workers"],
        )

        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=training_config["learning_rate"]
        )

        self.loss = SSDLoss(alpha=training_config["alpha"])

        self.epochs = training_config["epochs"]
        self.log = LogHandler(training_config["loggers"])

    def train(self):
        """
        Train the model
        """

        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch, self.epochs - 1))
            print("-" * 10)

            train_records = self.train_one_epoch()

            val_records = self.validate_one_epoch()

            records = {**train_records, **val_records}

            self.log(records, epoch)

    def train_one_epoch(self):
        """
        Run the model through one epoch of training
        """

        epoch_conf_loss = 0
        epoch_loc_loss = 0
        epoch_loss = 0

        self.model.train()

        for images, targets in tqdm(self.train_dataloader):
            # Compute prediction and loss
            predictions = self.model(images)

            conf_loss, loc_loss, loss = self.loss(predictions, targets)

            epoch_conf_loss += conf_loss
            epoch_loc_loss += loc_loss
            epoch_loss += loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {
            "train_conf_loss": epoch_conf_loss / len(self.train_dataloader),
            "train_loc_loss": epoch_loc_loss / len(self.train_dataloader),
            "train_total_loss": epoch_loss / len(self.train_dataloader),
        }

    def validate_one_epoch(self):
        """
        Run the model through one epoch of validation
        """

        epoch_val_conf_loss = 0
        epoch_val_loc_loss = 0
        epoch_val_loss = 0

        self.model.eval()

        with torch.no_grad():

            for images, targets in tqdm(self.val_dataloader):

                predictions = self.model(images)

                conf_loss, loc_loss, loss = self.loss(predictions, targets)

                epoch_val_conf_loss += conf_loss
                epoch_val_loc_loss += loc_loss
                epoch_val_loss += loss

            # TODO: Calculate mAP

        return {
            "val_conf_loss": epoch_val_conf_loss / len(self.val_dataloader),
            "val_loc_loss": epoch_val_loc_loss / len(self.val_dataloader),
            "val_total_loss": epoch_val_loss / len(self.val_dataloader),
        }
