"""
trainer.py
---------------------------------------------

Contains a trainer to train an SSD model with the specified dataset.

"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loggers.log_handler import LogHandler
from models.loss.ssd import SSDLoss
from utils.nms import non_maximum_supression


class Trainer:
    """
    The trainer takes a model and datasets as an argument
    and trains the model according to the training configurations
    """

    def __init__(
        self, model, train_dataset, val_dataset, training_config, device
    ) -> None:

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
        self.save_path = Path(training_config["model_save_path"])
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.iou_threshold = training_config["iou_threshold"]

        self.device = device

    def train(self) -> None:
        """
        Train the model
        """

        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch, self.epochs - 1))
            print("-" * 10)

            train_records = self.train_one_epoch()

            val_records = self.validate_one_epoch()

            self.save_best_model(epoch, val_records)

            records = {**train_records, **val_records}

            self.log(records, epoch)

        # Save Model
        last_model_path = self.save_path / "last_model.pth"
        torch.save(self.model.state_dict(), last_model_path)

    def train_one_epoch(self) -> dict:
        """
        Run the model through one epoch of training
        """

        epoch_conf_loss = 0
        epoch_loc_loss = 0
        epoch_loss = 0

        self.model.train()

        for images, targets in tqdm(self.train_dataloader):
            # Compute prediction and loss
            confidences, localizations = self.model(images)

            maximum_confidences = []
            maximum_localizations = []
            for idx, confidence in enumerate(confidences):

                max_confidence, max_localization = non_maximum_supression(
                    confidence,
                    localizations[idx],
                    self.iou_threshold,
                    device=self.device,
                )

                maximum_confidences.append(max_confidence)
                maximum_localizations.append(max_localization)

            predictions = (maximum_confidences, maximum_localizations)

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

    def validate_one_epoch(self) -> dict:
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

                predictions = non_maximum_supression(predictions, self.iou_threshold)

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

    def save_best_model(self, epoch: int, val_records: dict) -> None:
        """
        If the current model has a lower validation loss than the previous
        epoch's model, then save it as the best model.

        Args:
            epoch (int): The current epoch
            val_records (dict): The validation records
        """

        best_model_path = self.save_path / "best_model.pth"

        validation_loss = val_records["val_total_loss"]

        if epoch == 0:

            self.lowest_validation_loss = validation_loss

        if validation_loss < self.lowest_validation_loss:

            self.lowest_validation_loss = validation_loss

            torch.save(self.model.state_dict(), best_model_path)
