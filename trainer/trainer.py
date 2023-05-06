"""
trainer.py
---------------------------------------------

Contains a trainer to train an SSD model with the specified dataset.

"""

from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from loggers.log_handler import LogHandler
from models.loss.ssd import SSDLoss
from utils.nms import non_maximum_supression
from models.metrics.map import mean_average_precision


class Trainer:
    """
    The trainer takes a model and datasets as an argument
    and trains the model according to the training configurations
    """

    def __init__(
        self, model, train_dataset, val_dataset, training_config, device
    ) -> None:

        self.model = model.to(device)

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

        self.label_indices = np.arange(1, len(train_dataset.classes) + 1)

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

            localizations = non_maximum_supression(confidences, localizations, self.iou_threshold, self.device)

            conf_loss, loc_loss, loss = self.loss(
                confidences,
                localizations,
                targets
            )

            epoch_conf_loss += conf_loss.item()
            epoch_loc_loss += loc_loss.item()
            epoch_loss += loss.item()

            print(f"Conf: {conf_loss.item()} ")
            print(f"Loc: {loc_loss.item()} ")
            print(f"Total: {loss.item()} ")

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
        mAP = 0

        max_confidences = []
        predictions = []
        ground_truths = []

        self.model.eval()

        with torch.no_grad():

            for images, targets in tqdm(self.val_dataloader):

                confidences, localizations = self.model(images)

                localizations = non_maximum_supression(confidences, localizations, self.iou_threshold, self.device)

                conf_loss, loc_loss, loss = self.loss(confidences, localizations, targets)

                epoch_val_conf_loss += conf_loss.item()
                epoch_val_loc_loss += loc_loss.item()
                epoch_val_loss += loss.item()

                confidence_tensor = torch.reshape(confidences, (-1, 21))
                indices = confidence_tensor.argmax(1)
                ground_truth = torch.reshape(targets[:, :, -1], (-1,))

                max_confidences.append(confidence_tensor.max(1).values.cpu().numpy())
                predictions.append(indices.cpu().numpy())
                ground_truths.append(ground_truth.cpu().numpy())

            records = {
                "val_conf_loss": epoch_val_conf_loss / len(self.val_dataloader),
                "val_loc_loss": epoch_val_loc_loss / len(self.val_dataloader),
                "val_total_loss": epoch_val_loss / len(self.val_dataloader),
            }

            max_confidences = np.concatenate(max_confidences, axis=0)
            predictions = np.concatenate(predictions, axis=0)
            ground_truths = np.concatenate(ground_truths, axis=0)

            # mAP += mean_average_precision(max_confidences, predictions, ground_truths, self.label_indices)
            # records["map"] = mAP / len(self.val_dataloader)

        return records

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
