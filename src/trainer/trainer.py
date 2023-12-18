"""
trainer.py
---------------------------------------------

Contains a trainer to train an SSD model with the specified dataset.

"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchmetrics.detection import MeanAveragePrecision

from src.loggers.log_handler import LogHandler
from src.models.loss.ssd import SSDLoss
from src.visualize.visualize_batch import visualize_batch


class Trainer:
    """
    The trainer takes a model and datasets as an argument
    and trains the model according to the training configurations
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        training_config,
        device,
    ) -> None:
        self.model = model.to(device)

        print(self.model)

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_config["batch_size"],
            num_workers=training_config["num_workers"],
            shuffle=True,
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=training_config["batch_size"],
            num_workers=training_config["num_workers"],
        )

        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=training_config["learning_rate"], momentum=0.9
        )

        self.loss = SSDLoss(
            training_config["alpha"],
            training_config["iou_threshold"],
            device,
            train_dataset.data_encoder,
        )

        self.batch_size = training_config["batch_size"]

        self.epochs = training_config["epochs"]
        self.log = LogHandler(training_config["loggers"])
        self.save_path = Path(training_config["model_save_path"])
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.classes = train_dataset.classes
        self.label_indices = np.arange(1, len(self.classes) + 1)

        self.device = device

        self.map_frequency = training_config["map_frequency"]

        self.map = MeanAveragePrecision()

        self.lowest_validation_loss = 100000000.0

    def train(self) -> None:
        """
        Train the model
        """

        for epoch in range(1, self.epochs):
            print("Epoch {}/{}".format(epoch, self.epochs - 1))
            print("-" * 10)

            train_records = self.train_one_epoch(epoch)

            val_records = self.validate_one_epoch(epoch)

            self.save_best_model(epoch, val_records)

            records = {**train_records, **val_records}

            self.log(records, epoch)

        # Save Model
        last_model_path = self.save_path / "last_model.pth"
        torch.save(self.model.state_dict(), last_model_path)

    def train_one_epoch(self, epoch) -> dict:
        """
        Run the model through one epoch of training
        """

        epoch_conf_loss = 0
        epoch_loc_loss = 0
        epoch_loss = 0

        self.model.train()

        for images, targets in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()

            images = images.to(self.device)
            targets = targets.to(self.device)

            # Compute prediction and loss
            confidences, localizations = self.model(images)

            loss, breakdown = self.loss(confidences, localizations, targets)

            # Backpropagation
            if loss > 0:
                loss.backward()
                self.optimizer.step()

            epoch_conf_loss += breakdown["conf"]
            epoch_loc_loss += breakdown["loc"]
            epoch_loss += loss.item()

        labels = confidences.argmax(dim=2).unsqueeze(2)
        predictions = torch.concat([localizations, labels], dim=2)

        visualize_batch(
            images.cpu().detach(),
            predictions.cpu().detach(),
            self.classes,
            targets.cpu().detach(),
        )

        return {
            "train_conf_loss": epoch_conf_loss / len(self.train_dataloader),
            "train_loc_loss": epoch_loc_loss / len(self.train_dataloader),
            "train_total_loss": epoch_loss / len(self.train_dataloader),
        }

    def validate_one_epoch(self, epoch) -> dict:
        """
        Run the model through one epoch of validation
        """

        epoch_val_conf_loss = 0
        epoch_val_loc_loss = 0
        epoch_val_loss = 0

        self.model.eval()

        with torch.no_grad():
            for images, targets in tqdm(self.val_dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                confidences, localizations = self.model(images)

                loss, breakdown = self.loss(confidences, localizations, targets)

                epoch_val_conf_loss += breakdown["conf"]
                epoch_val_loc_loss += breakdown["loc"]
                epoch_val_loss += loss.item()

                # TODO: Only pass non-zero predictions (zero is for background)
                if epoch % self.map_frequency == 0:
                    max_confidences = confidences.max(dim=2)
                    max_values = max_confidences.values
                    max_indices = max_confidences.indices

                    preds = []
                    ground_truths = []

                    for indices, values, loc, target in zip(
                        max_indices, max_values, localizations, targets
                    ):
                        object_indices = target[:, -1].type(torch.int32) > 0

                        preds.append(
                            dict(
                                boxes=loc[object_indices],
                                scores=values[object_indices],
                                labels=indices[object_indices].type(torch.int32),
                            )
                        )
                        ground_truths.append(
                            dict(
                                boxes=target[:, 0:4][object_indices],
                                labels=target[:, 4][object_indices].type(torch.int32),
                            )
                        )

                    self.map.update(preds, ground_truths)

            records = {
                "val_conf_loss": epoch_val_conf_loss / len(self.val_dataloader),
                "val_loc_loss": epoch_val_loc_loss / len(self.val_dataloader),
                "val_total_loss": epoch_val_loss / len(self.val_dataloader),
            }

            if epoch % self.map_frequency == 0:
                metrics = self.map.compute()

                records["map"] = metrics["map"]
                records["map_50"] = metrics["map_50"]
                records["map_75"] = metrics["map_75"]
                records["map_large"] = metrics["map_large"]
                records["map_medium"] = metrics["map_medium"]
                records["map_small"] = metrics["map_small"]

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
