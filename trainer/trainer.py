"""
trainer.py
---------------------------------------------

Contains a trainer to train an SSD model with the specified dataset.

"""

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

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

    def train(self):
        """
        Train the model
        """

        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch, self.epochs - 1))
            print("-" * 10)

            #self.train_one_epoch()

            try:

                self.validate_one_epoch()

            except Exception as e:

                print(e)

                
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

        epoch_conf_loss /= len(self.train_dataloader)
        epoch_loc_loss /= len(self.train_dataloader)
        epoch_loss /= len(self.train_dataloader)

        print(f"Training Confidence Loss: {epoch_conf_loss}")
        print(f"Training Localization Loss: {epoch_loc_loss}")
        print(f"Training Total Loss: {epoch_loss}")


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

        epoch_val_conf_loss /= len(self.val_dataloader)
        epoch_val_loc_loss /= len(self.val_dataloader)
        epoch_val_loss /= len(self.val_dataloader)
        
        print(f"Validation Confidence Loss: {epoch_val_conf_loss}")
        print(f"Validation Localization Loss: {epoch_val_loc_loss}")
        print(f"Validation Total Loss: {epoch_val_loss}")