"""
trainer.py
---------------------------------------------

Contains a trainer to train an SSD model with the specified dataset.

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.loss.ssd import SSDLoss

class Trainer:
    def __init__(self, model, dataset, default_boxes, training_config):

        self.model = model

        self.dataloader = DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            num_workers=training_config["num_workers"],
        )

        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=training_config["learning_rate"]
        )

        
        self.alpha = training_config["alpha"]
        self.default_boxes = default_boxes

        self.epochs = training_config["epochs"]

    def train(self):

        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch, self.epochs - 1))
            print("-" * 10)

            for images, targets in self.dataloader:
                # Compute prediction and loss
                pred_loc, pred_conf = self.model(images)

                conf_loss, loc_loss, loss = self.loss(predictions, targets)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
