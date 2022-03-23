"""
trainer.py
---------------------------------------------

Contains a trainer to train an SSD model with the specified dataset.

"""

import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, dataset, loss_function, training_config):

        self.model = model

        self.dataloader = DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            num_workers=training_config["num_workers"],
        )

        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=training_config["learning_rate"]
        )

        self.loss_function = loss_function

        self.epochs = training_config["epochs"]

    def train(self):

        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch, self.epochs - 1))
            print("-" * 10)

            for X, y in self.dataloader:
                # Compute prediction and loss
                pred = self.model(X)
                loss = self.loss_function(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
