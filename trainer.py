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
            num_workers=training_config["num_workers"]
        )

        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_config["learning_rate"]
        )

        self.loss_function = loss_function

        self.epochs = training_config["epochs"]

    def train(self):

        size = len(self.dataloader.dataset)

        for batch, (X, y) in enumerate(self.dataloader):
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_function(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
