from src.models.ssd import SSD
from src.models.loss.ssd import SSDLoss

import torch
import lightning as L
from torchmetrics.detection import MeanAveragePrecision


class LightningSSD(L.LightningModule):

    def __init__(self, model_config, training_config, data_encoder, num_classes):
        super().__init__()

        self.model = SSD(model_config, num_classes, data_encoder)

        self.loss = SSDLoss(
            training_config["alpha"],
            training_config["iou_threshold"],
            data_encoder,
        )

        self.map_frequency = training_config["map_frequency"]

        self.lr = training_config["learning_rate"]
        self.momentum = training_config["momentum"]
        self.weight_decay = training_config["weight_decay"]

        self.map = MeanAveragePrecision()

    def forward(self, x):

        return self.model(x)

        
    def training_step(self, batch, batch_idx):

        images, targets = batch

        confidences, localizations = self(images)

        loss, breakdown = self.loss(confidences, localizations, targets)

        return {
            "loss": loss,
            "confidence_loss": breakdown["conf"],
            "localization_loss": breakdown["loc"]
        }
    

    def evaluate(self, batch, batch_idx, stage=None):

        images, targets = batch

        confidences, localizations = self(images)

        loss, breakdown = self.loss(confidences, localizations, targets)

        records = {
            "loss": loss,
            "confidence_loss": breakdown["conf"],
            "localization_loss": breakdown["loc"]
        }

        if batch_idx % self.map_frequency == 0 and batch_idx > 0:
            metrics = self.map.compute()

            records["map"] = metrics["map"]
            records["map_50"] = metrics["map_50"]
            records["map_75"] = metrics["map_75"]
            records["map_large"] = metrics["map_large"]
            records["map_medium"] = metrics["map_medium"]
            records["map_small"] = metrics["map_small"]

        return records

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay

        )

        return { "optimizer": optimizer }