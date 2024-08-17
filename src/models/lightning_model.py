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
            model_config["alpha"],
            model_config["iou_threshold"],
            data_encoder.default_boxes,
            data_encoder.scale_xy,
            data_encoder.scale_wh
        )

        self.map_frequency = training_config["map_frequency"]

        self.lr = training_config["learning_rate"]
        self.weight_decay = model_config["weight_decay"]

        self.map = MeanAveragePrecision()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        confidences, localizations = self(images)

        loss, loc_loss, conf_loss = self.loss(confidences, localizations, targets)

        records = {
            "loss": loss,
            "confidence_loss": loc_loss,
            "localization_loss": conf_loss,
        }

        self.log_dict(records, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return records

    def evaluate(self, batch, batch_idx, stage=None):
        images, targets = batch

        confidences, localizations = self(images)

        loss, loc_loss, conf_loss = self.loss(confidences, localizations, targets)

        records = {
            "loss": loss,
            "confidence_loss": loc_loss,
            "localization_loss": conf_loss,
        }

        self.log_dict(records, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return records

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, batch_idx, "val")

    def on_validation_epoch_end(self):

        metrics = self.map.compute()

        records = {
            "map": metrics["map"],
            "map_50": metrics["map_50"],
            "map_75": metrics["map_75"],
            "map_large": metrics["map_large"],
            "map_medium": metrics["map_medium"],
            "map_small": metrics["map_small"]
        }

        self.log_dict(records, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        self.evaluate(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        return {"optimizer": optimizer}
