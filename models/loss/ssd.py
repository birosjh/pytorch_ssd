import torch
import torch.nn as nn

from models.loss.localization import LocalizationLoss


class SSDLoss(nn.Module):
    def __init__(self):

        self.confidence_loss = nn.CrossEntropyLoss()
        self.localization_loss = LocalizationLoss()

    def forward(self, predictions, targets, default_boxes):

        pred_confidences, pred_localizations = predictions
        pred_confidences = torch.argmax(pred_confidences)

        target_confidences = targets[:, 0, :]
        target_localizations = targets[:, 1:, :]

        confidence_loss = self.confidence_loss(pred_confidences, target_confidences)

        matched_pred_localizations = pred_localizations[
            pred_confidences == target_confidences
        ]
        matched_target_localizations = target_localizations[
            pred_confidences == target_confidences
        ]

        localization_loss = self.localization_loss(
            matched_pred_localizations, matched_target_localizations
        )

        num_matched = torch.sum(pred_confidences == target_confidences)

        loss = (1 / num_matched) * (confidence_loss + self.alpha * localization_loss)

        return confidence_loss, localization_loss, loss
