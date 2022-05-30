import torch
import torch.nn as nn

from models.loss.localization import LocalizationLoss


class SSDLoss(nn.Module):
    def __init__(self, alpha):
        super(SSDLoss, self).__init__()

        self.alpha = alpha

        self.confidence_loss = nn.CrossEntropyLoss()
        self.localization_loss = LocalizationLoss()

    def forward(self, predictions, targets):

        pred_confidences, pred_localizations = predictions
        pred_confidences = torch.argmax(pred_confidences, dim=2).type(torch.float32)

        target_confidences = targets[:, :, -1]
        target_localizations = targets[:, :, 0:-1]

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

        confidence_loss /= num_matched
        localization_loss /= num_matched

        loss = confidence_loss + self.alpha * localization_loss

        return confidence_loss, localization_loss, loss
