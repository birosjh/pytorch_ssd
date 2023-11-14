import torch
import torch.nn as nn

from src.models.loss.localization import LocalizationLoss


class SSDLoss(nn.Module):
    def __init__(self, alpha) -> None:
        super(SSDLoss, self).__init__()

        self.alpha = alpha

        self.confidence_loss = nn.CrossEntropyLoss()
        self.localization_loss = LocalizationLoss()

    def forward(
        self,
        pred_confidences: torch.Tensor,
        pred_localizations: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple:
        target_confidences = targets[:, -1].type(torch.int32)
        target_localizations = targets[:, 0:-1]

        confidence_loss = self.confidence_loss(pred_confidences, target_confidences)

        predictions = pred_confidences.max(dim=1).indices

        matched_pred_localizations = pred_localizations[
            predictions == target_confidences
        ]
        matched_target_localizations = target_localizations[
            predictions == target_confidences
        ]

        localization_loss = self.localization_loss(
            matched_pred_localizations, matched_target_localizations
        )

        return confidence_loss, self.alpha * localization_loss
