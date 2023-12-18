import torch
import torch.nn as nn

from torchvision.ops import generalized_box_iou_loss
# from src.models.loss.localization import LocalizationLoss


class SSDLoss(nn.Module):
    def __init__(self, alpha, iou_threshold, data_encoder) -> None:
        super(SSDLoss, self).__init__()

        self.confidence_loss = nn.CrossEntropyLoss(reduction="none")
        self.localization_loss = generalized_box_iou_loss

        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.data_encoder = data_encoder

    def forward_per_item(
        self,
        pred_confidences: torch.Tensor,
        pred_localizations: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple:
        target_confidences = targets[:, -1].type(torch.int64)
        target_localizations = targets[:, 0:4]

        object_exists = target_confidences > 0

        num_matched = object_exists.int().sum()

        loss = self.calculate_confidence_loss(
            pred_confidences, target_confidences, object_exists, num_matched
        )

        conf_loss = loss.float().item()

        localization_loss = self.calculate_localization_loss(
            pred_localizations[object_exists],
            target_localizations[object_exists],
        )

        loss += self.alpha * localization_loss
        loc_loss = localization_loss.float().item()

        return loss, conf_loss, loc_loss, num_matched

    def calculate_confidence_loss(
        self, pred_confidences, target_confidences, positive_matches, num_matched
    ):
        positive_loss = self.calculate_positives_confidence_loss(
            pred_confidences[positive_matches], target_confidences[positive_matches]
        )

        negative_loss = self.calculate_negatives_confidence_loss(
            pred_confidences[~positive_matches], num_matched
        )

        loss = positive_loss + negative_loss

        return loss

    def calculate_positives_confidence_loss(
        self, matched_confidences, matched_target_confidences
    ):
        # Calculate Positive Confidence Losses
        positive_confidence_losses = self.confidence_loss(
            matched_confidences, matched_target_confidences
        )

        return positive_confidence_losses.sum()

    def calculate_negatives_confidence_loss(self, unmatched_confidences, num_matched):
        # Calculate Negative Confidence Losses
        unsorted_negative_confidence_losses = self.confidence_loss(
            unmatched_confidences,
            torch.zeros(unmatched_confidences.shape[0], dtype=int).to(
                unmatched_confidences.device
            ),
        )
        negative_confidence_losses = unsorted_negative_confidence_losses.sort(
            descending=True
        ).values

        return negative_confidence_losses[0 : 3 * num_matched].sum()

    def calculate_localization_loss(
        self, matched_localizations, matched_target_localizations
    ):
        localization_loss = self.localization_loss(
            matched_localizations, matched_target_localizations, reduction="sum"
        )

        return localization_loss

    def forward(self, confidences, localizations, targets):
        batch_conf_loss = 0
        batch_loc_loss = 0
        batch_loss = 0
        num_hits = 0

        for conf, loc, target in zip(confidences, localizations, targets):
            loss, conf_loss, loc_loss, hit_count = self.forward_per_item(
                conf, loc, target
            )

            batch_conf_loss += conf_loss
            batch_loc_loss += loc_loss
            batch_loss += loss
            num_hits += hit_count

        if num_hits == 0:
            return torch.tensor(0), {"conf": 0, "loc": 0}

        batch_conf_loss /= num_hits
        batch_loc_loss /= num_hits
        batch_loss /= num_hits

        breakdown = {"conf": batch_conf_loss, "loc": batch_loc_loss}

        return batch_loss, breakdown
