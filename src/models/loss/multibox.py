import torch
import torch.nn as nn

from functools import partial
from torchvision.ops import generalized_box_iou_loss
# from src.models.loss.localization import LocalizationLoss


class MultiBoxLoss(nn.Module):
    def __init__(self, alpha, iou_threshold, default_boxes, scale_xy, scale_wh) -> None:
        super(MultiBoxLoss, self).__init__()

        self.confidence_loss = nn.CrossEntropyLoss(reduction="none")
        self.localization_loss = partial(generalized_box_iou_loss, reduction='none')

        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.default_boxes = default_boxes.unsqueeze(dim=0)

        self.scale_xy = scale_xy
        self.scale_wh = scale_wh

    def create_offsets(self, boxes):
        """
        Create Location Offsets
        """

        wh = self.default_boxes[:, :, 2:] - self.default_boxes[:, :, :2]


        xy_offsets = self.scale_xy * (boxes[:, :, :2] - self.default_boxes[:, :, :2]) / wh
        wh_offsets = self.scale_wh * (boxes[:, :, 2:]/self.default_boxes[:, :, 2:]).log()

        return torch.cat((xy_offsets, wh_offsets), dim=2).contiguous()


    def forward(self, confidences: torch.Tensor, localizations: torch.Tensor, targets: torch.Tensor):
        """_summary_

        Args:
            confidences (torch.Tensor): Confidence values outputed from the model
            localizations (torch.Tensor): Localization values outputed from the model
            targets (torch.Tensor): _description_

        Returns:
            loss: Total Loss
            conf_loss: Confidence Loss
            loc_loss: Localization Loss
        """

        boxes = targets[:, :, :4]
        labels = targets[:, :, 4].long()
        
        non_zero_label_indices = labels > 0
        num_non_zero_targets = non_zero_label_indices.sum(dim=1)
        
        # location_offsets = self.create_offsets(boxes)

        loc_loss = self.localization_loss(localizations, boxes)

        # Retain only loss of non-background targets
        loc_loss = (non_zero_label_indices.float() * loc_loss).sum(dim=1)

        conf_loss = self.confidence_loss(confidences.permute((0, 2, 1)), labels)

        negative_conf_loss = conf_loss.clone()

        # Don't select positive targets
        negative_conf_loss[non_zero_label_indices] = 0

        _, conf_idx = negative_conf_loss.sort(dim=1, descending=True)
        _, conf_rank = conf_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(
            3 * num_non_zero_targets,
            max=non_zero_label_indices.size(1)
        ).unsqueeze(-1)

        neg_target_indices = conf_rank < neg_num

        conf_loss = (conf_loss * ((non_zero_label_indices + neg_target_indices).float())).sum(dim=1)

        # avoid no object detected
        total_loss = loc_loss + (self.alpha * conf_loss)
        num_mask = (num_non_zero_targets > 0).float()
        num_non_zero_targets = num_non_zero_targets.float().clamp(min=1e-6)
        loss = (total_loss * num_mask / num_non_zero_targets).mean(dim=0)

        return loss, loc_loss.detach().mean(), self.alpha * conf_loss.detach().mean()
