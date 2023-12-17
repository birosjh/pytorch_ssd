import torch
import torch.nn as nn


class LocalizationLoss(nn.Module):
    """
    Bounding box loss for only the boxes that had content categorized correctly.
    """

    def __init__(self):
        super(LocalizationLoss, self).__init__()

        self.smooth_l1 = nn.SmoothL1Loss(reduction="sum")

    def forward(self, predictions, targets):
        # As specified in the SSD paper, set to zero when no
        # examples are predicted correctly
        if len(predictions) == 0:
            return torch.tensor(0)

        return self.smooth_l1(predictions, targets)
