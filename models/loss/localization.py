import torch.nn as nn


class LocalizationLoss(nn.Module):
    """
    Bounding box loss for only the boxes that had content categorized correctly.
    """

    def __init__(self):
        super(LocalizationLoss, self).__init__()

        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, predictions, targets):

        return self.smooth_l1(predictions, targets)
