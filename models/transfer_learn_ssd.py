"""
transfer_learn_ssd.py
---------------------------------------------

Contains models that load a pretrained model from torch hub.

"""

import torch
import torch.nn as nn


class SSD(nn.Module):
    def __init__(self):

        self.base_model = torch.hub.load(
            "pytorch/vision:v0.6.0", "vgg16", pretrained=True
        ).features[0:30]
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.extras = create_extras()

    def forward(self, x):

        x = self.base_model(x)

        x = self.pool5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.extras(x)

    # def create_extras():

    #     # create extras here

    def class_predictor(self, out_channels, num_anchors, num_classes):

        return nn.Conv2d(
            out_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )

    def bbox_predictor(self, out_channels, num_anchors):

        return nn.Conv2d(out_channels, num_anchors * 4, kernel_size=3, padding=1)
