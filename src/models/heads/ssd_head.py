import torch
import torch.nn as nn


class SSDHead(nn.Module):
    def __init__(self, output_channels, num_anchors, num_classes) -> None:
        super(SSDHead, self).__init__()

        self.loc_head = self.bbox_predictor(output_channels, num_anchors)

        self.conf_head = self.class_predictor(output_channels, num_anchors, num_classes)

    def class_predictor(self, out_channels, num_anchors, num_classes):
        return nn.Conv2d(
            out_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )

    def bbox_predictor(self, out_channels, num_anchors):
        return nn.Conv2d(out_channels, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        conf = self.conf_head(x).permute(0, 2, 3, 1).contiguous()

        loc = self.loc_head(x).permute(0, 2, 3, 1).contiguous()

        return (conf, loc)
