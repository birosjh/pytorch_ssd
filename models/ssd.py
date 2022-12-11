"""
explicit_ssd.py
---------------------------------------------

Contains a model with the layers explicitly written out.

"""

import torch
import torch.nn as nn

from models.backbone.backbone_loader import backbone_loader
from utils.default_box import number_of_default_boxes_per_cell


class SSD(nn.Module):
    def __init__(self, config: dict, num_classes: int, device: str) -> None:
        # Always have to do this when making a new model
        super(SSD, self).__init__()

        self.feature_map_extractor = backbone_loader(config).to(device)
        self.num_classes = num_classes

        self.loc_layers = []
        self.conf_layers = []

        num_defaults_per_cell = number_of_default_boxes_per_cell(
            config["aspect_ratios"]
        )

        output_channels = self.feature_map_extractor.output_channels()

        for num_anchors, output_channels in zip(num_defaults_per_cell, output_channels):

            self.loc_layers.append(
                self.bbox_predictor(
                    output_channels, num_anchors, device
                ).to(device)
            )
            self.conf_layers.append(
                self.class_predictor(
                    output_channels, num_anchors, num_classes, device
                ).to(device)
            )

    def forward(self, x):
        feature_maps = []
        loc = []
        conf = []

        # Collect feature_maps
        for section in self.feature_map_extractor.model:

            x = section(x)
            feature_maps.append(x)

        for idx, feature_map in enumerate(feature_maps):
            loc.append(
                self.loc_layers[idx](feature_map).permute(0, 2, 3, 1).contiguous()
            )
            conf.append(
                self.conf_layers[idx](feature_map).permute(0, 2, 3, 1).contiguous()
            )

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        return (conf, loc)

    def class_predictor(self, out_channels, num_anchors, num_classes, device):

        return nn.Conv2d(
                out_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )

    def bbox_predictor(self, out_channels, num_anchors, device):

        return nn.Conv2d(
            out_channels, num_anchors * 4, kernel_size=3, padding=1
        )
