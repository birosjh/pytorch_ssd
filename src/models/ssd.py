"""
explicit_ssd.py
---------------------------------------------

Contains a model with the layers explicitly written out.

"""

import torch
import torch.nn as nn

from src.models.backbone.backbone_loader import backbone_loader
from src.utils.data_encoder import DataEncoder


class SSD(nn.Module):
    def __init__(
        self,
        config: dict,
        num_classes: int,
        data_encoder: DataEncoder,
    ) -> None:
        # Always have to do this when making a new model
        super(SSD, self).__init__()

        self.feature_map_extractor = backbone_loader(config)
        self.num_classes = num_classes

        self.data_encoder = data_encoder

        self.loc_layers = nn.ModuleList([])
        self.conf_layers = nn.ModuleList([])

        num_defaults_per_cell = self.data_encoder.number_of_default_boxes_per_cell()

        output_channels = self.feature_map_extractor.output_channels()

        for num_anchors, output_channels in zip(num_defaults_per_cell, output_channels):
            self.loc_layers.append(self.bbox_predictor(output_channels, num_anchors))
            self.conf_layers.append(
                self.class_predictor(output_channels, num_anchors, num_classes)
            )

    def forward(self, x):
        loc = []
        conf = []

        feature_maps = self.feature_map_extractor(x)

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
        loc = self.convert_to_box(loc)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        return (conf, loc)

    def class_predictor(self, out_channels, num_anchors, num_classes):
        conf_head = nn.Conv2d(
            out_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )

        nn.init.xavier_uniform_(conf_head.weight.data)
        if conf_head.bias is not None:
            nn.init.constant_(conf_head.bias, 0.0)

        return conf_head

    def bbox_predictor(self, out_channels, num_anchors):
        loc_head = nn.Conv2d(out_channels, num_anchors * 4, kernel_size=3, padding=1)

        nn.init.xavier_uniform_(loc_head.weight.data)
        if loc_head.bias is not None:
            nn.init.constant_(loc_head.bias, 0.0)

        return loc_head

    def convert_to_box(self, loc):
        default_boxes = self.data_encoder.default_boxes.to(loc.device)

        new_locs = torch.zeros(loc.shape).to(loc.device)

        for idx in range(loc.shape[0]):
            new_locs[idx][:, 0] = loc[idx][:, 0] * (
                default_boxes[:, 2] - default_boxes[:, 0]
            )
            new_locs[idx][:, 1] = loc[idx][:, 1] * (
                default_boxes[:, 3] - default_boxes[:, 1]
            )
            new_locs[idx][:, 2] = loc[idx][:, 2] * (
                default_boxes[:, 2] - default_boxes[:, 0]
            )
            new_locs[idx][:, 3] = loc[idx][:, 3] * (
                default_boxes[:, 3] - default_boxes[:, 1]
            )

            new_locs[idx][:, 0] += default_boxes[:, 0]
            new_locs[idx][:, 1] += default_boxes[:, 1]
            new_locs[idx][:, 2] += default_boxes[:, 2]
            new_locs[idx][:, 3] += default_boxes[:, 3]

        return new_locs
