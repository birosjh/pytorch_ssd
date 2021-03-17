"""
explicit_ssd.py
---------------------------------------------

Contains a model with the layers explicitly written out.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSD(nn.Module):

    def __init__(self, aspect_ratio_setting_per_feature_map: list, num_classes: int):
        # Always have to do this when making a new model
        super(SSD, self).__init__()


        self.loc_layers = []
        self.conf_layers = []

        self.base = self.base_net([3, 16, 32, 64])

        num_output_channels_per_layer = [64, 128, 128, 128, 128]

        num_defaults_per_feature_map = []

        for aspect_ratio_setting in aspect_ratio_setting_per_feature_map:
            num_defaults = 2 + len(aspect_ratio_setting) * 2

            num_defaults_per_feature_map.append(num_defaults)

        num_defaults_per_feature_map.append(num_defaults)
        
        for num_anchors, output_channels in zip(num_defaults_per_feature_map, num_output_channels_per_layer):

            self.loc_layers += self.bbox_predictor(output_channels, num_anchors)
            self.conf_layers += self.class_predictor(
                output_channels, num_anchors, num_classes)

        self.block_1 = self.down_sample_block(64, 128)
        self.block_2 = self.down_sample_block(128, 128)
        self.block_3 = self.down_sample_block(128, 128)


    def forward(self, x):
        feature_maps = []
        loc = []
        conf = []

        x = self.base(x)
        feature_maps.append(x)

        x = self.block_1(x)
        feature_maps.append(x)

        x = self.block_2(x)
        feature_maps.append(x)

        x = self.block_3(x)
        feature_maps.append(x)

        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        feature_maps.append(x)

        for idx, feature_map in enumerate(feature_maps):
            loc.append(self.loc_layers[idx](feature_map).permute(0, 2, 3, 1).contiguous())
            conf.append(self.conf_layers[idx](feature_map).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        return (loc, conf)

    def base_net(self, filter_sizes):
        base = []

        in_channel_list = filter_sizes[:-1]
        out_channel_list = filter_sizes[1:]

        for idx, (in_channels, out_channels) in enumerate(zip(in_channel_list, out_channel_list)):

            ciel = False

            if idx == 2:
                ciel = True
                
            base.append(self.down_sample_block(
                in_channels, out_channels, ceil=ciel))

        return nn.Sequential(*base)

    def down_sample_block(self, in_channels, out_channels, ceil=False):

        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=ceil)
        )

        return block


    def class_predictor(self, out_channels, num_anchors, num_classes):

        return [nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size=3, padding=1)]

    def bbox_predictor(self, out_channels, num_anchors):

        return [nn.Conv2d(out_channels, num_anchors * 4, kernel_size=3, padding=1)]
