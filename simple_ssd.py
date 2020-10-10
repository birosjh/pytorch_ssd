import torch
import torch.nn as nn
import torch.nn.functional as F


class SSD(nn.Module):

    def __init__(self, num_classes):
        # Always have to do this when making a new model
        super(SSD, self).__init__()
        sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
                 [0.88, 0.961]]


        ratios = [[1, 2, 0.5]] * 5
        num_anchors = len(sizes[0]) + len(ratios[0]) - 1


        self.loc_layers = []
        self.conf_layers = []

        self.base = self.base_net([3, 16, 32, 64])
        self.loc_layers += self.bbox_predictor(64, num_anchors)
        self.conf_layers += self.class_predictor(64, num_anchors, num_classes)

        self.block_1 = self.down_sample_block(64, 128)
        self.loc_layers += self.bbox_predictor(128, num_anchors)
        self.conf_layers += self.class_predictor(128, num_anchors, num_classes)

        self.block_2 = self.down_sample_block(128, 128)
        self.loc_layers += self.bbox_predictor(128, num_anchors)
        self.conf_layers += self.class_predictor(128, num_anchors, num_classes)

        self.block_3 = self.down_sample_block(128, 128)
        self.loc_layers += self.bbox_predictor(128, num_anchors)
        self.conf_layers += self.class_predictor(128, num_anchors, num_classes)

        self.loc_layers += self.bbox_predictor(128, num_anchors)
        self.conf_layers += self.class_predictor(128, num_anchors, num_classes)

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

        for in_channels, out_channels in zip(in_channel_list, out_channel_list):
            base.append(self.down_sample_block(in_channels, out_channels))

        return nn.Sequential(*base)

    def down_sample_block(self, in_channels, out_channels):

        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        return block


    def class_predictor(self, out_channels, num_anchors, num_classes):

        return [nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size=3, padding=1)]

    def bbox_predictor(self, out_channels, num_anchors):

        return [nn.Conv2d(out_channels, num_anchors * 4, kernel_size=3, padding=1)]
