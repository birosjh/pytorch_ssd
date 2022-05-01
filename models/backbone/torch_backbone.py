import torch.nn as nn

from models.layers.downsample_block import DownsampleBlock


class SimpleBackbone(nn.Module):
    def __init__(self, layers):

        super(SimpleBackbone, self).__init__()

        base = []

        in_channel_list = layers[:-1]
        out_channel_list = layers[1:]

        for in_channels, out_channels in zip(in_channel_list, out_channel_list):

            base.append(DownsampleBlock(in_channels, out_channels))

        self.layers = nn.Sequential(*base)

    def forward(self, x):

        return self.layers(x)
