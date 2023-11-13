import torchvision.models as models
from torch.nn import MaxPool2d, Module, Sequential

from src.models.layers.downsample_block import DownsampleBlock


class Vgg16(Module):
    def __init__(self, config):
        super(Vgg16, self).__init__()

        vgg16 = models.vgg16(weights="IMAGENET1K_V1")

        self.model = Sequential(
            Sequential(*vgg16.features[0:17]),
            Sequential(*vgg16.features[17:24]),
            Sequential(*vgg16.features[24:]),
            DownsampleBlock(512, 256),
            DownsampleBlock(256, 128),
            MaxPool2d(2),
        )

    def forward(self, x):
        return self.model(x)

    def output_channels(self):
        return [256, 512, 512, 256, 128, 128]
