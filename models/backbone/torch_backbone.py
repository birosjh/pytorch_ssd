import torch.models as models

from models.layers.downsample_block import DownsampleBlock


class TorchBackbone(nn.Module):
    def __init__(self, config):

        super(TorchBackbone, self).__init__()

        load_model = getattr(models, config["backbone"])

        self.model = load_model(pretrained=config["pretrained"])

        self.model = self.model.features

    def forward(self, x):

        return self.model(x)
