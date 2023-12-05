import timm
import torch
import torch.nn as nn

def initialize(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight.data)
        layer.bias.data.zero_()

class VGG16(nn.Module):

    def __init__(self, pretrained=True):
        super(VGG16, self).__init__()

        self.model = timm.create_model(
            "vgg16",
            pretrained=pretrained,
            features_only=True,
            out_indices=[3]
        )

        self.blocks = nn.ModuleList([
            self.block_2(),
            self.block_3(),
            self.block_4(),
            self.block_5(),
            self.block_6(),
        ])

    
    def forward(self, x):

        layer_outputs = []

        out = self.model(x)[0]
        layer_outputs.append(out)

        for block in self.blocks:

            out = block(out)
            layer_outputs.append(out)

        return layer_outputs

    def block_2(self):
        block = nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        
        block.apply(initialize)

        return block
    
    def block_3(self):
        block =  nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        block.apply(initialize)

        return block
    
    def block_4(self):
        block = nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        block.apply(initialize)

        return block
    
    def block_5(self):
        block = nn.Sequential(
            torch.nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        block.apply(initialize)

        return block
    
    def block_6(self):
        block =  nn.Sequential(
            torch.nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        return block
    
    def output_channels(self):

        return [512, 512, 512, 1024, 1024, 2048]
