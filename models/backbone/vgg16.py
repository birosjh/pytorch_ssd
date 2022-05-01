from torch.nn import MaxPool2d, Module
import torchvision.models as models

class Vgg16(Module):
    def __init__(self, config):

        super(Vgg16, self).__init__()

        vgg16 = models.vgg16(pretrained=config["pretrained"])

        vgg16.features[30] = MaxPool2d(
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            ceil_mode=False
        )

        self.output_1 = vgg16.features[0:24]

        self.output_2 = vgg16.features[24:]


    def forward(self, x):

        out = self.output_1(x)

        return self.output_2(out)