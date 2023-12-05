from src.models.backbone.vgg16 import VGG16


def backbone_loader(config):
    model_options = {"vgg16": VGG16}

    return model_options[config["backbone"]](config)
