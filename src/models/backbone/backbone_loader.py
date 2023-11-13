from src.models.backbone.vgg16 import Vgg16


def backbone_loader(config):
    model_options = {"vgg16": Vgg16}

    return model_options[config["backbone"]](config)
