import os

import cv2
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import torch
import xmltodict
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self, data_config, data_encoder, transform=False, mode="train", visualize=False
    ):

        # Set the mode (train/val)
        self.mode = mode

        self.visualize = visualize

        # Read in necessary configs
        file_data_path = data_config[mode]
        self.image_directory = data_config["image_directory"]
        self.annotation_directory = data_config["annotation_directory"]
        self.classes = tuple(data_config["classes"])

        self.file_list = self.create_file_list(file_data_path)

        self.transform = transform

        self.transformations = iaa.Noop()

        self.height = data_config["figure_size"]
        self.width = data_config["figure_size"]

        self.resize_transformation = iaa.Resize(
            {"height": self.height, "width": self.width}
        )

        self.data_encoder = data_encoder
        self.default_boxes = self.data_encoder.default_boxes

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        # Select filename from list
        filename = self.file_list[idx]

        # Load image
        image = self.load_image(filename)

        # Load annotations for image
        labels = self.load_labels_from_annotation(filename)

        labels = BoundingBoxesOnImage(labels, shape=image.shape)

        # Perform transformations on the data
        if self.transform:

            image, labels = self.transformations(image=image, bounding_boxes=labels)

        # Resize data regardless of train or test
        image, labels = self.resize_transformation(image=image, bounding_boxes=labels)

        labels = labels.bounding_boxes

        labels = torch.Tensor(
            np.array([np.append(box.coords.flatten(), box.label) for box in labels])
        )

        image = torch.Tensor(image)

        if self.mode == "train":
            labels = self.data_encoder.encode(labels)

        if self.visualize:

            return (image, labels)

        # This seems bad, but I will revist it later when I check for bottlenecks
        image = image.permute(2, 0, 1)

        return (image, labels)

    def create_file_list(self, file_data_path: str) -> list:

        df = pd.read_csv(file_data_path, names=["filename"])

        df["filename"] = df["filename"].str.replace(r"\s\s\d", "")

        file_list = df["filename"].unique().tolist()

        return file_list

    def load_image(self, filename: str) -> np.array:

        path_to_file = os.path.join(self.image_directory, filename + ".jpg")
        image = cv2.imread(path_to_file)

        return image

    def load_labels_from_annotation(self, filename: str) -> list:

        path_to_file = os.path.join(self.annotation_directory, filename + ".xml")

        with open(path_to_file) as fd:
            annotation = xmltodict.parse(fd.read())

        objects = annotation["annotation"]["object"]

        if type(objects) is not list:
            objects = [objects]

        labels = []

        for obj in objects:

            box = obj["bndbox"]

            labels.append(
                BoundingBox(
                    x1=box["xmin"],
                    y1=box["ymin"],
                    x2=box["xmax"],
                    y2=box["ymax"],
                    label=self.classes.index(obj["name"]),
                )
            )

        return labels
