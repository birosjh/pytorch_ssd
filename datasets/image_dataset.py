import os
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import xmltodict
from torch.utils.data import Dataset

from datasets.transformations import Transformations
from utils.data_encoder import DataEncoder


class ImageDataset(Dataset):
    """
    A PyTorch Dataset for Pascal VOC Data
    """

    def __init__(
        self,
        data_config: dict,
        data_encoder: DataEncoder,
        mode: str = "train",
        visualize: bool = False,
        device: str = "cpu",
        iou_threshold: float = 0.5,
    ):
        self.visualize = visualize

        # Read in necessary configs
        file_data_path = data_config[mode]
        self.image_directory = data_config["image_directory"]
        self.annotation_directory = data_config["annotation_directory"]
        self.classes = tuple(data_config["classes"])

        self.file_list = self.create_file_list(file_data_path)

        self.transformations = Transformations(data_config["transformations"], mode)

        self.height = data_config["transformations"]["figure_size"]
        self.width = data_config["transformations"]["figure_size"]

        self.data_encoder = data_encoder
        self.default_boxes = self.data_encoder.default_boxes

        self.device = device

        self.iou_threshold = iou_threshold

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        # Select filename from list
        filename = self.file_list[idx]

        # Load image
        image = self.load_image(filename)

        # Load annotations for image
        labels = self.load_labels_from_annotation(filename)

        # Perform transformations on the data
        transformed_data = self.transformations(image=image, bounding_boxes=labels)

        image = transformed_data["image"]
        labels = transformed_data["bboxes"]

        image = torch.Tensor(image)
        label_tensor = torch.Tensor(labels)

        bounding_boxes = label_tensor[:, 0:4]
        labels_in = label_tensor[:, 4]

        encoded_boxes, labels_out = self.data_encoder.encode(
            bounding_boxes, labels_in, self.iou_threshold
        )

        # Rejoin the encoded boxes and labels
        label_tensor = torch.cat((encoded_boxes, labels_out.unsqueeze(1)), 1).to(
            self.device
        )

        if self.visualize:
            return (image, label_tensor)

        # This seems bad, but I will revist it later when I check for bottlenecks
        image = image.permute(2, 0, 1).to(self.device)

        return (image, label_tensor)

    def create_file_list(self, file_data_path: str) -> Any:
        """
        Creates a list of files from the specified text file

        Args:
            file_data_path (str): Path to the file containing data file names

        Returns:
            list: Returns a list of the names of all files in the dataset
        """

        df = pd.read_csv(file_data_path, names=["filename"])

        df["filename"] = df["filename"].str.replace(r"\s\s\d", "", regex=True)

        file_list = df["filename"].unique().tolist()

        return file_list

    def load_image(self, filename: str) -> np.array:
        """
        Loads a single image from the image directory

        Args:
            filename (str): Filename of the image to be loaded

        Returns:
            np.array: An image in the format of a numpy array
        """

        path_to_file = os.path.join(self.image_directory, filename + ".jpg")
        image = cv2.imread(path_to_file)

        return image

    def load_labels_from_annotation(self, filename: str) -> list:
        """
        Loads labels from an annotation file

        Args:
            filename (str): Name of the annotation file

        Returns:
            list: A list of labels
        """

        path_to_file = os.path.join(self.annotation_directory, filename + ".xml")

        with open(path_to_file) as fd:
            annotation = xmltodict.parse(fd.read())

        objects = annotation["annotation"]["object"]

        if not isinstance(objects, list):
            objects = [objects]

        labels = []

        for obj in objects:
            box = obj["bndbox"]

            labels.append(
                [
                    float(box["xmin"]),
                    float(box["ymin"]),
                    float(box["xmax"]),
                    float(box["ymax"]),
                    self.classes.index(obj["name"]),
                ]
            )

        return labels
