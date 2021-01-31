import os
import cv2
import json
import torch

import numpy as np
import pandas as pd
import imgaug.augmenters as iaa

from torch.utils.data import Dataset
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from data_encoder import DataEncoder

class ImageDataset(Dataset):

    def __init__(self, data_config, transform=True, mode="train"):

        # Read in necessary configs
        file_data_path = data_config[mode]
        self.image_directory = data_config["image_directory"]
        self.annotations_directory = data_config["annotations_directory"]

        self.file_list = self.create_file_list(file_data_path)

        self.transform = transform

        self.height = data_config["figure_size"]
        self.width = data_config["figure_size"]

        self.resize_transformation = iaa.Resize({
            "height": self.height,
            "width": self.width
        })

        self.data_encoder = DataEncoder(data_config)


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        # Load image
        filename = self.file_list[idx]
        path_to_file = os.path.join(self.image_directory, filename + ".jpg")
        image = cv2.imread(path_to_file)

        # Fetch annotations for that image (they are already in bbs format)
        labels = load_labels_from_annotation(filename)

        # Perform transformations on the data
        if self.transform:
            labels = BoundingBoxesOnImage(labels, shape=(self.height, self.width))

            image, labels = self.resize_transformation(
                image=image, 
                bounding_boxes=labels
            )

            labels = labels.bounding_boxes
        
        labels = np.array([np.append(box.coords.flatten(), box.label) for box in labels])

        if self.train:
            labels = self.data_encoder.encode(labels)

        return (image, labels)

    def create_file_list(self, file_data_path: str) -> list:

        df = pd.read_csv(file_data_path, names=["filename"])

        df['filename'] = df['filename'].str.replace(r"\s\s\d", "")

        file_list = df['filename'].tolist()

        return file_list

    def load_labels_from_annotation(self, filename: str): -> list:

        labels = []

        annotations = []

        for annotation in annotations:

            labels.append(BoundingBox(
                x1=annotation["x1"],
                y1=annotation["y1"],
                x2=annotation["x2"],
                y2=annotation["y2"],
                label=x1=annotation["label"]
            ))


        return labels

