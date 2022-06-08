from typing import Any

import albumentations as A
import numpy as np


class Transformations:
    """
    Sets up an albumentations transformation composition when initialized.
    It returns a transformed image when called
    """

    def __init__(self, config: dict, mode: str) -> None:

        if config["transform"] and mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Resize(config["figure_size"], config["figure_size"]),
                ],
                bbox_params=A.BboxParams(format="pascal_voc"),
            )
        else:
            self.transform = A.Compose(
                [A.Resize(config["figure_size"], config["figure_size"])],
                bbox_params=A.BboxParams(format="pascal_voc"),
            )

    def __call__(self, image: np.ndarray, bounding_boxes: list) -> Any:
        """
        Apply the albumentations transformation and return the transformed image

        Args:
            image (np.ndarray): Image to be transformed
            bounding_boxes (list): Bounding boxes for the image

        Returns:
            dict: An albumentations dictionary containg image and bounding boxes
        """

        return self.transform(image=image, bboxes=bounding_boxes)
