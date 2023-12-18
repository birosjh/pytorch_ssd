from pathlib import Path
import argparse

import cv2
import matplotlib.cm as cm
import numpy as np
import yaml
from torch.utils.data import DataLoader

from src.datamodules.datasets.image_dataset import ImageDataset
from src.utils.data_encoder import DataEncoder

BOX_COLOR = (0, 0, 0)
TEXT_COLOR = (0, 0, 0)


def get_color_from_list(class_id: int, color_list: list) -> tuple:
    return tuple((color_list[class_id] * 256)[0:-1].astype(int).tolist())


def visualize_bbox(
    image: np.ndarray,
    label: np.ndarray,
    color_list: list,
    class_names: list,
    thickness=2,
) -> np.ndarray:
    """
    Visualizes a single bounding box on the image

    Args:
        image (np.ndarray): Image to apply bounding boxes to
        label (np.ndarray): A label containing bounding box coordinates and a class id
        color_list (list): A list of colors for bounding boxes of each class
        class_names (list): A list of class names
        thickness (int, optional): Thickness of . Defaults to 2.

    Returns:
        np.ndarray: The image with the box drawn on it
    """

    x_min, y_min, x_max, y_max, class_id = label
    class_id = int(class_id)
    class_name = class_names[class_id]

    color = get_color_from_list(class_id, color_list)

    cv2.rectangle(
        image, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness
    )

    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
    )

    cv2.rectangle(
        image,
        (x_min, y_min),
        (x_min + text_width, y_min + int(1.3 * text_height)),
        color=color,
        thickness=-1,
    )

    cv2.putText(
        image,
        text=class_name,
        org=(x_min, y_min + int(text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return image


def visualize_batch(images, labels, classes, targets=None) -> None:
    """
    Visualize a batch of images exactly as they are returned from the dataloader
    This includes any transformations that may have been performed

    Args:
        config_path (str): Path to the desired config  file
        val (bool, optional): Whether to use a validation data loader or not.
            Defaults to False.
    """

    if targets is not None:
        classes = ["background"] + list(classes)

    bbs_applied_images = []

    cmap = cm.viridis
    color_list = cmap(range(len(classes)))

    for idx, (image, labelset) in enumerate(zip(images, labels)):
        if targets is not None:
            object_exists = targets[idx][:, -1] > 0

            # Find Positive Matches Between Preds and Targets

            np_labels = labelset[object_exists].numpy().astype(int)

        else:
            np_labels = labelset[labelset[:, -1] > 0].numpy().astype(int)

        np_image = image.permute(1, 2, 0).numpy().copy().astype(np.uint8)

        for label in np_labels:
            np_image = visualize_bbox(np_image, label, color_list, classes)

        bbs_applied_images.append(np_image)

    batch_image = cv2.vconcat(bbs_applied_images)

    file_name = Path("src/visualize") / "visualized_batch.jpg"

    cv2.imwrite(str(file_name.absolute()), batch_image)


def visualize_a_batch(config, val):
    with open(config) as file:
        config = yaml.safe_load(file)

    model_config = config["model_configuration"]
    training_config = config["training_configuration"]
    data_config = config["data_configuration"]

    mode = "train"

    if val:
        mode = "val"

    data_encoder = DataEncoder(model_config)

    dataset = ImageDataset(
        data_config=data_config, data_encoder=data_encoder, mode=mode
    )

    dataloader = DataLoader(
        dataset, batch_size=training_config["batch_size"], num_workers=0, shuffle=True
    )

    images, labels = next(iter(dataloader))

    classes = data_config["classes"]

    visualize_batch(images, labels, classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", action="store", required=True)
    parser.add_argument("--val", action="store", required=True)
    arguments = parser.parse_args()

    visualize_a_batch(arguments.config, arguments.val)
