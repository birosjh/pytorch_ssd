import argparse

import cv2
import yaml
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torch.utils.data import DataLoader

from datasets.image_dataset import ImageDataset
from utils.data_encoder import DataEncoder

BOX_COLOR = (0, 0, 0)
TEXT_COLOR = (0, 0, 0)

def get_color_from_list(class_id, color_list):

    return tuple((color_list[class_id] * 256)[0:-1].astype(int).tolist())

def visualize_bbox(img, bbox, color_list, class_names, thickness=2):
    """Visualizes a single bounding box on the image"""

    x_min, y_min, x_max, y_max, class_id = bbox
    class_id = int(class_id)
    class_name = class_names[class_id]

    color = get_color_from_list(class_id, color_list)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img

def visualize_batch(config_file, use_val=False):

    with open(config_file) as file:
        config = yaml.safe_load(file)

    model_config = config["model_configuration"]
    training_config = config["training_configuration"]
    data_config = config["data_configuration"]

    mode = "train"

    if use_val:
        mode = "val"

    data_encoder = DataEncoder(model_config)

    dataset = ImageDataset(
        data_config=data_config,
        data_encoder=data_encoder,
        mode=mode
    )

    dataloader = DataLoader(
        dataset, batch_size=training_config["batch_size"], num_workers=0, shuffle=True
    )

    images, labels = next(iter(dataloader))

    bbs_applied_images = []

    cmap = cm.viridis
    color_list = cmap(range(len(data_config["classes"])))

    for image, labelset in zip(images, labels):

        np_image = image.permute(1, 2, 0).numpy().copy().astype(np.uint8)

        np_labels = labelset[labelset[:,-1] > 0].numpy().astype(int)

        for label in np_labels:

            np_image = visualize_bbox(np_image, label, color_list, data_config["classes"])

        bbs_applied_images.append(
            np_image
        )

    batch_image = cv2.vconcat(bbs_applied_images)

    file_name = Path("visualize") / "visualized_batch.jpg"
    
    cv2.imwrite(str(file_name.absolute()), batch_image)
