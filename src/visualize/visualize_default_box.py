import argparse
import random

import cv2
import imgaug.augmenters as iaa
import yaml
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from src.utils.data_encoder import DataEncoder


def get_box_groups(encoder_config, default_boxes):
    box_groups = {}

    counter = 0

    for idx, map_size in enumerate(encoder_config["feature_map_sizes"]):
        num_aspect_ratios = 2 + len(encoder_config["aspect_ratios"][idx]) * 2
        num_boxes = map_size * map_size * num_aspect_ratios

        boxes_for_this_group = default_boxes[counter : counter + num_boxes]

        box_groups[map_size] = [
            boxes_for_this_group[i : i + num_aspect_ratios]
            for i in range(0, len(boxes_for_this_group), num_aspect_ratios)
        ]

        counter += num_boxes

    return box_groups


def visualize_default_box(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)

    model_config = config["model_configuration"]
    data_encoder = DataEncoder(model_config)

    image_path = "data/JPEGImages/2010_001718.jpg"
    image = cv2.imread(image_path)

    aug = iaa.Resize({"height": 256, "width": 256})

    resized_image = aug(image=image)

    default_boxes = data_encoder.default_boxes

    box_groups = get_box_groups(model_config, default_boxes)

    print(box_groups.keys())

    set_of_boxes = random.choice(box_groups[32])

    set_of_boxes = set_of_boxes

    print(set_of_boxes)

    vis_boxes = []

    for box in set_of_boxes:
        vis_boxes.append(BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))

    bbs = BoundingBoxesOnImage(vis_boxes, shape=resized_image.shape)

    image_with_box = bbs.draw_on_image(resized_image, size=2)

    cv2.imwrite("example.png", image_with_box)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", action="store", required=True)
    arguments = parser.parse_args()

    visualize_default_box(arguments.config)
