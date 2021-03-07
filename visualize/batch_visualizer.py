import argparse
import cv2
import torch
import yaml

from image_dataset import ImageDataset
from torch.utils.data import DataLoader

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def load_configurations():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', action="store", required=True)
    parser.add_argument('--val', action="store_true")

    arguments = parser.parse_args()

    with open(arguments.config) as file:
        config = yaml.safe_load(file)

    return config, arguments.val

def convert_to_bounding_boxes(labels):

    bounding_boxes = []

    for label_group in labels:

        boxes_group = []

        for label in label_group:
        
            if label[4] != 0:
                boxes_group.append(
                    BoundingBox(
                        x1=label[0],
                        y1=label[1],
                        x2=label[2],
                        y2=label[3],
                        label=label[4]
                    )
                )
        
        bounding_boxes.append(boxes_group)
        
    return bounding_boxes


def main():

    config, use_val = load_configurations()

    training_config = config["training_configuration"]
    data_config = config["data_configuration"]

    mode = "train"

    if use_val:
        mode = "val"


    dataset = ImageDataset(
        data_config=data_config,
        transform=True,
        mode=mode
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        num_workers=0
    )

    images, labels = next(iter(dataloader))

    bounding_boxes = convert_to_bounding_boxes(labels)

    bbs = BoundingBoxesOnImage(bounding_boxes[0], shape=images[0].shape)

    image_after = bbs.draw_on_image(images[0], size=2, color=[0, 0, 255])

    file_name = "image.jpg"
    cv2.imwrite(file_name, image_after)


if __name__ == "__main__":

    main()
