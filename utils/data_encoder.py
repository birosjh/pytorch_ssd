"""
data_encoder.py
---------------------------------------------

Contains functions for encoding bounding boxes in terms of default boxes.

"""
from itertools import product
from math import sqrt

import numpy as np
import torch


class DataEncoder:
    def __init__(self, default_box_config):

        self.default_boxes = self.create_default_boxes(default_box_config)

    def create_default_boxes(self, default_box_config: dict) -> torch.Tensor:

        figure_size = default_box_config["figure_size"]
        feature_map_sizes = default_box_config["feature_map_sizes"]
        steps = default_box_config["steps"]
        scales = default_box_config["scales"]
        aspect_ratios = default_box_config["aspect_ratios"]

        default_boxes = []

        # The size of the k-th square feature map
        fk = figure_size / np.array(steps)

        for idx, size in enumerate(feature_map_sizes):

            # Scale for Aspect Ratio 1
            sk = scales[idx] / figure_size
            # Additional Scale for Aspect Ratio 1
            sk_prime = sqrt(sk * scales[idx + 1] / figure_size)

            box_dimensions = [(sk, sk), (sk_prime, sk_prime)]

            for ratio in aspect_ratios[idx]:
                width = sk * sqrt(ratio)
                height = sk / sqrt(ratio)

                # For ratio = n
                box_dimensions.append((width, height))
                # For ratio = 1/n
                box_dimensions.append((height, width))

            # Loop through every point in the feature map
            for i, j in product(range(size), repeat=2):
                for width, height in box_dimensions:
                    # Calculate
                    cx = (i + 0.5) / fk[idx]
                    cy = (j + 0.5) / fk[idx]

                    default_boxes.append([cx, cy, width, height])

        dboxes = torch.tensor(default_boxes, dtype=torch.float32)

        # Convert Default Boxes to x_min, y_min, x_max, y_max
        dboxes_ltrb = dboxes.clone()
        dboxes_ltrb[:, 0] = dboxes[:, 0] - 0.5 * dboxes[:, 2]
        dboxes_ltrb[:, 1] = dboxes[:, 1] - 0.5 * dboxes[:, 3]
        dboxes_ltrb[:, 2] = dboxes[:, 0] + 0.5 * dboxes[:, 2]
        dboxes_ltrb[:, 3] = dboxes[:, 1] + 0.5 * dboxes[:, 3]

        dboxes_ltrb = dboxes_ltrb * figure_size

        return dboxes_ltrb

    def encode(self, boxes_and_labels, criteria: float = 0.5) -> torch.Tensor:

        bounding_boxes = boxes_and_labels[:, 0:4]
        labels_in = boxes_and_labels[:, 4]

        num_default_boxes = self.default_boxes.size(0)

        # Returns a matrix of ious for each bbox and each dbox
        ious = self.calculate_iou(bounding_boxes, self.default_boxes)

        # Get the values and indices of the best bbox ious for dboxes
        best_dbox_ious, best_dbox_idx = ious.max(0)

        # Get the values and inices of the best dbox ious for bboxes
        best_bbox_ious, best_bbox_idx = ious.max(1)

        # Set locations where both dbox and bbox ious were the best to 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        passes_criteria = best_dbox_ious > criteria

        labels_out = torch.zeros(num_default_boxes, dtype=torch.float32)
        labels_out[passes_criteria] = labels_in[best_dbox_idx[passes_criteria]]

        encoded_boxes = self.default_boxes.clone()
        encoded_boxes[passes_criteria, :] = bounding_boxes[
            best_dbox_idx[passes_criteria], :
        ]

        # Rejoin the encoded boxes and labels
        encoded_boxes_and_labels = torch.cat(
            (encoded_boxes, labels_out.unsqueeze(1)), 1
        )

        return encoded_boxes_and_labels

    def calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:

        N = box1.size(0)
        M = box2.size(0)

        left_top = torch.max(
            # [N,2] -> [N,1,2] -> [N,M,2]
            box1[:, :2].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),
        )

        right_bottom = torch.min(
            # [N,2] -> [N,1,2] -> [N,M,2]
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),
        )

        width_height = right_bottom - left_top  # [N,M,2]
        width_height[width_height < 0] = 0  # clip at 0

        # Area of the intersection of the two boxes
        intersection = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

        # Areas of each box
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]

        area1 = area1.unsqueeze(1).expand_as(intersection)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(intersection)  # [M,] -> [1,M] -> [N,M]

        # Union is the combined area of the two boxes without counting overlap twice
        union = area1 + area2 - intersection

        return intersection / union
