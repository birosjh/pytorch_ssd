"""
data_encoder.py
---------------------------------------------

Contains functions for encoding bounding boxes in terms of default boxes.

"""
from itertools import product
from math import sqrt

import numpy as np
import torch

from torchvision.ops import box_iou


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

    def encode(self, ground_truth, criteria) -> torch.Tensor:
        bounding_boxes = ground_truth[:, 0:4]
        labels_in = ground_truth[:, -1]

        num_default_boxes = self.default_boxes.size(0)

        # Returns a matrix of ious for each bbox and each dbox
        ious = box_iou(bounding_boxes, self.default_boxes)

        # Get the values and indices of the best bbox ious for dboxes
        best_dbox_ious, best_dbox_idx = ious.max(0)

        # Get the values and inices of the best dbox ious for bboxes
        best_bbox_ious, best_bbox_idx = ious.max(1)

        # Set locations where both dbox and bbox ious were the best to 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx] = idx

        # filter IoU > 0.5
        passes_criteria = best_dbox_ious > criteria

        labels_out = torch.zeros(num_default_boxes, dtype=torch.float32)
        labels_out[passes_criteria] = labels_in[best_dbox_idx[passes_criteria]]

        encoded_boxes = self.default_boxes.clone()
        encoded_boxes[passes_criteria, :] = bounding_boxes[
            best_dbox_idx[passes_criteria], :
        ]

        # encoded_boxes = self.normalize(encoded_boxes)

        encoded_ground_truth = torch.cat(
            [encoded_boxes, labels_out.unsqueeze(dim=1)], dim=1
        )

        return encoded_ground_truth

    def normalize(self, boxes, device=None):
        normalized_boxes = boxes.clone()

        default_boxes = self.default_boxes.clone()

        if device is not None:
            default_boxes = default_boxes.to(device)

        normalized_boxes[:, 0] -= default_boxes[:, 0]
        normalized_boxes[:, 1] -= default_boxes[:, 1]
        normalized_boxes[:, 2] -= default_boxes[:, 2]
        normalized_boxes[:, 3] -= default_boxes[:, 3]

        normalized_boxes[:, 0] /= default_boxes[:, 2] - default_boxes[:, 0]
        normalized_boxes[:, 1] /= default_boxes[:, 3] - default_boxes[:, 1]
        normalized_boxes[:, 2] /= default_boxes[:, 2] - default_boxes[:, 0]
        normalized_boxes[:, 3] /= default_boxes[:, 3] - default_boxes[:, 1]

        return normalized_boxes

    def denormalize(self, boxes, device=None):
        denormalized_boxes = boxes.clone()

        default_boxes = self.default_boxes.clone()

        if device is not None:
            default_boxes = default_boxes.to(device)

        denormalized_boxes[:, 0] *= default_boxes[:, 2] - default_boxes[:, 0]
        denormalized_boxes[:, 1] *= default_boxes[:, 3] - default_boxes[:, 1]
        denormalized_boxes[:, 2] *= default_boxes[:, 2] - default_boxes[:, 0]
        denormalized_boxes[:, 3] *= default_boxes[:, 3] - default_boxes[:, 1]

        denormalized_boxes[:, 0] += default_boxes[:, 0]
        denormalized_boxes[:, 1] += default_boxes[:, 1]
        denormalized_boxes[:, 2] += default_boxes[:, 2]
        denormalized_boxes[:, 3] += default_boxes[:, 3]

        return boxes
