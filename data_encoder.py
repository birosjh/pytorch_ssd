"""
data_encoder.py
---------------------------------------------

Contains functions for encoding bounding boxes in terms of default boxes.

"""
from math import sqrt
from itertools import product
import numpy as np
import torch


class DataEncoder:
    def __init__(self, default_box_config):
        
        self.default_boxes = self.create_default_boxes(default_box_config)

    def create_default_boxes(self, default_box_config: dict) -> list:

        figure_size = default_box_config['figure_size']
        feature_map_sizes = default_box_config['feature_map_sizes']
        steps = default_box_config['steps']
        scales = default_box_config['scales']
        aspect_ratios = default_box_config['aspect_ratios']

        default_boxes = []

        # The size of the k-th square feature map
        fk = figure_size / np.array(steps)

        for idx, size in enumerate(feature_map_sizes):
            
            # Scale for Aspect Ratio 1
            sk = scales[idx]/figure_size
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

        dboxes = torch.tensor(default_boxes, dtype=torch.float)

        # Convert Default Boxes to x_min, y_min, x_max, y_max
        dboxes_ltrb = dboxes.clone()
        dboxes_ltrb[:, 0] = dboxes[:, 0] - 0.5 * dboxes[:, 2]
        dboxes_ltrb[:, 1] = dboxes[:, 1] - 0.5 * dboxes[:, 3]
        dboxes_ltrb[:, 2] = dboxes[:, 0] + 0.5 * dboxes[:, 2]
        dboxes_ltrb[:, 3] = dboxes[:, 1] + 0.5 * dboxes[:, 3]

        return dboxes_ltrb

    def encode(self, bounding_boxes):

        encoded_boxes = []

        default_boxes = self.default_boxes
        num_default_boxes = default_boxes.size(0)

        ious = self.calculate_iou(bounding_boxes, default_boxes)

        iou, max_idx = ious.max(0)  # [1,8732]
        max_idx.squeeze_(0)        # [8732,]
        iou.squeeze_(0)            # [8732,]



        return encoded_boxes

    def calculate_iou(self, box1, box2):

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
        area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
        area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]

        area1 = area1.unsqueeze(1).expand_as(
            intersection
        )  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(
            intersection
        )  # [M,] -> [1,M] -> [N,M]

        # Union is the combined area of the two boxes without counting overlap twice
        union = area1 + area2 - intersection

        return intersection / union
