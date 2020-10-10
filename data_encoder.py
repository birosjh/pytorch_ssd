
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
        fk = figure_size / np.array(feature_map_sizes)

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

                    default_boxes.append([cx, cy, cx + width, cy + height])

        return torch.tensor(default_boxes, dtype=torch.float)
