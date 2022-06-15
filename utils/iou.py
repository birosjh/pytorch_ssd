import torch


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:

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
