import torch


def intersection_over_union(
    first_boxes: torch.Tensor, second_boxes: torch.Tensor
) -> torch.Tensor:
    first_area = (first_boxes[:, 2] - first_boxes[:, 0]) * (
        first_boxes[:, 3] - first_boxes[:, 1]
    )

    second_area = (second_boxes[:, 2] - second_boxes[:, 0]) * (
        second_boxes[:, 3] - second_boxes[:, 1]
    )

    # Create a grid of max and min for each box dimension
    left = torch.max(first_boxes[:, 0].unsqueeze(dim=1), second_boxes[:, 0])
    top = torch.max(first_boxes[:, 1].unsqueeze(dim=1), second_boxes[:, 1])
    right = torch.min(first_boxes[:, 2].unsqueeze(dim=1), second_boxes[:, 2])
    bottom = torch.min(first_boxes[:, 3].unsqueeze(dim=1), second_boxes[:, 3])

    # Subtract right to left to get x_diff
    x_diff = right - left
    x_diff[x_diff < 0] = 0

    # Subtract bottom to top to get y_diff
    y_diff = bottom - top
    y_diff[y_diff < 0] = 0

    intersecting_area = x_diff * y_diff

    # Union is the combined area of the two boxes without counting overlap twice
    union = first_area.unsqueeze(dim=1) + second_area - intersecting_area

    return intersecting_area / union
