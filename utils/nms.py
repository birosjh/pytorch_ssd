import torch

from utils.iou import intersection_over_union

def non_maximum_supression(confidences, boxes, threshold, device):

    iou_grid = intersection_over_union(boxes, boxes)

    maximum_indices = []

    # Get higest confidence value for each box
    highest_confidence = confidences.max(dim=1).values

    # Create a list of indices for each box
    confidence_indices = torch.arange(len(highest_confidence)).to(device)

    while len(confidence_indices) > 0:

        # Select highest confidence value in list
        index_of_max = highest_confidence[confidence_indices].argmax()
        maximum_indices.append(index_of_max)

        # Remove max from remaining indices
        confidence_indices = confidence_indices[confidence_indices != index_of_max]

        # Get max iou values
        iou = iou_grid[index_of_max][confidence_indices]

        # Remove overlapping
        confidence_indices = confidence_indices[iou < threshold]

    
    return (confidences[maximum_indices], boxes[maximum_indices])
