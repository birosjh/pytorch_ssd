import torch

from utils.iou import intersection_over_union

def non_maximum_supression(confidences, boxes, threshold):

    iou = intersection_over_union(boxes, boxes)

    maximum_indices = []

    confidence_indices = torch.arange(len(confidences))

    while len(confidence_indices) > 0:

        index_of_max = confidences[confidence_indices].argmax(dim=0)
        maximum_indices.append(index_of_max)

        confidence_indices = confidence_indices[confidence_indices != index_of_max]

        # Remove overlapping
        confidence_indices = confidence_indices[iou < threshold]
        iou = iou[iou < threshold]

    
    return boxes[maximum_indices]
