import torch

from utils.iou import intersection_over_union


def non_maximum_supression(confidences, localizations, iou_threshold, device):

    for idx in range(len(confidences)):

        boxes = localizations[idx]
        confs = confidences[idx]

        iou_grid = intersection_over_union(boxes, boxes)

        maximum_indices = []

        # Get higest confidence value for each box
        highest_confidence = confs.max(dim=1).values

        # Create a list of indices for each box
        confidence_indices = torch.arange(len(highest_confidence)).to(device)

        print("hi")

        while len(confidence_indices) > 0 and len(maximum_indices) < len(confidence_indices):

            # Select highest confidence value in list
            index_of_max = torch.max(highest_confidence[confidence_indices], 0).indices # Argmax in mps has a bug
            maximum_indices.append(index_of_max)

            # Remove max from remaining indices
            confidence_indices = confidence_indices[confidence_indices != index_of_max]

            # Get max iou values
            iou = iou_grid[index_of_max][confidence_indices]

            # Remove overlapping
            confidence_indices = confidence_indices[iou < iou_threshold]

        maximum_indices = list(set(maximum_indices))

        min_localizations = torch.ones(localizations.shape[1], dtype=bool)
        min_localizations[maximum_indices] = False

        localizations[idx][min_localizations] = 0

    return confidences, localizations