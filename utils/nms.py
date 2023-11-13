import torch

from utils.iou import intersection_over_union
from utils.profiler import profiler


@profiler("loss")
def non_maximum_supression(confidences, localizations, iou_threshold, device):
    for idx in range(len(confidences)):
        boxes = localizations[idx]
        confs = confidences[idx]

        iou_grid = intersection_over_union(boxes, boxes)

        is_overlapping = ~torch.isclose(
            iou_grid.sum(dim=0), iou_grid.max(dim=0).values
        ).to(device)

        maximum_indices = []

        # Get higest confidence value for each box
        highest_confidence = confs.max(dim=1).values.to(device)

        # Create a list of indices for each box
        confidence_indices = torch.arange(len(highest_confidence)).to(device)
        overlapping_indices = confidence_indices[is_overlapping]
        non_overlapping_indices = confidence_indices[~is_overlapping]

        while len(overlapping_indices) > 0:
            # Select highest confidence value in list
            index_of_max = torch.max(
                highest_confidence[overlapping_indices], 0
            ).indices  # Argmax in mps has a bug
            maximum_indices.append(overlapping_indices[index_of_max])

            # Remove max from remaining indices
            overlapping_indices = overlapping_indices[
                overlapping_indices != overlapping_indices[index_of_max]
            ]

            # Get max iou values
            iou = iou_grid[index_of_max][overlapping_indices]

            # Remove overlapping
            overlapping_indices = overlapping_indices[iou < iou_threshold]

        maximum_indices = torch.tensor(maximum_indices).to(device)
        maximum_indices = torch.cat([maximum_indices, non_overlapping_indices])

        min_localizations = torch.ones(localizations.shape[1], dtype=bool)
        min_localizations[maximum_indices] = False

        localizations[idx][min_localizations] = 0

    return localizations
