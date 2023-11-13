import numpy as np
import torch

from src.utils.nms import non_maximum_supression


class MeanAveragePrecision:
    def __init__(self, classes, device, iou_threshold, map_type="coco"):
        self.classes = classes
        self.device = device
        self.map_type = map_type
        self.iou_threshold = iou_threshold

    def __call__(self, confidences, localizations, ground_truths):
        if self.map_type == "pascal":
            return self.mean_average_precision(
                confidences, localizations, ground_truths, self.iou_threshold
            )

        return self.coco_mean_average_precision(
            confidences, localizations, ground_truths
        )

    def average_precision(
        self, confidences: torch.tensor, is_true_positive: torch.tensor
    ):
        ap = 0

        if len(confidences) == 0:
            return ap

        # Sort by confidence
        sorted_indices = torch.argsort(confidences)

        # Calculate the row-wise precision and recall
        tp_total = torch.sum(is_true_positive)

        tp_count = 0
        fp_count = 0

        precision = 0
        precisions = [1]
        recalls = [0]

        for is_tp in is_true_positive[sorted_indices]:
            tp_count += is_tp.long()
            fp_count += (~is_tp).long()

            precisions.append(tp_count / (tp_count + fp_count))
            recalls.append(tp_count / tp_total)

        for idx in range(len(precisions) - 1, 0, -1):
            precision = (
                precisions[idx].item()
                if precisions[idx].item() > precision
                else precision
            )

            ap += (recalls[idx] - recalls[idx - 1]) * precision

        return ap

    def mean_average_precision(
        self,
        confidences: torch.tensor,
        localizations: torch.tensor,
        ground_truths: torch.tensor,
        iou_threshold: float,
    ) -> float:
        average_precisions = []

        localizations = non_maximum_supression(
            confidences, localizations, iou_threshold, self.device
        )

        for confidence, localization, ground_truth in zip(
            confidences, localizations, ground_truths
        ):
            filtered_confidences = confidence[(localization.sum(dim=1) != 0)]
            filtered_ground_truths = ground_truth[(localization.sum(dim=1) != 0)]

            for class_id in range(len(self.classes)):
                positives = filtered_confidences.argmax(dim=1) == class_id

                positive_confidences = filtered_confidences[positives]
                ground_truths_of_positives = filtered_ground_truths[positives]

                is_true_positive = positive_confidences.argmax(
                    dim=1
                ) == ground_truths_of_positives.argmax(dim=1)

                ap = self.average_precision(
                    positive_confidences.argmax(dim=1),
                    is_true_positive,
                )

                average_precisions.append(ap)

        mAP = sum(average_precisions) / len(self.classes)

        return mAP

    def coco_mean_average_precision(
        self,
        confidences: torch.tensor,
        localizations: torch.tensor,
        ground_truths: torch.tensor,
    ) -> float:
        iou_thresholds = np.arange(0.05, 0.95, 0.05)

        coco_mAP = 0

        for threshold in iou_thresholds:
            coco_mAP += self.mean_average_precision(
                confidences, localizations, ground_truths, threshold
            )

        return coco_mAP / len(iou_thresholds)
