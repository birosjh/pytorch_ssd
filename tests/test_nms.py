import unittest
import yaml
import torch

from utils.nms import non_maximum_supression

class TestNonMaximumSupression(unittest.TestCase):

    def test_that_nms_deletes_one_overlapping(self):

        iou_threshold = 0.5

        confidences = torch.tensor([[
            [0.9, 0, 0, 0, 0],
            [0, 0.6, 0, 0, 0],
        ]])

        # boxes have iou of 0.532
        boxes = torch.tensor([[
            [10, 10, 70, 70],
            [20, 20, 80, 80],
        ]])

        output = non_maximum_supression(
            confidences,
            boxes,
            iou_threshold,
            "cpu"
        )

        answer = torch.tensor([[
            [10, 10, 70, 70],
            [0, 0, 0, 0],
        ]])

        self.assertTrue(torch.allclose(output, answer))

    def test_that_nms_deletes_multiple_overlapping(self):

        iou_threshold = 0.5

        confidences = torch.tensor([[
            [0.9, 0, 0, 0, 0],
            [0.89, 0, 0, 0, 0],
            [0, 0.6, 0, 0, 0],
        ]])

        # boxes have iou of 0.532
        boxes = torch.tensor([[
            [10, 10, 70, 70],
            [11, 11, 70, 70],
            [20, 20, 80, 80],
        ]])

        output = non_maximum_supression(
            confidences,
            boxes,
            iou_threshold,
            "cpu"
        )

        answer = torch.tensor([[
            [10, 10, 70, 70],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]])

        self.assertTrue(torch.allclose(output, answer))

    def test_that_nms_leaves_overlapping_below_thresh(self):

        iou_threshold = 0.5

        confidences = torch.tensor([[
            [0.9, 0, 0, 0, 0],
            [0, 0.6, 0, 0, 0],
            [0, 0, 0.7, 0, 0],
        ]])

        # boxes have iou of 0.532
        boxes = torch.tensor([[
            [10, 10, 70, 70],
            [20, 20, 80, 80],
            [60, 60, 120, 120],
        ]])

        output = non_maximum_supression(
            confidences,
            boxes,
            iou_threshold,
            "cpu"
        )

        answer = torch.tensor([[
            [10, 10, 70, 70],
            [0, 0, 0, 0],
            [60, 60, 120, 120],
        ]])

        self.assertTrue(torch.allclose(output, answer))

    def test_nms_leaves_non_overlapping(self):

        iou_threshold = 0.5

        confidences = torch.tensor([[
            [0.9, 0, 0, 0, 0],
            [0, 0.6, 0, 0, 0],
            [0, 0, 0.7, 0, 0],
        ]])

        # boxes have iou of 0.532
        boxes = torch.tensor([[
            [10, 10, 70, 70],
            [20, 20, 80, 80],
            [90, 90, 120, 120],
        ]])

        output = non_maximum_supression(
            confidences,
            boxes,
            iou_threshold,
            "cpu"
        )

        answer = torch.tensor([[
            [10, 10, 70, 70],
            [0, 0, 0, 0],
            [90, 90, 120, 120],
        ]])

        self.assertTrue(torch.allclose(output, answer))

    def test_that_nms_deletes_exact_overlap(self):

        iou_threshold = 0.5

        confidences = torch.tensor([[
            [0.9, 0, 0, 0, 0],
            [0.9, 0, 0, 0, 0],
        ]])

        # boxes have iou of 0.532
        boxes = torch.tensor([[
            [10, 10, 70, 70],
            [10, 10, 70, 70],
        ]])

        output = non_maximum_supression(
            confidences,
            boxes,
            iou_threshold,
            "cpu"
        )

        answer = torch.tensor([[
            [10, 10, 70, 70],
            [0, 0, 0, 0],
        ]])

        self.assertTrue(torch.allclose(output, answer))