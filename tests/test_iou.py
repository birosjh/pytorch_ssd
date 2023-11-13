import unittest

import torch

from utils.iou import intersection_over_union

torch.set_printoptions(precision=10)


class TestIou(unittest.TestCase):
    def test_single_iou_output_is_correct(self):

        box1 = torch.tensor([[200.0, 300.0, 350.0, 400.0]])
        box2 = torch.tensor([[250.0, 250.0, 400.0, 350.0]])

        ious = intersection_over_union(box1, box2)

        # Compares each value, then gives a single boolean for whether they all match or not # noqa: E501
        equivalence_check = torch.all(torch.isclose(ious, torch.tensor([0.2])))

        self.assertTrue(equivalence_check.numpy())

    def test_multiple_iou_output_is_correct(self):

        box1 = torch.tensor(
            [[200.0, 300.0, 350.0, 400.0], [250.0, 250.0, 400.0, 350.0]]
        )
        box2 = torch.tensor(
            [[250.0, 250.0, 400.0, 350.0], [350.0, 300.0, 550.0, 400.0]]
        )

        ious = intersection_over_union(box1, box2)

        # Compares each value, then gives a single boolean for whether they all match or not # noqa: E501
        # One box exists in both box1 and box2 so its value is 1
        # One set of boxes don't overlap at all so they come out to zero
        equivalence_check = torch.all(
            torch.isclose(
                ious,
                torch.tensor(
                    [
                        [0.2000, 0.0000],  # box1 row 1 ious for box2
                        [1.0000, 0.076923],  # box1 row 2 ious for box 2
                    ]
                ),
            )
        )

        self.assertTrue(equivalence_check.numpy())

    def test_sandbox(self):

        boxes = torch.tensor(
            [
                [20.0, 30.0, 35.0, 40.0],
                [25.0, 25.0, 40.0, 35.0],
                [25.0, 25.0, 45.0, 35.0],
                [100.0, 100.0, 120.0, 120.0],
                [35.0, 30.0, 55.0, 40.0],
            ]
        )

        ious = intersection_over_union(boxes, boxes)

        print(ious)

        print(ious.max(dim=0))
        print(ious.sum(dim=0))

        print(~torch.isclose(ious.sum(dim=0), ious.max(dim=0).values))
        print(
            (~torch.isclose(ious.sum(dim=0), ious.max(dim=0).values)).nonzero(
                as_tuple=True
            )[0]
        )
