from functools import reduce
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class HeatMapFocalLoss(nn.Module):
    """
    Implements a penalty-reduced pixel-wise logistic regression with focal loss,
    as described in https://arxiv.org/pdf/1904.07850.pdf
    """

    _EPSILON = torch.as_tensor(0.0001)
    """
    Small constant value to avoid log(0).
    """

    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        positive_loss_weight: float = 1.0,
    ):
        """
        Args:
            alpha: Alpha parameter for the focal loss.
            beta: Beta parameter for the focal loss.
            positive_loss_weight: Additional weight to give the positive
                component of the loss. This is to help balance the
                preponderance of negative samples.
        """
        __constants__ = ["alpha", "beta", "positive_loss_weight"]  # noqa: F841

        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.positive_loss_weight = positive_loss_weight

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        # Loss we use at "positive" locations.
        positive_loss = torch.pow(1.0 - y_pred, self.alpha) * torch.log(
            torch.maximum(y_pred, self._EPSILON)
        )
        positive_loss *= self.positive_loss_weight
        # Loss we use at "negative" locations.
        negative_loss = (
            torch.pow(1.0 - y_true, self.beta)
            * torch.pow(y_pred, self.alpha)
            * torch.log(torch.maximum(1.0 - y_pred, self._EPSILON))
        )

        # Figure out which locations are positive and which are negative.
        positive_mask = y_true >= 1.0
        pixel_wise_loss = torch.where(
            positive_mask, positive_loss, negative_loss
        )

        mean_loss = -pixel_wise_loss.sum()
        # Normalize by the number of keypoints.
        num_points = torch.count_nonzero(positive_mask)

        if num_points.item() > 0:
            return mean_loss / num_points.to(torch.float)
        else:
            # Avoid division by zero.
            return mean_loss


def sparse_l1_loss(
    *,
    predictions: Tensor,
    targets: List[Tensor],
    target_points: List[Tensor],
) -> Tensor:
    """
    A custom sparse L1 loss specifically designed to work on the size
    or offset outputs from the CenterNet. These outputs are unique because
    the predictions come in the form of a dense feature map, while the
    ground-truth comes in the form of a sparse vector.

    Args:
        predictions: The dense predictions for the sizes or offsets.
            Should have shape `[batch, num_channels, height, width]`.
        targets: The target values for the sizes or offsets. For each
            input images, a tensor of shape `[num_points, num_channels]`
            should be provided.
        target_points: The ground-truth object center point locations.
            Should have the same shape as `targets`. The two columns should
            be the x and y coordinates.

    Returns:
        The L1 loss at the target points.

    """
    points_per_example = [p.shape[0] for p in target_points]
    # We're going to grab all the target points in one go, so we can
    # concatenate them into a giant list.
    targets = torch.cat(targets, dim=0)
    target_points = torch.cat(target_points, dim=0)

    # Compute the indices in the dense predictions that we're interested in.
    points_index = target_points * torch.as_tensor(
        predictions.shape[2:][::-1], device=target_points.device
    )
    points_index = points_index.to(torch.long)

    # Compute the indices for the examples in the batch.
    batch_size = predictions.shape[0]
    batch_index = [[i] * points_per_example[i] for i in range(batch_size)]
    batch_index = reduce(lambda x, y: x + y, batch_index, [])
    batch_index = torch.as_tensor(batch_index, dtype=torch.long)

    # Extract the relevant points in one go.
    predictions_at_points = predictions[
        batch_index, :, points_index[:, 0], points_index[:, 1]
    ]

    # Compute the loss at those points.
    return F.l1_loss(predictions_at_points, targets)


class SparseL1Loss(nn.Module):
    """
    Custom L1 loss designed to work on the size and offset outputs from
    the CenterNet detection model.
    """

    def __init__(self, *, size_weight: float, offset_weight: float):
        """
        Args:
            size_weight: The weight to use for the size loss.
            offset_weight: The weight to use for the offset loss.

        """
        __constants__ = ["size_weight", "offset_weight"]  # noqa: F841

        super().__init__()

        self.size_weight = size_weight
        self.offset_weight = offset_weight

    def forward(
        self,
        *,
        predicted_sizes: Tensor,
        predicted_offsets: Tensor,
        target_boxes: List[Tensor],
    ) -> Tensor:
        """
        Args:
            predicted_sizes: The dense size output from the model. Should have
                the shape `[batch, 2, h, w]`.
            predicted_offsets: The predicted offsets from the model. Should have
                the shape `[batch, 2, x, y]`.
            target_boxes: The target bounding boxes. Each tensor contains the
                boxes for one example in the batch, and should have the shape
                `[num_boxes, 6]`. The six columns should be
                `[center_x, center_y, width, height, offset_x, offset_y]`.

        Returns:
            The total L1 loss.

        """
        # Extract each of the individual target attributes.
        center_points = [b[:, :2] for b in target_boxes]
        sizes = [b[:, 2:4] for b in target_boxes]
        offsets = [b[:, 4:] for b in target_boxes]

        # Compute the losses.
        size_loss = sparse_l1_loss(
            predictions=predicted_sizes,
            targets=sizes,
            target_points=center_points,
        )
        offset_loss = sparse_l1_loss(
            predictions=predicted_offsets,
            targets=offsets,
            target_points=center_points,
        )

        return self.size_weight * size_loss + self.offset_weight * offset_loss
