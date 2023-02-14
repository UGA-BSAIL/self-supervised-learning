"""
Custom losses for temporal order verification.
"""


from typing import Any

import torch
from torch import Tensor, linalg, nn


class RegularizedTripletLoss(nn.Module):
    """
    Margin triplet loss that also has a regularization term to keep
    representations small.
    """

    def __init__(
        self, margin: float = 1.0, regularization: float = 1.0, **kwargs: Any
    ):
        """
        Args:
            margin: The margin to use.
            regularization: The regularization coefficient to use.
            **kwargs: Will be forwarded to `TripletMarginLoss`.

        """
        __constants__ = ["regularization"]  # noqa: F841

        super().__init__()

        self.regularization = regularization
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, **kwargs)

    def forward(
        self, anchor: Tensor, positive: Tensor, negative: Tensor
    ) -> Tensor:
        """
        Args:
            anchor: The anchor representation vectors.
            positive: The positive representation vectors.
            negative: The negative representation vectors.

        Returns:
            The computed triplet loss.

        """
        triplet_loss = self.triplet_loss(anchor, positive, negative)

        # Compute regularization.
        norms = [
            linalg.vector_norm(r, dim=1) for r in (anchor, positive, negative)
        ]
        total_norm = torch.sum(sum(norms))

        return triplet_loss + self.regularization * total_norm
