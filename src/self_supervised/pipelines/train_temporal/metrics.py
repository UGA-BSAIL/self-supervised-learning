"""
Custom metrics for temporal order verification.
"""


from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, linalg
from torchmetrics.classification import BinaryAccuracy


class ContrastiveAccuracy(BinaryAccuracy):
    """
    Computes the accuracy for a contrastive learning problem where
    we have anchor, positive, and negative examples.
    """

    def __init__(self, **kwargs: Any):
        """
        Args:
            **kwargs: Will be forwarded to the superclass.

        """

        # We always have two classes.
        super().__init__(num_classes=2, **kwargs)

    def forward(
        self, anchor_reps: Tensor, positive_reps: Tensor, negative_reps: Tensor
    ) -> Tensor:
        """
        Computes the accuracy.

        Args:
            anchor_reps: The representations of the anchor frames.
            positive_reps: The representations of the positive frames.
            negative_reps: The representation of the negative frames.

        Returns:
            The computed accuracy.

        """
        # Compute the distance between the representations.
        positive_distance = linalg.vector_norm(
            positive_reps - anchor_reps, dim=1, keepdim=True
        )
        negative_distance = linalg.vector_norm(
            negative_reps - anchor_reps, dim=1, keepdim=True
        )

        # Treat these distances as predictions in a classification problem.
        predictions = torch.cat([positive_distance, negative_distance], 1)
        predictions = F.softmax(predictions, dim=1)[:, 1]
        print(predictions)

        # We always want to predict the negative sample as being farther away.
        batch_size = positive_distance.shape[0]
        targets = torch.ones(
            batch_size, dtype=torch.int, device=predictions.device
        )

        return super().forward(predictions, targets)
