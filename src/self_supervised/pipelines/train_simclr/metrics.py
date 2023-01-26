"""
Custom metrics for SimCLR.
"""


import torch
from torch import Tensor
from torchmetrics.classification import BinaryAccuracy

from .losses import compute_all_similarities


class ProxyClassAccuracy(BinaryAccuracy):
    """
    Computes the accuracy on the proxy classification task.
    """

    def forward(self, left_features: Tensor, right_features: Tensor) -> Tensor:
        """

        Args:
            left_features: Predictions from the left branch.
            right_features: Predictions from the right branch.

        Returns:
            The computed accuracy.

        """
        similarities = compute_all_similarities(left_features, right_features)

        # The positive pairs should be on the diagonals.
        num_examples, _ = similarities.shape
        targets = torch.eye(num_examples, device=similarities.device)

        return super().forward(similarities, targets)
