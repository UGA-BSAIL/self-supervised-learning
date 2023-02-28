"""
Custom metrics for SimCLR.
"""


import itertools
from typing import List

import torch
from torch import Tensor
from torchmetrics.classification import BinaryAccuracy

from .losses import compute_all_similarities


class ProxyClassAccuracy(BinaryAccuracy):
    """
    Computes the accuracy on the proxy classification task.
    """

    def __pair_accuracy(
        self, left_features: Tensor, right_features: Tensor
    ) -> Tensor:
        """
        Computes the accuracy for a single pair of features.

        Args:
            left_features: Predictions from the left branch.
            right_features: Predictions from the right branch.

        Returns:
            The computed accuracy.

        """
        similarities = compute_all_similarities(left_features, right_features)
        # Treat the similarities as probabilities.
        predicted_classes = similarities >= self.threshold
        predicted_classes = predicted_classes.to(torch.int)

        # The positive pairs should be on the diagonals.
        num_examples, _ = similarities.shape
        targets = torch.eye(
            num_examples, device=similarities.device, dtype=torch.int
        )

        return super().forward(predicted_classes, targets)

    def forward(self, feature_set: List[Tensor]) -> Tensor:
        """
        Args:
            feature_set: The set of all the corresponding features.

        Returns:
            The computed accuracy.

        """
        total_acc = torch.scalar_tensor(0, device=feature_set[0].device)

        # Compute the accuracy for every pair.
        all_pairs = itertools.combinations(feature_set, 2)
        i = 0
        for i, (left, right) in enumerate(all_pairs):
            total_acc += self.__pair_accuracy(left, right)

        return total_acc / (i + 1)
