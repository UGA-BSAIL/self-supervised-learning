import itertools
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def compute_all_similarities(
    left_features: Tensor, right_features: Tensor
) -> Tensor:
    """
    Computes all pairwise similarities between features in a batch.

    Args:
        left_features: The first set of features we are comparing. Should
            have shape `(batch, num_features)`.
        right_features: The second set of features we are comparing. Should
            be equivalent to `left_features`. For instance, maybe they are
            produced from the same input, just augmented in a different way.

    Returns:
        The pairwise similarities. Will be a matrix of shape `(batch, batch)`.

    """
    assert left_features.shape == right_features.shape
    assert (
        left_features.device == right_features.device
    ), "Features must be on same device."
    batch_size, _ = left_features.shape

    left_repeated = torch.repeat_interleave(left_features, batch_size, dim=0)
    right_tiled = torch.tile(right_features, (batch_size, 1))

    similarities = F.cosine_similarity(left_repeated, right_tiled, dim=1)
    # Put it back into the expected 2D shape.
    return torch.reshape(similarities, (batch_size, batch_size))


def compute_loss_all_similarities(similarities: Tensor) -> Tensor:
    """
    Computes the cross-entropy loss across all similarities.
    In on-diagonal cases, it assumes that the similarity should be high.
    In contrast, for off-diagonal cases, it assumes that it should be low.

    Args:
        similarities: The matrix of similarities for this batch. Should
            have shape `(2N, 2N)`. If we are using a temperature parameter, they
            should have already been scaled by that.

    Returns:
        The reduced loss for all the similarities.

    """
    num_examples, _ = similarities.shape

    def _per_example_losses(exp_similarities_: Tensor) -> Tensor:
        # Computes losses for each positive pair.
        denom_mask = 1.0 - torch.eye(
            num_examples, device=exp_similarities_.device
        )

        # Compute the denominators.
        denom_terms = denom_mask * exp_similarities_
        denominators = torch.sum(denom_terms, dim=1)

        # Compute the numerators. These will not be on the main diagonal,
        # (which is all ones), but instead the diagonal of either the
        # top right or bottom left quadrant, which represents the similarities
        # between corresponding left and right features.
        numerators = torch.diagonal(exp_similarities_, num_examples // 2)
        # Since the matrix is symmetric, we can just read one of these
        # diagonals and then tile it.
        numerators = numerators.tile(2)

        # Compute per-example losses.
        return -torch.log(numerators / denominators)

    exp_similarities = torch.exp(similarities)
    return torch.mean(_per_example_losses(exp_similarities))


class NtXentLoss(nn.Module):
    """
    Implements the NT-Xent loss from
    http://proceedings.mlr.press/v119/chen20j/chen20j.pdf
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: The temperature parameter to use for the loss.

        """
        __constants__ = ["temperature"]  # noqa: F841

        super().__init__()

        self.temperature = temperature

    def forward(self, left_features: Tensor, right_features: Tensor) -> Tensor:
        """
        Computes the loss.

        Args:
            left_features: The first set of features, with one set of
                augmentations.
            right_features: The features from the same images with different
                augmentations.

        Returns:
            The computed loss.

        """
        # Compute both similarities between right and left features, and
        # similarities between negative pairs on the same side.
        combined = torch.cat((left_features, right_features))
        similarities = compute_all_similarities(combined, combined)
        similarities /= self.temperature

        return compute_loss_all_similarities(similarities)


class FullGraphLoss(nn.Module):
    """
    Generalizes a pair-wise loss to a set by computing the loss over
    all possible pairs in the set.
    """

    def __init__(self, pair_loss: nn.Module):
        """
        Args:
            pair_loss: The pair-wise loss to wrap.

        """
        super().__init__()

        self.pair_loss = pair_loss

    def forward(self, feature_set: List[Tensor]) -> Tensor:
        """
        Computes the loss.

        Args:
            feature_set: The set of all the corresponding features.

        Returns:
            The computed loss.

        """
        total_loss = torch.scalar_tensor(0, device=feature_set[0].device)

        # Compute similarities between every pair.
        all_pairs = itertools.combinations(feature_set, 2)
        for left, right in all_pairs:
            total_loss += self.pair_loss(left, right)

        return total_loss
