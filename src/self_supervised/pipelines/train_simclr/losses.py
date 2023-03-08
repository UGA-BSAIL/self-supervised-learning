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


def compute_loss_all_similarities(
    similarities: Tensor, num_views: int = 2
) -> Tensor:
    """
    Computes the cross-entropy loss across all similarities.
    In on-diagonal cases, it assumes that the similarity should be high.
    In contrast, for off-diagonal cases, it assumes that it should be low.

    Args:
        similarities: The matrix of similarities for this batch. Should
            have shape `(2N, 2N)` (if `num_views` == 2). If we are using a
            temperature parameter, they should have already been scaled by that.
        num_views: The total number of views present in this similarity
            matrix. Setting this >2 allows us to generalize to more than two
            views. In this case, the input should have a shape that is a
            multiple of `num_views`.

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
        # (which is all ones). The diagonals that we use for these depend on
        # the number of views, and indicate positions with corresponding
        # elements from two different views.
        numerators = torch.tensor(
            [], device=exp_similarities_.device, dtype=exp_similarities_.dtype
        )
        ordered_denominators = torch.tensor(numerators)
        diagonal_step = num_examples // num_views
        for diagonal_index in range(
            diagonal_step, num_examples, diagonal_step
        ):
            diagonal = torch.diagonal(exp_similarities_, diagonal_index)
            diagonal_length = diagonal.shape[0]
            # The matrix is symmetric, so add the numerator twice to account
            # for both sides.
            numerators = torch.cat((numerators, diagonal, diagonal))
            # Set the corresponding denominators as well, for both sides.
            ordered_denominators = torch.cat(
                (
                    ordered_denominators,
                    denominators[0:diagonal_length],
                    denominators[(num_examples - diagonal_length) :],
                )
            )

        # Compute per-example losses.
        return -torch.log(numerators / ordered_denominators)

    exp_similarities = torch.exp(similarities)
    return torch.mean(_per_example_losses(exp_similarities))


class NtXentLoss(nn.Module):
    """
    Implements the NT-Xent loss from
    http://proceedings.mlr.press/v119/chen20j/chen20j.pdf

    Note that I have extended this to support arbitrary numbers of
    corresponding features.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: The temperature parameter to use for the loss.

        """
        __constants__ = ["temperature"]  # noqa: F841

        super().__init__()

        self.temperature = temperature

    def forward(self, *feature_set: Tensor) -> Tensor:
        """
        Computes the loss.

        Args:
            feature_set: The set of all the corresponding features.

        Returns:
            The computed loss.

        """
        # Compute both similarities between right and left features, and
        # similarities between negative pairs on the same side.
        combined = torch.cat(feature_set)
        similarities = compute_all_similarities(combined, combined)
        similarities /= self.temperature

        return compute_loss_all_similarities(
            similarities, num_views=len(feature_set)
        )
