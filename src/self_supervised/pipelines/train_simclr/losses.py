from torch import Tensor
import torch
from torch import nn
import torch.nn.functional as F
import itertools


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
        The pairwise similarities. Will be a symmetric matrix of shape
        `(batch, batch)`.

    """
    assert left_features.shape == right_features.shape
    assert (
        left_features.device == right_features.device
    ), "Features must be on same device."
    batch_size, num_features = left_features.shape
    similarities = torch.zeros(
        size=(batch_size,) * 2, device=left_features.device
    )

    # Fill in the similarities for the upper triangle.
    for left_i, right_i in itertools.combinations(range(batch_size), 2):
        similarities[left_i, right_i] = nn.functional.cosine_similarity(
            left_features[left_i], right_features[right_i], dim=0
        )
    # `combinations` does not produce repeated elements, so we need to fill the
    # diagonal separately.
    for i in range(batch_size):
        similarities[i, i] = nn.functional.cosine_similarity(
            left_features[i], right_features[i], dim=0
        )

    # Copy the upper triangle into the lower triangle.
    return similarities + similarities.triu(diagonal=1).T


def compute_loss_all_similarities(similarities: Tensor) -> Tensor:
    """
    Computes the cross-entropy loss across all similarities.
    In on-diagonal cases, it assumes that the similarity should be high.
    In contrast, for off-diagonal cases, it assumes that it should be low.

    Args:
        similarities: The matrix of similarities for this batch. Should
            have shape `(N, N)`. If we are using a temperature parameter, they
            should have already been scaled by that.

    Returns:
        The reduced loss for all the similarities.

    """
    # Compute the target distributions. These are essentially just one-hot
    # vectors that are 1 where i==j.
    num_examples, _ = similarities.shape
    targets = torch.eye(num_examples, device=similarities.device)

    return F.cross_entropy(similarities, targets)


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
        __constants__ = ["temperature"]

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
        similarities = compute_all_similarities(left_features, right_features)
        similarities /= self.temperature

        return compute_loss_all_similarities(similarities)
