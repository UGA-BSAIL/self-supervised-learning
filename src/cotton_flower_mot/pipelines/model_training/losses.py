"""
Defines custom losses.
"""


from typing import Any, Dict

import tensorflow as tf

from ..schemas import ModelTargets
from .loss_metric_utilities import MaybeRagged, correct_ragged_mismatch


class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Implementation of binary cross-entropy loss that allows us to weight
    positive and negative samples differently.
    """

    _EPSILON = tf.constant(0.0001)
    """
    Small constant value to avoid log(0).
    """

    def __init__(
        self,
        *args: Any,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        **kwargs: Any
    ):
        """
        Args:
            *args: Will be forwarded to superclass.
            positive_weight: Weight for positive samples.
            negative_weight: Weight for negative samples.
            **kwargs: Will be forwarded to superclass.

        """
        super().__init__(*args, **kwargs)

        # We use only single underscores to avoid AutoGraph issues.
        self._positive_weight = tf.constant(positive_weight, dtype=tf.float32)
        self._negative_weight = tf.constant(negative_weight, dtype=tf.float32)

    def call(self, y_true: MaybeRagged, y_pred: MaybeRagged) -> tf.Tensor:
        y_true, y_pred = correct_ragged_mismatch(y_true, y_pred)

        positive_samples = y_true * tf.math.log(y_pred + self._EPSILON)
        negative_samples = (1.0 - y_true) * tf.math.log(
            1.0 - y_pred + self._EPSILON
        )

        weighted_losses = (
            self._positive_weight * positive_samples
            + self._negative_weight * negative_samples
        )
        num_samples = tf.cast(tf.size(positive_samples), tf.float32)
        return -tf.reduce_sum(weighted_losses) / num_samples


def make_losses(
    positive_sample_weight: float = 1.0,
) -> Dict[str, tf.keras.losses.Loss]:
    """
    Creates the losses to use for the model.

    Args:
        positive_sample_weight: How much to weight positive samples in the loss.

    Returns:
        The losses that it created.

    """
    return {
        ModelTargets.SINKHORN.value: WeightedBinaryCrossEntropy(
            positive_weight=positive_sample_weight
        )
    }
