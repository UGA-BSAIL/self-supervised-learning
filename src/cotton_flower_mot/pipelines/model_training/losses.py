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

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Args:
            *args: Will be forwarded to superclass.
            **kwargs: Will be forwarded to superclass.

        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def _flatten(tensor: MaybeRagged) -> MaybeRagged:
        """
        Flattens a tensor, leaving the batch dimension intact.

        Args:
            tensor: The tensor or `RaggedTensor` to flatten.

        Returns:
            The same tensor, in 2D.

        """
        num_dims = len(tensor.shape)
        if num_dims <= 2:
            # We're already flat.
            return tensor

        if type(tensor) == tf.RaggedTensor:
            return tensor.merge_dims(1, num_dims - 1)
        else:
            batch_size = tf.shape(tensor)[0]
            flat_shape = tf.stack([batch_size, -1])
            return tf.reshape(tensor, flat_shape)

    def call(self, y_true: MaybeRagged, y_pred: MaybeRagged) -> tf.Tensor:
        y_true, y_pred = correct_ragged_mismatch(y_true, y_pred)
        y_true = self._flatten(y_true)
        y_pred = self._flatten(y_pred)

        # Calculate the fraction of samples that are positive.
        num_positive = tf.reduce_sum(y_true, axis=1)
        if type(y_true) == tf.RaggedTensor:
            num_total = y_true.row_lengths()
        else:
            num_total = tf.shape(y_true)[1]
        # Determine the weights.
        negative_weights = num_positive / tf.cast(num_total, tf.float32)
        negative_weights = tf.maximum(negative_weights, self._EPSILON)
        # Make sure it broadcasts correctly.
        negative_weights = tf.expand_dims(negative_weights, axis=-1)

        positive_samples = y_true * tf.math.log(y_pred + self._EPSILON)
        negative_samples = (1.0 - y_true) * tf.math.log(
            1.0 - y_pred + self._EPSILON
        )

        weighted_losses = (
            positive_samples + negative_weights * negative_samples
        )
        num_samples = tf.cast(tf.size(positive_samples), tf.float32)
        return -tf.reduce_sum(weighted_losses) / num_samples


def make_losses() -> Dict[str, tf.keras.losses.Loss]:
    """
    Creates the losses to use for the model.

    Returns:
        The losses that it created.

    """
    return {ModelTargets.SINKHORN.value: WeightedBinaryCrossEntropy()}
