"""
Defines custom losses.
"""


from typing import Any, Dict, Tuple, Union

import tensorflow as tf

from ..schemas import ModelTargets

MaybeRagged = Union[tf.Tensor, tf.RaggedTensor]
"""
Represents something that might be a RaggedTensor.
"""


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

    @staticmethod
    def _correct_ragged_mismatch(
        y_true: MaybeRagged, y_pred: MaybeRagged
    ) -> Tuple[MaybeRagged, MaybeRagged]:
        """
        Workaround for a bug in TF 2.4 where we can't input `y_true` as a
        ragged tensor. However, `y_pred` can still be ragged, so in this
        situation, we convert `y_true` to a ragged tensor based on the row
        lengths from `y_pred`.

        Args:
            y_true: The ground-truth labels.
            y_pred: The predicted labels.

        Returns:
            The same true and predicted labels. In the case where `y_pred` is
            ragged and `y_true` is not, `y_true` will be made ragged. Otherwise,
            both will be returned unchanged.


        """
        if isinstance(y_pred, tf.RaggedTensor) and isinstance(
            y_true, tf.Tensor
        ):
            y_true = tf.RaggedTensor.from_tensor(
                y_true, lengths=y_pred.row_lengths()
            )

        return y_true, y_pred

    def call(self, y_true: MaybeRagged, y_pred: MaybeRagged) -> tf.Tensor:
        y_true, y_pred = self._correct_ragged_mismatch(y_true, y_pred)

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
