"""
Custom metrics for the model.
"""


from typing import Any, Dict, Optional

import tensorflow as tf

from ..schemas import ModelTargets
from .loss_metric_utilities import MaybeRagged, correct_ragged_mismatch


def id_switches(
    true_assignment: tf.RaggedTensor, pred_assignment: tf.RaggedTensor
) -> tf.Tensor:
    """
    Calculates the number of ID switches for a batch of image pairs.

    Args:
        true_assignment: The true hard assignment matrices, of shape
            `[batch_size, n_tracklets * n_detections]`.
        pred_assignment: The predicted hard assignment matrices, of shape
            `[batch_size, n_tracklets * n_detections]`.

    Returns:
        The number of ID switches for each item in the batch.

    """
    switched = tf.math.logical_xor(true_assignment, pred_assignment)
    switched = tf.cast(switched, tf.int32)
    # Division by 2 is because every time the assignment changes, it results
    # in one incorrect True value and one incorrect False value.
    return tf.reduce_sum(switched, axis=1) // 2


class IdSwitches(tf.keras.metrics.Metric):
    """
    A metric that tracks the number of ID switches.
    """

    def __init__(self, name="id_switches", **kwargs: Any):
        """
        Args:
            name: The name of this metric.
            **kwargs: Will be forwarded to the base class constructor.

        """
        super().__init__(name=name, **kwargs)

        self._num_id_switches = self.add_weight(
            name="num_switches", initializer="zeros", dtype=tf.int32
        )

    def update_state(
        self,
        y_true: MaybeRagged,
        y_pred: MaybeRagged,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> None:
        y_true, y_pred = correct_ragged_mismatch(y_true, y_pred)

        _id_switches = id_switches(y_true, y_pred)

        if sample_weight is not None:
            # Weight each sample.
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, _id_switches.shape)
            _id_switches *= sample_weight

        self._num_id_switches.assign_add(tf.reduce_sum(_id_switches))

    def result(self) -> tf.Tensor:
        return self._num_id_switches


def make_metrics() -> Dict[str, tf.keras.metrics.Metric]:
    """
    Creates the metrics to use for the model.

    Returns:
        The metrics that it created.

    """
    # return {ModelTargets.ASSIGNMENT.value: IdSwitches()}
    return {}
