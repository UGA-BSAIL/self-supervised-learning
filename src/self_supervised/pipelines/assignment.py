"""
Implementation of the Sinkhorn-Kopp algorithm in TensorFlow.
"""

from functools import partial
from typing import Optional, Tuple, Union

import tensorflow as tf
from scipy import optimize

_EPSILON = tf.constant(0.0001)
"""
Small value to use to avoid dividing by zero.
"""


def _maybe_float_to_tensor(maybe_float: Union[float, tf.Tensor]) -> tf.Tensor:
    """
    If a value is specified as a float, this converts it to a float tensor.
    If it is already a tensor, it does nothing.

    Args:
        maybe_float: The value that might be a float or tensor.

    Returns:
        The same value as a tensor.

    """
    if not tf.is_tensor(maybe_float):
        return tf.convert_to_tensor(maybe_float, dtype=tf.float32)
    else:
        return maybe_float


def solve_optimal_transport(
    cost: tf.Tensor,
    *,
    row_sums: Optional[tf.Tensor] = None,
    column_sums: Optional[tf.Tensor] = None,
    lamb: Union[tf.Tensor, float],
    epsilon: Union[tf.Tensor, float] = tf.constant(0.01),
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Solves an optimal-transport problem using the Sinkhorn algorithm.

    Args:
        cost: The cost matrix. Should have the shape `[batch_size, n, m]`.
        row_sums: Values we want each row in the optimal transport matrix to
            sum to. Should have shape `[batch_size, n]`. If not specified, it
            will default to all ones.
        column_sums: Values we want each column in the optimal transport matrix
            to sum to. Should have shape `[batch_size, m]`. If not specified,
            it will default to all ones.
        lamb: 0-d tensor controlling strength of entropic regularization.
        epsilon: 0-d tensor, convergence threshold.

    Returns:
        The optimal transport matrix, with shape `[batch_size, n, m]`, along
        with the Sinkhorn distance.

    """
    lamb = _maybe_float_to_tensor(lamb)
    epsilon = _maybe_float_to_tensor(epsilon)
    cost = tf.cast(cost, tf.float32)

    cost_3d = tf.assert_rank(cost, 3)

    with tf.control_dependencies([cost_3d]):
        # Default to ones for rows and column sums.
        cost_shape = tf.shape(cost)
        if row_sums is None:
            row_sums = tf.ones(cost_shape[:2])
        if column_sums is None:
            column_sums = tf.ones(
                tf.stack([cost_shape[0], cost_shape[2]], axis=0)
            )

    row_sums_2d = tf.assert_rank(row_sums, 2)
    column_sums_2d = tf.assert_rank(column_sums, 2)
    lambda_0d = tf.assert_rank(lamb, 0)
    epsilon_0d = tf.assert_rank(epsilon, 0)

    with tf.control_dependencies(
        [cost_3d, row_sums_2d, column_sums_2d, lambda_0d, epsilon_0d]
    ):
        # Create initial optimal transport matrix.
        regularized_cost = -lamb * cost
        # Clip anything that's too large to guard against numerical instability.
        regularized_cost = tf.minimum(regularized_cost, 10)
        transport = tf.exp(regularized_cost)
        transport /= tf.reduce_sum(transport)

        # Add extra dimension to row and column sums for easy multiplication.
        row_sums = tf.expand_dims(row_sums, axis=2)
        column_sums = tf.expand_dims(column_sums, axis=1)

        last_actual_row_sum = tf.zeros_like(row_sums)

        def _cond(
            _last_actual_row_sum: tf.Tensor, _transport: tf.Tensor
        ) -> tf.Tensor:
            """
            Whether we should continue running the loop.

            Args:
                _last_actual_row_sum: The previous row sums.
                _transport: The current optimal transport matrix.

            Returns:
                0-d boolean tensor, true if we should continue.

            """
            # Check for convergence.
            new_row_sums = tf.reduce_sum(_transport, axis=2, keepdims=True)
            max_change = tf.reduce_max(
                tf.abs(_last_actual_row_sum - new_row_sums)
            )
            return tf.greater(max_change, epsilon)

        def _body(_, _transport: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            """
            Loop body.

            Args:
                _transport: The current optimal transport matrix.

            Returns:
                New values for the loop variables.

            """
            next_row_sum = tf.reduce_sum(_transport, axis=2, keepdims=True)
            # Avoid zero division.
            next_row_sum = tf.maximum(next_row_sum, _EPSILON)
            next_transport = _transport * (row_sums / next_row_sum)

            next_column_sum = tf.reduce_sum(
                next_transport, axis=1, keepdims=True
            )
            next_column_sum = tf.maximum(next_column_sum, _EPSILON)
            # This broadcast is needed so that Tensorflow can statically
            # verify that the shape of the Sinkhorn matrix doesn't change
            # between iterations.
            column_scale = tf.broadcast_to(
                column_sums / next_column_sum, tf.shape(_transport)
            )
            next_transport *= column_scale

            return next_row_sum, next_transport

    _, transport = tf.while_loop(
        _cond, _body, [last_actual_row_sum, transport], name="sinkhorn"
    )

    # Calculate the sinkhorn distance.
    return transport, tf.reduce_sum(transport * cost)


def construct_gt_sinkhorn_matrix(
    *, detection_ids: tf.Tensor, tracklet_ids: tf.Tensor
) -> tf.Tensor:
    """
    Utility for constructing ground-truth Sinkhorn matrix based on lists
    of corresponding object IDs for two consecutive frames.

    Args:
        detection_ids: The list of IDs for the detections. Should have shape
            `[n_detections]`.
        tracklet_ids: The list of IDs for the tracklets. Should have shape
            `[n_tracklets]`.

    Returns:
        The ground-truth Sinkhorn matrix. Will have the shape
        `[n_tracklets, n_detections]`.

    """
    detections_1d = tf.assert_rank(detection_ids, 1)
    tracklets_1d = tf.assert_rank(tracklet_ids, 1)

    with tf.control_dependencies([detections_1d, tracklets_1d]):
        # Create the base matrix.
        sinkhorn_matrix = tf.equal(
            tf.expand_dims(tracklet_ids, axis=1),
            tf.expand_dims(detection_ids, axis=0),
        )

    # Convert booleans to floats.
    return tf.cast(sinkhorn_matrix, tf.float32)


def do_hard_assignment(
    sinkhorn: tf.Tensor, threshold: float = 0.5
) -> tf.Tensor:
    """
    Converts the "soft" Sinkhorn assignment matrix into a hard one by using
    thresholding and the Hungarian algorithm.

    Args:
        sinkhorn: The sinkhorn matrix. Should have shape
            `[n_tracklets, n_detections]`.
        threshold: The threshold value to use. Anything above this will be
            treated as a one.

    Returns:
        The hard assignment matrix.

    """
    # Binarize the Sinkhorn matrix.
    threshold = tf.constant(threshold, dtype=tf.float32)
    binarized = tf.where(
        sinkhorn >= threshold, tf.ones_like(sinkhorn), tf.zeros_like(sinkhorn)
    )

    # Apply Hungarian matching.
    maximize_affinity = partial(optimize.linear_sum_assignment, maximize=True)
    row_indices, col_indices = tf.numpy_function(
        maximize_affinity, [binarized], (tf.int64, tf.int64), name="hungarian"
    )

    # Create the assignment matrix.
    sparse_indices = tf.stack((row_indices, col_indices), axis=1)
    num_assignments = tf.shape(row_indices)[0]
    values = tf.ones((num_assignments,), dtype=tf.bool)
    dense_shape = tf.cast(tf.shape(sinkhorn), tf.int64)

    sparse_assignment = tf.sparse.SparseTensor(
        sparse_indices, values, dense_shape
    )
    return tf.sparse.to_dense(sparse_assignment)
