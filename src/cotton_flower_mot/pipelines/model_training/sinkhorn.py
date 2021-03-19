"""
Implementation of the Sinkhorn-Kopp algorithm in TensorFlow.
"""

from typing import Optional, Tuple, Union

import tensorflow as tf


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

    cost_3d = tf.assert_rank(cost, 3)

    with tf.control_dependencies([cost_3d]):
        # Default to ones for rows and column sums.
        cost_shape = tf.shape(cost)
        if row_sums is None:
            row_sums = tf.ones(cost_shape[:2])
        if column_sums is None:
            column_sums = tf.ones(
                tf.concat([cost_shape[0], cost_shape[2]], axis=0)
            )

    row_sums_2d = tf.assert_rank(row_sums, 2)
    column_sums_2d = tf.assert_rank(column_sums, 2)
    lambda_0d = tf.assert_rank(lamb, 0)
    epsilon_0d = tf.assert_rank(epsilon, 0)

    with tf.control_dependencies(
        [cost_3d, row_sums_2d, column_sums_2d, lambda_0d, epsilon_0d]
    ):
        # Create initial optimal transport matrix.
        transport = tf.exp(-lamb * cost)
        transport /= tf.reduce_sum(transport)

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
            new_row_sums = tf.reduce_sum(_transport, axis=2)
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
            next_row_sum = tf.reduce_sum(_transport, axis=2)
            next_transport = _transport * tf.expand_dims(
                (row_sums / next_row_sum), axis=2
            )

            next_column_sum = tf.reduce_sum(next_transport, axis=1)
            next_transport *= tf.expand_dims(
                column_sums / next_column_sum, axis=1
            )

            return next_row_sum, next_transport

    _, transport = tf.while_loop(
        _cond, _body, [last_actual_row_sum, transport], name="sinkhorn"
    )

    # Calculate the sinkhorn distance.
    return transport, tf.reduce_sum(transport * cost)
