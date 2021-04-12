from functools import partial
from typing import Any, Dict, Tuple

import tensorflow as tf

from ...assignment import do_hard_assignment, solve_optimal_transport


class AssociationLayer(tf.keras.layers.Layer):
    """
    Custom layer that computes association matrices.
    """

    def __init__(self, sinkhorn_lambda: float = 10.0):
        """
        Args:
            sinkhorn_lambda: The Sinkhorn lambda value.

        """
        super().__init__()

        self._lambda = sinkhorn_lambda

    @staticmethod
    def _compute_row_or_column_sum(length: tf.Tensor) -> tf.Tensor:
        """
        For Sinkhorn normalization, we expect rows and columns to sum to one,
        except for the births/deaths row/column, which we expect to sum to their
        own length.

        Args:
            length: The length of the row or column.

        Returns:
            The expected sums. This will have the form `[1, 1, ..., length]`.

        """
        row_shape = tf.expand_dims(length, axis=0)
        sums = tf.ones(row_shape, dtype=tf.float32)

        # Add the last element.
        return tf.concat((sums, tf.cast(row_shape, tf.float32)), axis=0)

    def call(
        self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:
        """

        Args:
            inputs:
                Has three elements:
                - affinity_scores: The affinity scores computed for each
                    possible tracklet/detection pair. Should have a shape of
                    `[batch_size, max_n_tracklets, max_n_detections]`.
                - num_detections: The number of detections in each example.
                    Should be a vector of shape `[batch_size]`.
                - num_tracklets: The number of tracklets in each example.
                    Should be a vector of shape `[batch_size]`.

        Returns:
            The normalized optimal transport matrix, and the corresponding hard
            assignment matrix. These will be`RaggedTensor`s where the second
            dimension is ragged, so it will have the shape
            `[batch_size, n_tracklets * n_detections]`.

        """
        affinity_scores, num_detections, num_tracklets = inputs

        def _normalize(
            element: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            affinity_matrix, _num_detections, _num_tracklets = element
            affinity_flat_length = tf.math.reduce_prod(
                tf.shape(affinity_matrix)
            )
            affinity_flat_length = tf.cast(affinity_flat_length, tf.int64)

            def _trim_and_flatten(association: tf.Tensor) -> tf.Tensor:
                # Remove births/deaths row/column.
                association = association[:-1, :-1]
                # Flatten.
                transport_flat = tf.reshape(association, (-1,))
                # Re-pad so the outputs all have the same size.
                padding = tf.stack(
                    (
                        0,
                        affinity_flat_length
                        - _num_tracklets * _num_detections,
                    )
                )
                return tf.pad(transport_flat, tf.expand_dims(padding, 0))

            # Remove the padding.
            affinity_un_padded = affinity_matrix[
                :_num_tracklets, :_num_detections
            ]
            # Add additional row and column for track births/deaths.
            affinity_expanded = tf.pad(affinity_un_padded, [[0, 1], [0, 1]])

            row_sums = self._compute_row_or_column_sum(_num_tracklets)
            column_sums = self._compute_row_or_column_sum(_num_detections)

            # Add fake batch dimension.
            affinity_expanded = tf.expand_dims(affinity_expanded, axis=0)
            row_sums = tf.expand_dims(row_sums, axis=0)
            column_sums = tf.expand_dims(column_sums, axis=0)

            # Normalize it.
            transport, _ = solve_optimal_transport(
                # Cost matrix is -affinity.
                -affinity_expanded,
                lamb=self._lambda,
                row_sums=row_sums,
                column_sums=column_sums,
            )
            # Remove extraneous batch dimension.
            transport = transport[0]
            assignment = do_hard_assignment(transport)

            return _trim_and_flatten(transport), _trim_and_flatten(assignment)

        # Unfortunately, we can't have padding for the affinity scores, because
        # it affects the optimization. Therefore, this process has to be done
        # with map_fn instead of vectorized.
        sinkhorn_dense, assignment_dense = tf.map_fn(
            _normalize,
            (affinity_scores, num_detections, num_tracklets),
            fn_output_signature=(
                tf.TensorSpec(shape=[None], dtype=tf.float32),
                tf.TensorSpec(shape=[None], dtype=tf.bool),
            ),
        )

        # Convert to a ragged tensor.
        row_lengths = num_detections * num_tracklets
        to_ragged = partial(tf.RaggedTensor.from_tensor, lengths=row_lengths)
        return to_ragged(sinkhorn_dense), to_ragged(assignment_dense)

    def get_config(self) -> Dict[str, Any]:
        return dict(sinkhorn_lambda=self._lambda)
