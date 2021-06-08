"""
Utilities for creating the heat maps used by CenterNet and related
keypoint-based detectors.
"""


from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa
from loguru import logger


def make_point_annotation_map(
    points: tf.Tensor, *, map_size: tf.Tensor
) -> tf.Tensor:
    """
    Creates dense point maps given a set of point locations. Each point
    corresponds to a pixel in the map that will be set to 1, while the rest
    will be set to zero.

    Args:
        points: The points to add to the map. Should be in the form (x, y).
        map_size: The size of the map, in the form (width, height).

    Returns:
        The point annotation map that it created.

    """
    points = tf.ensure_shape(points, (None, 2))
    map_size = tf.ensure_shape(map_size, (2,))
    points = tf.cast(points, tf.float32)

    # Quantize the annotations to convert from frame fractions to actual
    # pixel values.
    pixel_points = points * tf.cast(map_size - tf.constant(1), tf.float32)
    pixel_points = tf.cast(tf.round(pixel_points), tf.int64)

    # Generate the output maps.
    num_non_zero_values = tf.shape(pixel_points)[0]
    non_zeros = tf.ones((num_non_zero_values,), dtype=tf.float32)
    sparse_maps = tf.SparseTensor(
        indices=pixel_points[:, ::-1],
        values=non_zeros,
        dense_shape=tf.cast(map_size[::-1], dtype=tf.int64),
    )
    # Reorder indices to conform with sparse tensor conventions.
    sparse_maps = tf.sparse.reorder(sparse_maps)

    dense = tf.sparse.to_dense(
        sparse_maps,
        default_value=0.0,
        # There might be duplicate indices, which we want to ignore.
        validate_indices=False,
    )
    # Add a dummy channel dimension.
    return tf.expand_dims(dense, 2)


def make_heat_map(
    points: tf.Tensor, *, map_size: tf.Tensor, sigma: float
) -> tf.Tensor:
    """
    Creates a gaussian heat map by splatting a set of points.

    Args:
        points: The points to add to the map. Should be in the form (x, y).
        map_size: The size of the map, in the form (width, height).
        sigma:
            The standard deviation in pixels to use for the applied gaussian
            filter.

    Returns:
        A tensor containing density maps, of the shape
        (height, width, 1)

    """
    # Compute our filter size so that it's odd and has sigma pixels on either
    # side.
    kernel_size = int(1 + 6 * sigma)
    logger.debug("Using {}-pixel kernel for gaussian blur.", kernel_size)

    # Obtain initial point annotations.
    dense_annotations = make_point_annotation_map(points, map_size=map_size)

    return tfa.image.gaussian_filter2d(
        dense_annotations,
        filter_shape=kernel_size,
        sigma=sigma,
    )
