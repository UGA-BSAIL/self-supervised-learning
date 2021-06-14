"""
Utilities for creating the heat maps used by CenterNet and related
keypoint-based detectors.
"""


from typing import Optional

import tensorflow as tf
import tensorflow_addons as tfa
from loguru import logger


def trim_out_of_bounds(points: tf.Tensor) -> tf.Tensor:
    """
    Convenience function to trim points that are not within `[0, 1]`. It is
    invoked automatically when creating heatmaps.

    Args:
        points: The points to trim, as a matrix of shape
            `[num_points, num_dims]`. It will assume that the first two
            dimensions are the x and y coordinates.

    Returns:
        The trimmed points.

    """
    x_valid = tf.logical_and(points[:, 0] >= 0.0, points[:, 0] <= 1.0)
    y_valid = tf.logical_and(points[:, 1] >= 0.0, points[:, 1] <= 1.0)
    # Both coordinates of a point must be valid to keep that point.
    point_valid = tf.logical_and(x_valid, y_valid)
    return tf.boolean_mask(points, point_valid)


def make_point_annotation_map(
    points: tf.Tensor,
    *,
    map_size: tf.Tensor,
    point_values: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Creates dense point maps given a set of point locations. Each point
    corresponds to a pixel in the map that will be set to a non-zero value,
    while the rest will be set to zero.

    Args:
        points: The points to add to the map. Should be in the form (x, y).
        map_size: The size of the map, in the form (width, height).
        point_values: If specified, these are the corresponding values which
            will be set at the location of each point. Should be a 1D tensor
            with the same length as `points`. If not specified, all values will
            be set to one.

    Returns:
        The point annotation map that it created. It will have the shape
        specified by `map_size`.

    """
    points = tf.ensure_shape(points, (None, 2))
    map_size = tf.ensure_shape(map_size, (2,))
    points = tf.cast(points, tf.float32)

    # Eliminate any points that are out-of-bounds.
    points = trim_out_of_bounds(points)

    # Quantize the annotations to convert from frame fractions to actual
    # pixel values.
    pixel_points = points * tf.cast(map_size - tf.constant(1), tf.float32)
    pixel_points = tf.cast(tf.round(pixel_points), tf.int64)

    # Generate the output maps.
    if point_values is None:
        # Use ones for all the values.
        num_non_zero_values = tf.shape(pixel_points)[0]
        point_values = tf.ones((num_non_zero_values,), dtype=tf.float32)
    sparse_maps = tf.SparseTensor(
        indices=pixel_points[:, ::-1],
        values=point_values,
        # See https://github.com/tensorflow/tensorrt/issues/118 for
        # explanation of indexing.
        dense_shape=tf.cast(map_size[..., ::-1], dtype=tf.int64),
    )
    # Reorder indices to conform with sparse tensor conventions.
    sparse_maps = tf.sparse.reorder(sparse_maps)

    dense = tf.sparse.to_dense(
        sparse_maps,
        default_value=0.0,
        # There might be duplicate indices, which we want to ignore.
        validate_indices=False,
    )
    return dense


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
    # Add a dummy channel dimension.
    dense_annotations = tf.expand_dims(dense_annotations, 2)

    return tfa.image.gaussian_filter2d(
        dense_annotations,
        filter_shape=kernel_size,
        sigma=sigma,
    )
