"""
Utilities for computing similarity metrics.
"""


from typing import Tuple

import tensorflow as tf

_EPSILON = tf.constant(1.0e-6)
"""
Small value to use to avoid division by zero.
"""


def compute_ious(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """
    Computes the IOUs between two sets of bounding boxes.

    Args:
        boxes1: First array of bounding boxes. It should have the shape
            `[n_boxes, 4]`, where the columns are the horizontal position
            of the box center, the vertical position of the box center,
            the box's width, and the box's height, in that order.
        boxes2: The second array of bounding boxes. Has the same shape and
            format as `boxes1`.

    Returns:
        An array of shape `[n_boxes,]` containing the IOUs between each pair
        of bounding boxes.

    """
    boxes1 = tf.ensure_shape(boxes1, (None, 4))
    boxes2 = tf.ensure_shape(boxes2, (None, 4))

    def _min_max(boxes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes minimum and maximum extents for bounding boxes.

        """
        boxes_center = boxes[:, :2]
        boxes_half_size = boxes[:, 2:] / 2.0

        boxes_min = boxes_center - boxes_half_size
        boxes_max = boxes_center + boxes_half_size
        return boxes_min, boxes_max

    boxes1_rank2 = tf.assert_rank(boxes1, 2)
    boxes2_rank2 = tf.assert_rank(boxes2, 2)

    with tf.control_dependencies([boxes1_rank2, boxes2_rank2]):
        boxes1_min, boxes1_max = _min_max(boxes1)
        boxes2_min, boxes2_max = _min_max(boxes2)

    box_boundaries = tf.stack(
        [boxes1_min, boxes1_max, boxes2_min, boxes2_max], axis=1
    )
    sorted_boundaries = tf.sort(box_boundaries, axis=1)

    # Extract the intersection and union.
    intersection = sorted_boundaries[:, 2, :] - sorted_boundaries[:, 1, :]
    union = sorted_boundaries[:, 3, :] - sorted_boundaries[:, 0, :]

    # Account for the case where the boxes are disjoint.
    total_size = boxes1[:, 2:] + boxes2[:, 2:]
    is_disjoint = union > total_size
    # If they are disjoint, the intersection is zero.
    intersection = tf.where(is_disjoint, 0.0, intersection)

    # Figure out the area.
    intersection_area = tf.reduce_prod(intersection, axis=1)
    union_area = tf.reduce_prod(union, axis=1)
    return intersection_area / (tf.maximum(union_area, _EPSILON))


def cosine_similarity(features1: tf.Tensor, features2: tf.Tensor) -> tf.Tensor:
    """
    Computes the cosine similarity between two sets of feature vectors.

    Args:
        features1: The first set of feature vectors, with shape
            `[batch_size, n_features]`.
        features2: The second set of feature vectors, with shape
            `[batch_size, n_features]`.

    Returns:
        The cosine similarities between corresponding features, with shape
        `[batch_size,]`.

    """
    feature_dot = tf.reduce_sum(features1 * features2, axis=-1)
    feature1_mag = tf.sqrt(tf.reduce_sum(tf.square(features1), axis=-1))
    feature2_mag = tf.sqrt(tf.reduce_sum(tf.square(features2), axis=-1))

    return feature_dot / (feature1_mag * feature2_mag + _EPSILON)
