"""
Tests for the `centernet_model` module.
"""


import numpy as np
import pytest
import tensorflow as tf

from src.cotton_flower_mot.pipelines.model_training import centernet_model


@pytest.mark.parametrize(
    ("confidence_mask", "sizes", "offsets", "expected_boxes"),
    [
        (
            np.array([[0, 0.75], [0.9, 0]]),
            np.array(
                [[[0.25, 0.20], [0.65, 0.72]], [[0.09, 0.43], [0.31, 0.08]]]
            ),
            np.array(
                [
                    [[0.001, 0.003], [0.002, 0.0012]],
                    [[0.01, 0.002], [0.005, 0.0043]],
                ]
            ),
            np.array(
                [
                    [0.502, 0.0012, 0.65, 0.72, 0.75],
                    [0.01, 0.502, 0.09, 0.43, 0.9],
                ]
            ),
        )
    ],
    ids=["standard"],
)
def test_compute_sparse_predictions(
    confidence_mask: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    expected_boxes: np.ndarray,
) -> None:
    """
    Tests that `compute_sparse_predictions` works.

    Args:
        confidence_mask: The mask for the center pixels to test with.
        sizes: The sizes for each pixel in the heatmap.
        offsets: The offsets for each pixel in the heatmap.
        expected_boxes: The expected resulting bounding boxes.

    """
    # Arrange.
    # Add in batch dimension.
    confidence_mask = tf.constant(np.expand_dims(confidence_mask, 0))
    confidence_mask = tf.expand_dims(confidence_mask, -1)
    sizes = tf.constant(np.expand_dims(sizes.astype(np.float32), 0))
    offsets = tf.constant(np.expand_dims(offsets.astype(np.float32), 0))
    expected_boxes = tf.constant(np.expand_dims(expected_boxes, 0))

    # Act.
    got_boxes = centernet_model.compute_sparse_predictions(
        confidence_masks=confidence_mask, sizes=sizes, offsets=offsets
    )

    # Assert.
    got_boxes = got_boxes.to_tensor().numpy()
    np.testing.assert_array_almost_equal(got_boxes, expected_boxes)
