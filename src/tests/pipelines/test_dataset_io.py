"""
Tests for the `dataset_io` module.
"""


import numpy as np
import pytest
import tensorflow as tf
from faker import Faker

from src.cotton_flower_mot.pipelines import dataset_io
from src.cotton_flower_mot.pipelines.schemas import ModelInputs, ModelTargets

from .data import TESTING_DATASET_PATH

_TEST_DATASET_SIZE = 4
"""
Number of elements that are in the test dataset.
"""


@pytest.mark.integration
def test_inputs_and_targets_from_dataset_smoke(faker: Faker) -> None:
    """
    Attempts to load actual data and makes sure that it works.

    Args:
        faker: The fixture to use for generating fake data.

    """
    # Arrange.
    # Load the raw dataset.
    raw_dataset = tf.data.TFRecordDataset([TESTING_DATASET_PATH.as_posix()])

    config = faker.model_config(image_shape=(100, 100, 3))

    # Act.
    dataset = dataset_io.inputs_and_targets_from_dataset(
        raw_dataset, config=config
    )

    # Assert.
    # Make sure we can get the items.
    batch = next(iter(dataset))
    # It should have made one batch.
    inputs, targets = batch

    # Make sure we have the right inputs.
    for element in ModelInputs:
        assert element.value in inputs
    for element in ModelTargets:
        assert element.value in targets

    # We expect the batch size to be one fewer than the number of elements, due
    # to windowing.
    expected_batch_size = _TEST_DATASET_SIZE - 1

    # Check the shapes of things.
    detections_shape = (
        inputs[ModelInputs.DETECTIONS.value].bounding_shape().numpy()
    )
    tracklets_shape = (
        inputs[ModelInputs.TRACKLETS.value].bounding_shape().numpy()
    )
    assert len(detections_shape) == len(tracklets_shape) == 5
    assert detections_shape[0] == tracklets_shape[0] == expected_batch_size
    assert np.all(detections_shape[2:] == (100, 100, 3))
    assert np.all(tracklets_shape[2:] == (100, 100, 3))

    detection_geometry_shape = (
        inputs[ModelInputs.DETECTION_GEOMETRY.value].bounding_shape().numpy()
    )
    tracklet_geometry_shape = (
        inputs[ModelInputs.TRACKLET_GEOMETRY.value].bounding_shape().numpy()
    )
    assert len(detection_geometry_shape) == len(tracklet_geometry_shape) == 3
    assert (
        detection_geometry_shape[0]
        == tracklet_geometry_shape[0]
        == expected_batch_size
    )
    assert detection_geometry_shape[2] == tracklet_geometry_shape[2] == 4

    # The number of nodes should be consistent.
    assert detections_shape[1] == detection_geometry_shape[1]
    assert tracklets_shape[1] == tracklet_geometry_shape[1]

    sinkhorn_shape = tf.shape(targets[ModelTargets.SINKHORN.value]).numpy()
    assert len(sinkhorn_shape) == 2
    assert sinkhorn_shape[0] == expected_batch_size
    # Sinkhorn matrix should have an entry for each detection/tracklet pair.
    assert sinkhorn_shape[1] == detections_shape[1] * tracklets_shape[1]
