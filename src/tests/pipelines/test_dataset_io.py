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

_TEST_DATASET_SIZE = 10
"""
Number of elements that are in the test dataset.
"""


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "include_frame", (True, False), ids=["include_frame", "no_frame"]
)
@pytest.mark.parametrize(
    "random_drop", (True, False), ids=["random_drop", "no_random_drop"]
)
@pytest.mark.parametrize(
    "include_heat_map", (True, False), ids=["heatmap", "no_heatmap"]
)
def test_inputs_and_targets_from_dataset_smoke(
    faker: Faker,
    include_frame: bool,
    random_drop: bool,
    include_heat_map: bool,
) -> None:
    """
    Attempts to load actual data and makes sure that it works.

    Args:
        faker: The fixture to use for generating fake data.
        include_frame: Whether to test including the full frame.
        random_drop: Whether to test dropping random examples.
        include_heat_map: Whether to test including the detection heat map.

    """
    # Arrange.
    # Load the raw dataset.
    raw_dataset = tf.data.TFRecordDataset([TESTING_DATASET_PATH.as_posix()])

    config = faker.model_config(image_shape=(100, 100, 3))

    # Act.
    expected_batch_size = 8
    drop_kwargs = {}
    if random_drop:
        drop_kwargs = dict(drop_probability=0.5, repeats=2)
    dataset = dataset_io.inputs_and_targets_from_dataset(
        raw_dataset,
        config=config,
        include_frame=include_frame,
        include_heat_map=include_heat_map,
        batch_size=8,
        **drop_kwargs
    )

    # Assert.
    # Make sure we can get the items.
    batch = next(iter(dataset))
    # It should have made one batch.
    inputs, targets = batch

    # Make sure we have the right inputs and targets.
    expected_inputs = set(ModelInputs)
    expected_targets = set(ModelTargets)
    if not include_frame:
        # We won't have a frame input in this case.
        expected_inputs -= {ModelInputs.DETECTIONS_FRAME}
    if not include_heat_map:
        expected_targets -= {ModelTargets.HEATMAP}
        expected_targets -= {ModelTargets.GEOMETRY}
    for element in expected_inputs:
        assert element.value in inputs
    for element in expected_targets:
        assert element.value in targets

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

    if include_frame:
        # Check the frame shape.
        frame_shape = tf.shape(
            inputs[ModelInputs.DETECTIONS_FRAME.value]
        ).numpy()
        assert len(frame_shape) == 4
        assert frame_shape[0] == expected_batch_size

    detection_geometry = inputs[ModelInputs.DETECTION_GEOMETRY.value]
    tracklet_geometry = inputs[ModelInputs.TRACKLET_GEOMETRY.value]
    detection_geometry_shape = detection_geometry.bounding_shape().numpy()
    tracklet_geometry_shape = tracklet_geometry.bounding_shape().numpy()
    assert len(detection_geometry_shape) == len(tracklet_geometry_shape) == 3
    assert (
        detection_geometry_shape[0]
        == tracklet_geometry_shape[0]
        == expected_batch_size
    )
    assert detection_geometry_shape[2] == tracklet_geometry_shape[2] == 4

    # The geometry should all be normalized.
    assert np.all(detection_geometry.to_tensor().numpy() >= 0.0)
    assert np.all(detection_geometry.to_tensor().numpy() <= 1.0)
    assert np.all(tracklet_geometry.to_tensor().numpy() >= 0.0)
    assert np.all(tracklet_geometry.to_tensor().numpy() <= 1.0)

    # The number of nodes should be consistent.
    assert detections_shape[1] == detection_geometry_shape[1]
    assert tracklets_shape[1] == tracklet_geometry_shape[1]

    # It should have specified the sequence ID.
    sequence_id_shape = tf.shape(inputs[ModelInputs.SEQUENCE_ID.value]).numpy()
    np.testing.assert_array_equal(sequence_id_shape, (expected_batch_size, 2))

    sinkhorn_shape = tf.shape(targets[ModelTargets.SINKHORN.value]).numpy()
    assert len(sinkhorn_shape) == 2
    assert sinkhorn_shape[0] == expected_batch_size
    # Sinkhorn matrix should have an entry for each detection/tracklet pair.
    assert sinkhorn_shape[1] == detections_shape[1] * tracklets_shape[1]

    # Hard assignment matrix should be equivalent to the Sinkhorn matrix.
    sinkhorn = targets[ModelTargets.SINKHORN.value].numpy()
    assignment = targets[ModelTargets.ASSIGNMENT.value].numpy()
    np.testing.assert_array_equal(sinkhorn.astype(np.bool), assignment)
