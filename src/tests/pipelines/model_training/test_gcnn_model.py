"""
Tests for the `gcnn_model` module.
"""


import numpy as np
import pytest
import tensorflow as tf
from faker import Faker

from src.cotton_flower_mot.pipelines.model_training import gcnn_model


@pytest.mark.integration
@pytest.mark.slow
def test_extract_appearance_features_smoke(faker: Faker) -> None:
    """
    Tests that `extract_appearance_features` builds a functional graph.

    Args:
        faker: The fixture for creating fake data.

    """
    # Arrange.
    # Create fake detections and tracklets.
    image_shape = (100, 100, 3)
    batch_size = faker.random_int(min=1, max=16)
    detections = faker.detected_objects(
        image_shape=image_shape, batch_size=batch_size
    )
    tracklets = faker.detected_objects(
        image_shape=image_shape, batch_size=batch_size
    )

    config = faker.model_config(image_shape=image_shape)

    # Act.
    input_shape = (None,) + image_shape
    detection_input = tf.keras.Input(input_shape, ragged=True)
    tracklet_input = tf.keras.Input(input_shape, ragged=True)

    (
        detection_features,
        tracklet_features,
    ) = gcnn_model.extract_appearance_features(
        detections=detection_input, tracklets=tracklet_input, config=config
    )

    # Turn this into a model.
    model = tf.keras.Model(
        inputs=[detection_input, tracklet_input],
        outputs=[detection_features, tracklet_features],
    )

    # Apply the model to the inputs we generated.
    got_detection_features, got_tracklet_features = model.predict(
        (detections, tracklets)
    )

    # Assert.
    # Make sure that our results are the right shape.
    detection_shape = got_detection_features.bounding_shape().numpy()
    tracklet_shape = got_tracklet_features.bounding_shape().numpy()
    assert len(detection_shape) == len(tracklet_shape) == 3
    # Batch size should be correct.
    assert detection_shape[0] == tracklet_shape[0] == batch_size
    # Number of features should be correct.
    assert (
        detection_shape[2]
        == tracklet_shape[2]
        == config.num_appearance_features
    )


@pytest.mark.integration
@pytest.mark.slow
def test_extract_interaction_features_smoke(faker: Faker) -> None:
    """
    Tests that `extract_interaction_features` builds a functional graph.

    Args:
        faker: The fixture to use for generating fake data.

    """
    # Arrange.
    image_shape = (100, 100, 3)
    batch_size = faker.random_int(min=1, max=16)
    detections = faker.detected_objects(
        image_shape=image_shape, batch_size=batch_size
    )
    tracklets = faker.detected_objects(
        image_shape=image_shape, batch_size=batch_size
    )

    # Create fake geometry features.
    num_geom_features = faker.random_int(min=1, max=100)
    detections_geometry = faker.ragged_tensor(
        row_lengths=detections.row_lengths(), inner_shape=(num_geom_features,)
    )
    tracklets_geometry = faker.ragged_tensor(
        row_lengths=detections.row_lengths(), inner_shape=(num_geom_features,)
    )

    config = faker.model_config(image_shape=image_shape)

    # Act.
    input_shape = (None,) + image_shape
    detection_input = tf.keras.Input(input_shape, ragged=True)
    tracklet_input = tf.keras.Input(input_shape, ragged=True)

    geom_input_shape = (None, num_geom_features)
    detection_geometry_input = tf.keras.Input(geom_input_shape, ragged=True)
    tracklet_geometry_input = tf.keras.Input(geom_input_shape, ragged=True)

    (
        tracklet_inter_features,
        detection_inter_features,
    ) = gcnn_model.extract_interaction_features(
        detections=detection_input,
        tracklets=tracklet_input,
        detections_geometry=detection_geometry_input,
        tracklets_geometry=tracklet_geometry_input,
        config=config,
    )

    # Turn this into a model.
    model = tf.keras.Model(
        inputs=[
            detection_input,
            tracklet_input,
            detection_geometry_input,
            tracklet_geometry_input,
        ],
        outputs=[tracklet_inter_features, detection_inter_features],
    )

    # Apply the model to the inputs we generated.
    got_tracklet_features, got_detection_features = model.predict(
        (detections, tracklets, detections_geometry, tracklets_geometry)
    )

    # Assert.
    # Make sure our results are the right shape.
    detection_shape = got_detection_features.bounding_shape()
    tracklet_shape = got_tracklet_features.bounding_shape()
    assert len(detection_shape) == len(tracklet_shape) == 3
    # It should have the correct number of features.
    assert detection_shape[2] == tracklet_shape[2] == config.num_gcn_channels
    # It should have the correct number of nodes.
    assert np.all(
        got_detection_features.row_lengths().numpy()
        == detections.row_lengths().numpy()
    )
    assert np.all(
        got_tracklet_features.row_lengths().numpy()
        == tracklets.row_lengths().numpy()
    )
