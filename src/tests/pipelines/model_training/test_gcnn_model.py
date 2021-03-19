"""
Tests for the `gcnn_model` module.
"""


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
    batch_size = faker.random_int(min=1, max=16)
    row_lengths_detections = [
        faker.random_int(max=8) for _ in range(batch_size)
    ]
    row_lengths_tracklets = [
        faker.random_int(max=8) for _ in range(batch_size)
    ]
    image_shape = (100, 100, 3)

    detections = faker.ragged_tensor(
        row_lengths=row_lengths_detections, inner_shape=image_shape
    )
    tracklets = faker.ragged_tensor(
        row_lengths=row_lengths_tracklets, inner_shape=image_shape
    )

    # Convert to integers to simulate how actual images are.
    detections = tf.cast(detections, tf.uint8)
    tracklets = tf.cast(tracklets, tf.uint8)

    config = gcnn_model.ModelConfig(
        image_input_shape=image_shape,
        num_appearance_features=faker.random_int(min=1, max=256),
        num_gcn_channels=faker.random_int(min=1, max=256),
    )

    # Act.
    input_shape = (None,) + image_shape
    detection_inputs = tf.keras.Input(input_shape, ragged=True)
    tracklet_inputs = tf.keras.Input(input_shape, ragged=True)

    (
        detection_features,
        tracklet_features,
    ) = gcnn_model.extract_appearance_features(
        detections=detection_inputs, tracklets=tracklet_inputs, config=config
    )

    # Turn this into a model.
    model = tf.keras.Model(
        inputs=[detection_inputs, tracklet_inputs],
        outputs=[detection_features, tracklet_features],
    )

    # Apply the model to the inputs we generated.
    got_detection_features, got_tracklet_features = model.predict(
        (detections, tracklets)
    )

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
