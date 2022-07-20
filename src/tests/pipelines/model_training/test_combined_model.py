"""
Tests for the `combined_model` module.
"""


import pytest
from faker import Faker

from src.cotton_flower_mot.pipelines.model_training import combined_model


@pytest.mark.integration
@pytest.mark.slow
def test_build_combined_model_smoke(faker: Faker) -> None:
    """
    Tests that we can build and use the combined model.

    Args:
        faker: The fixture to use for generating fake data.

    """
    # Arrange.
    detection_shape = (540, 960, 3)

    # Create fake input data.
    batch_size = faker.random_int(max=4)
    current_frames = faker.tensor((batch_size,) + detection_shape)
    previous_frames = faker.tensor((batch_size,) + detection_shape)
    detection_geometry = faker.bounding_boxes(batch_size=batch_size)
    tracklet_geometry = faker.bounding_boxes(batch_size=batch_size)

    config = faker.model_config(detection_input_shape=detection_shape)

    # Act.
    model = combined_model.build_combined_model(config)

    # Test the model.
    model(
        (
            current_frames,
            previous_frames,
            tracklet_geometry,
            detection_geometry,
        )
    )
