"""
Tests for the `pooling` module.
"""

import numpy as np
import pytest
import tensorflow as tf
from faker import Faker
from pydantic.dataclasses import dataclass

from src.cotton_flower_mot.pipelines import heat_maps
from src.cotton_flower_mot.pipelines.model_training.layers import pooling
from src.cotton_flower_mot.type_helpers import ArbitraryTypesConfig

# Do not use mixed precision during tests as it can degrade accuracy.
tf.keras.mixed_precision.set_global_policy("float32")


class TestPeakLayer:
    """
    Tests for the `PeakLayer` class.
    """

    @dataclass(frozen=True, config=ArbitraryTypesConfig)
    class ConfigForTests:
        """
        Encapsulates standard configuration for most tests.

        Attributes:
            peak_layer: The `PeakLayer` under test.
            has_confidence: Whether the layer is configured to include
                confidence values.
        """

        peak_layer: pooling.PeakLayer
        has_confidence: bool

    @classmethod
    @pytest.fixture(params=[False, True], ids=["no_confidence", "confidence"])
    def config(cls, request) -> ConfigForTests:
        """
        Generates standard configuration for most tests.

        Args:
            request: The `pytest` request object.

        Returns:
            The configuration that it generated.

        """
        use_confidence = request.param
        layer = pooling.PeakLayer(with_confidence=use_confidence)

        return cls.ConfigForTests(
            peak_layer=layer, has_confidence=use_confidence
        )

    def test_point_mask(self, config: ConfigForTests, faker: Faker) -> None:
        """
        Tests that a simple point mask goes through this layer unchanged.

        Args:
            config: The configuration to use for testing.
            faker: The fixture to use for generating fake data.

        """
        # Arrange.
        # Create the point mask.
        points = faker.tensor((10, 2), min_value=0.0, max_value=1.0)
        values = faker.tensor((10,), min_value=0.0)
        point_mask = heat_maps.make_point_annotation_map(
            points, map_size=tf.constant((100, 100)), point_values=values
        )
        # Expand to 4D.
        point_mask = tf.expand_dims(point_mask, axis=0)
        point_mask = tf.expand_dims(point_mask, axis=-1)

        # Act.
        result = config.peak_layer.call(point_mask)

        # Assert.
        if not config.has_confidence:
            # Maxima at zero are trivial.
            result = tf.logical_and(result, tf.not_equal(point_mask, 0.0))
            # It should choose each point.
            expected_mask = tf.greater(point_mask, 0)
            np.testing.assert_array_equal(
                expected_mask.numpy(), result.numpy()
            )

        else:
            # Maxima at zero are trivial.
            result = result * tf.cast(
                tf.not_equal(point_mask, 0.0), result.dtype
            )
            # It should choose each point.
            np.testing.assert_array_equal(point_mask.numpy(), result.numpy())

    def test_keypoints(self, config: ConfigForTests, faker: Faker) -> None:
        """
        Tests that it detects the maxima of a simple heatmap with some gaussian
        keypoints.

        Args:
            config: The configuration to use for testing.
            faker: The fixture to use for generating fake data.

        """
        # Arrange.
        # Create the point mask.
        points = faker.tensor((10, 2), min_value=0.0, max_value=1.0)
        sigmas = faker.tensor((10,), min_value=3.0, max_value=10.0)
        point_mask = heat_maps.make_point_annotation_map(
            points, map_size=tf.constant((100, 100))
        )
        heatmap = heat_maps.make_heat_map(
            points,
            map_size=tf.constant((100, 100)),
            sigmas=sigmas,
            normalized=False,
        )
        # Expand to 4D.
        heatmap = tf.expand_dims(heatmap, axis=0)
        point_mask = tf.reshape(point_mask, heatmap.shape)

        # Act.
        result = config.peak_layer.call(heatmap)

        # Assert.
        if not config.has_confidence:
            # Maxima at zero are trivial.
            result = tf.logical_and(result, tf.not_equal(heatmap, 0.0))
            # It should choose each point.
            expected_mask = tf.greater(point_mask, 0)
            np.testing.assert_array_equal(
                expected_mask.numpy(), result.numpy()
            )

        else:
            # Maxima at zero are trivial.
            result = result * tf.cast(tf.not_equal(heatmap, 0.0), result.dtype)
            # It should choose each point.
            np.testing.assert_array_equal(point_mask.numpy(), result.numpy())


class TestRoiPoolingLayer:
    """
    Tests for the `RoiPooling` class.
    """

    @dataclass(frozen=True, config=ArbitraryTypesConfig)
    class ConfigForTests:
        """
        Encapsulates standard configuration for most tests.

        Attributes:
            pooling_layer: The `RoiPooling` layer under test.
            pool_size: The size that we are using for the pooled output.

        """

        pooling_layer: pooling.RoiPooling
        pool_size: int

    @classmethod
    @pytest.fixture
    def config(cls, faker: Faker) -> ConfigForTests:
        """
        Generates standard configuration for most tests.

        Args:
            faker: The fixture to use for generating fake data.

        Returns:
            The configuration that it generated.

        """
        pool_size = faker.random_int(max=11)
        layer = pooling.RoiPooling(pool_size=pool_size)

        return cls.ConfigForTests(pooling_layer=layer, pool_size=pool_size)

    def test_pooling(self, config: ConfigForTests, faker: Faker) -> None:
        """
        Tests that the basic ROI pooling operation works.

        Args:
            config: The configuration to use for testing.
            faker: The fixture to use for generating fake data.

        """
        # Arrange.
        batch_size = faker.random_int(min=1, max=10)
        # Create fake image features.
        images = faker.tensor((batch_size, 100, 100, 3))

        # Create fake ROIs.
        row_lengths = [faker.random_int(max=10) for _ in range(batch_size)]
        # Ensure there's at least one zero-length row for testing purposes.
        row_lengths[-1] = 0
        rois = faker.ragged_tensor(row_lengths=row_lengths, inner_shape=(4,))
        # Convert to normalized coordinates.
        rois -= tf.reduce_min(rois)
        rois /= tf.reduce_max(rois)

        # Act.
        roi_crops = config.pooling_layer((images, rois))

        # Assert.
        # We should have the correct number of extracted crops for each image.
        assert list(roi_crops.row_lengths()) == row_lengths
        # The crops should be the correct size.
        assert tuple(roi_crops.shape[-3:-1]) == (
            config.pool_size,
            config.pool_size,
        )
