"""
Implementation of the CenterNet detector model.
"""


from functools import partial
from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers

from ..config import ModelConfig
from ..schemas import ModelInputs, ModelTargets
from .layers import BnActConv, HdaStage, TransitionLayer, UpSamplingIda

# Use mixed precision to speed up training.
tf.keras.mixed_precision.set_global_policy("mixed_float16")


def _build_backbone(
    normalized_input: tf.Tensor, *, config: ModelConfig
) -> tf.Tensor:
    """
    Builds the backbone for CenterNet, which in this case, is based on DLA.

    Args:
        normalized_input: The normalized input images.
        config: Model configuration.

    Returns:
        The batch of extracted features.

    """
    hda_stage = partial(HdaStage, agg_filter_size=3, activation="relu")

    # Create initial reduction stages.
    reduced_input = normalized_input
    num_channels = 3
    for i in range(config.num_reduction_stages):
        reduced_input = hda_stage(
            agg_depth=0,
            num_channels=num_channels,
            name=f"reduction_stage_{i}",
        )(reduced_input)
        reduced_input = layers.MaxPool2D()(reduced_input)

        num_channels *= 2

    # Create the main stages of the model.
    hda1 = hda_stage(
        agg_depth=1,
        num_channels=64,
        name="hda_stage_1",
    )(reduced_input)
    transition1 = TransitionLayer()(hda1)
    hda2 = hda_stage(
        agg_depth=2,
        num_channels=128,
        name="hda_stage_2",
    )(transition1)
    transition2 = TransitionLayer()(hda2)
    hda3 = hda_stage(
        agg_depth=2,
        num_channels=256,
        name="hda_stage_3",
    )(transition2)
    transition3 = TransitionLayer()(hda3)
    hda4 = hda_stage(
        agg_depth=1,
        num_channels=512,
        name="hda_stage_4",
    )(transition3)

    # Create the decoder side.
    return UpSamplingIda(
        agg_filter_size=3, activation="relu", name="up_sample"
    )((hda1, hda2, hda3, hda4))


def _build_prediction_head(
    features: tf.Tensor, *, output_channels: int, name: Optional[str] = None
) -> tf.Tensor:
    """
    Builds a prediction head for the model.

    Args:
        features: The features from the backbone.
        output_channels: The number of output channels we want.
        name: Name for the output layer.

    Returns:
        The prediction maps.

    """
    conv1_1 = BnActConv(64, 3, activation="relu", padding="same")(features)
    conv1_2 = BnActConv(64, 1, activation="relu", padding="same")(conv1_1)
    return BnActConv(output_channels, 1, padding="same", name=name)(conv1_2)


def build_model(config: ModelConfig) -> tf.keras.Model:
    """
    Builds the detection model.

    Args:
        config: The model configuration.

    Returns:
        The model that it created.

    """
    images = layers.Input(
        shape=(1080, 1920, 3), name=ModelInputs.DETECTIONS_FRAME.value
    )

    def _normalize(_images: tf.Tensor) -> tf.Tensor:
        # Normalize the images before putting them through the model.
        float_images = tf.cast(_images, tf.keras.backend.floatx())
        return tf.image.per_image_standardization(float_images)

    normalized = layers.Lambda(_normalize, name="normalize")(images)

    # Build the model.
    features = _build_backbone(normalized, config=config)
    heatmap = _build_prediction_head(features, output_channels=1)
    sizes = _build_prediction_head(features, output_channels=2)
    offsets = _build_prediction_head(features, output_channels=2)

    # The loss expects sizes and offsets to be merged.
    geometry = layers.Concatenate(
        name=ModelTargets.GEOMETRY.value, dtype=tf.float32
    )((sizes, offsets))
    # Since we treat heatmaps like a pixel-wise classification problem,
    # apply sigmoid activation.
    heatmap = layers.Activation(
        "sigmoid", name=ModelTargets.HEATMAP.value, dtype=tf.float32
    )(heatmap)

    return tf.keras.Model(inputs=images, outputs=[heatmap, geometry])
