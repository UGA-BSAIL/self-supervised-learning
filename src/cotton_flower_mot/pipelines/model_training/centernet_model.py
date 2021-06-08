"""
Implementation of the CenterNet detector model.
"""


import tensorflow as tf

from ..config import ModelConfig
from .layers import HdaStage, TransitionLayer, UpSamplingIda


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
    # Create initial reduction stages.
    reduced_input = normalized_input
    for i in range(config.num_reduction_stages):
        reduced_input = HdaStage(
            agg_depth=0,
            num_channels=32,
            agg_filter_size=3,
            activation="relu",
            name=f"reduction_stage_{i}",
        )(reduced_input)
        reduced_input = TransitionLayer()(reduced_input)
