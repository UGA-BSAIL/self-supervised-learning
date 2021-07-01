"""
Custom layers for the CenterNet model.
"""


import itertools
from functools import partial
from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras import layers

from . import HdaStage


class ReductionStages(layers.Layer):
    """
    Wraps initial reduction stages into a single package with reduced memory
    usage.
    """

    def __init__(
        self,
        *,
        num_reduction_stages: int,
        initial_num_channels: int = 8,
        **kwargs: Any,
    ):
        """
        Args:
            num_reduction_stages: The number of reduction stages to use.
            initial_num_channels: Number of channels in the lowest stage,
                which will be doubled in each subsequent stage.
            **kwargs: Will be forwarded to superclass.

        """
        super().__init__(**kwargs)

        self._num_reduction_stages = num_reduction_stages
        self._initial_num_channels = initial_num_channels

        hda_stage = partial(
            HdaStage,
            agg_filter_size=3,
            activation="relu",
        )

        # Create initial reduction stages.
        num_channels = initial_num_channels
        self._reduction_layers = []
        # Use dilation for the first stage.
        dilation_rates = itertools.chain((2,), itertools.repeat(1))
        for i, dilation in zip(range(num_reduction_stages), dilation_rates):
            reduction_stage = hda_stage(
                agg_depth=0,
                num_channels=num_channels,
                dilation_rate=dilation,
                name=f"reduction_stage_{i}",
            )
            pool = layers.MaxPool2D()

            self._reduction_layers.extend((reduction_stage, pool))
            num_channels *= 2

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        # These layers are very big, so use gradient checkpointing here to
        # reduce memory usage.
        @tf.recompute_grad
        def _apply_layers(_inputs: tf.Tensor) -> tf.Tensor:
            outputs = _inputs
            for layer in self._reduction_layers:
                outputs = layer(outputs, **kwargs)

            return outputs

        return _apply_layers(inputs)

    def get_config(self) -> Dict[str, Any]:
        return dict(
            num_reduction_stages=self._num_reduction_stages,
            initial_num_channels=self._initial_num_channels,
            name=self.name,
        )
