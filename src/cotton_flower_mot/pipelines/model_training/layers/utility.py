"""
Common utility layers used in various places.
"""


from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow.keras import layers


class BnReluConv(layers.Layer):
    """
    Small helper layer that applies batch normalization, ReLU, and convolution
    in that order.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Args:
            *args: Will be forwarded to `Conv2D()`.
            **kwargs: Will be forwarded to `Conv2D()`.
        """
        super().__init__()

        self._conv_args = args
        self._conv_kwargs = kwargs

        # Initialize the sub-layers.
        self._conv = layers.Conv2D(*args, **kwargs)
        self._norm = layers.BatchNormalization()
        self._relu = layers.Activation("relu")

    def call(
        self, inputs: tf.Tensor, training: Optional[bool] = None
    ) -> tf.Tensor:
        return self._conv(self._relu(self._norm(inputs, training=training)))

    def get_config(self) -> Dict[str, Any]:
        return dict(conv_args=self._conv_args, conv_kwargs=self._conv_kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BnReluConv":
        return cls(*config["conv_args"], **config["conv_kwargs"])


class BnReluDense(layers.Layer):
    """
    Small helper layer that applies batch normalization, ReLU, and a dense
    layer in that order.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Args:
            *args: Will be forwarded to `Dense()`.
            **kwargs: Will be forwarded to `Dense()`.
        """
        super().__init__()

        self._dense_args = args
        self._dense_kwargs = kwargs

        # Initialize the sub-layers.
        self._dense = layers.Dense(*args, **kwargs)
        self._norm = layers.BatchNormalization()
        self._relu = layers.Activation("relu")

    def call(
        self, inputs: tf.Tensor, training: Optional[bool] = None
    ) -> tf.Tensor:
        return self._dense(self._relu(self._norm(inputs, training=training)))

    def get_config(self) -> Dict[str, Any]:
        return dict(
            dense_args=self._dense_args, dense_kwargs=self._dense_kwargs
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BnReluDense":
        return cls(*config["dense_args"], **config["dense_kwargs"])
