"""
Custom pooling layers.
"""


from typing import Any

import tensorflow as tf
from tensorflow.keras import layers


class PeakLayer(layers.Layer):
    """
    A layer that detects local maxima in an input image. It outputs a mask
    where local maximum pixels are 1, and all other pixels are zero. A pixel
    is considered a peak when it is the maximum of all its neighbors in a 3x3
    neighborhood.
    """

    _DILATION_KERNEL = tf.expand_dims(
        tf.constant([[0, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=tf.float32),
        axis=2,
    )
    """
    Dilation kernel to use for peak finding.
    """

    def __init__(
        self, *args: Any, with_confidence: bool = False, **kwargs: Any
    ):
        """
        Args:
            *args:  Will be forwarded to superclass.
            with_confidence: If true, it will set the positive values in the
                resulting peak mask to the confidence scores for that peak,
                instead of merely one.
            **kwargs: Will be forwarded to superclass.
        """
        super().__init__(*args, **kwargs)

        self._with_confidence = with_confidence

        # Create the sub-layers.
        self._is_peak = layers.Lambda(self._find_peaks, name="peak_finding")

    def _find_peaks(self, image: tf.Tensor) -> tf.Tensor:
        """
        Finds true local peaks within an image. To be a peak, a pixel must
        be greater than all surrounding pixels.

        Args:
            image: The image to find peaks in.

        Returns:
            The peak mask, where peak pixels are set to 1, and all others are
            zero.

        """
        kernel = tf.cast(self._DILATION_KERNEL, image.dtype)

        # Custom dilation for finding peaks.
        dilated = tf.nn.dilation2d(
            image,
            kernel,
            [1, 1, 1, 1],
            "SAME",
            "NHWC",
            [1, 1, 1, 1],
            name="peak_finding_dilation",
        )
        mask = tf.greater(image, dilated)

        if self._with_confidence:
            # Get the actual confidence at those points.
            return image * tf.cast(mask, image.dtype)
        return mask

    def call(self, inputs: tf.Tensor, **_: Any) -> tf.Tensor:
        return self._is_peak(inputs)
