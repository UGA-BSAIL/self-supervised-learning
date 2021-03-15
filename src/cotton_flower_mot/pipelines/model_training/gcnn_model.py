"""
Implements a model inspired by GCNNTrack.
https://arxiv.org/pdf/2010.00067.pdf
"""


from typing import Any, Callable

import tensorflow as tf
from loguru import logger
from tensorflow.keras import layers

from .layers.dense import DenseBlock, TransitionLayer


def _bn_relu_conv(
    *args: Any, **kwargs: Any
) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Small helper function that builds a bn-relu-conv block.

    Args:
        *args: Forwarded to `Conv2D()`.
        **kwargs: Forwarded to `Conv2D()`.

    Returns:
        The block, which can be called to apply it to some input, similar to
        a Keras layer.

    """
    conv = layers.Conv2D(*args, **kwargs)
    norm = layers.BatchNormalization()
    relu = layers.Activation("relu")

    def _apply_block(block_input: tf.Tensor) -> tf.Tensor:
        return conv(relu(norm(block_input)))

    return _apply_block


def _build_appearance_feature_extractor(
    normalized_input: tf.Tensor, num_features: int
) -> tf.Tensor:
    """
    Builds a CNN for extracting appearance features from detection images.

    Args:
        normalized_input: The normalized input detections.
        num_features: The number of appearance features we want to output
            per image.

    Returns:
        A batch of corresponding appearance features.

    """
    logger.debug("Appearance features will have length {}.", num_features)

    # Input convolution layers.
    conv1_1 = _bn_relu_conv(48, 3, padding="same")(normalized_input)
    conv1_2 = _bn_relu_conv(48, 3, padding="same")(conv1_1)
    pool1 = layers.MaxPool2D()(conv1_2)

    # Dense blocks.
    dense1 = DenseBlock(3, growth_rate=4)(pool1)
    transition1 = TransitionLayer()(dense1)

    dense2 = DenseBlock(6, growth_rate=4)(transition1)
    transition2 = TransitionLayer()(dense2)

    dense3 = DenseBlock(12, growth_rate=4)(transition2)
    transition3 = TransitionLayer()(dense3)

    dense4 = DenseBlock(8, growth_rate=4)(transition3)

    # Fully-connected layer to generate feature vector.
    fc4_1 = layers.Dense(num_features)(dense4)
    norm4_1 = layers.BatchNormalization()(fc4_1)
    return layers.Activation("relu")(norm4_1)


def _update_affinity_matrix(
    *, affinity_matrix: tf.Tensor, node_features: tf.Tensor
) -> tf.Tensor:
    """
    Updates the affinity matrix for the next layer according to the method
    specified in https://arxiv.org/pdf/2010.00067.pdf.

    Args:
        affinity_matrix: The affinity matrix from the previous layer.
            Should have the shape `[batch_size, n_nodes, n_nodes, 1]`.
        node_features: The node features from the previous layer. Should
            have the shape `[batch_size, n_nodes, n_features]`.

    Returns: The affinity matrix for the next layer.

    """
