import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Cropping2D,
    Dropout,
    ReLU,
    UpSampling2D,
)
from tensorflow.python.keras.regularizers import l2

from ...config import ModelConfig


def efficientnet(
    *,
    image_input: tf.Tensor,
    config: ModelConfig,
    pretrained: bool = True,
) -> tf.Tensor:
    """
    Creates a new EfficientNet-based model.

    Args:
        image_input: The tensor containing (non-normalized) input images.
        config: The model configuration to use.
        pretrained: If true, it will use pretrained `ImageNet` weights and
            fix those portions of the network. Otherwise, it will initialize
            randomly, and the whole thing will be trainable.

    Returns:
        The top layer of the model.

    """
    efficientnet = EfficientNetV2S(
        include_top=False,
        input_tensor=image_input,
        input_shape=config.detection_model_input_shape,
        weights="imagenet" if pretrained else None,
    )
    efficientnet.trainable = not pretrained

    block2 = efficientnet.get_layer("block2d_add").get_output_at(0)
    block3 = efficientnet.get_layer("block3d_add").get_output_at(0)
    block5 = efficientnet.get_layer("block5i_add").get_output_at(0)
    top = efficientnet.get_layer("top_activation").get_output_at(0)

    top = Dropout(rate=0.5)(top)
    block5 = Dropout(rate=0.4)(block5)
    block3 = Dropout(rate=0.3)(block3)
    block2 = Dropout(rate=0.2)(block2)
    x = top

    # decoder
    x = Conv2D(
        256,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(UpSampling2D()(x))
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([block5, x])
    x = Conv2D(
        256,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(
        128,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(UpSampling2D()(x))
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([block3, x])
    x = Conv2D(
        128,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(
        64,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(UpSampling2D()(x))
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Because of the way the resnet layer sizes shake out, we need to add
    # some slight cropping here.
    x = Cropping2D(cropping=((1, 0), (0, 0)))(x)
    x = Concatenate()([block2, x])
    x = Conv2D(
        64,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(5e-4),
    )(x)
    x = BatchNormalization()(x)
    return ReLU()(x)
