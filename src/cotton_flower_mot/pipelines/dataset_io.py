from multiprocessing import cpu_count
from typing import Dict, Tuple

import tensorflow as tf

from .schemas import ObjectTrackingFeatures as Otf

_FEATURE_DESCRIPTION = {
    Otf.IMAGE_HEIGHT.value: tf.io.FixedLenFeature([1], tf.dtypes.int64),
    Otf.IMAGE_WIDTH.value: tf.io.FixedLenFeature([1], tf.dtypes.int64),
    Otf.IMAGE_FILENAME.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
    Otf.IMAGE_SOURCE_ID.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
    Otf.IMAGE_ENCODED.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
    Otf.IMAGE_FORMAT.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
    Otf.OBJECT_BBOX_X_MIN.value: tf.io.RaggedFeature(tf.dtypes.float32),
    Otf.OBJECT_BBOX_X_MAX.value: tf.io.RaggedFeature(tf.dtypes.float32),
    Otf.OBJECT_BBOX_Y_MIN.value: tf.io.RaggedFeature(tf.dtypes.float32),
    Otf.OBJECT_BBOX_Y_MAX.value: tf.io.RaggedFeature(tf.dtypes.float32),
    Otf.OBJECT_CLASS_TEXT.value: tf.io.RaggedFeature(tf.dtypes.string),
    Otf.OBJECT_CLASS_LABEL.value: tf.io.RaggedFeature(tf.dtypes.int64),
    Otf.OBJECT_ID.value: tf.io.RaggedFeature(tf.dtypes.int64),
    Otf.IMAGE_SEQUENCE_ID.value: tf.io.RaggedFeature(tf.dtypes.int64),
    Otf.IMAGE_FRAME_NUM.value: tf.io.RaggedFeature(tf.dtypes.int64),
}
"""
Descriptions of the features found in the dataset containing flower annotations.
"""

_INPUT_FEATURES = [Otf.IMAGE_ENCODED.value]
"""
The features that will be used as input to the model.
"""

_NUM_THREADS = cpu_count()
"""
Number of threads to use for multi-threaded operations.
"""


def _decode_jpegs(jpeg_batch: tf.Tensor) -> tf.Tensor:
    """
    Decodes JPEG images from a feature dictionary.

    Args:
        jpeg_batch: The batch of JPEG images.

    Returns:
        The 4D batch of decoded images.

    """
    # This is going to have a batch dimension, so we need to map it.
    return tf.map_fn(
        lambda j: tf.io.decode_jpeg(j[0]),
        jpeg_batch,
        dtype=tf.dtypes.uint8,
        parallel_iterations=_NUM_THREADS,
    )


def _to_bounding_boxes(
    *,
    x_min: tf.RaggedTensor,
    y_min: tf.RaggedTensor,
    x_max: tf.RaggedTensor,
    y_max: tf.RaggedTensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Converts a batch of bounding boxes from the format in TFRecords to a
    single-tensor format.

    Args:
        x_min: The minimum x-coordinates of the bounding boxes.
        y_min: The minimum y-coordinates of the bounding boxes.
        x_max: The maximum x-coordinates of the bounding boxes.
        y_max: The maximum y-coordinates of the bounding boxes.

    Returns:
        - A tensor of shape [N, 4], where N is the total number of bounding
        boxes in the input. The ith row specifies the coordinates of the ith
        bounding box in the form [y1, x1, y2, x2].
        - A tensor of shape [N], where N is the total number of bounding boxes
        in the input. The ith element specifies the index of the image that
        the ith bounding box corresponds to.

    """
    # Figure out the corresponding images for each bounding box.
    row_lengths = x_min.nested_row_lengths()
    # These tensors should have just one ragged dimension.
    row_lengths = tf.ensure_shape(row_lengths, [None])


def _inputs_and_targets_from_feature_dict(
    features: tf.data.Dataset,
) -> tf.data.Dataset:
    """
    Separates inputs and targets into two separate dictionaries for input to
    Keras.

    Args:
        features: The raw (combined) feature dictionary.

    Returns:
        A dictionary with just the inputs and one with just the targets.

    """
    return features.map(lambda f: ())


def inputs_and_targets_from_dataset(
    raw_dataset: tf.data.Dataset, *, batch_size: int = 32
) -> tf.data.Dataset:
    """
    Deserializes raw data from a `Dataset`, and coerces it into the form used by
    the model.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        batch_size: The size of the batches that we generate.

    Returns:
        A dataset that produces input images and target bounding boxes.

    """
    # Deserialize it.
    batched_raw_dataset = raw_dataset.batch(batch_size)
    return batched_raw_dataset.map(
        lambda s: tf.io.parse_example(s, _FEATURE_DESCRIPTION),
        num_parallel_calls=_NUM_THREADS,
    )
