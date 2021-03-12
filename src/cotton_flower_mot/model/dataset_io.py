from multiprocessing import cpu_count

import tensorflow as tf

_FEATURE_DESCRIPTION = {
    "image/height": tf.io.FixedLenFeature([1], tf.dtypes.int64),
    "image/width": tf.io.FixedLenFeature([1], tf.dtypes.int64),
    "image/filename": tf.io.FixedLenFeature([1], tf.dtypes.string),
    "image/source_id": tf.io.FixedLenFeature([1], tf.dtypes.string),
    "image/encoded": tf.io.FixedLenFeature([1], tf.dtypes.string),
    "image/format": tf.io.FixedLenFeature([1], tf.dtypes.string),
    "image/object/bbox/xmin": tf.io.RaggedFeature(tf.dtypes.float32),
    "image/object/bbox/xmax": tf.io.RaggedFeature(tf.dtypes.float32),
    "image/object/bbox/ymin": tf.io.RaggedFeature(tf.dtypes.float32),
    "image/object/bbox/ymax": tf.io.RaggedFeature(tf.dtypes.float32),
    "image/object/class/text": tf.io.RaggedFeature(tf.dtypes.string),
    "image/object/class/label": tf.io.RaggedFeature(tf.dtypes.int64),
}
"""
Descriptions of the features found in the dataset containing flower annotations.
"""

_NUM_THREADS = cpu_count()
"""
Number of threads to use for multi-threaded operations.
"""


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
