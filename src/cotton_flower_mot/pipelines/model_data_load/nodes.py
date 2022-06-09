"""
Nodes for the `model_data_load` pipeline.
"""


from typing import Any

import tensorflow as tf


def concat_datasets(*args: Any) -> tf.data.Dataset:
    """
    Concatenates the specified datasets.

    Args:
        *args: The datasets to concatenate.

    Returns:
        The concatenated dataset.

    """
    combined = args[0]
    for dataset in args[1:]:
        combined = combined.concatenate(dataset)

    return combined
