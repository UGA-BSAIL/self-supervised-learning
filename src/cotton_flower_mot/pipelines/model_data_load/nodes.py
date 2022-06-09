"""
Nodes for the `model_data_load` pipeline.
"""


from typing import Any, Callable, Dict

import tensorflow as tf
from tqdm import tqdm

from ..dataset_io import rot_net_inputs_and_targets_from_dataset


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
