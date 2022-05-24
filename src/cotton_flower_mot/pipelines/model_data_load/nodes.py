"""
Nodes for the `model_data_load` pipeline.
"""


from typing import Dict, Callable, Any
import tensorflow as tf
from tqdm import tqdm
from ..dataset_io import rot_net_inputs_and_targets_from_datasets


def build_rotnet_pipeline(
    unannotated_datasets: Dict[str, Callable[[], tf.data.Dataset]],
    **kwargs: Any
) -> tf.data.Dataset:
    """
    Builds the pipeline for loading `RotNet` data.

    Args:
        unannotated_datasets: The raw, unannotated data to use as input.
        **kwargs: Will be forwarded to
            `rot_net_inputs_and_targets_from_datasets`.

    Returns:
        The dataset for training `RotNet`.

    """
    # Load the actual datasets.
    datasets = []
    for loader in tqdm(
        unannotated_datasets.values(), desc="Loading unannotated clips"
    ):
        datasets.append(loader())

    return rot_net_inputs_and_targets_from_datasets(datasets, **kwargs)
