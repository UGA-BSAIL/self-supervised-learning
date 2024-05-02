"""
Nodes for the `recompute_flower_numbers` pipeline.
"""


from pathlib import Path

import pandas as pd
from tqdm import trange
from ultralytics import YOLO

from ..common_nodes import num_flowers_in_image
from ..schemas import MarsMetadata
from ..train_simclr.dataset_io import SingleFrameDataset


def recompute_flower_numbers(
    *, mars_metadata: pd.DataFrame, image_folder: Path, detector: YOLO
) -> pd.DataFrame:
    """
    Recomputes the flower numbers for the entire dataset using the specified
    flower detection model.

    Args:
        mars_metadata: The metadata for the dataset.
        image_folder: The folder containing the dataset images.
        detector: The detection model to use.

    Returns:
        The updated metadata for the dataset.

    """
    dataset = SingleFrameDataset(
        mars_metadata=mars_metadata, image_folder=image_folder
    )
    new_metadata = dataset.metadata
    flower_column = new_metadata.columns.to_list().index(
        MarsMetadata.NUM_FLOWERS.value
    )

    for i in trange(len(dataset)):
        image = dataset[i]
        num_flowers = num_flowers_in_image(
            image.permute(1, 2, 0).numpy(), detector=detector
        )

        new_metadata.iloc[i, flower_column] = num_flowers

    return new_metadata
