"""
Utilities for loading supervised image data.
"""


import enum
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader
from yolov5.utils.dataloaders import create_dataloader


@enum.unique
class Split(enum.Enum):
    """
    Available splits in the dataset.
    """

    TRAINING = ("train",)
    TESTING = ("test",)
    VALIDATION = "valid"


def get_data_loader(
    dataset: Path,
    *,
    hyperparams: Dict[str, Any],
    split: Split = Split.TRAINING
) -> DataLoader:
    """
    Creates a new `DataLoader` for loading the data. This just borrows the
    data loading code from YOLOv5, since that's the most "official" parser
    for YOLO format datasets that I can find.

    Args:
        dataset: The path to the `dataset.yml` file.
        hyperparams: The hyperparameters to use for data augmentation.
        split: Which split to load.

    Returns:
        The corresponding `DataLoader`.

    """
