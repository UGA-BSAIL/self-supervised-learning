"""
Utilities for loading supervised image data.
"""


import enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset
from yolov5.utils.dataloaders import LoadImagesAndLabels
from yolov5.utils.general import check_dataset

from .heatmaps import heatmap_from_points


@enum.unique
class Split(enum.Enum):
    """
    Available splits in the dataset.
    """

    TRAINING = "train"
    TESTING = "test"
    VALIDATION = "valid"


class HeatmapDataset(Dataset):
    """
    Dataset that produces heatmaps as labels as well as bounding
    box coordinates. It also adds the offset values to the bounding box
    coordinates.
    """

    def __init__(
        self, image_label_dataset: Dataset, *, heatmap_size: Tuple[int, int]
    ):
        """
        Args:
            image_label_dataset: Dataset producing images and labels that we
                will wrap.
            heatmap_size: The size of the heatmaps to output (h, w).
        """
        self.__dataset = image_label_dataset
        self.__heatmap_size = heatmap_size

    @staticmethod
    def __add_offsets(labels: Tensor, *, image_shape: torch.Size) -> Tensor:
        """
        Computes the offsets to use for given bounding boxes.

        Args:
            labels: The raw labels.
            image_shape: The shape of the input image.

        Returns:
            The same labels, with offset info added.

        """
        centers = labels[:, 2:4]
        image_width_height = torch.as_tensor(image_shape[-2:][::-1])
        centers_px = centers * image_width_height

        # CenterNet downsamples by a factor of 4, so offsets are between
        # 0 and 3 pixels.
        offsets_px = centers_px.to(torch.int) % 4
        offsets_norm = offsets_px.to(torch.float) / image_width_height

        return torch.cat((labels, offsets_norm), dim=1)

    def __len__(self) -> int:
        # Same as the underlying image dataset.
        return len(self.__dataset)

    def __getitem__(
        self, item: int
    ) -> Tuple[Tensor, Tensor, Tensor, str, Optional[Tuple]]:
        # Get the underlying element.
        image, labels, file, shapes = self.__dataset[item]

        # Create the heatmap.
        boxes = labels[:, 2:]
        heatmap = heatmap_from_points(boxes, output_size=self.__heatmap_size)

        labels = self.__add_offsets(labels, image_shape=image.shape)

        return image, heatmap, labels, file, shapes


def get_dataset(
    dataset: Union[Path, str],
    *,
    image_size: int,
    heatmap_size: int,
    batch_size: int,
    augment: bool = True,
    cache: bool = False,
    hyperparams: Dict[str, Any],
    split: Split = Split.TRAINING
) -> Dataset:
    """
    Creates a new `Dataset` for loading the data. This just borrows the
    data loading code from YOLOv5, since that's the most "official" parser
    for YOLO format datasets that I can find.

    Args:
        dataset: The path to the `dataset.yml` file.
        image_size: Square image size to output, in pixels.
        heatmap_size: Square heatmap size to output, in pixels.
        batch_size: The batch size to use.
        augment: Whether to enable data augmentation.
        cache: Whether to enable in-RAM cacheing.
        hyperparams: The hyperparameters to use for data augmentation.
        split: Which split to load.

    Returns:
        The corresponding `DataSet`.

    """
    data_dict = check_dataset(Path(dataset).as_posix())
    split_path = data_dict[split.value]

    base_dataset = LoadImagesAndLabels(
        split_path,
        img_size=image_size,
        batch_size=batch_size,
        augment=augment,
        cache_images=cache,
        hyp=hyperparams,
    )
    # Generate heatmaps as well.
    heatmap_dataset = HeatmapDataset(
        base_dataset, heatmap_size=(heatmap_size,) * 2
    )

    return heatmap_dataset
