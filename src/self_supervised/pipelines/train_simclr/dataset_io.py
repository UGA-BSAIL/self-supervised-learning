"""
Utilities for loading the image data.
"""


from torch.utils.data import Dataset
import pandas as pd
from ..schemas import MarsMetadata
from pathlib import Path
from loguru import logger
from torchvision.io import read_image
from torch import Tensor
from torch import nn
import torch


class SingleFrameDataset(Dataset):
    """
    Dataset that just reads single frames.
    """

    def __init__(self, *, mars_metadata: pd.DataFrame, image_folder: Path):
        """
        Args:
            mars_metadata: The metadata, which describes where to find the
                dataset files.
            image_folder: The folder that contains all the dataset images.

        """
        self.__metadata = mars_metadata
        logger.info("Loading dataset images from {}.", image_folder)
        self.__image_folder = image_folder

    def __len__(self) -> int:
        return len(self.__metadata)

    def __getitem__(self, item: int) -> Tensor:
        """
        Args:
            item: The index for the item to get.

        Returns:
            The corresponding image from the dataset.

        """
        # Figure out which file we should read.
        example_row = self.__metadata.iloc[item]
        file_id = example_row[MarsMetadata.FILE_ID.value]
        file_path = self.__image_folder / f"{file_id}.jpg"
        return read_image(file_path.as_posix())


class PairedAugmentedDataset(Dataset):
    """
    Reads images, and provides two different augmentations of the same image.
    """

    def __init__(
        self,
        *,
        image_dataset: Dataset,
        augmentation: nn.Module,
    ):
        """

        Args:
            image_dataset: The dataset to draw raw images from. Should produce
                single images.
            augmentation: The random augmentation module to apply to the images.

        """
        self.__dataset = image_dataset
        self.__augmentation = augmentation

    def __len__(self) -> int:
        # Same as the underlying image dataset.
        return len(self.__dataset)

    def __getitem__(self, item: int) -> Tensor:
        """
        Args:
            item: The index of the image in the dataset.

        Returns:
            The two augmentations for this image, as a tensor with a batch
            dimension of size 2.

        """
        image = self.__dataset[item]

        # Perform two different augmentations.
        augmentation_1 = self.__augmentation(image)
        augmentation_2 = self.__augmentation(image)

        return torch.stack((augmentation_1, augmentation_2))
