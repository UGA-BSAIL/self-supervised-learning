"""
Utilities for loading the image data.
"""


from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg, read_file, read_image

from ..frame_selector import FrameSelector
from ..schemas import MarsMetadata


class SingleFrameDataset(Dataset):
    """
    Dataset that just reads single frames.
    """

    def __init__(
        self,
        *,
        mars_metadata: pd.DataFrame,
        image_folder: Path,
        augmentation: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        """
        Args:
            mars_metadata: The metadata, which describes where to find the
                dataset files.
            image_folder: The folder that contains all the dataset images.
            augmentation: Augmentation to apply to output images. Defaults to
                nothing.

        """
        self.__metadata = mars_metadata
        logger.info("Loading dataset images from {}.", image_folder)
        self.__image_folder = image_folder
        self.__augmentation = augmentation

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
        image = read_image(file_path.as_posix())

        if self.__augmentation is not None:
            # Apply the augmentation.
            image = self.__augmentation(image)
        return image


class PairedAugmentedDataset(Dataset):
    """
    Reads images, and provides two different augmentations of the same image.
    """

    def __init__(
        self,
        *,
        image_dataset: Dataset,
        augmentation: Callable[[Tensor], Tensor],
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


class MultiViewDataset(Dataset):
    """
    Creates image pairs from multiple corresponding camera views.
    """

    def __init__(
        self,
        *,
        frames: FrameSelector,
        image_folder: Path,
        augmentation: Callable[[Tensor], Tensor] = lambda x: x,
        max_jitter: int = 0,
        decode_device: str = "cpu",
    ):
        """
        Args:
            frames: The frame selector to use for extracting frames.
            image_folder: The folder that contains all the dataset images.
            augmentation: The data augmentation to apply to the frames.
            max_jitter: Maximum number of frames to jitter the camera views
                by, in either direction. This can add some more variation to
                the data.
            decode_device: The device to use for decoding images.

        """
        self.__frames = frames
        logger.info("Loading dataset images from {}.", image_folder)
        self.__image_folder = image_folder
        self.__augmentation = augmentation
        self.__decode_device = decode_device
        self.__max_jitter = max_jitter

    def __len__(self) -> int:
        return self.__frames.num_frames

    def __read_single_image(self, file_id: str) -> Tensor:
        """
        Reads a single image from the dataset.

        Args:
            file_id: The file ID of the image to read.

        Returns:
            The image that it read.

        """
        file_path = self.__image_folder / f"{file_id}.jpg"
        image_compressed = read_file(file_path.as_posix())
        image = decode_jpeg(image_compressed, device=self.__decode_device)

        # Apply augmentations.
        return self.__augmentation(image)

    def __getitem__(self, index: int) -> List[Tensor]:
        """
        Args:
            index: The index of the image in the dataset.

        Returns:
            The frames for this example, as a tensor with a batch
            dimension.

        """
        frame_ids = self.__frames.get_all_views(
            index, jitter_by=self.__max_jitter
        )

        # Read the images.
        return [self.__read_single_image(f) for f in frame_ids]
