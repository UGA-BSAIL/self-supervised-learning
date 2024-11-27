"""
Utilities for loading the image data.
"""


import itertools
import random
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import pandas as pd
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
        downsample_size: Optional[int] = None,
        augmentation: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            mars_metadata: The metadata, which describes where to find the
                dataset files.
            image_folder: The folder that contains all the dataset images.
            downsample_size: If specified, will downsample the dataset to
                have at most this number of frames.
            augmentation: Augmentation to apply to output images. Defaults to
                nothing.

        """
        self.__metadata = mars_metadata
        logger.info("Loading dataset images from {}.", image_folder)
        self.__image_folder = Path(image_folder)
        self.augmentation = augmentation

        if downsample_size is not None:
            self.__metadata = self.__sample_data_set(downsample_size)

    def __sample_data_set(self, dataset_size: int) -> pd.DataFrame:
        """
        Randomly downsamples the dataset.

        Args:
            dataset_size: The desired total size of the dataset.

        Returns:
            The metadata for the sampled dataset.

        """
        if dataset_size > len(self.__metadata):
            # Don't sample.
            return self.__metadata

        # Start with all the images containing flowers.
        flower_data = self.__metadata[
            self.__metadata[MarsMetadata.NUM_FLOWERS.value] > 0
        ]
        logger.debug("Selecting {} images with flowers.", len(flower_data))

        # Sample the rest at random.
        num_to_sample = max(dataset_size - len(flower_data), 0)
        sampled_data = self.__metadata.sample(n=num_to_sample, random_state=0)

        sampled_data = pd.concat(
            [flower_data, sampled_data], ignore_index=True
        )
        logger.debug("Downsampled dataset to {} examples.", len(sampled_data))
        return sampled_data

    def __len__(self) -> int:
        return len(self.__metadata)

    @property
    def metadata(self) -> pd.DataFrame:
        return self.__metadata.copy()

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

        if self.augmentation is not None:
            # Apply the augmentation.
            image = self.augmentation(image)
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
        self.single_frame_dataset = image_dataset
        self.augmentation = augmentation

    def __len__(self) -> int:
        # Same as the underlying image dataset.
        return len(self.single_frame_dataset)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            item: The index of the image in the dataset.

        Returns:
            The two augmentations for this image, as a tensor with a batch
            dimension of size 2.

        """
        image = self.single_frame_dataset[item]

        # Perform two different augmentations.
        augmentation_1 = self.augmentation(image, data_index=item)
        augmentation_2 = self.augmentation(image, data_index=item)

        return augmentation_1, augmentation_2


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
        views: int | Sequence[int] | None = None,
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
            views: If this is a sequence, it should specify the exact views
                to use. If it is a single integer, it will select that many
                views randomly. By default, it will produce all the views
                available.

        """
        self.__frames = frames
        logger.info("Loading dataset images from {}.", image_folder)
        self.__image_folder = image_folder
        self.augmentation = augmentation
        self.__decode_device = decode_device
        self.__max_jitter = max_jitter
        self.__views = views

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
        return self.augmentation(image)

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
        if type(self.__views) is int:
            # Select views randomly.
            frame_ids = random.choices(frame_ids, k=self.__views)
        elif self.__views is not None:
            # Select specific views.
            try:
                frame_ids = [frame_ids[i] for i in self.__views]
            except IndexError:
                # Fall back on choosing frames randomly if these cameras are
                # not present in this example.
                frame_ids = random.choices(frame_ids, k=len(self.__views))

        # Read the images.
        return [self.__read_single_image(f) for f in frame_ids]


class TemporalMultiViewDataset(MultiViewDataset):
    """
    A variation of the `MultiViewDataset` that also includes nearby
    frames as additional views.
    """

    def __init__(
        self,
        *args: Any,
        frame_step_range: Tuple[int, int] = (-3, 3),
        num_extra_views: int,
        **kwargs: Any,
    ):
        """

        Args:
            *args: Will be forwarded to the superclass.
            frame_step_range: Minimum and maximum number of frames to step
                forward in time to generate additional views.
            num_extra_views: The number of extra views to generate.
            **kwargs: Will be forwarded to the superclass.

        """
        super().__init__(*args, **kwargs)

        self.__frame_step_range = frame_step_range
        self.__num_extra_views = num_extra_views

    def __get_nearby_index(self, base_index: int) -> int:
        """
        Gets a nearby index to use for additional views.

        Args:
            base_index: The base index we are sampling from.

        Returns:
            The new index to use.

        """
        index = base_index + random.randint(*self.__frame_step_range)
        index = min(index, len(self) - 1)
        index = max(0, index)

        return index

    def __getitem__(self, index: int) -> List[Tensor]:
        views = super().__getitem__(index)

        other_view_indices = [
            self.__get_nearby_index(index)
            for _ in range(self.__num_extra_views)
        ]

        # This sampling strategy is to ensure we pull evenly from all the
        # cameras.
        camera_indices = list(range(len(views)))
        random.shuffle(camera_indices)
        camera_indices = itertools.cycle(camera_indices)

        for camera_index, frame_index in zip(
            camera_indices, other_view_indices
        ):
            other_views = super().__getitem__(frame_index)
            views.append(other_views[camera_index])

        return views
