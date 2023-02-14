from pathlib import Path
from typing import Callable, Tuple

from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image

from .frame_selector import FrameSelector


class TripletDataset(Dataset):
    """
    Dataset that produces frame triplets for training.
    """

    def __init__(
        self,
        frames: FrameSelector,
        *,
        image_folder: Path,
        multi_camera: bool = True,
        augmentation: Callable[[Tensor], Tensor] = lambda x: x,
    ):
        """
        Args:
            frames: The `FrameSelector` used to produce dataset frames.
            image_folder: The folder that contains all the dataset images.
            multi_camera: Whether to allow frames from different cameras in
                the same triplet.
            augmentation: The data augmentation to apply to all images.

        """
        self.__frames = frames
        logger.info("Loading dataset images from {}.", image_folder)
        self.__image_folder = image_folder
        self.__multi_camera = multi_camera
        self.__augmentation = augmentation

    def __len__(self) -> int:
        return self.__frames.num_anchor_frames

    def __read_image(self, file_id: str) -> Tensor:
        """
        Reads the specified image file from the disk, with data
        augmentation applied.

        Args:
            file_id: The ID of the image to read.

        Returns:
            The image that it read.

        """
        file_path = self.__image_folder / f"{file_id}.jpg"
        image = read_image(file_path.as_posix())
        return self.__augmentation(image)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            index: The index of the anchor frame from the dataset.

        Returns:
            The anchor, positive, and negative examples.

        """
        anchor_id, positive_id, negative_id = self.__frames.get_example(
            anchor_index=index, multi_camera=self.__multi_camera
        )

        # Read the relevant image files.
        return (
            self.__read_image(anchor_id),
            self.__read_image(positive_id),
            self.__read_image(negative_id),
        )
