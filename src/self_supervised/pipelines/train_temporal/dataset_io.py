from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset

from .frame_selector import FrameSelector


class TripletDataset(Dataset):
    """
    Dataset that produces frame triplets for training.
    """

    def __init__(self, frames: FrameSelector):
        """
        Args:
            frames: The `FrameSelector` used to produce dataset frames.

        """
        self.__frames = frames
        self.__metadata = frames.metadata

    def __len__(self) -> int:
        return self.__frames.num_anchor_frames

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            index: The index of the anchor frame from the dataset.

        Returns:

        """
