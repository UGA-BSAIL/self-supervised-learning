"""
Utilities for loading the image data.
"""


from torch.utils.data import Dataset
from .frame_selector import FrameSelector


class MarsDataset(Dataset):
    """
    Torch dataset that loads MARS data.
    """

    def __init__(self, frame_selector: FrameSelector):
        """
        Args:
            frame_selector: Used for loading
        """