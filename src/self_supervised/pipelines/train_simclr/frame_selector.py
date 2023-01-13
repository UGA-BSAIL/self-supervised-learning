"""
Class that helps with selecting frame pairs for the self-supervised learning
task.
"""


import pandas as pd
from typing import Tuple
from ..schemas import MarsMetadata
from loguru import logger
import random


class FrameSelector:
    """
    Class that helps with selecting frame pairs for the self-supervised learning
    task.
    """

    def __init__(
        self,
        mars_metadata: pd.DataFrame,
        positive_frame_range: Tuple[int, int] = (1, 11),
        negative_frame_range: Tuple[int, int] = (30, 51),
    ):
        """
        Args:
            mars_metadata: The metadata for the MARS dataset.
            positive_frame_range: Specifies how many frames away from the anchor
                frame we will search for a positive pair.
            negative_frame_range: Specifies how many frames away from the anchor
                frame we will search for a negative pair.

        """
        self.__metadata = self.__filter_short_clips(mars_metadata)
        self.__positive_frame_range = positive_frame_range
        self.__negative_frame_range = negative_frame_range

        # Re-index metadata to make the search operations faster.
        self.__metadata.set_index(
            [
                MarsMetadata.CLIP.value,
                MarsMetadata.FRAME_NUM.value,
                MarsMetadata.CAMERA.value,
            ],
            inplace=True,
        )

    def __filter_short_clips(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Filters clips from the dataset that are too short for proper example
        generation.

        Args:
            metadata: The metadata for the dataset.

        Returns:
            The filtered metadata.

        """
        # Compute lengths for each clip. Limit to a single camera, so we get
        # correct lengths. (All cameras should be synchronized and thus have
        # the same number of frames.)
        single_camera = metadata[metadata[MarsMetadata.CAMERA.value] == 0]
        by_clip = single_camera.groupby(MarsMetadata.CLIP.value)
        clip_lengths = by_clip.size()

        # Throw out anything that's too short for proper example selection.
        _, negative_example_max = self.__negative_frame_range
        short_clips = clip_lengths[clip_lengths < negative_example_max]
        metadata = metadata[
            not metadata[MarsMetadata.CLIP.value].isin(short_clips.index)
        ]
        logger.debug(
            "Have {} clips of sufficient length.",
            len(metadata[MarsMetadata.CLIP.value].unique()),
        )

        return metadata

    def select_example(self) -> Tuple[str, str, str]:
        """
        Selects a new training example.

        Returns:
            The file IDs of the anchor frame, positive pair frame, and negative
            pair frame that it selected.

        """
        # Select the anchor frame randomly.
        anchor_index = random.randint(0, len(self.__metadata))
        anchor_row = self.__metadata.iloc[anchor_index]

        # Choose the positive and negative pairs from within the same clip.
        anchor_clip, anchor_frame, _ = anchor_row
        clip_frames = self.__metadata.loc[anchor_clip]

        positive_index = random.randint(*self.__positive_frame_range)
        positive_row = clip_frames.iloc[positive_index]
        negative_index = random.randint(*self.__negative_frame_range)
        negative_row = clip_frames.iloc[negative_index]
