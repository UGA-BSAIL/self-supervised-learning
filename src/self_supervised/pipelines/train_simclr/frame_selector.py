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
        positive_time_range: Tuple[float, float] = (0.0, 0.5),
        negative_time_range: Tuple[float, float] = (1.5, 2.5),
    ):
        """
        Args:
            mars_metadata: The metadata for the MARS dataset.
            positive_time_range: Specifies how many seconds away from the anchor
                frame we will search for a positive pair.
            negative_time_range: Specifies how many seconds away from the anchor
                frame we will search for a negative pair.

        """
        self.__positive_time_range = positive_time_range
        self.__negative_time_range = negative_time_range
        self.__metadata = self.__filter_short_clips(mars_metadata)

        # Re-index metadata to make the search operations faster.
        self.__metadata.set_index(
            [
                MarsMetadata.CLIP.value,
                MarsMetadata.FRAME_NUM.value,
            ],
            inplace=True,
        )
        # Groups the metadata by camera.
        self.__by_camera = self.__metadata.groupby(MarsMetadata.CAMERA.value)

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
        single_camera = single_camera[
            [MarsMetadata.CLIP.value, MarsMetadata.TIMESTAMP.value]
        ]
        by_clip = single_camera.groupby(MarsMetadata.CLIP.value)
        clip_lengths = by_clip.max() - by_clip.min()

        # Throw out anything that's too short for proper example selection.
        _, negative_example_max = self.__negative_time_range
        short_clips = clip_lengths[clip_lengths < negative_example_max]
        short_clips.dropna(inplace=True)
        metadata = metadata[
            ~metadata[MarsMetadata.CLIP.value].isin(short_clips.index)
        ]
        logger.debug(
            "Have {} clips of sufficient length.",
            len(metadata[MarsMetadata.CLIP.value].unique()),
        )

        return metadata

    @staticmethod
    def __random_row(frame: pd.DataFrame) -> pd.Series:
        """
        Helper function that selects a random row from a `DataFrame`.

        Args:
            frame: The `DataFrame` to get the row from.

        Returns:
            The random row.

        """
        row_index = random.randint(0, len(frame) - 1)
        return frame.iloc[row_index]

    def __get_camera_metadata(
        self, multi_camera: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Gets the camera-specific metadata to use for the anchor, positive, and
        negative examples.

        Args:
            multi_camera: Whether to use different cameras across examples.
                If false, all three cameras will be the same.

        Returns:
            The camera-specific metadata for the anchor, positive, and negative
            examples.

        """
        # Choose the cameras randomly.
        cameras = random.choices(list(self.__by_camera.groups.keys()), k=3)
        anchor_metadata, positive_metadata, negative_metadata = [
            self.__by_camera.get_group(c) for c in cameras
        ]

        if not multi_camera:
            # Use the same camera for all.
            return anchor_metadata, anchor_metadata, anchor_metadata
        else:
            return anchor_metadata, positive_metadata, negative_metadata

    def select_example(
        self, multi_camera: bool = True
    ) -> Tuple[str, str, str]:
        """
        Selects a new training example.

        Args:
            multi_camera: Whether to use different cameras within the same
                triplet. If false, all three examples will use the same camera.

        Returns:
            The file IDs of the anchor frame, positive pair frame, and negative
            pair frame that it selected.

        """
        # Choose the camera randomly.
        (
            anchor_metadata,
            positive_metadata,
            negative_metadata,
        ) = self.__get_camera_metadata(multi_camera=multi_camera)

        # Select the anchor clip randomly.
        anchor_clip, _ = self.__random_row(anchor_metadata).name
        clip_metadata = anchor_metadata.loc[anchor_clip]

        # Select the anchor frame, making sure we leave enough space for
        # positive/negative example generation.
        _, negative_example_max = self.__negative_time_range
        clip_timestamps = clip_metadata[MarsMetadata.TIMESTAMP.value]
        max_anchor_timestamp = clip_timestamps.max() - negative_example_max
        possible_anchor_frames = clip_metadata[
            clip_timestamps <= max_anchor_timestamp
        ]
        anchor_row = self.__random_row(possible_anchor_frames)
        anchor_timestamp = anchor_row[MarsMetadata.TIMESTAMP.value]

        # Choose the positive and negative pairs `from within the same clip.
        anchor_offsets = clip_timestamps - anchor_timestamp
        positive_min, positive_max = self.__positive_time_range
        possible_positive_frames = positive_metadata.loc[anchor_clip][
            anchor_offsets.between(positive_min, positive_max)
        ]
        positive_row = self.__random_row(possible_positive_frames)

        negative_min, negative_max = self.__negative_time_range
        possible_negative_frames = negative_metadata.loc[anchor_clip][
            anchor_offsets.between(negative_min, negative_max)
        ]
        negative_row = self.__random_row(possible_negative_frames)

        id_key = MarsMetadata.FILE_ID.value
        return anchor_row[id_key], positive_row[id_key], negative_row[id_key]
