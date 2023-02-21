"""
Class that helps with selecting frame pairs for the self-supervised learning
task.
"""


import random
from functools import cached_property
from typing import Tuple

import pandas as pd
from loguru import logger
from pandas.core.groupby.generic import DataFrameGroupBy

from .schemas import MarsMetadata


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
        row_index = random.randrange(0, len(frame))
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
        if multi_camera:
            # Choose the cameras randomly.
            cameras = random.choices(list(self.__by_camera.groups.keys()), k=3)
        else:
            # Use the same camera for all.
            cameras = (
                random.choices(list(self.__by_camera.groups.keys()), k=1) * 3
            )

        # Draw anchor metadata from the set of possible anchor frames.
        anchor_metadata = self.__anchors_by_camera.get_group(cameras[0])
        positive_metadata, negative_metadata = [
            self.__by_camera.get_group(c) for c in cameras[1:]
        ]

        return anchor_metadata, positive_metadata, negative_metadata

    @cached_property
    def __possible_anchor_frames(self) -> pd.DataFrame:
        """
        Returns:
            A subset of the metadata frames that can reasonably be used
            as anchor frames.

        """
        possible_anchor_frames = []

        clips = self.__metadata.index.to_frame()[
            MarsMetadata.CLIP.value
        ].unique()
        for clip in clips:
            clip_metadata = self.__metadata.xs(clip, drop_level=False)

            # Find the frames that we can use as anchors while still having
            # enough space to select a negative example.
            _, negative_example_max = self.__negative_time_range
            clip_timestamps = clip_metadata[MarsMetadata.TIMESTAMP.value]
            max_anchor_timestamp = clip_timestamps.max() - negative_example_max
            possible_anchor_frames.append(
                clip_metadata[clip_timestamps < max_anchor_timestamp]
            )

        return pd.concat(possible_anchor_frames)

    @cached_property
    def __by_camera(self) -> DataFrameGroupBy:
        """
        Returns:
            The metadata grouped by camera.

        """
        return self.__metadata.groupby(MarsMetadata.CAMERA.value)

    @cached_property
    def __anchors_by_camera(self) -> DataFrameGroupBy:
        """
        Returns:
            The possible anchor frames grouped by camera.

        """
        return self.__possible_anchor_frames.groupby(MarsMetadata.CAMERA.value)

    @cached_property
    def num_anchor_frames(self) -> int:
        """
        Returns:
            The total number of unique anchor frames in this dataset. Frames
            visible from multiple cameras are counted once.

        """
        # Get data from a single camera.
        camera = list(self.__anchors_by_camera.groups.keys())[0]
        camera_data = self.__anchors_by_camera.get_group(camera)

        return len(camera_data)

    @cached_property
    def num_frames(self) -> int:
        """
        Returns:
            The total number of frames in this dataset. Frames visible
            from multiple cameras are counted once.

        """
        # Get data from a single camera.
        camera = list(self.__by_camera.groups.keys())[0]
        camera_data = self.__by_camera.get_group(camera)

        return len(camera_data)

    @property
    def metadata(self) -> pd.DataFrame:
        """
        Returns:
            The underlying metadata.

        """
        return self.__metadata.copy()

    def get_triplet(
        self, *, anchor_index: int, multi_camera: bool = True
    ) -> Tuple[str, str, str]:
        """
        Selects a triplet using a specific anchor frame.

        Args:
            anchor_index: The index of the anchor frame to use.
            multi_camera: Whether to use difference cameras within the same
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

        # Select the anchor frame.
        anchor_row = anchor_metadata.iloc[anchor_index]
        anchor_clip, _ = anchor_row.name
        anchor_timestamp = anchor_row[MarsMetadata.TIMESTAMP.value]

        clip_metadata = positive_metadata.loc[anchor_clip]
        clip_timestamps = clip_metadata[MarsMetadata.TIMESTAMP.value]

        # Choose the positive and negative pairs from within the same clip.
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

    def get_pair(self, frame_index: int) -> Tuple[str, str]:
        """
        Selects a pair (the same image from multiple cameras) using a specific
        frame. Cameras will be chosen randomly.

        Args:
            frame_index: The index of the frame.

        Returns:
            The file IDs of the two frames it selected.

        """
        # Choose the cameras randomly.
        camera1, camera2 = random.choices(
            list(self.__by_camera.groups.keys()), k=2
        )
        metadata1 = self.__by_camera.get_group(camera1)
        metadata2 = self.__by_camera.get_group(camera2)

        # Select the frames.
        row_index = random.randrange(0, len(metadata1))
        row1 = metadata1.iloc[row_index]
        row2 = metadata2.iloc[row_index]

        id_key = MarsMetadata.FILE_ID.value
        return row1[id_key], row2[id_key]

    def select_random_triplet(
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
        return self.get_triplet(
            anchor_index=random.randint(0, self.num_anchor_frames),
            multi_camera=multi_camera,
        )
