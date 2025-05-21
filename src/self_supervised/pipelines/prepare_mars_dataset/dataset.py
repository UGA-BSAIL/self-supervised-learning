"""
Standard interface for interacting with MARS datasets.
"""


import abc
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
from loguru import logger

LazyFrame = Callable[[], np.array]
"""
Type alias for a function that returns a frame.
"""


class Camera(abc.ABC):
    """
    Represents data corresponding to a single camera.
    """

    @abc.abstractmethod
    def frame_at_timestamp(
        self, timestamp: float, tolerance: float = 0.05
    ) -> LazyFrame:
        """
        Gets the frame from the video that's closest to a particular timestamp.

        Args:
            timestamp: The timestamp that we are looking for the frame at.
            tolerance: Maximum difference between the desired timestamp and the
                actual frame timestamp that we allow.

        Returns:
            The corresponding frame image.

        Raises:
            `KeyError` if it cannot find a frame at this timestamp.

        """

    @property
    @abc.abstractmethod
    def path(self) -> Path:
        """
        Returns:
            The path to the raw camera data.

        """

    @abc.abstractmethod
    def frames(self) -> Iterable[Tuple[float, LazyFrame]]:
        """
        Yields:
            The timestamp and corresponding frame for each frame of the video,
            in order.

        """


class Session(abc.ABC):
    """
    Represents a session in the dataset.
    """

    @property
    @abc.abstractmethod
    def cameras(self) -> List[Camera]:
        """
        Returns:
            The camera data from this session.

        """

    @property
    @abc.abstractmethod
    def session_folder(self) -> Path:
        """
        Returns:
            The folder where the session files are stored.

        """

    def synchronized_frames(
        self, tolerance: float = 0.05
    ) -> Iterable[Tuple[float, List[LazyFrame]]]:
        """
        Iterates through all the frames in this session, ensuring that all
        frames produced correspond for every camera in the session. Frames
        from one camera that don't match all the others will be dropped.

        Args:
            tolerance: The time tolerance to use when finding matching frames.

        Yields:
            The timestamp, and corresponding frames at that timestamp. If not
            all cameras have frames at a particular timestamp, it will just
            yield the timestamp and an empty list of frames.

        """
        # We'll use one camera as an anchor.
        anchor_camera = self.cameras[0]
        other_cameras = self.cameras[1:]

        for timestamp, frame in anchor_camera.frames():
            # Get corresponding frames from other cameras.
            frames = [frame]
            for camera in other_cameras:
                try:
                    frames.append(
                        camera.frame_at_timestamp(
                            timestamp, tolerance=tolerance
                        )
                    )
                except KeyError:
                    logger.debug(
                        "No corresponding frame from {} at {}.",
                        camera.path,
                        timestamp,
                    )
                    # We only want to yield images if ALL cameras have
                    # corresponding frames, so there's no point in checking the
                    # other ones in this case.
                    frames = []
                    break

            yield timestamp, frames


class Dataset(abc.ABC):
    """
    Represents a complete collection of MARS data.
    """

    def __init__(self, *, sessions: List[Session]):
        """
        Args:
            sessions: The data for each session in the dataset.

        """
        self.__sessions = sessions

    @property
    def sessions(self) -> List[Session]:
        """
        Returns:
            The sessions in this dataset.

        """
        return self.__sessions[:]


class YamlRep(abc.ABC):
    """
    Class that represents data from the YAML file.
    """

    @classmethod
    @abc.abstractmethod
    def from_yaml(
        cls,
        spec: Dict[str, Any],
        **kwargs: Any,
    ) -> "YamlRep":
        """
        Creates a new instance of this class from a YAML specification.

        Args:
            spec: The YAML specification to use.
            **kwargs: Custom arguments that can be specified by subclasses.

        Returns:
            The instance it created.

        """


def merge_datasets(*datasets: Dataset) -> Dataset:
    """
    Merges multiple datasets into a single one.

    Args:
        *datasets: The datasets to merge.

    Returns:
        The merged dataset, with the combined data.

    """
    all_sessions = []
    for dataset in datasets:
        all_sessions.extend(dataset.sessions)

    return Dataset(sessions=all_sessions)
