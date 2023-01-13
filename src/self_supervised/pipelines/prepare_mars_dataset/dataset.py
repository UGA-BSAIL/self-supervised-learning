"""
Classes for interacting with the MARS dataset.
"""


from pathlib import Path
from functools import cached_property
import enum
from typing import Dict, Any, List, Iterable, Tuple
import abc

import cv2
from methodtools import lru_cache

import pandas as pd
import numpy as np
from loguru import logger


class _YamlRep(abc.ABC):
    """
    Class that represents data from the YAML file.
    """

    @classmethod
    @abc.abstractmethod
    def from_yaml(
        cls, spec: Dict[str, Any], *, parent_folder: Path
    ) -> "_YamlRep":
        """
        Creates a new instance of this class from a YAML specification.

        Args:
            spec: The YAML specification to use.
            parent_folder: The parent folder that all paths in this section are
                relative to.

        Returns:
            The instance it created.

        """


class Camera(_YamlRep):
    """
    Represents data corresponding to a single camera.
    """

    _FRAME_SEEK_THRESHOLD = 5
    """
    When we make incremental changes to the position in the video file
    (below this number of frames), we use the seeking method instead of
    CAP_PROP_POS_FRAMES.
    """

    @enum.unique
    class TimestampCol(enum.Enum):
        """
        Names of the columns in the timestamp files.
        """

        FRAME_NUM = "frame_num"
        """
        The frame number.
        """
        TIMESTAMP = "timestamp"
        """
        The timestamp of that frame.
        """

    def __init__(
        self,
        *,
        parent_folder: Path,
        video_name: str,
        timestamp_name: str,
    ):
        """
        Args:
            parent_folder: The path to the folder containing the camera files.
            video_name: The name of the video file for this camera.
            timestamp_name: The name of the timestamp file for this camera.

        """
        self.__root = parent_folder
        self.__video_name = video_name
        self.__timestamp_name = timestamp_name

        # Used to keep track internally of which frame the video capture is set
        # at.
        self.__capture_frame = 0

    @classmethod
    def from_yaml(
        cls, spec: Dict[str, Any], *, parent_folder: Path
    ) -> "Camera":
        logger.debug("Loading camera from {}.", parent_folder)
        return cls(
            parent_folder=parent_folder,
            video_name=spec["video"],
            timestamp_name=spec["times"],
        )

    @cached_property
    def video_path(self) -> Path:
        """
        Returns:
            The path to the video file.

        """
        return self.__root / self.__video_name

    @cached_property
    def timestamp_path(self) -> Path:
        """
        Returns:
            The path to the timestamp file.

        """
        return self.__root / self.__timestamp_name

    @cached_property
    def __timestamps(self) -> pd.DataFrame:
        """
        Returns:
            The loaded timestamps for this camera.

        """
        timestamps = pd.read_csv(
            self.timestamp_path,
            sep=" ",
            names=[c.value for c in self.TimestampCol],
        )
        # Remove timestamps with a value of zero, which we occasionally end up with. (This must be a quirk of the
        # software I used to convert the Rosbags.)
        timestamps = timestamps[timestamps[self.TimestampCol.TIMESTAMP.value] > 0.0]
        # Use the timestamps as an index for easy querying.
        timestamps.set_index(self.TimestampCol.TIMESTAMP.value, inplace=True)

        return timestamps

    @cached_property
    def __video_capture(self) -> cv2.VideoCapture:
        """
        Returns:
            A `VideoCapture` object for the video.

        Raises:
            `OSError` if it fails to open the video.

        """
        capture = cv2.VideoCapture(self.video_path.as_posix())
        if not capture.isOpened():
            raise OSError(
                f"Failed to open video file at {self.video_path}. Is it valid?"
            )

        self.__capture_frame = 0

        return capture

    def __frame_num_at_timestamp(
        self, timestamp: float, tolerance: float = 0.05
    ) -> int:
        """
        Gets the frame number from the video that's closest to a particular
        timestamp.

        Args:
            timestamp: The timestamp that we are looking for the frame at.
            tolerance: Maximum difference between the desired timestamp and the
                actual frame timestamp that we allow.

        Returns:
            The corresponding frame index.

        Raises:
            `KeyError` if it cannot find a frame at this timestamp.

        """
        match_indices = self.__timestamps.index.get_indexer(
            [timestamp], method="nearest", tolerance=tolerance
        )
        match_locations = self.__timestamps.index.take(
            match_indices, fill_value=pd.NA
        )
        try:
            return int(
                self.__timestamps.loc[match_locations][
                    self.TimestampCol.FRAME_NUM.value
                ]
            )
        except KeyError:
            raise KeyError(f"No frame found for timestamp {timestamp}.")

    @lru_cache(maxsize=32)
    def __frame_at_index(self, frame_index: int) -> np.ndarray:
        """
        Gets the frame at a particular index in the video.

        Args:
            frame_index: The index of the frame to get.

        Returns:
            The frame that it got.

        """
        # Get the corresponding image.
        skip_frames = frame_index - self.__capture_frame
        if 0 <= skip_frames <= self._FRAME_SEEK_THRESHOLD:
            # This is an optimization because the CAP_PROP_POS_FRAMES method
            # does not seem to be very efficient. In practice, with short seeks
            # in the correct direction, it's much faster to just grab them.
            for _ in range(skip_frames):
                self.__video_capture.grab()
        else:
            self.__video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        status, frame = self.__video_capture.read()
        self.__capture_frame = frame_index + 1

        assert status, "Failed to read frame from video."
        return frame

    def frame_at_timestamp(
        self, timestamp: float, tolerance: float = 0.05
    ) -> np.ndarray:
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
        frame_index = self.__frame_num_at_timestamp(
            timestamp, tolerance=tolerance
        )
        return self.__frame_at_index(frame_index)

    def frames(self) -> Iterable[Tuple[float, np.ndarray]]:
        """
        Yields:
            The timestamp and corresponding frame for each frame of the video,
            in order.

        """
        for timestamp, frame_num in self.__timestamps.itertuples():
            frame = self.__frame_at_index(frame_num)

            yield timestamp, frame


class Session(_YamlRep):
    """
    Represents a session in the dataset.
    """

    def __init__(
        self,
        *,
        session_folder: Path,
        cameras: List[Camera],
    ):
        """
        Args:
            session_folder: Path to the folder containing session files.
            cameras: The cameras associated with this session.

        """
        self.__session_folder = session_folder
        self.__cameras = cameras

    @classmethod
    def from_yaml(
        cls, spec: Dict[str, Any], *, parent_folder: Path
    ) -> "Session":
        session_folder = parent_folder / Path(spec["path"])
        logger.debug("Loading session from {}.", session_folder)

        # Load the cameras in the session.
        cameras = []
        for camera_spec in spec["cameras"].values():
            cameras.append(
                Camera.from_yaml(camera_spec, parent_folder=session_folder)
            )

        return cls(session_folder=session_folder, cameras=cameras)

    @cached_property
    def cameras(self) -> List[Camera]:
        """
        Returns:
            The camera data from this session.

        """
        return self.__cameras[:]

    @property
    def session_folder(self) -> Path:
        """
        Returns:
            The folder where the session files are stored.

        """
        return self.__session_folder

    def synchronized_frames(
        self, tolerance: float = 0.05
    ) -> Iterable[Tuple[float, List[np.ndarray]]]:
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
        anchor_camera = self.__cameras[0]
        other_cameras = self.__cameras[1:]

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
                        camera.video_path,
                        timestamp,
                    )
                    # We only want to yield images if ALL cameras have
                    # corresponding frames, so there's no point in checking the
                    # other ones in this case.
                    frames = []
                    break

            yield timestamp, frames


class Dataset(_YamlRep):
    """
    Represents a complete dataset of video files.
    """

    def __init__(self, *, dataset_folder: Path, sessions: List[Session]):
        """
        Args:
            dataset_folder: The path to the main dataset folder.
            sessions: The data for each session in the dataset.

        """
        self.__dataset_folder = dataset_folder
        self.__sessions = sessions

    @classmethod
    def from_yaml(
        cls, spec: Dict[str, Any], *, parent_folder: Path = Path("/")
    ) -> "Dataset":
        dataset_folder = parent_folder / spec["root"]
        logger.debug("Loading dataset from {}.", dataset_folder)

        # Load the sessions.
        sessions = []
        for session_spec in spec["sessions"]:
            sessions.append(
                Session.from_yaml(session_spec, parent_folder=dataset_folder)
            )

        return cls(dataset_folder=dataset_folder, sessions=sessions)

    @cached_property
    def sessions(self) -> List[Session]:
        """
        Returns:
            The sessions in this dataset.

        """
        return self.__sessions[:]
