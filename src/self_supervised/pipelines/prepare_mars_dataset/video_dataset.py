"""
Classes for interacting with the MARS video dataset.
"""

import enum
from functools import cached_property, partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from methodtools import lru_cache
from tqdm import tqdm
from upath import UPath

from .dataset import Camera, Dataset, LazyFrame, Session, YamlRep
from .mallard_utils import ensure_user_logged_in, streaming_video_download


class VideoCamera(YamlRep, Camera):
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
        camera_id: str,
        video_path: UPath,
        timestamp_path: Optional[UPath] = None,
    ):
        """
        Args:
            camera_id: The name of this camera.
            video_path: The path to the video file for this camera.
            timestamp_path: The path to the timestamp file for this camera.
                If not provided, timestamps from the video will be used.

        """
        self.__camera_id = camera_id
        self.__video_path = video_path
        self.__timestamp_path = timestamp_path

        # Used to keep track internally of which frame the video capture is set
        # at.
        self.__capture_frame = 0
        # Temporary directory to use for downloading videos.
        self.__download_dir = TemporaryDirectory()

    @classmethod
    def from_yaml(
        cls,
        spec: Dict[str, Any],
        *,
        video_parent_folder: UPath = None,
        ts_parent_folder: UPath = None,
    ) -> "VideoCamera":
        """
        Creates a new instance of this class from a YAML specification.

        Args:
            spec: The YAML specification to use.
            video_parent_folder: The parent folder that all video paths in this
                session are relative to.
            ts_parent_folder: The parent folder that all timestamp data in
                this session is relative to.

        Returns:
            The instance it created.

        """
        logger.debug("Loading camera from {}.", video_parent_folder)
        ts_path = None
        if "times" in spec:
            ts_path = ts_parent_folder / spec["times"]
        return cls(
            camera_id=spec["camera_id"],
            video_path=video_parent_folder / spec["video"],
            timestamp_path=ts_path,
        )

    @property
    def path(self) -> UPath:
        """
        Returns:
            The path to the video file.

        """
        if self.__video_path.parts[0].startswith("http"):
            video_path = UPath(self.__download_dir.name) / "video.mp4"
            if video_path.exists():
                return video_path

            # Re-create the temporary directory to be safe, as it can get
            # cleaned up when copying.
            self.__download_dir = TemporaryDirectory()
            video_path = UPath(self.__download_dir.name) / "video.mp4"

            # We need to download this file.
            logger.debug("Downloading video from {}.", self.__video_path)
            ensure_user_logged_in()
            streaming_video_download(self.__video_path.as_posix(), video_path)

            return video_path

        return self.__video_path

    @cached_property
    def timestamp_path(self) -> Optional[UPath]:
        """
        Returns:
            The path to the timestamp file, or None if there is no timestamp
            file.

        """
        return self.__timestamp_path

    def __timestamps_from_file(self) -> pd.DataFrame:
        """
        Loads timestamp data from a separate file.

        Returns:
            The loaded timestamps for this camera.

        """
        timestamps = pd.read_csv(
            self.timestamp_path,
            sep=" ",
            names=[c.value for c in self.TimestampCol],
        )
        # Remove timestamps with a value of zero, which we occasionally end up
        # with. (This must be a quirk of the software I used to convert the
        # Rosbags.)
        timestamps = timestamps[
            timestamps[self.TimestampCol.TIMESTAMP.value] > 0.0
        ]
        # Use the timestamps as an index for easy querying.
        timestamps.set_index(self.TimestampCol.TIMESTAMP.value, inplace=True)

        # Sometimes we also have more timestamps than frames...
        num_actual_frames = int(
            self.__video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        if (
            num_actual_frames
            <= timestamps[self.TimestampCol.FRAME_NUM.value].max()
        ):
            logger.warning(
                "{} has fewer frames ({}) that we have timestamps for. "
                "Truncating timestamps.",
                self.path,
                num_actual_frames,
            )
            timestamps = timestamps[
                timestamps[self.TimestampCol.FRAME_NUM.value]
                < num_actual_frames
            ]

        return timestamps

    def __timestamps_from_metadata(self) -> pd.DataFrame:
        """
        Loads timestamp data from a camera metadata file.

        Returns:
            The loaded timestamps for this camera.

        """
        num_documents = self.timestamp_path.read_text().count("---")
        metadata = yaml.load_all(
            self.timestamp_path.open(), Loader=yaml.CLoader
        )

        data = []
        for frame in tqdm(
            metadata, desc="Loading metadata", total=num_documents
        ):
            if frame["frame_id"] != self.__camera_id:
                # This is from another camera.
                continue
            data.append([frame["frame_num"], frame["stamp"]])

        timestamps_frame = pd.DataFrame(
            data=data,
            columns=[
                self.TimestampCol.FRAME_NUM.value,
                self.TimestampCol.TIMESTAMP.value,
            ],
        )
        # Use the timestamps as an index for easy querying.
        timestamps_frame.set_index(
            self.TimestampCol.TIMESTAMP.value, inplace=True
        )
        timestamps_frame.sort_index(inplace=True)
        return timestamps_frame

    def __timestamps_from_video(self) -> pd.DataFrame:
        """
        Generates timestamp data from the video file.

        Returns:
            The generated timestamps.

        """
        # Generate evenly-spaced timestamps based on the video.
        num_video_frames = int(
            self.__video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        video_fps = self.__video_capture.get(cv2.CAP_PROP_FPS)
        logger.debug(
            "Video {} has {} frames at {} FPS.",
            self.__video_path.name,
            num_video_frames,
            video_fps,
        )
        timestamps = np.linspace(
            0, num_video_frames / video_fps, num_video_frames
        )

        timestamps_frame = pd.DataFrame(
            data={
                self.TimestampCol.FRAME_NUM.value: np.arange(num_video_frames),
                self.TimestampCol.TIMESTAMP.value: timestamps,
            }
        )
        # Use the timestamps as an index for easy querying.
        timestamps_frame.set_index(
            self.TimestampCol.TIMESTAMP.value, inplace=True
        )
        return timestamps_frame

    @cached_property
    def __timestamps(self) -> pd.DataFrame:
        """
        Returns:
            The timestamps for the video.

        """
        if self.__timestamp_path is not None:
            # Read from the file.
            if self.__timestamp_path.name.endswith(".txt"):
                return self.__timestamps_from_file()
            else:
                return self.__timestamps_from_metadata()
        else:
            # Generate from the video.
            return self.__timestamps_from_video()

    @cached_property
    def __video_capture(self) -> cv2.VideoCapture:
        """
        Returns:
            A `VideoCapture` object for the video.

        Raises:
            `OSError` if it fails to open the video.

        """
        capture = cv2.VideoCapture(self.path.as_posix())
        if not capture.isOpened():
            raise OSError(
                f"Failed to open video file at {self.path}. Is it valid?"
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

        if not status:
            logger.error("Failed to read frame at index {}.", frame_index)
            raise ValueError(f"Failed to read frame at index {frame_index}.")

        # Fix RGB vs. BGR.
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def frame_at_timestamp(
        self, timestamp: float, tolerance: float = 0.05
    ) -> LazyFrame:
        frame_index = self.__frame_num_at_timestamp(
            timestamp, tolerance=tolerance
        )
        return partial(self.__frame_at_index, frame_index)

    def frames(self) -> Iterable[Tuple[float, LazyFrame]]:
        for timestamp, frame_num in self.__timestamps.itertuples():
            frame = partial(self.__frame_at_index, frame_num)
            yield timestamp, frame

    def release(self) -> None:
        """
        Releases the video capture instances associated with this camera,
        freeing some memory. There will be a time penalty the next time this
        camera is used as the capture instances have to be recreated.

        """
        self.__frame_at_index.cache_clear()

        try:
            del self.__video_capture
            logger.debug("Releasing capture for {}.", self.path)
        except AttributeError:
            # It wasn't initialized. Ignore.
            pass


class VideoSession(YamlRep, Session):
    """
    Represents a session in the dataset.
    """

    def __init__(
        self,
        *,
        session_folder: Path,
        cameras: List[VideoCamera],
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
        cls,
        spec: Dict[str, Any],
        *,
        video_parent_folder: UPath = None,
        ts_parent_folder: UPath = None,
    ) -> "VideoSession":
        if "path" in spec:
            video_parent_folder = video_parent_folder / spec["path"]
            ts_parent_folder = ts_parent_folder / spec["path"]
        else:
            video_parent_folder = video_parent_folder / spec.get(
                "video_path", ""
            )
            ts_parent_folder = ts_parent_folder / spec.get("ts_path", "")

        logger.debug("Loading session from {}.", video_parent_folder)

        # Load the cameras in the session.
        cameras = []
        for camera_id, camera_spec in spec["cameras"].items():
            camera_spec["camera_id"] = camera_id
            cameras.append(
                VideoCamera.from_yaml(
                    camera_spec,
                    video_parent_folder=video_parent_folder,
                    ts_parent_folder=ts_parent_folder,
                )
            )

        return cls(session_folder=video_parent_folder, cameras=cameras)

    @property
    def cameras(self) -> List[VideoCamera]:
        return self.__cameras[:]

    @property
    def session_folder(self) -> Path:
        return self.__session_folder

    def synchronized_frames(
        self, tolerance: float = 0.05
    ) -> Iterable[Tuple[float, List[LazyFrame]]]:
        yield from super().synchronized_frames(tolerance)

        # Release camera memory.
        for camera in self.__cameras:
            camera.release()


class VideoDataset(YamlRep, Dataset):
    """
    Represents a complete dataset of video files.
    """

    @classmethod
    def from_yaml(
        cls,
        spec: Dict[str, Any],
        *,
        video_parent_folder: Optional[UPath] = None,
        ts_parent_folder: Optional[UPath] = None,
    ) -> "VideoDataset":
        if "root" in spec:
            video_dataset_folder = UPath(spec["root"])
            ts_dataset_folder = UPath(spec["root"])
        else:
            video_dataset_folder = UPath(spec.get("video_root", ""))
            ts_dataset_folder = UPath(spec.get("ts_root", ""))
        if video_parent_folder is not None:
            video_dataset_folder = video_parent_folder / video_dataset_folder
        if ts_parent_folder is not None:
            ts_dataset_folder = ts_parent_folder / ts_dataset_folder
        logger.debug("Loading dataset from {}.", video_dataset_folder)

        # Load the sessions.
        sessions = []
        for session_spec in spec["sessions"]:
            sessions.append(
                VideoSession.from_yaml(
                    session_spec,
                    video_parent_folder=video_dataset_folder,
                    ts_parent_folder=ts_dataset_folder,
                )
            )

        return cls(sessions=sessions)
