"""
A dataset generated using ISAAC Sim.
"""

from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from loguru import logger
from methodtools import lru_cache
from PIL import Image

from .dataset import Camera, Dataset, LazyFrame, Session, YamlRep


class IsaacCamera(Camera):
    """
    Represents data corresponding to a single camera.
    """

    def __init__(self, *, camera_dir: Path):
        """
        Args:
            camera_dir: The path to the directory containing data for this
            camera.

        """
        # Contains RGB images.
        self.__camera_dir = camera_dir
        self.__rgb_dir = camera_dir / "rgb"
        self.__num_frames = len(list(self.__rgb_dir.iterdir()))
        logger.debug(
            "Found {} total frames in {}.", self.__num_frames, self.__rgb_dir
        )

    @property
    def path(self) -> Path:
        return self.__camera_dir

    @lru_cache(maxsize=32)
    def __frame_at_index(self, index: int) -> np.array:
        """
        Gets the frame at a particular index.

        Args:
            index: The index.

        Returns:
            The frame data.

        """
        frame_path = self.__rgb_dir / f"rgb_{index:04d}.png"
        return np.array(Image.open(frame_path).convert("RGB"))

    def frame_at_timestamp(
        self, timestamp: float, tolerance: float = 0.05
    ) -> LazyFrame:
        # In this case, timestamps are discrete.
        index = int(timestamp)
        return partial(self.__frame_at_index, index)

    def frames(self) -> Iterable[Tuple[float, LazyFrame]]:
        for i in range(self.__num_frames):
            yield i, self.frame_at_timestamp(i)


class IsaacSession(Session):
    """
    Represents a session in the dataset.
    """

    def __init__(self, *, session_dir: Path):
        """
        Args:
            session_dir: Path to the folder containing the session data.

        """
        self.__session_dir = session_dir

        # Add all of the camera data.
        self.__cameras = [
            IsaacCamera(camera_dir=camera_dir)
            for camera_dir in session_dir.iterdir()
            if camera_dir.is_dir()
        ]

    @property
    def cameras(self) -> List[IsaacCamera]:
        return self.__cameras[:]

    @property
    def session_folder(self) -> Path:
        return self.__session_dir


class IsaacDataset(YamlRep, Dataset):
    """
    Represents a complete collection of synthetic MARS data.
    """

    def __init__(self, *, dataset_dir: Path):
        """
        Args:
            dataset_dir: The path to the root dataset directory.

        """
        sessions = [
            IsaacSession(session_dir=session_dir)
            for session_dir in dataset_dir.iterdir()
            if session_dir.is_dir()
        ]
        super().__init__(sessions=sessions)

    @classmethod
    def from_yaml(
        cls,
        spec: Dict[str, Any],
        **kwargs: Any,
    ) -> "IsaacDataset":
        return cls(
            dataset_dir=Path(spec["root"]),
        )
