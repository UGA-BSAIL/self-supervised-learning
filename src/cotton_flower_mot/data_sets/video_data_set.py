"""
Loads/stores a video from/to a sequence of frame images.
"""


from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from kedro.io import AbstractVersionedDataSet, Version
from loguru import logger


class VideoDataSet(AbstractVersionedDataSet):
    """
    Loads/stores a video from/to a sequence of frame images.
    """

    def __init__(
        self,
        filepath: PurePosixPath,
        version: Optional[Version] = None,
        codec: str = "mp4v",
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
    ):
        """
        Args:
            filepath: The path to the output video.
            version: The version information for the `DataSet`.
            codec: FourCC code code to use for video encoding.
            fps: The FPS to use when writing the video.
            resolution: The output resolution of the video, in the form
                `(width, height)`.
        """
        super().__init__(PurePosixPath(filepath), version)

        self.__codec = codec
        self.__fps = fps
        self.__resolution = resolution

    def _load(self) -> Iterable[np.ndarray]:
        """
        Loads the video frame-by-frame.

        Yields:
            Each frame of the video, in order.

        """
        reader = cv2.VideoCapture(self._get_load_path().as_posix())

        while reader.isOpened():
            status, frame = reader.read()
            if not status:
                logger.warning("Failed to read frame, skipping.")
                continue

            yield frame

        reader.release()

    def _save(self, data: Iterable[np.ndarray]) -> None:
        """
        Saves the video frame-by-frame.

        Args:
            data: An iterable of frames to save as a video.

        """
        # Make sure the save directory exists.
        save_path = Path(self._get_save_path())
        save_dir = save_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        codec = cv2.VideoWriter_fourcc(*self.__codec)
        writer = cv2.VideoWriter(
            self._get_save_path().as_posix(),
            codec,
            self.__fps,
            self.__resolution,
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to open video writer.")

        for frame in data:
            writer.write(frame)

        writer.release()

    def _exists(self) -> bool:
        path = self._get_load_path()
        return Path(path).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(
            codec=self.__codec,
            fps=self.__fps,
            resolution=self.__resolution,
        )
