"""
Loads/stores a video from/to a sequence of frame images.
"""


from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional, Tuple, Callable, Protocol

import cv2
import numpy as np
from kedro.io import AbstractVersionedDataSet, Version
from loguru import logger


class FrameReader(Protocol):
    """
    A function that can be used to read frames starting at a particular spot.
    """

    def __call__(
        self, frame_num: int, max_frames: Optional[int] = ..., /
    ) -> Iterable[np.ndarray]:
        ...


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
        bgr_color: bool = True,
        skip_frames: int = 0,
    ):
        """
        Args:
            filepath: The path to the output video.
            version: The version information for the `DataSet`.
            codec: FourCC code to use for video encoding.
            fps: The FPS to use when writing the video.
            resolution: The output resolution of the video, in the form
                `(width, height)`.
            bgr_color: If true, it will load and save images in the BGR color
                space. Otherwise, it will load and save images in the RGB color
                space.
        """
        super().__init__(PurePosixPath(filepath), version)

        self.__codec = codec
        self.__fps = fps
        self.__resolution = resolution
        self.__bgr_color = bgr_color

    def _load(self) -> FrameReader:
        """
        Loads the video frame-by-frame.

        Returns:
            A function that can be used to read frames starting at a particular
            point, and reading up to a maximum number of frames.

        """

        def _read_frame(
            frame_num: int, max_frames: Optional[int] = None
        ) -> Iterable[np.ndarray]:
            logger.info(
                "Reading video at {} from frame {}.",
                self._get_load_path(),
                frame_num,
            )
            reader = cv2.VideoCapture(self._get_load_path().as_posix())
            # Get the total number of frames.
            total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.debug("Video has {} frames.", total_frames)

            # Seek to the starting point.
            if frame_num >= total_frames:
                raise ValueError(
                    f"Frame {frame_num} requested, but video has only"
                    f" {total_frames} frames."
                )
            set_success = reader.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            assert set_success

            # Read the frames.
            num_read = 0
            while reader.isOpened() and (
                max_frames is None or num_read < max_frames
            ):
                status, frame = reader.read()
                num_read += 1
                if not status:
                    logger.warning("Failed to read frame, skipping.")
                    continue

                if not self.__bgr_color:
                    # OpenCV works with BGR images, but we need RGB.
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                yield frame

            reader.release()

        return _read_frame

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
            if self.__bgr_color:
                # OpenCV works with BGR images, but VideoWriter expects RGB.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
