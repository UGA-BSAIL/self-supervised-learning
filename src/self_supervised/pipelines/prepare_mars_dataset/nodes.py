"""
Nodes for the `prepare_mars_dataset` pipeline.
"""


from typing import Dict, Any, Iterable, List, Tuple
from pathlib import Path

import cv2
import kedro.io
import pandas as pd
from .dataset import Dataset
import numpy as np
from kedro.io import PartitionedDataSet
from loguru import logger
from ..schemas import MarsMetadata
from PIL import Image


def _file_id(*, clip: int, frame: int, camera: int) -> str:
    """
    Creates a unique ID for a particular frame.

    Args:
        clip: The clip that the frame is from.
        frame: The frame number within the clip.
        camera: The camera that the frame is from.

    Returns:
        The ID that it created.

    """
    return f"clip{clip}_cam{camera}_frame{frame}"


def _quantify_motion(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Quantifies the amount of motion between two frames.

    Args:
        frame1: The first frame.
        frame2: The second frame.

    Returns:
        The average difference between pixels in the two frames.

    """

    def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
        # Downsample the frames a lot to speed up the calculation.
        frame = cv2.resize(frame, (100, 100))

        # Convert to grayscale.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur slightly.
        frame = cv2.GaussianBlur(src=frame, ksize=(5, 5), sigmaX=0)

        return frame

    frame1 = _preprocess_frame(frame1)
    frame2 = _preprocess_frame(frame2)

    # Find the pixels that are different between the two frames.
    frame_diff = cv2.absdiff(frame1, frame2)
    return float(np.mean(frame_diff))


def _minimum_motion(
    frames1: List[np.ndarray], frames2: List[np.ndarray]
) -> float:
    """
    Quantifies the motion across multiple cameras, and then takes the minimum.

    Args:
        frames1: The first set of frames from all the cameras.
        frames2: The second set of frames from all the cameras.

    Returns:
        The minimum amount of motion across all cameras.

    """
    motions = [_quantify_motion(f1, f2) for f1, f2 in zip(frames1, frames2)]
    return np.min(motions)


def _write_until_clip_end(
    frame_iter: Iterable[Tuple[float, List[np.ndarray]]],
    *,
    frame_dataset: PartitionedDataSet,
    clip_num: int,
    max_gap: float,
    motion_threshold: float,
) -> Tuple[bool, pd.DataFrame]:
    """
    Writes a complete clip to the disk. Will continue to write until it detects
    a sufficient gap to end the clip.

    Args:
        frame_iter: The iterator that produces synchronized frames.
        frame_dataset: The dataset where the frames will be written.
        clip_num: The clip number that we are writing.
        max_gap: The maximum gap in timestamps before we consider the clip
            to have ended.
        motion_threshold: Minimum amount of motion we expect between images. If
            not reached, the images will be considered stationary and ignored.

    Returns:
        Whether there are any more frames to read, and metadata for the clip.

    """
    have_more_frames = True
    metadata = pd.DataFrame(columns=[c.value for c in MarsMetadata])
    # Last timestamp at which we saw a frame.
    last_frame_timestamp = np.inf
    # Most recent set of frames from the cameras.
    previous_frames = None

    for frame_num, (timestamp, frames) in enumerate(frame_iter):
        if timestamp - last_frame_timestamp > max_gap:
            logger.info("Reached end of clip {}.", clip_num)
            break
        if len(frames) == 0:
            # We have no frames at this timestamp.
            continue

        # Check that we have sufficient motion between frames.
        if previous_frames is None:
            # Wait to check motion before writing anything.
            previous_frames = frames
            continue
        if _minimum_motion(previous_frames, frames) < motion_threshold:
            logger.debug(
                "Dropping frame {} because it is stationary.", frame_num
            )
            continue

        previous_frames = frames
        last_frame_timestamp = timestamp

        # Save the frames.
        frame_files = {}
        for camera, frame in enumerate(frames):
            file_id = _file_id(clip=clip_num, camera=camera, frame=frame_num)

            # Fix the color and resize before saving.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (None, None), fx=0.5, fy=0.5)

            frame_files[file_id] = Image.fromarray(frame)

            # Update the metadata.
            metadata.loc[len(metadata.index)] = (
                clip_num,
                frame_num,
                camera,
                timestamp,
                file_id,
            )
        frame_dataset.save(frame_files)

    else:
        logger.info("Ending clip {} because session is over.", clip_num)
        have_more_frames = False
    return have_more_frames, metadata


def build_dataset(
    dataset_spec: Dict[str, Any],
    *,
    image_dataset_path: str,
    sync_tolerance: float = 0.05,
    max_timestamp_gap: float = 0.5,
    motion_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Builds the dataset, preprocessing the images and writing them out to disk.

    Args:
        dataset_spec: The YAML specification for the dataset.
        image_dataset_path: Where to write the dataset of image files.
        sync_tolerance: The maximum difference in timestamps allowed for
            synchronized frames.
        max_timestamp_gap: The maximum difference in timestamps allowed
            between consecutive frames in a clip.
        motion_threshold: Minimum amount of motion we expect between images. If
            not reached, the images will be considered stationary and ignored.

    Returns:
        A table containing the metadata for the dataset.

    """
    Path(image_dataset_path).mkdir(exist_ok=True)
    image_dataset = kedro.io.PartitionedDataSet(
        path=image_dataset_path,
        dataset="pillow.ImageDataSet",
        filename_suffix=".jpg",
    )

    dataset = Dataset.from_yaml(dataset_spec["dataset"])

    # Write out the frames from the dataset.
    clip_num = 0
    metadata = pd.DataFrame(columns=[c.value for c in MarsMetadata])
    for session in dataset.sessions:
        logger.info(
            "Writing clips from session at {}.", session.session_folder
        )
        frame_iter = session.synchronized_frames(tolerance=sync_tolerance)

        have_more_frames = True
        while have_more_frames:
            have_more_frames, clip_metadata = _write_until_clip_end(
                frame_iter,
                frame_dataset=image_dataset,
                clip_num=clip_num,
                max_gap=max_timestamp_gap,
                motion_threshold=motion_threshold,
            )
            metadata = pd.concat([metadata, clip_metadata], ignore_index=True)

            clip_num += 1

    return metadata
