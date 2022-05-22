import itertools
from functools import partial
from typing import Any, Callable, Dict, Iterable, List

import cv2
import numpy as np
import tensorflow as tf
from loguru import logger

from ...data_sets.video_data_set import FrameReader
from ..schemas import UnannotatedFeatures as Uf
from ..tfrecords_utils import bytes_feature, int_feature

_FEATURES_TO_FACTORIES = {
    Uf.IMAGE_ENCODED: bytes_feature,
    Uf.IMAGE_SEQUENCE_ID: int_feature,
    Uf.IMAGE_FRAME_NUM: int_feature,
}


def _make_example(
    *, image: np.ndarray, frame_num: int, sequence_id: int
) -> tf.train.Example:
    """
    Creates a TF `Example` for a single frame.

    Args:
        image: The compressed image data for the frame.
        frame_num: The frame number.
        sequence_id: The sequence ID.

    Returns:
        The example that it created.

    """
    # Create the feature dictionary.
    features = {
        Uf.IMAGE_ENCODED.value: _FEATURES_TO_FACTORIES[Uf.IMAGE_ENCODED](
            image
        ),
        Uf.IMAGE_SEQUENCE_ID.value: _FEATURES_TO_FACTORIES[
            Uf.IMAGE_SEQUENCE_ID
        ]((sequence_id,)),
        Uf.IMAGE_FRAME_NUM.value: _FEATURES_TO_FACTORIES[Uf.IMAGE_FRAME_NUM](
            (frame_num,)
        ),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def _split_clips(
    video: Iterable[np.ndarray], *, clip_length: int
) -> Iterable[List[np.ndarray]]:
    """
    Splits a video into clips of at most a certain length.

    Args:
        video: The input video.
        clip_length: The maximum length of each clip.

    Yields:
        Each clip, one-by-one.

    """
    clip = []
    for frame in video:
        clip.append(frame)

        if len(clip) >= clip_length:
            yield clip
            clip = []

    # If we have any extra, this will be a last, partial clip.
    if clip:
        yield clip


def _generate_clip_examples(
    clip: Iterable[np.ndarray], *, sequence_id: int
) -> Iterable[tf.train.Example]:
    """
    Generates TFRecord examples for a particular clip.

    Args:
        clip: The clip to generate examples for.
        sequence_id: The sequence ID of the clip.

    Yields:
        Example for each frame in the clip.

    """
    logger.info("Generating examples for clip {}.", sequence_id)

    for frame_num, frame in enumerate(clip):
        # Encode the frame.
        encoded, frame_jpeg = cv2.imencode(".jpg", frame)
        assert encoded, "Failed to encode image"

        yield _make_example(
            image=frame_jpeg, frame_num=frame_num, sequence_id=sequence_id
        )


def _generate_examples(
    video: FrameReader,
    *,
    video_name: str,
    clip_length: int,
    skip_initial_frames: int = 0,
) -> Dict[str, Callable[[], Iterable[tf.train.Example]]]:
    """
    Generates TFRecord examples for an entire video, splitting it up into
    clips.

    Args:
        video: The video to generate annotations for.
        video_name: Unique name for the video.
        clip_length: The length of each clip, in frames.
        skip_initial_frames: How many frames to skip at the beginning of the
            video.

    Returns:
        A dictionary of functions mapping file names to
        generators of TFRecord examples for each clip.

    """

    def _read_and_process_clip(
        _sequence_id: int, _start_frame: int
    ) -> Iterable[tf.train.Example]:
        """
        Reads and generates examples for a particular clip.

        Args:
            _sequence_id: The sequence ID.
            _start_frame: The clip's starting frame.

        Yields:
            The examples for the clip.

        """
        clip = video.read(_start_frame)
        clip = itertools.islice(clip, clip_length)
        yield from _generate_clip_examples(clip, sequence_id=_sequence_id)

    clips = {}
    for sequence_id, start_frame in enumerate(
        range(skip_initial_frames, video.num_frames, clip_length)
    ):
        clip_name = f"{video_name}_clip_{sequence_id}.tfrecord"
        clips[clip_name] = partial(
            _read_and_process_clip, sequence_id, start_frame
        )

    return clips


def generate_multiple_video_examples(
    videos: Dict[str, Callable[[], Any]],
    *,
    clip_length: int,
    skip_initial_frames: int = 0,
) -> Dict[str, Callable[[], Iterable[tf.train.Example]]]:
    """

    Args:
        videos: The videos to generate examples for.
        clip_length: The length of the clips to generate.
        skip_initial_frames: How many frames to skip at the

    Returns:
        A dictionary mapping clip names to generators that produce examples
        for each clip.

    """
    clips = {}
    for video_name, video in videos.items():
        clips.update(
            _generate_examples(
                video(),
                video_name=video_name,
                clip_length=clip_length,
                skip_initial_frames=skip_initial_frames,
            )
        )

    return clips


def combine_session_examples(
    *args: Any,
) -> Dict[str, Callable[[], Iterable[tf.train.Example]]]:
    """
    Combines examples from a bunch of clips into a single dictionary.

    Args:
        *args: The examples to combine.

    Returns:
        The combined examples.

    """
    combined = {}
    for session_num, session_examples in enumerate(args):
        for clip_name, examples in session_examples.items():
            # Prefix with unique video names.
            combined[f"session_{session_num}_{clip_name}"] = examples

    return combined
