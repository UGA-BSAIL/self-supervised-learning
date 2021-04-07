import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from pycvat import Task

from ..schemas import ObjectTrackingFeatures as Otf
from ..tfrecords_utils import bytes_feature, float_feature, int_feature

_FEATURES_TO_FACTORIES = {
    Otf.IMAGE_HEIGHT.value: int_feature,
    Otf.IMAGE_WIDTH.value: int_feature,
    Otf.IMAGE_FILENAME.value: bytes_feature,
    Otf.IMAGE_SOURCE_ID.value: bytes_feature,
    Otf.IMAGE_ENCODED.value: bytes_feature,
    Otf.IMAGE_FORMAT.value: bytes_feature,
    Otf.OBJECT_BBOX_X_MIN.value: float_feature,
    Otf.OBJECT_BBOX_X_MAX.value: float_feature,
    Otf.OBJECT_BBOX_Y_MIN.value: float_feature,
    Otf.OBJECT_BBOX_Y_MAX.value: float_feature,
    Otf.OBJECT_CLASS_TEXT.value: bytes_feature,
    Otf.OBJECT_CLASS_LABEL.value: int_feature,
    Otf.OBJECT_ID.value: int_feature,
    Otf.IMAGE_SEQUENCE_ID.value: int_feature,
    Otf.IMAGE_FRAME_NUM.value: int_feature,
}


def random_splits(
    annotations: pd.DataFrame, *, train_fraction: float, test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits annotation data into training and testing components.

    Args:
        annotations: The annotations to split. Should already be organized
            into clips.
        train_fraction: The fraction of the data to use for training.
        test_fraction: The fraction of the data to use for testing.

    Returns:
        The individual train, test, and validation datasets.

    """
    # Set the index to the sequence ID to speed up filtering operations.
    annotations.set_index(
        Otf.IMAGE_SEQUENCE_ID.value, inplace=True, drop=False
    )

    train_clips = []
    test_clips = []
    validation_clips = []
    for sequence_id in annotations.index.unique():
        clip = annotations.iloc[annotations.index == sequence_id]

        sample = random.random()
        if sample > train_fraction + test_fraction:
            validation_clips.append(clip)
        elif sample > train_fraction:
            test_clips.append(clip)
        else:
            train_clips.append(clip)

    # Concatenate the clips back into single dataframes.
    def concat(clips: List[pd.DataFrame]) -> pd.DataFrame:
        if len(clips) > 0:
            return pd.concat(clips, ignore_index=True)
        else:
            return pd.DataFrame([], columns=[e.value for e in Otf])

    return concat(train_clips), concat(test_clips), concat(validation_clips)


def _get_missing_columns(
    frame_annotations: pd.DataFrame,
) -> Dict[str, tf.train.Feature]:
    """
    Gets a dictionary containing default values for any missing features.

    Args:
        frame_annotations: The frame annotations.

    Returns:
        A dict containing the default values.

    """
    default_values = {
        int_feature: [-1],
        bytes_feature: b"",
        float_feature: [0.0],
    }

    missing_features = {}
    for column, feature_type in _FEATURES_TO_FACTORIES.items():
        if column not in frame_annotations:
            missing_features[column] = feature_type(
                default_values[feature_type]
            )

    return missing_features


def _make_example(
    *, image: np.ndarray, frame_annotations: pd.DataFrame, frame_num: int
) -> tf.train.Example:
    """
    Creates a TF `Example` for a single frame.

    Args:
        image: The compressed image data for the frame.
        frame_annotations: The annotations for the frame.
        frame_num: The frame number.

    Returns:
        The example that it created.

    """
    # Shuffle the order of the rows to add some variation to Sinkhorn matrices.
    frame_annotations = frame_annotations.sample(frac=1.0)

    # Remove the frame number column since that info is provided manually.
    frame_annotations = frame_annotations.drop(
        columns=[Otf.IMAGE_FRAME_NUM.value]
    )

    # Create the feature dictionary.
    features = {}
    for column_name, column_data in frame_annotations.items():
        features[column_name] = _FEATURES_TO_FACTORIES[column_name](
            column_data
        )

    # Add default values for missing columns.
    features.update(_get_missing_columns(frame_annotations))

    # Add the image data.
    features[Otf.IMAGE_ENCODED.value] = _FEATURES_TO_FACTORIES[
        Otf.IMAGE_ENCODED.value
    ](image)
    # Add the frame number.
    features[Otf.IMAGE_FRAME_NUM.value] = _FEATURES_TO_FACTORIES[
        Otf.IMAGE_FRAME_NUM.value
    ]((frame_num,))

    return tf.train.Example(features=tf.train.Features(feature=features))


def _generate_clip_examples(
    video_frames: Task, annotations: pd.DataFrame
) -> Iterable[tf.train.Example]:
    """
    Generates TFRecord examples from annotations and corresponding video frames.

    Args:
        video_frames: The CVAT `Task` to source frames from.
        annotations: The loaded annotations, transformed to the TF format.

    Yields:
        Corresponding TFRecord examples for each frame. The examples are
        correspond to the TF Object Detection API format, except with the
        addition of an "image/object/id" label to group detections that
        are part of the same track.

    """
    frame_nums = annotations[Otf.IMAGE_FRAME_NUM.value]
    first_frame = frame_nums.min()
    last_frame = frame_nums.max()

    for frame_num in range(first_frame, last_frame + 1):
        logger.debug("Generating example for frame {}.", frame_num)

        # Get all the annotations for that frame.
        frame_annotations = annotations[
            annotations[Otf.IMAGE_FRAME_NUM.value] == frame_num
        ]
        logger.debug(
            "Have {} annotations for frame {}.",
            len(frame_annotations),
            frame_num,
        )
        # Get the actual frame image.
        frame_image = video_frames.get_image(frame_num, compressed=True)

        yield _make_example(
            image=frame_image,
            frame_annotations=frame_annotations,
            frame_num=frame_num,
        )


def generate_examples(
    video_frames: Task, annotations: pd.DataFrame
) -> Iterable[Iterable[tf.train.Example]]:
    """
    Generates TFRecord examples from annotations and corresponding video
    frames for all clips.

    Args:
        video_frames: The CVAT `Task` to source frames from.
        annotations: The loaded annotations, transformed to the TF format.

    Yields:
        Iterables of TFRecord examples for each clip.

    """
    # Set the index to the sequence ID to speed up filtering operations.
    annotations.set_index(
        Otf.IMAGE_SEQUENCE_ID.value, inplace=True, drop=False
    )

    for sequence_id in annotations.index.unique():
        logger.info("Generating TFRecords for clip {}.", sequence_id)
        yield _generate_clip_examples(
            video_frames, annotations.iloc[annotations.index == sequence_id]
        )
