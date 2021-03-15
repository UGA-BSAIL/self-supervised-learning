"""
Nodes for the `data_engineering` pipeline.
"""

import random
from functools import partial
from typing import Any, Dict, Iterable, Tuple

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


def cut_video(
    annotations_mot: pd.DataFrame, *, new_length: int
) -> pd.DataFrame:
    """
    Cuts a video down to the first N frames.

    Args:
        annotations_mot: The annotations, in MOT 1.1 format, from a single
            video.
        new_length: All frames beyond this one will be dropped.

    Returns:
        The same annotations, with extraneous frames dropped.

    """
    # We assume frames are one-indexed, as per MOT standard.
    frame_condition = annotations_mot["frame"] <= new_length
    return annotations_mot[frame_condition]


def merge_annotations(*args: Any) -> pd.DataFrame:
    """
    Merges annotations from multiple DataSets, setting the
    `IMAGE_SEQUENCE_ID` to differentiate between them.

    Args:
        *args: The datasets to merge from.

    Returns:
        The merged annotations.

    """
    # Set a unique sequence ID for each of them.
    sequence_id = 0
    for dataset in args:
        dataset[Otf.IMAGE_SEQUENCE_ID.value] = sequence_id
        sequence_id += 1

    # Merge them.
    return pd.concat(args, ignore_index=True)


def split_clips(
    annotations_mot: pd.DataFrame, *, max_clip_length: int
) -> pd.DataFrame:
    """
    Splits data into clips that have a specified maximum length. Will update
    the `IMAGE_SEQUENCE_ID` column to differentiate different clips.

    Args:
        annotations_mot: The annotations to split.
        max_clip_length: The maximum length in frames of the clips.

    Returns:
        The modified annotations.

    """
    old_sequence_id = -1
    new_sequence_id = -1
    current_clip_length = 0

    new_sequence_ids = []
    for sequence_id in annotations_mot[Otf.IMAGE_SEQUENCE_ID.value]:
        if sequence_id != old_sequence_id:
            # This is a new video in the underlying data.
            current_clip_length = 0
            new_sequence_id += 1
        old_sequence_id = sequence_id

        current_clip_length += 1
        if current_clip_length > max_clip_length:
            # We should split a new clip here.
            current_clip_length = 0
            new_sequence_id += 1

        new_sequence_ids.append(new_sequence_id)

    # Set the new sequence IDs.
    annotations_mot[Otf.IMAGE_SEQUENCE_ID.value] = new_sequence_ids

    return annotations_mot


def shuffle_clips(annotations_mot: pd.DataFrame) -> pd.DataFrame:
    """
    Shuffles the order of clips.

    Args:
        annotations_mot: The annotations to shuffle. They should already have
            been split into clips.

    Returns:
        The same annotations, with the clips shuffled.

    """
    # Set the index to the sequence ID to speed up filtering operations.
    annotations_mot.set_index(
        Otf.IMAGE_SEQUENCE_ID.value, inplace=True, drop=False
    )

    clip_annotations = []
    for sequence_id in annotations_mot.index.unique():
        clip_annotations.append(
            annotations_mot.iloc[annotations_mot.index == sequence_id]
        )

    # Shuffle the clips.
    random.shuffle(clip_annotations)
    # Re-compose them back into a single dataframe.
    return pd.concat(clip_annotations, ignore_index=True)


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
    concat = partial(pd.concat, ignore_index=True)
    return concat(train_clips), concat(test_clips), concat(validation_clips)


def _transform_bounding_boxes(mot_annotations: pd.DataFrame) -> None:
    """
    Transforms bounding boxes from the x, y, width, height format that MOT 1.1
    uses to the x_min, x_max, y_min, y_max format that we use in TensorFlow.

    Args:
        mot_annotations: The annotations, in MOT 1.1 format. They will be
            modified in-place. It will use the TensorFlow column names.

    """
    x_min = mot_annotations["bb_left"]
    y_min = mot_annotations["bb_top"]
    x_max = x_min + mot_annotations["bb_width"]
    y_max = y_min + mot_annotations["bb_height"]

    # Use the right column names.
    mot_annotations.rename(
        {
            "bb_left": Otf.OBJECT_BBOX_X_MIN.value,
            "bb_top": Otf.OBJECT_BBOX_Y_MIN.value,
        },
        axis=1,
        inplace=True,
    )
    # Add the additional columns.
    mot_annotations[Otf.OBJECT_BBOX_X_MAX.value] = x_max
    mot_annotations[Otf.OBJECT_BBOX_Y_MAX.value] = y_max
    # Remove the extraneous columns.
    mot_annotations.drop(["bb_width", "bb_height"], axis=1, inplace=True)


def mot_to_object_detection_format(
    mot_annotations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converts annotations from the MOT 1.1 format to a format format that's
    more compatible with the TF Object Detection API.

    Args:
        mot_annotations: The raw annotations, in MOT 1.1 format.

    Returns:
        The transformed annotations.

    """
    # Update the bounding box format.
    _transform_bounding_boxes(mot_annotations)

    # Rename the remaining columns.
    mot_annotations.rename(
        {"frame": Otf.IMAGE_FRAME_NUM.value, "id": Otf.OBJECT_ID.value},
        axis=1,
        inplace=True,
    )
    # Drop columns that we don't care about.
    mot_annotations.drop(["conf", "x", "y", "z"], axis=1, inplace=True)

    return mot_annotations


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
    *, image: np.ndarray, frame_annotations: pd.DataFrame
) -> tf.train.Example:
    """
    Creates a TF `Example` for a single frame.

    Args:
        image: The compressed image data for the frame.
        frame_annotations: The annotations for the frame.

    Returns:
        The example that it created.

    """
    # Create the feature dictionary.
    features = {}
    for column_name, column_data in frame_annotations.items():
        # Collapse the frame number column since all frame numbers should be the
        # same.
        if column_name == Otf.IMAGE_FRAME_NUM.value:
            column_data = column_data.unique()

        features[column_name] = _FEATURES_TO_FACTORIES[column_name](
            column_data
        )

    # Add default values for missing columns.
    features.update(_get_missing_columns(frame_annotations))

    # Add the image data.
    features[Otf.IMAGE_ENCODED.value] = _FEATURES_TO_FACTORIES[
        Otf.IMAGE_ENCODED.value
    ](image)

    return tf.train.Example(features=tf.train.Features(feature=features))


def generate_examples(
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
    frame_nums = annotations[Otf.IMAGE_FRAME_NUM.value].unique().tolist()
    for frame_num in frame_nums:
        logger.debug("Generating example for frame {}.", frame_num)

        # Get all the annotations for that frame.
        frame_annotations = annotations[
            annotations[Otf.IMAGE_FRAME_NUM.value] == frame_num
        ]
        # Get the actual frame image.
        frame_image = video_frames.get_image(frame_num, compressed=True)

        yield _make_example(
            image=frame_image, frame_annotations=frame_annotations
        )
