import itertools
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import combine_session_examples, generate_multiple_video_examples

_WHEEL_SESSIONS = [
    "2021-09-03_left",
    "2021-09-03_right",
    "2021-09-10_left",
    "2021-09-10_right",
    "2021-09-24_left",
    "2021-09-24_right",
    "2021-10-15_left",
    "2021-10-15_right",
    "2021-10-22_left",
    "2021-10-22_right",
]
"""
The names of all the sessions we are processing which have parts of the robot
structure visible.
"""

_NO_WHEEL_SESSIONS = [
    "2021-08-25_left",
    "2021-08-25_right",
    "2021-09-03_right",
    "2021-09-10_left",
    "2021-09-10_right",
    "2021-09-24_left",
    "2021-09-24_right",
    "2021-10-01_right",
    "2021-10-15_left",
    "2021-10-15_right",
    "2021-10-22_left",
    "2021-10-22_right",
]
"""
The names of all the sessions we are processing which have none of the robot
structure visible.
"""


def _make_dataset_pipeline(dataset_slug: str) -> Pipeline:
    """
    Generates a pipeline for a particular dataset.

    Args:
        dataset_slug: The unique part of the dataset name.

    Returns:
        The pipeline that it generated.

    """
    return pipeline(
        [
            node(
                generate_multiple_video_examples,
                dict(
                    videos=f"mars_{dataset_slug}",
                    clip_length="params:clip_length",
                    skip_initial_frames="params:skip_initial_frames",
                    frame_shape="params:unannotated_frame_shape",
                    histogram_shape="params:histogram_shape",
                ),
                f"{dataset_slug}_tfrecord_unannotated",
            )
        ]
    )


def create_pipeline(**_) -> Pipeline:
    # Generate all the dataset names for each session.
    wheel_dataset_names = [f"{s}_wheel" for s in _WHEEL_SESSIONS]
    no_wheel_dataset_names = [f"{s}_no_wheel" for s in _NO_WHEEL_SESSIONS]

    wheel_pipeline = sum(
        [_make_dataset_pipeline(n) for n in wheel_dataset_names]
    )
    no_wheel_pipeline = sum(
        [_make_dataset_pipeline(n) for n in no_wheel_dataset_names]
    )

    wheel_pipeline += pipeline(
        [
            # Combine all the session examples.
            node(
                combine_session_examples,
                [f"{n}_tfrecord_unannotated" for n in wheel_dataset_names],
                "tfrecord_unannotated_wheel",
            )
        ]
    )
    no_wheel_pipeline += pipeline(
        [
            # Combine all the session examples.
            node(
                combine_session_examples,
                [f"{n}_tfrecord_unannotated" for n in no_wheel_dataset_names],
                "tfrecord_unannotated_no_wheel",
            )
        ]
    )

    return wheel_pipeline + no_wheel_pipeline
