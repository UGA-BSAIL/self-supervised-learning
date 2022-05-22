from kedro.pipeline import Pipeline, node, pipeline

from .nodes import combine_session_examples, generate_multiple_video_examples

_SESSIONS = [
    "2021-08-25_left",
    "2021-08-25_right",
    "2021-09-03_left",
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
The names of all the sessions we are processing.
"""


def _make_session_pipeline(session_name: str) -> Pipeline:
    """
    Generates a pipeline for a particular session.

    Args:
        session_name: The name of the session.

    Returns:
        The pipeline that it generated.

    """
    return pipeline(
        [
            node(
                generate_multiple_video_examples,
                dict(
                    videos=f"mars_{session_name}",
                    clip_length="params:clip_length",
                    skip_initial_frames="params:skip_initial_frames",
                ),
                f"{session_name}_tfrecord_unannotated",
            )
        ]
    )


def create_pipeline(**_) -> Pipeline:
    my_pipeline = sum([_make_session_pipeline(name) for name in _SESSIONS])
    my_pipeline += pipeline(
        [
            # Combine all the session examples.
            node(
                combine_session_examples,
                [f"{s}_tfrecord_unannotated" for s in _SESSIONS],
                "tfrecord_unannotated",
            )
        ]
    )

    return my_pipeline
