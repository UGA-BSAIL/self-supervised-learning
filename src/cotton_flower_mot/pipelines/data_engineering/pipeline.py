from functools import partial
from typing import List

from kedro.pipeline import Pipeline, node
from kedro.pipeline.node import Node

from .nodes import (
    merge_annotations,
    mot_to_object_detection_format,
    record_task_id,
    shuffle_clips,
    split_clips,
)

_TASK_IDS = [169, 170, 172, 173, 174, 175, 190, 191]
"""
All task IDs of the data we are processing.
"""


def _task_specific_nodes(task_id: int) -> List[Node]:
    """
    Creates nodes that are specific to a particular task.

    Args:
        task_id: The task ID.

    Returns:
        The list of task-specific nodes.

    """
    return [
        node(
            partial(record_task_id, task_id=task_id),
            f"annotations_mot_1_1_{task_id}",
            f"annotations_mot_{task_id}_ex",
        ),
    ]


def create_pipeline(**kwargs):
    # Create all the task-specific nodes.
    task_specific_nodes = []
    for task_id in _TASK_IDS:
        task_specific_nodes.extend(_task_specific_nodes(task_id))

    return Pipeline(
        task_specific_nodes
        + [
            # This node is needed so we can pass an iterable of datasets as
            # input to the next node.
            node(
                lambda *args: args,
                [f"cotton_videos_{i}" for i in _TASK_IDS],
                "cotton_videos",
            ),
            # Merge all annotations into one.
            node(
                merge_annotations,
                [f"annotations_mot_{t}_ex" for t in _TASK_IDS],
                "annotations_mot_merged",
            ),
            node(
                split_clips,
                dict(
                    annotations_mot="annotations_mot_merged",
                    max_clip_length="params:max_clip_length",
                ),
                "annotations_mot_clips",
            ),
            node(
                shuffle_clips,
                "annotations_mot_clips",
                "annotations_mot_shuffled",
            ),
            node(
                mot_to_object_detection_format,
                "annotations_mot_shuffled",
                "annotations_pandas",
            ),
        ]
    )
