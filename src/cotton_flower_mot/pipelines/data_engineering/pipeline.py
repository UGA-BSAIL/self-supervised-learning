from functools import partial

from kedro.pipeline import Pipeline, node

from .nodes import (
    cut_video,
    merge_annotations,
    mot_to_object_detection_format,
    record_task_id,
    shuffle_clips,
    split_clips,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # Add the task IDs to the annotations.
            node(
                partial(record_task_id, task_id=169),
                "annotations_mot_1_1_169",
                "annotations_mot_169_ex",
            ),
            node(
                partial(record_task_id, task_id=170),
                "annotations_mot_1_1_170",
                "annotations_mot_170_ex",
            ),
            # Merge all annotations into one.
            node(
                merge_annotations,
                ["annotations_mot_169_ex", "annotations_mot_170_ex"],
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
