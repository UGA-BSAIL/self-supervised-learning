from kedro.pipeline import Pipeline, node

from .nodes import (
    cut_video,
    merge_annotations,
    mot_to_object_detection_format,
    shuffle_clips,
    split_clips,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                cut_video,
                dict(
                    annotations_mot="annotations_mot_1_1",
                    new_length="params:use_first_frames",
                ),
                "annotations_mot_1_1_clipped",
            ),
            node(
                merge_annotations,
                ["annotations_mot_1_1_clipped"],
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
