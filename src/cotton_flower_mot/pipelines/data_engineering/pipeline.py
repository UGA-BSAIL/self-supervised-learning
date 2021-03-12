from kedro.pipeline import Pipeline, node

from .nodes import (
    cut_video,
    generate_examples,
    merge_annotations,
    mot_to_object_detection_format,
    random_splits,
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
                "annotations_tf",
            ),
            node(
                random_splits,
                dict(
                    annotations="annotations_tf",
                    train_fraction="params:train_fraction",
                    test_fraction="params:test_fraction",
                ),
                [
                    "annotations_tf_train",
                    "annotations_tf_test",
                    "annotations_tf_valid",
                ],
            ),
            # Generate TFRecords from all splits.
            node(
                generate_examples,
                dict(
                    video_frames="cotton_videos",
                    annotations="annotations_tf_train",
                ),
                "tfrecord_train",
            ),
            node(
                generate_examples,
                dict(
                    video_frames="cotton_videos",
                    annotations="annotations_tf_test",
                ),
                "tfrecord_test",
            ),
            node(
                generate_examples,
                dict(
                    video_frames="cotton_videos",
                    annotations="annotations_tf_valid",
                ),
                "tfrecord_valid",
            ),
        ]
    )
