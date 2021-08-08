from kedro.pipeline import Pipeline, node

from .nodes import generate_examples, split_specific


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # Split into training, testing, and validation.
            node(
                split_specific,
                dict(
                    annotations="annotations_pandas",
                    test_clips="params:test_clips",
                    valid_clips="params:valid_clips",
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
                    annotations="annotations_tf_test",
                    config="model_config",
                ),
                "tfrecord_test",
            ),
            node(
                generate_examples,
                dict(
                    video_frames="cotton_videos",
                    annotations="annotations_tf_valid",
                    config="model_config",
                ),
                "tfrecord_valid",
            ),
            node(
                generate_examples,
                dict(
                    video_frames="cotton_videos",
                    annotations="annotations_tf_train",
                    config="model_config",
                ),
                "tfrecord_train",
            ),
        ]
    )
