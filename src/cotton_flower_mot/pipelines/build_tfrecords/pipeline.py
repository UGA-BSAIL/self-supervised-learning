from kedro.pipeline import Pipeline, node

from .nodes import generate_examples, random_splits


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                random_splits,
                dict(
                    annotations="annotations_pandas",
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
            node(
                generate_examples,
                dict(
                    video_frames="cotton_videos",
                    annotations="annotations_tf_train",
                ),
                "tfrecord_train",
            ),
        ]
    )
