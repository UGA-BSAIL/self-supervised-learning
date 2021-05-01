from kedro.pipeline import Pipeline, node

from .nodes import generate_examples, split_random


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                split_random,
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
            # This node is needed so we can pass an iterable of datasets as
            # input to the next node.
            node(
                lambda *args: args,
                ["cotton_videos_169", "cotton_videos_170"],
                "cotton_videos",
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
