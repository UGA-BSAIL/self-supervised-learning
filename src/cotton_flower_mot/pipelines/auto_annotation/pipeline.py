"""
Pipeline for generating automatic annotations using a pre-trained
model.
"""

from kedro.pipeline import Pipeline, node

from .nodes import annotate_video, convert_to_mot


def create_pipeline(**_):
    return Pipeline(
        [
            node(
                annotate_video,
                dict(video="cotton_videos_190", model="trained_model"),
                "raw_detections_190",
            ),
            node(
                convert_to_mot,
                dict(
                    detections="raw_detections_190",
                    video="cotton_videos_190",
                    min_confidence="params:min_confidence",
                ),
                "annotations_mot_1_1_190",
                tags="post_process",
            ),
        ]
    )
