"""
Pipeline for generating automatic annotations using a pre-trained
model.
"""

from kedro.pipeline import Pipeline, node

from .nodes import annotate_video, convert_to_mot

_TASK_NUMBER = 191
"""
The task to auto-annotate images from.
"""


def create_pipeline(**_):
    return Pipeline(
        [
            node(
                annotate_video,
                dict(
                    video=f"cotton_videos_{_TASK_NUMBER}",
                    model="trained_model",
                ),
                f"raw_detections_{_TASK_NUMBER}",
            ),
            node(
                convert_to_mot,
                dict(
                    detections=f"raw_detections_{_TASK_NUMBER}",
                    video=f"cotton_videos_{_TASK_NUMBER}",
                    min_confidence="params:min_confidence",
                    iou_threshold="params:iou_threshold",
                ),
                f"auto_annotations_mot_1_1_{_TASK_NUMBER}",
                tags="post_process",
            ),
        ]
    )
