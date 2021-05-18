"""
Pipeline definition for annotation conversion.
"""

from kedro.pipeline import Pipeline, node

from .nodes import convert_to_mot_format, merge_frame_annotations


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # Merge annotations from each frame.
            node(
                merge_frame_annotations,
                "annotations_mot_chenjiao_1",
                "annotations_merged_1",
            ),
            node(
                merge_frame_annotations,
                "annotations_mot_chenjiao_2",
                "annotations_merged_2",
            ),
            node(
                merge_frame_annotations,
                "annotations_mot_chenjiao_3",
                "annotations_merged_3",
            ),
            node(
                merge_frame_annotations,
                "annotations_mot_chenjiao_4",
                "annotations_merged_4",
            ),
            # Convert to MOT 1.1 format.
            node(
                convert_to_mot_format,
                dict(
                    annotations="annotations_merged_1",
                    image_size="params:chenjiao_data_image_size",
                ),
                "converted_annotations_mot_1_1_video_1",
            ),
            node(
                convert_to_mot_format,
                dict(
                    annotations="annotations_merged_2",
                    image_size="params:chenjiao_data_image_size",
                ),
                "converted_annotations_mot_1_1_video_2",
            ),
            node(
                convert_to_mot_format,
                dict(
                    annotations="annotations_merged_3",
                    image_size="params:chenjiao_data_image_size",
                ),
                "converted_annotations_mot_1_1_video_3",
            ),
            node(
                convert_to_mot_format,
                dict(
                    annotations="annotations_merged_4",
                    image_size="params:chenjiao_data_image_size",
                ),
                "converted_annotations_mot_1_1_video_4",
            ),
        ]
    )
