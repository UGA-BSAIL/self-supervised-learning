"""
Pipeline for generating a dataset of MARS images.
"""


from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_dataset


def create_pipeline(**_) -> Pipeline:
    return pipeline(
        [
            node(
                build_dataset,
                dict(
                    dataset_spec="mars_dataset_spec",
                    image_dataset_path="params:image_dataset_path",
                    sync_tolerance="params:sync_tolerance",
                    max_timestamp_gap="params:max_timestamp_gap",
                    motion_threshold="params:motion_threshold",
                    green_threshold="params:green_threshold",
                ),
                "mars_dataset_meta",
            )
        ]
    )
