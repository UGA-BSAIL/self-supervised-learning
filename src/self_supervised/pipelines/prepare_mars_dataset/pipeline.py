"""
Pipeline for generating a dataset of MARS images.
"""


from kedro.pipeline import Pipeline, node, pipeline

from .dataset import merge_datasets
from .nodes import build_dataset, load_from_spec


def create_pipeline(**_) -> Pipeline:
    return pipeline(
        [
            # Read dataset specifications.
            node(
                load_from_spec,
                "mars_flower_dataset_spec",
                "mars_flower_dataset",
            ),
            node(
                load_from_spec,
                "mars_flower_dataset_rs_spec",
                "mars_flower_dataset_rs",
            ),
            node(
                load_from_spec,
                "gpheno_2020_dataset_spec",
                "gpheno_flower_dataset",
            ),
            # Combine into one.
            node(
                merge_datasets,
                [
                    "mars_flower_dataset_rs",
                    "gpheno_flower_dataset",
                    "mars_flower_dataset",
                ],
                "mars_combined_dataset",
            ),
            # Build the dataset.
            node(
                build_dataset,
                dict(
                    dataset="mars_combined_dataset",
                    image_dataset_path="params:image_dataset_path",
                    sync_tolerance="params:sync_tolerance",
                    max_timestamp_gap="params:max_timestamp_gap",
                    motion_threshold="params:motion_threshold",
                    green_threshold="params:green_threshold",
                ),
                "mars_dataset_meta",
            ),
        ]
    )
