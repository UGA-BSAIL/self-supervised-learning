"""
Pipeline for generating a dataset of MARS images.
"""


from kedro.pipeline import Pipeline, node, pipeline

from .dataset import merge_datasets
from .nodes import build_dataset, load_from_spec

_DATASETS = [
    # "mars_flower_dataset",
    # "mars_flower_dataset_rs",
    # "gpheno_2020_dataset",
    # "mars_boll_dataset",
    # "mars_boll_2024_dataset",
    "synthetic_boll_dataset",
]


def create_pipeline(**_) -> Pipeline:
    # Read dataset specifications.
    datasets_nodes = [node(load_from_spec, f"{d}_spec", d) for d in _DATASETS]
    return pipeline(
        datasets_nodes
        + [
            # Combine into one.
            node(
                merge_datasets,
                _DATASETS,
                "mars_combined_dataset",
            ),
            # Build the dataset.
            node(
                build_dataset,
                dict(
                    dataset="mars_combined_dataset",
                    image_dataset_path="params:image_dataset_path",
                    # detection_model="yolov8l_moco_round_1",
                    sync_tolerance="params:sync_tolerance",
                    max_timestamp_gap="params:max_timestamp_gap",
                    motion_threshold="params:motion_threshold",
                    green_threshold="params:green_threshold",
                ),
                "mars_dataset_meta",
            ),
        ]
    )
