"""
This pipeline recomputes the flower numbers of an existing dataset, only
updating the metadata.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import recompute_flower_numbers


def create_pipeline(**_) -> Pipeline:
    return pipeline(
        [
            node(
                recompute_flower_numbers,
                dict(
                    mars_metadata="mars_dataset_meta",
                    image_folder="params:mars_image_folder",
                    detector="yolov8l_moco_round_1",
                ),
                "mars_dataset_meta_updated",
            ),
            node(
                lambda x: x,
                "mars_dataset_meta_updated",
                "mars_dataset_metadata",
                name="dummy_save_metadata",
            ),
        ]
    )
