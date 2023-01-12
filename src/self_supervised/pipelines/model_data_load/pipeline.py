"""
Pipeline for loading TFRecords data.
"""

from functools import partial

from kedro.pipeline import Pipeline, node

from ..dataset_io import (
    DataAugmentationConfig,
    colorization_inputs_and_targets_from_dataset,
    rot_net_inputs_and_targets_from_dataset,
)
from .nodes import concat_datasets


def create_pipeline(**_):
    # Common configuration for standard dataset loading.
    loading_config = dict(
        config="model_config",
        batch_size="params:batch_size",
        drop_probability="params:drop_probability",
        repeats="params:repeats",
    )

    return Pipeline(
        [
            node(
                DataAugmentationConfig,
                dict(
                    max_brightness_delta="params:max_brightness_delta",
                    max_hue_delta="params:max_hue_delta",
                    min_contrast="params:min_contrast",
                    max_contrast="params:max_contrast",
                    min_saturation="params:min_saturation",
                    max_saturation="params:max_saturation",
                    flip="params:flip",
                    max_bbox_jitter="params:bbox_jitter_fraction",
                ),
                "data_augmentation_config",
            ),
            node(
                rot_net_inputs_and_targets_from_dataset,
                dict(
                    unannotated_dataset="tfrecord_unannotated_no_wheel_combined",
                    config="model_config",
                    batch_size="params:rot_net_batch_size",
                ),
                "rotnet_unannotated_data",
            ),
            node(
                concat_datasets,
                [
                    "tfrecord_unannotated_no_wheel_combined",
                    "tfrecord_unannotated_wheel_combined",
                ],
                "tfrecord_unannotated_all",
            ),
            node(
                colorization_inputs_and_targets_from_dataset,
                dict(
                    unannotated_dataset="tfrecord_unannotated_all",
                    config="model_config",
                    batch_size="params:colorization_batch_size",
                ),
                "colorization_unannotated_data",
            ),
        ]
    )
