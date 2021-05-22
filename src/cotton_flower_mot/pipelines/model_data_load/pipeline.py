"""
Pipeline for loading TFRecords data.
"""

from functools import partial

from kedro.pipeline import Pipeline, node

from ..dataset_io import (
    DataAugmentationConfig,
    inputs_and_targets_from_datasets,
)
from .nodes import make_model_config


def create_pipeline(**kwargs):
    # Preset for loading testing and validation data that doesn't randomize
    # or interleave clips.
    load_clips = partial(
        inputs_and_targets_from_datasets,
        interleave=False,
        include_empty=True,
        include_frame=True,
        batch_size=1,
    )
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
                make_model_config,
                dict(
                    image_input_shape="params:image_input_shape",
                    num_appearance_features="params:num_appearance_features",
                    num_gcn_channels="params:num_gcn_channels",
                    sinkhorn_lambda="params:sinkhorn_lambda",
                ),
                "model_config",
            ),
            node(
                DataAugmentationConfig,
                dict(
                    max_bbox_jitter="params:bbox_jitter_fraction",
                    max_brightness_delta="params:max_brightness_delta",
                    max_hue_delta="params:max_hue_delta",
                    min_contrast="params:min_contrast",
                    max_contrast="params:max_contrast",
                    min_saturation="params:min_saturation",
                    max_saturation="params:max_saturation",
                ),
                "data_augmentation_config",
            ),
            # Load the datasets.
            node(
                inputs_and_targets_from_datasets,
                dict(
                    raw_datasets="tfrecord_train",
                    augmentation_config="data_augmentation_config",
                    **loading_config
                ),
                "training_data",
            ),
            node(
                inputs_and_targets_from_datasets,
                dict(raw_datasets="tfrecord_test", **loading_config),
                "testing_data",
            ),
            node(
                inputs_and_targets_from_datasets,
                dict(raw_datasets="tfrecord_valid", **loading_config),
                "validation_data",
            ),
            # Create clip datasets.
            node(
                load_clips,
                dict(
                    raw_datasets="tfrecord_test",
                    config="model_config",
                ),
                "testing_data_clips",
            ),
            node(
                load_clips,
                dict(
                    raw_datasets="tfrecord_valid",
                    config="model_config",
                ),
                "validation_data_clips",
            ),
        ]
    )
