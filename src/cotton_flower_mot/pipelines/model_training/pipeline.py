"""
Pipeline definition for model training.
"""

from kedro.pipeline import Pipeline, node

from ..dataset_io import inputs_and_targets_from_datasets
from .nodes import create_model, make_callbacks, make_model_config, train_model


def create_pipeline(**kwargs):
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
            # Load the datasets.
            node(
                inputs_and_targets_from_datasets,
                dict(
                    raw_datasets="tfrecord_train",
                    config="model_config",
                    batch_size="params:batch_size",
                ),
                "training_data",
            ),
            node(
                inputs_and_targets_from_datasets,
                dict(
                    raw_datasets="tfrecord_test",
                    config="model_config",
                    batch_size="params:batch_size",
                ),
                "testing_data",
            ),
            node(
                inputs_and_targets_from_datasets,
                dict(
                    raw_datasets="tfrecord_valid",
                    config="model_config",
                    batch_size="params:batch_size",
                ),
                "validation_data",
            ),
            node(create_model, "model_config", "initial_model"),
            node(
                make_callbacks,
                dict(
                    tensorboard_output_dir="params:tensorboard_output_dir",
                    histogram_period="params:histogram_period",
                    update_period="params:update_period",
                ),
                "callbacks",
            ),
            node(
                train_model,
                dict(
                    model="initial_model",
                    training_data="training_data",
                    testing_data="testing_data",
                    learning_phases="params:learning_phases",
                    positive_sample_weight="params:positive_sample_weight",
                    callbacks="callbacks",
                ),
                "trained_model",
            ),
        ]
    )
