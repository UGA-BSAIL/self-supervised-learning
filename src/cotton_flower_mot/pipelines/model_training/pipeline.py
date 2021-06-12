"""
Pipeline definition for model training.
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    create_model,
    make_callbacks,
    set_check_numerics,
    train_model,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(set_check_numerics, "params:enable_numeric_checks", None),
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
                    testing_data="validation_data",
                    learning_phases="params:learning_phases",
                    loss_params="params:loss_params",
                    callbacks="callbacks",
                ),
                "trained_model",
            ),
        ]
    )
