"""
Pipeline for pre-training the model on the RotNet task.
"""

from kedro.pipeline import Pipeline, node, pipeline

from ..training_utils import set_check_numerics
from .nodes import create_model, train_model


def create_pipeline(**_) -> Pipeline:
    return pipeline(
        [
            node(set_check_numerics, "params:enable_numeric_checks", None),
            node(create_model, "model_config", "initial_rotnet_model"),
            node(
                train_model,
                dict(
                    model="initial_rotnet_model",
                    training_data="unannotated_data",
                    learning_phases="params:learning_phases",
                    tensorboard_output_dir="params:tensorboard_output_dir",
                    histogram_period="params:histogram_period",
                    update_period="params:update_period",
                    lr_patience_epochs="params:lr_patience_epochs",
                    min_lr="params:min_lr",
                ),
                "pretrained_rotnet_model",
            ),
        ]
    )
