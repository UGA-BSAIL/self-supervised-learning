"""
Pipeline definition for model training.
"""

from kedro.pipeline import Pipeline, node

from .nodes import create_model, set_check_numerics, train_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(set_check_numerics, "params:enable_numeric_checks", None),
            node(create_model, "model_config", "initial_model"),
            node(
                train_model,
                dict(
                    model="initial_model",
                    training_data="training_data",
                    testing_data="validation_data",
                    learning_phases="params:learning_phases",
                    loss_params="params:loss_params",
                    heatmap_loss_weight="params:heatmap_loss_weight",
                    geometry_loss_weight="params:geometry_loss_weight",
                    tensorboard_output_dir="params:tensorboard_output_dir",
                    histogram_period="params:histogram_period",
                    update_period="params:update_period",
                    heatmap_size="params:heatmap_size",
                    heatmap_period="params:heatmap_period",
                    num_heatmap_batches="params:num_heatmap_batches",
                    num_heatmap_images="params:num_heatmap_images",
                    lr_patience_epochs="params:lr_patience_epochs",
                    min_lr="params:min_lr",
                ),
                "trained_model",
            ),
        ]
    )
