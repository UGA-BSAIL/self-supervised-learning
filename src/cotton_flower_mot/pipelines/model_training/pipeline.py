"""
Pipeline definition for model training.
"""

from kedro.pipeline import Pipeline, node

from ..training_utils import set_check_numerics
from .nodes import create_model, prepare_pretrained_encoder, train_model


def create_pipeline(init_rotnet: bool = False):
    """
    Args:
        init_rotnet: If true, will build a pipeline that initializes from
            a pretrained RotNet model.

    Returns:

    """
    if not init_rotnet:
        # This is the default model loading process.
        init_pipeline = Pipeline(
            [node(create_model, "model_config", "initial_model")]
        )
    else:
        # This is the model loading process using a pretrained RotNet encoder.
        init_pipeline = Pipeline(
            [
                node(
                    prepare_pretrained_encoder,
                    dict(
                        encoder="pretrained_rotnet_model",
                        config="model_config",
                        freeze_fraction="params:freeze_fraction",
                    ),
                    "encoder_model",
                ),
                node(
                    create_model,
                    dict(config="model_config", encoder="encoder_model"),
                    "initial_model",
                ),
            ],
        )

    return init_pipeline + Pipeline(
        [
            node(set_check_numerics, "params:enable_numeric_checks", None),
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
                ),
                "trained_model",
            ),
        ],
    )
