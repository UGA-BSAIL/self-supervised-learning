"""
Pipeline definition for supervised validation.
"""


from functools import partial

from kedro.pipeline import Pipeline, node, pipeline

from ..common_nodes import init_wandb
from .dataset_io import Split, get_dataset
from .nodes import build_model, get_pretrained_encoder, train_model


def create_pipeline(**_) -> Pipeline:
    get_training_dataset = partial(get_dataset, split=Split.TRAINING)
    get_testing_dataset = partial(
        get_dataset, split=Split.TESTING, augment=False
    )

    # Common parameters when loading datasets.
    dataset_params = dict(
        dataset="params:super_dataset_path",
        image_size="params:super_input_size",
        heatmap_size="params:super_heatmap_size",
        batch_size="params:batch_size",
        hyperparams="valid_hyperparams",
        cache="params:enable_cache",
    )

    return pipeline(
        [
            # Initialize WandB.
            node(
                init_wandb,
                dict(
                    entity="params:wandb_entity",
                    num_epochs="params:super_num_epochs",
                    batch_size="params:super_batch_size",
                    learning_rate="params:super_learning_rate",
                    focal_alpha="params:super_focal_alpha",
                    focal_beta="params:super_focal_beta",
                    size_loss_weight="params:super_size_loss_weight",
                    offset_loss_weight="params:super_offset_loss_weight",
                    input_image_size="params:super_input_size",
                    heatmap_size="params:super_heatmap_size",
                    warmup_steps="params:super_warmup_steps",
                ),
                None,
            ),
            # Load the datasets.
            node(
                get_training_dataset,
                dataset_params,
                "training_data",
            ),
            node(
                get_testing_dataset,
                dataset_params,
                "testing_data",
            ),
            # Train the model.
            node(get_pretrained_encoder, None, "imagenet_encoder"),
            node(
                build_model,
                dict(
                    encoder="imagenet_encoder",
                ),
                "initial_model",
            ),
            node(
                train_model,
                dict(
                    model="initial_model",
                    training_data="training_data",
                    testing_data="testing_data",
                    num_epochs="params:super_num_epochs",
                    batch_size="params:super_batch_size",
                    learning_rate="params:super_learning_rate",
                    warmup_steps="params:super_warmup_steps",
                    focal_alpha="params:super_focal_alpha",
                    focal_beta="params:super_focal_beta",
                    size_loss_weight="params:super_size_loss_weight",
                    offset_loss_weight="params:super_offset_loss_weight",
                ),
                "trained_model",
            ),
        ]
    )
