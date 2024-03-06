"""
Pipeline definition for `train_simclr`.
"""

from kedro.pipeline import Pipeline, node, pipeline

from ..common_nodes import init_wandb
from .nodes import build_model, load_dataset, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                init_wandb,
                dict(
                    entity="params:wandb_entity",
                    num_epochs="params:num_epochs",
                    batch_size="params:batch_size",
                    learning_rate="params:learning_rate",
                    temperature="params:temperature",
                    max_frame_jitter="params:max_frame_jitter",
                ),
                None,
            ),
            # Start the training.
            node(
                load_dataset,
                dict(
                    image_folder="params:mars_image_folder",
                    metadata="mars_dataset_meta",
                    max_frame_jitter="params:max_frame_jitter",
                ),
                "training_data",
            ),
            node(build_model, "yolov8_l_description", "initial_model"),
            node(
                train_model,
                dict(
                    model="initial_model",
                    training_data="training_data",
                    num_epochs="params:num_epochs",
                    batch_size="params:batch_size",
                    learning_rate="params:learning_rate",
                    temperature="params:temperature",
                ),
                "trained_model",
            ),
        ]
    )
