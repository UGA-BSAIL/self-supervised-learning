"""
Nodes for the model training pipeline.
"""


from typing import Any, Dict, List, Tuple

import tensorflow as tf
from loguru import logger

from ..callbacks import LogHeatmaps
from ..config import ModelConfig
from ..schemas import ModelTargets
from ..training_utils import (
    get_log_dir,
    make_common_callbacks,
    make_learning_rate,
)
from .centernet_model import build_detection_model
from .losses import make_losses
from .metrics import make_metrics


def create_model(config: ModelConfig) -> tf.keras.Model:
    """
    Builds the model to use.

    Args:
        config: The model configuration.

    Returns:
        The model that it created.

    """
    model = build_detection_model(config)
    logger.info("Model has {} parameters.", model.count_params())

    return model


def _make_callbacks(
    *,
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    tensorboard_output_dir: str,
    heatmap_size: Tuple[int, int],
    heatmap_period: int,
    num_heatmap_batches: int,
    num_heatmap_images: int,
    **kwargs: Any,
) -> List[tf.keras.callbacks.Callback]:
    """
    Creates callbacks to use when training the model.

    Args:
        model: The model to log heatmaps from.
        dataset: The dataset to use for logging heatmaps.
        tensorboard_output_dir: The directory to use for storing Tensorboard
            logs.
        heatmap_size: Size of the logged heatmap visualizations.
            (width, height)
        heatmap_period: Period at which to generate heatmap visualizations,
            in epochs.
        num_heatmap_batches: Total number of batches to log heatmap data from.
        num_heatmap_images: Total number of heatmap images to include in each
            batch.
        **kwargs: Will be forwarded to `make_common_callbacks`.

    Returns:
        The list of callbacks.

    """
    common_callbacks = make_common_callbacks(
        tensorboard_output_dir=tensorboard_output_dir, **kwargs
    )

    log_dir = get_log_dir(tensorboard_output_dir)
    heatmap_callback = LogHeatmaps(
        model=model,
        dataset=dataset,
        log_dir=log_dir / "heatmaps",
        resize_images=heatmap_size,
        log_period=heatmap_period,
        max_num_batches=num_heatmap_batches,
        num_images_per_batch=num_heatmap_images,
    )

    return common_callbacks + [heatmap_callback]


def _remove_unused_targets(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Removes unused targets from a dataset.

    Args:
        dataset: The dataset to remove targets from.

    Returns:
        The modified dataset.

    """

    def _remove_targets(inputs: Dict, targets: Dict) -> Tuple[Dict, Dict]:
        targets.pop(ModelTargets.SINKHORN.value)
        targets.pop(ModelTargets.ASSIGNMENT.value)

        return inputs, targets

    return dataset.map(_remove_targets)


def train_model(
    model: tf.keras.Model,
    *,
    training_data: tf.data.Dataset,
    testing_data: tf.data.Dataset,
    learning_phases: List[Dict[str, Any]],
    validation_frequency: int = 1,
    loss_params: Dict[str, Any],
    heatmap_loss_weight: float = 1.0,
    geometry_loss_weight: float = 1.0,
    **kwargs: Any,
) -> tf.keras.Model:
    """
    Trains a model.

    Args:
        model: The model to train.
        training_data: The `Dataset` containing pre-processed training data.
        testing_data: The `Dataset` containing pre-processed testing data.
        learning_phases: List of hyperparameter configurations for each training
            stage, in order.
        validation_frequency: Number of training epochs after which to run
            validation.
        loss_params: Parameters to pass to the loss functions.
        heatmap_loss_weight: The loss weight for the heatmap focal loss.
        geometry_loss_weight: The loss weight for the L1 geometry loss.
        **kwargs: Will be forwarded to `_make_callbacks()`.

    Returns:
        The trained model.

    """
    # Clear unused tracking targets.
    training_data = _remove_unused_targets(training_data)
    testing_data = _remove_unused_targets(testing_data)

    for phase in learning_phases:
        logger.info("Starting new training phase.")
        logger.debug("Using phase parameters: {}", phase)

        callbacks = _make_callbacks(
            model=model,
            dataset=testing_data,
            training_params=phase,
            **kwargs,
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=make_learning_rate(phase["learning_rate"]),
        )
        model.compile(
            optimizer=optimizer,
            loss=make_losses(**loss_params),
            loss_weights={
                ModelTargets.HEATMAP.value: heatmap_loss_weight,
                ModelTargets.GEOMETRY_DENSE_PRED.value: geometry_loss_weight,
            },
            metrics=make_metrics(),
        )
        model.fit(
            training_data,
            validation_data=testing_data,
            epochs=phase["num_epochs"],
            callbacks=callbacks,
            validation_freq=validation_frequency,
        )

    return model
