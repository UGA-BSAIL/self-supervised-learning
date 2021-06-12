"""
Nodes for the model training pipeline.
"""


from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import tensorflow as tf
import tensorflow.keras.optimizers.schedules as schedules
from loguru import logger

from ..config import ModelConfig
from .centernet_model import build_model
from .losses import make_losses
from .metrics import make_metrics


def _make_learning_rate(
    config: Dict[str, Any]
) -> Union[float, schedules.LearningRateSchedule]:
    """
    Creates the learning rate to use for optimization, based on the user
    configuration.

    Args:
        config: The configuration for the learning rate.

    Returns:
        Either a float for a fixed learning rate, or a `LearningRateSchedule`.

    """
    initial_rate = config["initial"]
    if not config.get("decay", False):
        # No decay is configured.
        logger.debug("Using fixed learning rate of {}.", initial_rate)
        return initial_rate

    logger.debug("Using decaying learning rate.")
    return tf.keras.experimental.CosineDecayRestarts(
        initial_rate,
        config["decay_steps"],
        t_mul=config["t_mul"],
        m_mul=config["m_mul"],
        alpha=config["min_learning_rate"],
    )


def set_check_numerics(enable: bool) -> None:
    """
    Sets whether to enable checking for NaNs and infinities.

    Args:
        enable: If true, will enable the checks.

    """
    if enable:
        logger.info("Enabling numeric checks. Training might be slow.")
        tf.debugging.enable_check_numerics()
    else:
        tf.debugging.disable_check_numerics()


def create_model(config: ModelConfig) -> tf.keras.Model:
    """
    Builds the model to use.

    Args:
        config: The model configuration.

    Returns:
        The model that it created.

    """
    model = build_model(config)
    logger.info("Model has {} parameters.", model.count_params())

    return model


def make_callbacks(
    *, tensorboard_output_dir: str, histogram_period: int, update_period: int
) -> List[tf.keras.callbacks.Callback]:
    """
    Creates callbacks to use when training the model.

    Args:
        tensorboard_output_dir: The directory to use for storing Tensorboard
            logs.
        histogram_period: Period at which to generate histograms for
            Tensorboard output, in epochs.
        update_period: Period in batches at which to log metrics.

    Returns:
        The list of callbacks.

    """
    # Create a callback for storing Tensorboard logs.
    log_dir = Path(tensorboard_output_dir) / datetime.now().isoformat()
    logger.debug("Writing Tensorboard logs to {}.", log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=histogram_period,
        update_freq=update_period,
    )

    nan_termination = tf.keras.callbacks.TerminateOnNaN()

    return [tensorboard_callback, nan_termination]


def train_model(
    model: tf.keras.Model,
    *,
    training_data: tf.data.Dataset,
    testing_data: tf.data.Dataset,
    learning_phases: List[Dict[str, Any]],
    callbacks: List[tf.keras.callbacks.Callback] = [],
    validation_frequency: int = 1,
    loss_params: Dict[str, Any],
) -> tf.keras.Model:
    """
    Trains a model.

    Args:
        model: The model to train.
        training_data: The `Dataset` containing pre-processed training data.
        testing_data: The `Dataset` containing pre-processed testing data.
        learning_phases: List of hyperparameter configurations for each training
            stage, in order.
        callbacks: The callbacks to use when training.
        validation_frequency: Number of training epochs after which to run
            validation.
        loss_params: Parameters to pass to the loss functions.

    Returns:
        The trained model.

    """
    for phase in learning_phases:
        logger.info("Starting new training phase.")
        logger.debug("Using phase parameters: {}", phase)

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=_make_learning_rate(phase["learning_rate"]),
            momentum=phase["momentum"],
            nesterov=True,
        )
        model.compile(
            optimizer=optimizer,
            loss=make_losses(**loss_params),
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
