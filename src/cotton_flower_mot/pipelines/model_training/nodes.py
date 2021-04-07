"""
Nodes for the model training pipeline.
"""


from typing import Any, Dict, List, Union

import tensorflow as tf
import tensorflow.keras.optimizers.schedules as schedules
from loguru import logger

from .gcnn_model import ModelConfig, build_model
from .losses import make_losses


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


def _make_metrics() -> Dict[str, List[tf.keras.metrics.Metric]]:
    """
    Creates the metrics to use for the model.

    Returns:
        The metrics that it created.

    """
    return {}


def make_model_config(**kwargs: Any) -> ModelConfig:
    """
    Creates a configuration to use for the model.

    Args:
        **kwargs: Will be forwarded to the `ModelConfig` constructor.

    Returns:
        The `ModelConfig` that it created.

    """
    logger.debug("Creating model configuration: {}", kwargs)
    return ModelConfig(**kwargs)


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


def train_model(
    model: tf.keras.Model,
    *,
    training_data: tf.data.Dataset,
    testing_data: tf.data.Dataset,
    learning_phases: List[Dict[str, Any]],
    callbacks: List[tf.keras.callbacks.Callback] = [],
    validation_frequency: int = 1,
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
            loss=make_losses(),
            metrics=_make_metrics(),
        )
        model.fit(
            training_data,
            validation_data=testing_data,
            epochs=phase["num_epochs"],
            callbacks=callbacks,
            validation_freq=validation_frequency,
        )

    return model
