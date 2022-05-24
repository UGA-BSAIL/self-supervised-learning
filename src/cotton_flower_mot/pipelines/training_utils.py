"""
Collection of miscellaneous utility functions used for training by various
pipelines.
"""


from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Union

import tensorflow as tf
import tensorflow.keras.optimizers.schedules as schedules
from loguru import logger

from .callbacks import ClearMemory


def make_learning_rate(
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


@lru_cache
def get_log_dir(tensorboard_output_dir: str) -> Path:
    """
    Gets the logging directory to use for this run.

    Args:
        tensorboard_output_dir: The base tensorboard output directory.

    Returns:
        The run-specific logging directory.

    """
    log_dir = Path(tensorboard_output_dir) / datetime.now().isoformat()
    logger.debug("Writing Tensorboard logs to {}.", log_dir)

    return log_dir


def make_common_callbacks(
    *,
    tensorboard_output_dir: str,
    histogram_period: int,
    update_period: int,
    lr_patience_epochs: int,
    min_lr: float,
) -> List[tf.keras.callbacks.Callback]:
    """
    Creates callbacks to use when training the model.

    Args:
        tensorboard_output_dir: The directory to use for storing Tensorboard
            logs.
        histogram_period: Period at which to generate histograms for
            Tensorboard output, in epochs.
        update_period: Period in batches at which to log metrics.
        lr_patience_epochs: Patience parameter to use for LR reduction.
        min_lr: The minimum learning rate to allow.

    Returns:
        The list of callbacks.

    """
    # Create a callback for storing Tensorboard logs.
    log_dir = get_log_dir(tensorboard_output_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=histogram_period,
        update_freq=update_period,
    )

    nan_termination = tf.keras.callbacks.TerminateOnNaN()

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        patience=lr_patience_epochs,
        verbose=1,
        min_lr=min_lr,
    )

    memory_callback = ClearMemory()

    return [
        tensorboard_callback,
        nan_termination,
        reduce_lr_callback,
        memory_callback,
    ]
