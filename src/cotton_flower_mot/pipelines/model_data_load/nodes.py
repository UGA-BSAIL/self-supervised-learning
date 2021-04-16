"""
Nodes for the `model_data_load` pipeline.
"""


from typing import Any

from loguru import logger

from ..config import ModelConfig


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
