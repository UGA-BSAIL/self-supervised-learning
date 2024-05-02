"""
Node definitions that are shared between pipelines.
"""


from typing import Any

import numpy as np
from ultralytics import YOLO

import wandb


def init_wandb(*, entity: str, **config: Any) -> None:
    """
    Initializes WandB at the beginning of a pipeline.

    Args:
        entity: The entity name to use. The name of the pipeline is generally
            a good choice.
        **config: These arguments will be interpreted as the configuration
            for WandB. Logging Kedro parameters here is often a good idea.

    """
    wandb.init(project="self_supervised", entity=entity, config=config)
    # Define "global_step" as the x-axis in all WanbB graphs.
    wandb.define_metric("*", step_metric="global_step")


def num_flowers_in_image(frame: np.ndarray, *, detector: YOLO) -> int:
    """
    Applies a YOLO model to a frame and returns the number of flowers detected.

    Args:
        frame: The frame to apply the model to.
        detector: The YOLO model to use.

    Returns:
        The number of flowers detected.

    """
    results = detector(frame, conf=0.1)
    return len(results[0].boxes)
