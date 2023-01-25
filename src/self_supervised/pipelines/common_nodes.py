"""
Node definitions that are shared between pipelines.
"""


from typing import Any

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
