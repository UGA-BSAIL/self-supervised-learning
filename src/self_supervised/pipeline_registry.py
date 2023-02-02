from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines import (
    eda,
    prepare_mars_dataset,
    supervised_validation,
    train_simclr,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    eda_pipeline = eda.create_pipeline()
    mars_pipeline = prepare_mars_dataset.create_pipeline()
    simclr_pipeline = train_simclr.create_pipeline()
    valid_pipeline = supervised_validation.create_pipeline()

    return {
        "__default__": simclr_pipeline,
        "eda": eda_pipeline,
        "prepare_mars_dataset": mars_pipeline,
        "train_simclr": simclr_pipeline,
        "supervised_validation": valid_pipeline,
    }
