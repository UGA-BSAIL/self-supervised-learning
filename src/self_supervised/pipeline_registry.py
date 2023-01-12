from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines import (
    build_unannotated_tfrecords,
    eda,
    model_config,
    model_data_load,
    model_evaluation,
    model_training,
    train_colorization,
    train_rotnet,
    prepare_mars_dataset,
)
from .pipelines.model_training.pipeline import ModelInit


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    config_pipeline = model_config.create_pipeline()
    unannotated_pipeline = build_unannotated_tfrecords.create_pipeline()
    eda_pipeline = eda.create_pipeline()
    data_load_pipeline = config_pipeline + model_data_load.create_pipeline()
    training_pipeline = data_load_pipeline + model_training.create_pipeline()
    training_pipeline_rotnet_init = (
        data_load_pipeline
        + model_training.create_pipeline(model_init=ModelInit.ROTNET)
    )
    training_pipeline_colorization_init = (
        data_load_pipeline
        + model_training.create_pipeline(model_init=ModelInit.COLORIZATION)
    )
    rotnet_pipeline = data_load_pipeline + train_rotnet.create_pipeline()
    colorization_pipeline = (
        data_load_pipeline + train_colorization.create_pipeline()
    )
    evaluation_pipeline = (
        data_load_pipeline + model_evaluation.create_pipeline()
    )
    mars_pipeline = prepare_mars_dataset.create_pipeline()

    return {
        "__default__": eda_pipeline + training_pipeline,
        "build_unannotated_tfrecords": unannotated_pipeline,
        "eda": eda_pipeline,
        "model_training": training_pipeline,
        "model_training_rotnet_init": training_pipeline_rotnet_init,
        "model_training_colorization_init": training_pipeline_colorization_init,
        "model_evaluation": evaluation_pipeline,
        "train_rotnet": rotnet_pipeline,
        "train_colorization": colorization_pipeline,
        "prepare_mars_dataset": mars_pipeline,
    }
