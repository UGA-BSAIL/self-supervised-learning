"""
Pipeline for active learning. It orders the images in a somewhat-optimal
order for labelling.
"""


from kedro.pipeline import Pipeline, node, pipeline

from .nodes import save_image_reps


def create_pipeline(**_) -> Pipeline:
    return pipeline(
        [
            node(
                save_image_reps,
                dict(
                    metadata="mars_dataset_meta",
                    root_path="params:mars_image_folder",
                    model="trained_model_input",
                ),
                "mars_image_reps",
            )
        ]
    )
