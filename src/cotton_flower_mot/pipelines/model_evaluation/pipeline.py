"""
Model evaluation pipeline.
"""

from kedro.pipeline import Pipeline, node

from .nodes import compute_tracks_for_clip


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                compute_tracks_for_clip,
                dict(model="trained_model", clip_dataset="validation_data"),
                "validation_tracks",
            )
        ]
    )
