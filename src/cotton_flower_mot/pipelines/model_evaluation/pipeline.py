"""
Model evaluation pipeline.
"""

from kedro.pipeline import Pipeline, node

from .nodes import compute_counts, compute_tracks_for_clip


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # Compute online tracks.
            node(
                compute_tracks_for_clip,
                dict(model="trained_model", clip_dataset="testing_data_clips"),
                "testing_tracks",
            ),
            node(
                compute_tracks_for_clip,
                dict(
                    model="trained_model", clip_dataset="validation_data_clips"
                ),
                "validation_tracks",
            ),
            # Create count reports.
            node(
                compute_counts,
                dict(
                    tracks_from_clips="testing_tracks",
                    annotations="annotations_pandas",
                ),
                "count_report_test",
            ),
            node(
                compute_counts,
                dict(
                    tracks_from_clips="validation_tracks",
                    annotations="annotations_pandas",
                ),
                "count_report_valid",
            ),
        ]
    )
