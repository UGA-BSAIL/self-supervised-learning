from kedro.pipeline import Pipeline, node

from .nodes import annotation_size, annotations_per_frame, track_length


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                annotation_size, "annotations_pandas", "bounding_box_size_dist"
            ),
            node(
                annotations_per_frame,
                "annotations_pandas",
                "annotations_per_frame",
            ),
            node(
                track_length,
                "annotations_pandas",
                "track_lengths",
            ),
        ]
    )
