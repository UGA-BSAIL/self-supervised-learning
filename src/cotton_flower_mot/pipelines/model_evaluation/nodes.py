"""
Nodes for model evaluation pipeline.
"""


from typing import List

import tensorflow as tf
from loguru import logger

from ..schemas import ModelInputs
from .online_tracker import OnlineTracker, Track


def compute_tracks_for_clip(
    *, model: tf.keras.Model, clip_dataset: tf.data.Dataset
) -> List[Track]:
    """
    Computes the tracks for a given clip.

    Args:
        model: The model to use for track computation.
        clip_dataset: The dataset containing detections for each frame in the
            clip.

    Returns:
        The tracks that it computed.

    """
    logger.info("Computing tracks for clip...")

    tracker = OnlineTracker(model)
    for inputs, _ in clip_dataset:
        tracker.add_new_detections(
            detections=inputs[ModelInputs.DETECTIONS.value],
            geometry=inputs[ModelInputs.DETECTION_GEOMETRY.value],
        )

    return tracker.tracks
