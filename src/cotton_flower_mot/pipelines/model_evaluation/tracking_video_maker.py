"""
Framework for creating tracking videos.
"""

from typing import Dict, Iterable, List

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from ..schemas import ModelInputs
from .online_tracker import Track

_BOX_COLORS = np.random.randint(0, 255, (10, 3))
"""
Bounding box colors to use.
"""


def _draw_bounding_box(
    artist: ImageDraw.ImageDraw, *, track: Track, box: np.ndarray
) -> None:
    """
    Draws a bounding box.

    Args:
        artist: The `ImageDraw` object to draw with.
        track: The track that we are drawing a box for.
        box: The box to be drawn, in the form
            `[center_x, center_y, width, height]`.
    """
    # Convert the box to a form we can draw with.
    center = box[:2]
    size = box[2:]
    min_point = center - size // 2
    max_point = center + size // 2
    min_point = tuple(min_point)
    max_point = tuple(max_point)

    # Choose a color.
    color_index = hash(track) % len(_BOX_COLORS)
    color = _BOX_COLORS[color_index]

    artist.rectangle([min_point, max_point], outline=tuple(color), width=3)


def draw_track_frame(
    frame: np.ndarray,
    *,
    frame_num: int,
    tracks: List[Track],
    geometry: np.ndarray
) -> np.ndarray:
    """
    Draws the tracks on a single frame.

    Args:
        frame: The frame to draw on. Will be modified in-place.
        frame_num: The frame number of this frame.
        tracks: The tracks to draw.
        geometry: The associated detection geometry for this frame. Should
            have shape `[num_detections, 4]`.

    Returns:
        The modified frame.

    """
    frame = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(frame)

    # Determine the associated bounding box for all the tracks.
    for track in tracks:
        detection_index = track.index_for_frame(frame_num)
        if detection_index is None:
            # No detection for this track at this frame.
            continue
        bounding_box = geometry[detection_index]

        # Draw the bounding box.
        _draw_bounding_box(draw, track=track, box=bounding_box)

    return np.array(frame)


def draw_tracks(
    inputs: Iterable[Dict[str, tf.Tensor]], *, tracks: List[Track]
) -> Iterable[np.ndarray]:
    """
    Draws the tracks on top of a video.

    Args:
        inputs: Dictionary containing the input data, organized according
            to the keys in `ModelInputs`.
        tracks: The tracks to draw.

    Yields:
        Each frame, with the tracks drawn on it.

    """
    for frame_num, feature_dict in enumerate(inputs):
        frame = feature_dict[ModelInputs.FRAME.value].numpy()
        geometry = feature_dict[ModelInputs.DETECTION_GEOMETRY.value].numpy()

        frame = draw_track_frame(
            frame, frame_num=frame_num, tracks=tracks, geometry=geometry
        )

        yield frame
