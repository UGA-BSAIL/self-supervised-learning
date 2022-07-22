"""
Implements a combined detection + tracking model.
"""


from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ..config import ModelConfig
from ..schemas import ModelInputs, ModelTargets
from .centernet_model import build_detection_model
from .gcnn_model import compute_association
from .layers.pooling import RoiPooling


def _extract_appearance_features(
    *,
    bbox_geometry: tf.RaggedTensor,
    image_features: tf.Tensor,
    config: ModelConfig
) -> tf.RaggedTensor:
    """
    Extracts the appearance features for the detections or tracklets.

    Args:
        bbox_geometry: The bounding box information. Should have
            shape `[batch_size, num_boxes, 4]`, where the second dimension is
            ragged, and the third is ordered `[x, y, width, height]`.
            tracklets.
        image_features: The raw image features from the detector.
        config: The model configuration to use.

    Returns:
        The extracted appearance features. They are a `RaggedTensor`
        with the shape `[batch_size, n_nodes, n_features]`, where the second
        dimension is ragged.

    """
    feature_crops = RoiPooling(config.roi_pooling_size)(
        (image_features, bbox_geometry)
    )

    # Coerce the features to the correct shape.
    def _flatten_features(_features: tf.RaggedTensor) -> tf.RaggedTensor:
        # We have to go through this annoying process to ensure that the
        # static shape remains correct.
        inner_shape = _features.shape[-3:]
        num_flat_features = np.prod(inner_shape)
        flat_features = tf.reshape(_features.values, (-1, num_flat_features))
        return _features.with_values(flat_features)

    features = layers.Lambda(_flatten_features)(feature_crops)
    return layers.Dense(config.num_appearance_features, activation="relu")(
        features
    )


def _apply_detector(
    detector: tf.keras.Model, *, frames: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.RaggedTensor]:
    """
    Applies the detector model to an input.

    Args:
        detector: The detector model.
        frames: The input frames.

    Returns:
        The heatmaps, dense geometry predictions, and bounding boxes.

    """
    heatmap, dense_geometry, bboxes = detector(frames)

    # Ensure that the resulting layers have the correct names when we set
    # them as outputs.
    heatmap = layers.Lambda(lambda x: x, name=ModelTargets.HEATMAP.value)(
        heatmap
    )
    dense_geometry = layers.Lambda(
        lambda x: x, name=ModelTargets.GEOMETRY_DENSE_PRED.value
    )(dense_geometry)
    bboxes = layers.Lambda(
        lambda x: x, name=ModelTargets.GEOMETRY_SPARSE_PRED.value
    )(bboxes)

    return heatmap, dense_geometry, bboxes


def build_combined_model(
    config: ModelConfig,
    encoder: Optional[tf.keras.Model] = None,
    is_training: bool = True,
) -> tf.keras.Model:
    """
    Builds the combined detection + tracking model.

    Args:
        config: The model configuration.
        encoder: A custom pretrained encoder which will be used for feature
            extraction.
        is_training: Whether to use the training or inference configuration
            for the model. The primary difference is that the inference
            configuration uses the inferred detections in the tracking step,
            but the training configuration uses ground-truth detections,
            which have to be input manually.

    Returns:
        The model it created.

    """
    # Build the detector/feature extractor.
    image_feature_extractor, detector = build_detection_model(
        config, encoder=encoder
    )

    # Input for the current video frame.
    current_frames = layers.Input(
        shape=config.detection_model_input_shape,
        name=ModelInputs.DETECTIONS_FRAME.value,
    )
    # Input for the previous video frame.
    previous_frames = layers.Input(
        shape=config.detection_model_input_shape,
        name=ModelInputs.TRACKLETS_FRAME.value,
    )

    # Apply the detection model to the input frames.
    (
        detection_heatmap,
        detection_dense_geometry,
        detection_bboxes,
    ) = _apply_detector(detector, frames=current_frames)

    geometry_input_shape = (None, 4)
    # In all cases, we need to manually provide the tracklet bounding boxes.
    # These can either be the ground-truth, during training, or the
    # detections from the previous frame, during online inference.
    tracklet_geometry = layers.Input(
        geometry_input_shape,
        ragged=True,
        name=ModelInputs.TRACKLET_GEOMETRY.value,
    )
    # Default to using
    model_inputs = [current_frames, previous_frames, tracklet_geometry]
    if is_training:
        # Create the GT detection bounding box inputs, which are needed for
        # training.
        detection_geometry_for_tracker = layers.Input(
            geometry_input_shape,
            ragged=True,
            name=ModelInputs.DETECTION_GEOMETRY.value,
        )
        model_inputs.append(detection_geometry_for_tracker)
    else:
        # Just grab the bounding boxes from the detector.
        detection_geometry_for_tracker = detection_bboxes
        # Remove the confidence scores.
        detection_geometry_for_tracker = detection_geometry_for_tracker[
            :, :, :4
        ]

    # Extract the image features.
    current_frame_features = image_feature_extractor(current_frames)
    previous_frame_features = image_feature_extractor(previous_frames)

    # Extract the appearance features.
    detection_features = _extract_appearance_features(
        bbox_geometry=detection_geometry_for_tracker,
        image_features=current_frame_features,
        config=config,
    )
    tracklet_features = _extract_appearance_features(
        bbox_geometry=tracklet_geometry,
        image_features=previous_frame_features,
        config=config,
    )

    sinkhorn, hungarian = compute_association(
        detections_app_features=detection_features,
        tracklets_app_features=tracklet_features,
        detections_geometry=detection_geometry_for_tracker,
        tracklets_geometry=tracklet_geometry,
        config=config,
    )
    return tf.keras.Model(
        inputs=model_inputs,
        outputs=[
            detection_heatmap,
            detection_dense_geometry,
            detection_bboxes,
            sinkhorn,
            hungarian,
        ],
        name="gcnnmatch_end_to_end",
    )
