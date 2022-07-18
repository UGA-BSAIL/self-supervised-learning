"""
Implements a combined detection + tracking model.
"""


from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from ..config import ModelConfig
from ..schemas import ModelInputs
from .centernet_model import build_detection_model
from .gcnn_model import compute_association
from .layers.pooling import RoiPooling
from .layers.utility import BnActDense


def _extract_appearance_features(
    *,
    bbox_geometry: tf.RaggedTensor,
    image_features: tf.Tensor,
    config: ModelConfig
) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:
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
    features = layers.Lambda(lambda r: r.merge_dims(2, -1))(feature_crops)
    return BnActDense(config.num_appearance_features, activation="relu")(
        features
    )


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

    geometry_input_shape = (None, 4)
    # In all cases, we need to manually provide the tracklet bounding boxes.
    # These can either be the ground-truth, during training, or the
    # detections from the previous frame, during online inference.
    tracklet_geometry = layers.Input(
        geometry_input_shape,
        ragged=True,
        name=ModelInputs.TRACKLET_GEOMETRY.value,
    )
    model_inputs = [current_frames, previous_frames, tracklet_geometry]
    if is_training:
        # Create the GT detection bounding box inputs, which are needed for
        # training.
        detection_geometry = layers.Input(
            geometry_input_shape,
            ragged=True,
            name=ModelInputs.DETECTION_GEOMETRY.value,
        )
        model_inputs.append(detection_geometry)
    else:
        # Just grab the bounding boxes from the detector.
        _, _, detection_geometry = detector(current_frames)
        # Remove the confidence scores.
        detection_geometry = detection_geometry[:, :, :4]

    # Extract the image features.
    current_frame_features = image_feature_extractor(current_frames)
    previous_frame_features = image_feature_extractor(previous_frames)

    # Extract the appearance features.
    detection_features = _extract_appearance_features(
        bbox_geometry=detection_geometry,
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
        detections_geometry=detection_geometry,
        tracklets_geometry=tracklet_geometry,
        config=config,
    )
    return tf.keras.Model(
        inputs=model_inputs,
        outputs=[sinkhorn, hungarian],
        name="gcnnmatch_end_to_end",
    )
