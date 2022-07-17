"""
Implements a combined detection + tracking model.
"""


from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers

from ..config import ModelConfig
from ..schemas import ModelInputs
from .centernet_model import build_detection_model


def _extract_appearance_features(
    *,
    detections_geometry: tf.RaggedTensor,
    tracklets_geometry: tf.RaggedTensor,
    image_features: tf.Tensor
) -> tf.RaggedTensor:
    """
    Extracts the raw
    Args:
        detections_geometry:
        tracklets_geometry:
        image_features:

    Returns:

    """


# def build_combined_model(
#     config: ModelConfig,
#     encoder: Optional[tf.keras.Model] = None,
#     is_training: bool = True,
# ) -> tf.keras.Model:
#     """
#     Builds the combined detection + tracking model.
#
#     Args:
#         config: The model configuration.
#         encoder: A custom pretrained encoder which will be used for feature
#             extraction.
#         is_training: Whether to use the training or inference configuration
#             for the model. The primary difference is that the inference
#             configuration uses the inferred detections in the tracking step,
#             but the training configuration uses ground-truth detections,
#             which have to be input manually.
#
#     Returns:
#         The model it created.
#
#     """
#     # Build the detector/feature extractor.
#     image_feature_extractor, detector = build_detection_model(
#         config, encoder=encoder
#     )
#
#     # Input for the current video frame.
#     images = layers.Input(
#         shape=config.detection_model_input_shape,
#         name=ModelInputs.DETECTIONS_FRAME.value,
#     )
#
#     geometry_input_shape = (None, 4)
#     # In all cases, we need to manually provide the tracklet bounding boxes.
#     # These can either be the ground-truth, during training, or the
#     # detections from the previous frame, during online inference.
#     tracklet_geometry = layers.Input(
#         geometry_input_shape,
#         ragged=True,
#         name=ModelInputs.TRACKLET_GEOMETRY.value,
#     )
#     if is_training:
#         # Create the GT detection bounding box inputs, which are needed for
#         # training.
#         detection_geometry = layers.Input(
#             geometry_input_shape,
#             ragged=True,
#             name=ModelInputs.DETECTION_GEOMETRY.value,
#         )
#     else:
#         # Just grab the bounding boxes from the detector.
#         _, _, detection_geometry = detector(images)
