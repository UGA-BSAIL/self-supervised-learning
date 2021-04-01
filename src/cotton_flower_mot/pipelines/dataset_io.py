import enum
from functools import partial
from multiprocessing import cpu_count
from typing import Dict, Iterable, Tuple, Union

import tensorflow as tf

from .model_training.gcnn_model import ModelConfig
from .model_training.sinkhorn import construct_gt_sinkhorn_matrix
from .schemas import ModelInputs, ModelTargets
from .schemas import ObjectTrackingFeatures as Otf

_FEATURE_DESCRIPTION = {
    Otf.IMAGE_HEIGHT.value: tf.io.FixedLenFeature([1], tf.dtypes.int64),
    Otf.IMAGE_WIDTH.value: tf.io.FixedLenFeature([1], tf.dtypes.int64),
    Otf.IMAGE_FILENAME.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
    Otf.IMAGE_SOURCE_ID.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
    Otf.IMAGE_ENCODED.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
    Otf.IMAGE_FORMAT.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
    Otf.OBJECT_BBOX_X_MIN.value: tf.io.RaggedFeature(tf.dtypes.float32),
    Otf.OBJECT_BBOX_X_MAX.value: tf.io.RaggedFeature(tf.dtypes.float32),
    Otf.OBJECT_BBOX_Y_MIN.value: tf.io.RaggedFeature(tf.dtypes.float32),
    Otf.OBJECT_BBOX_Y_MAX.value: tf.io.RaggedFeature(tf.dtypes.float32),
    Otf.OBJECT_CLASS_TEXT.value: tf.io.RaggedFeature(tf.dtypes.string),
    Otf.OBJECT_CLASS_LABEL.value: tf.io.RaggedFeature(tf.dtypes.int64),
    Otf.OBJECT_ID.value: tf.io.RaggedFeature(tf.dtypes.int64),
    Otf.IMAGE_SEQUENCE_ID.value: tf.io.RaggedFeature(tf.dtypes.int64),
    Otf.IMAGE_FRAME_NUM.value: tf.io.RaggedFeature(tf.dtypes.int64),
}
"""
Descriptions of the features found in the dataset containing flower annotations.
"""


Feature = Dict[str, tf.Tensor]
"""
Feature dictionary that contains only normal tensors.
"""
RaggedFeature = Dict[str, tf.RaggedTensor]
"""
Feature dictionary that contains only ragged tensors.
"""
MaybeRaggedFeature = Dict[str, Union[tf.Tensor, tf.RaggedTensor]]
"""
Feature dictionary that may contain normal or ragged tensors.
"""


@enum.unique
class FeatureNames(enum.Enum):
    """
    Standard key names for processed features.
    """

    DETECTIONS = "detections"
    """
    Extracted detection crops.
    """
    GEOMETRY = "geometry"
    """
    Geometric features.
    """
    OBJECT_IDS = "object_ids"
    """
    Object IDs.
    """
    FRAME_NUM = "frame_num"
    """
    The frame number in the underlying video clip.
    """


_INPUT_FEATURES = [Otf.IMAGE_ENCODED.value]
"""
The features that will be used as input to the model.
"""

_NUM_THREADS = cpu_count()
"""
Number of threads to use for multi-threaded operations.
"""


def _get_geometric_features(bbox_coords: tf.Tensor) -> tf.Tensor:
    """
    Converts a batch of bounding boxes from the format in TFRecords to the
    single-tensor geometric feature format.

    Args:
        bbox_coords: The bounding box coordinates. Should have the shape
            `[N, 4]`, where each row takes the form
            `[min_y, min_x, max_y, max_x]`.

    Returns:
        A tensor of shape [N, 4], where N is the total number of bounding
        boxes in the input. The ith row specifies the coordinates of the ith
        bounding box in the form [center_x, center_y, width, height].

    """
    x_min = bbox_coords[1]
    x_max = bbox_coords[3]
    y_min = bbox_coords[0]
    y_max = bbox_coords[2]

    width_x = x_max - x_min
    width_y = y_max - y_min
    center_x = x_min + width_x / tf.constant(2.0)
    center_y = y_min + width_y / tf.constant(2.0)

    return tf.stack([center_x, center_y, width_x, width_y], axis=1)


def _extract_detection_images(
    *, bbox_coords: tf.Tensor, image: tf.Tensor, config: ModelConfig
) -> tf.Tensor:
    """
    Extracts detections from an image.

    Args:
        bbox_coords: The normalized bounding box coordinates. Should have
            the shape `[N, 4]`, where each row takes the form
            `[min_y, min_x, max_y, max_x]`.
        image: The input image to extract from. Should have shape
            `[height, width, num_channels]`.
        config: The model configuration to use.

    Returns:
        The extracted detections, resized to the specified detection size.
        Will have shape `[num_detections, d_height, d_width, num_channels]`.

    """
    # We only have one image...
    image = tf.expand_dims(image, axis=0)
    num_detections = tf.shape(bbox_coords)[0]
    box_indices = tf.zeros((num_detections,), dtype=tf.int32)

    detection_size = config.image_input_shape[:2]
    return tf.image.crop_and_resize(
        image, bbox_coords, box_indices, crop_size=detection_size
    )


def _load_single_image_features(
    features: tf.data.Dataset, *, config: ModelConfig
) -> tf.data.Dataset:
    """
    Loads the features that can be extracted from a single image.

    Args:
        features: The raw (combined) feature dictionary for one image.
        config: The model configuration to use.

    Returns:
        A dataset with elements that are dictionaries with the single-image
        features.

    """

    def _process_image(
        feature_dict: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        # Get bounding box coordinates as a single tensor.
        x_min = feature_dict[Otf.OBJECT_BBOX_X_MIN.value]
        x_max = feature_dict[Otf.OBJECT_BBOX_X_MAX.value]
        y_min = feature_dict[Otf.OBJECT_BBOX_Y_MIN.value]
        y_max = feature_dict[Otf.OBJECT_BBOX_Y_MAX.value]
        bbox_coords = tf.stack([y_min, x_min, y_max, x_max], axis=1)

        # Compute the geometric features.
        geometric_features = _get_geometric_features(bbox_coords)

        # Extract the detection crops.
        image_encoded = feature_dict[Otf.IMAGE_ENCODED.value]
        image = tf.io.decode_jpeg(image_encoded[0])
        detections = _extract_detection_images(
            bbox_coords=bbox_coords, image=image, config=config
        )

        object_ids = feature_dict[Otf.OBJECT_ID.value]
        frame_num = feature_dict[Otf.IMAGE_FRAME_NUM.value][0]
        return {
            FeatureNames.DETECTIONS.value: detections,
            FeatureNames.GEOMETRY.value: geometric_features,
            FeatureNames.OBJECT_IDS.value: object_ids,
            FeatureNames.FRAME_NUM.value: frame_num,
        }

    return features.map(_process_image, num_parallel_calls=_NUM_THREADS)


def _load_pair_features(features: tf.data.Dataset) -> tf.data.Dataset:
    """
    Loads the features that need to be extracted from a consecutive image
    pair.

    Args:
        features: The single-image features for a batch of two consecutive
            frames.

    Returns:
        A dataset with elements that contain a dictionary of input features
        and a dictionary of target features for each frame pair.

    """

    def _process_pair(
        feature_batch: Dict[str, tf.Tensor]
    ) -> Tuple[Dict[str, tf.Tensor]]:
        # Compare frame numbers to ensure that these frames actually are
        # sequential.
        first_frame_num = feature_batch[FeatureNames.FRAME_NUM.value][0]
        next_frame_num = feature_batch[FeatureNames.FRAME_NUM.value][1]
        are_consecutive = tf.assert_equal(
            first_frame_num, next_frame_num - tf.constant(1, dtype=tf.int64)
        )
        with tf.control_dependencies([are_consecutive]):
            # Compute the ground-truth Sinkhorn matrix.
            tracklet_ids = feature_batch[FeatureNames.OBJECT_IDS.value][0]
            detection_ids = feature_batch[FeatureNames.OBJECT_IDS.value][1]
            sinkhorn = construct_gt_sinkhorn_matrix(
                detection_ids=detection_ids, tracklet_ids=tracklet_ids
            )
            # The sinkhorn matrix produced by the model is flattened.
            sinkhorn = tf.reshape(sinkhorn, (-1,))

            # Merge everything into input and target feature dictionaries.
            tracklets = feature_batch[FeatureNames.DETECTIONS.value][0]
            detections = feature_batch[FeatureNames.DETECTIONS.value][1]
            tracklet_geometry = feature_batch[FeatureNames.GEOMETRY.value][0]
            detection_geometry = feature_batch[FeatureNames.GEOMETRY.value][1]

            inputs = {
                ModelInputs.DETECTIONS.value: detections,
                ModelInputs.TRACKLETS.value: tracklets,
                ModelInputs.DETECTION_GEOMETRY.value: detection_geometry,
                ModelInputs.TRACKLET_GEOMETRY.value: tracklet_geometry,
            }
            targets = {ModelTargets.SINKHORN.value: sinkhorn}

            return inputs, targets

    return features.map(_process_pair, num_parallel_calls=_NUM_THREADS)


def _ensure_ragged(
    features: tf.data.Dataset,
    *,
    input_keys: Iterable[str],
    target_keys: Iterable[str]
) -> tf.data.Dataset:
    """
    `Dataset.map()` does not guarantee that the output will be a `RaggedTensor`.
    Since downstream code expects `RaggedTensor` inputs, this function ensures
    that the proper features are ragged.

    Args:
        features: Dataset containing input and target feature dictionaries.
        input_keys: The keys in the input that we want to make ragged.
        target_keys: The keys in the targets that we want to make ragged.

    Returns:
        Dataset with the same features, but ensuring that the specified ones
        are ragged.

    """

    def _ensure_element_ragged(
        inputs: MaybeRaggedFeature, targets: MaybeRaggedFeature
    ) -> Tuple[MaybeRaggedFeature, MaybeRaggedFeature]:
        def _validate_feature(
            feature: Union[tf.Tensor, tf.RaggedTensor]
        ) -> tf.RaggedTensor:
            if isinstance(feature, tf.RaggedTensor):
                # Already ragged.
                return feature
            # Otherwise, make it ragged.
            return tf.RaggedTensor.from_tensor(feature)

        for key in input_keys:
            inputs[key] = _validate_feature(inputs[key])
        for key in target_keys:
            targets[key] = _validate_feature(targets[key])

        return inputs, targets

    return features.map(
        _ensure_element_ragged, num_parallel_calls=_NUM_THREADS
    )


def inputs_and_targets_from_dataset(
    raw_dataset: tf.data.Dataset, *, config: ModelConfig, batch_size: int = 32
) -> tf.data.Dataset:
    """
    Deserializes raw data from a `Dataset`, and coerces it into the form used by
    the model.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        config: Model configuration we are loading data for.
        batch_size: The size of the batches that we generate.

    Returns:
        A dataset that produces input images and target bounding boxes.

    """
    # Deserialize it.
    deserialized = raw_dataset.map(
        lambda s: tf.io.parse_single_example(s, _FEATURE_DESCRIPTION),
        num_parallel_calls=_NUM_THREADS,
    )

    # Extract the features.
    single_image_features = _load_single_image_features(
        deserialized, config=config
    )
    # Break into pairs.
    image_pairs = single_image_features.batch(2)
    inputs_and_targets = _load_pair_features(image_pairs)

    # Construct batches.
    batched = inputs_and_targets.batch(batch_size)
    return _ensure_ragged(
        batched,
        input_keys=[e.value for e in ModelInputs],
        target_keys=[e.value for e in ModelTargets],
    )
