import enum
from multiprocessing import cpu_count
from typing import Any, Dict, Iterable, Tuple, Union

import tensorflow as tf

from .assignment import construct_gt_sinkhorn_matrix
from .config import ModelConfig
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

    FRAME_IMAGE = "frame_image"
    """
    Full frame image.
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
    SEQUENCE_ID = "sequence_id"
    """
    The sequence ID of the clip.
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
    bbox_coords = tf.ensure_shape(bbox_coords, (None, 4))

    def _extract_features() -> tf.Tensor:
        x_min = bbox_coords[:, 1]
        x_max = bbox_coords[:, 3]
        y_min = bbox_coords[:, 0]
        y_max = bbox_coords[:, 2]

        width_x = x_max - x_min
        width_y = y_max - y_min
        center_x = x_min + width_x / tf.constant(2.0)
        center_y = y_min + width_y / tf.constant(2.0)

        return tf.stack([center_x, center_y, width_x, width_y], axis=1)

    # Handle the case where we have no detections.
    return tf.cond(
        tf.shape(bbox_coords)[0] > 0,
        _extract_features,
        lambda: tf.zeros((0, 4), dtype=tf.float32),
    )


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
    bbox_coords = tf.ensure_shape(bbox_coords, (None, 4))

    # Convert from pixel to normalized coordinates.
    image_height_width = tf.shape(image)[:2]
    image_height_width = tf.cast(image_height_width, tf.float32)
    image_height_width = tf.tile(image_height_width, (2,))
    bbox_coords = bbox_coords / image_height_width

    # We only have one image...
    image = tf.expand_dims(image, axis=0)
    num_detections = tf.shape(bbox_coords)[0]
    box_indices = tf.zeros((num_detections,), dtype=tf.int32)

    detection_size = config.image_input_shape[:2]
    extracted = tf.image.crop_and_resize(
        image, bbox_coords, box_indices, crop_size=detection_size
    )
    # Make sure the result has the expected shape.
    extracted = tf.ensure_shape(extracted, (None,) + config.image_input_shape)

    # Convert back to uint8s.
    return tf.cast(extracted, tf.uint8)


def _load_single_image_features(
    features: tf.data.Dataset,
    *,
    config: ModelConfig,
    include_frame: bool = False,
) -> tf.data.Dataset:
    """
    Loads the features that can be extracted from a single image.

    Args:
        features: The raw (combined) feature dictionary for one image.
        config: The model configuration to use.
        include_frame: If true, include the full frame image in the features.

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
        sequence_id = feature_dict[Otf.IMAGE_SEQUENCE_ID.value][0]

        loaded_features = {
            FeatureNames.DETECTIONS.value: detections,
            FeatureNames.GEOMETRY.value: geometric_features,
            FeatureNames.OBJECT_IDS.value: object_ids,
            FeatureNames.FRAME_NUM.value: frame_num,
            FeatureNames.SEQUENCE_ID.value: sequence_id,
        }
        if include_frame:
            loaded_features[FeatureNames.FRAME_IMAGE.value] = image
        return loaded_features

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
        pair_features: Dict[str, tf.data.Dataset],
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        # Windowing combines features into sub-datasets with two elements. To
        # access them, we will batch them into a single element and then
        # extract them.
        def _as_single_element(
            feature_key: str,
        ) -> Union[tf.Tensor, tf.RaggedTensor]:
            window_dataset = pair_features[feature_key]
            window_dataset = window_dataset.apply(
                tf.data.experimental.dense_to_ragged_batch(2)
            )
            return tf.data.experimental.get_single_element(window_dataset)

        frame_nums = _as_single_element(FeatureNames.FRAME_NUM.value)
        object_ids = _as_single_element(FeatureNames.OBJECT_IDS.value)
        detections = _as_single_element(FeatureNames.DETECTIONS.value)
        geometry = _as_single_element(FeatureNames.GEOMETRY.value)
        sequence_ids = _as_single_element(FeatureNames.SEQUENCE_ID.value)
        frame_images = None
        if FeatureNames.FRAME_IMAGE.value in pair_features:
            frame_images = _as_single_element(FeatureNames.FRAME_IMAGE.value)

        # Compare frame numbers to ensure that these frames actually are
        # sequential.
        first_frame_num = frame_nums[0]
        next_frame_num = frame_nums[1]
        are_consecutive = tf.assert_equal(
            first_frame_num,
            next_frame_num - tf.constant(1, dtype=tf.int64),
            message="Pair frame numbers are not consecutive.",
        )

        with tf.control_dependencies([are_consecutive]):
            # Compute the ground-truth Sinkhorn matrix.
            tracklet_ids = object_ids[0]
            detection_ids = object_ids[1]
            sinkhorn = construct_gt_sinkhorn_matrix(
                detection_ids=detection_ids, tracklet_ids=tracklet_ids
            )
            # The sinkhorn matrix produced by the model is flattened.
            sinkhorn = tf.reshape(sinkhorn, (-1,))
            # Assignment target is the same as the sinkhorn matrix, just not a
            # float.
            assignment = tf.cast(sinkhorn, tf.bool)

            tracklets = detections[0]
            detections = detections[1]
            tracklet_geometry = geometry[0]
            detection_geometry = geometry[1]

            # Merge everything into input and target feature dictionaries.
            inputs = {
                ModelInputs.DETECTIONS.value: detections,
                ModelInputs.TRACKLETS.value: tracklets,
                ModelInputs.DETECTION_GEOMETRY.value: detection_geometry,
                ModelInputs.TRACKLET_GEOMETRY.value: tracklet_geometry,
                ModelInputs.SEQUENCE_ID.value: sequence_ids,
            }
            if frame_images is not None:
                # Frame images should both be identical.
                inputs[ModelInputs.FRAME.value] = frame_images[0]
            targets = {
                ModelTargets.SINKHORN.value: sinkhorn,
                ModelTargets.ASSIGNMENT.value: assignment,
            }

            return inputs, targets

    return features.map(_process_pair, num_parallel_calls=_NUM_THREADS)


def _filter_empty(features: tf.data.Dataset) -> tf.data.Dataset:
    """
    Filters out examples that contain no detections at all.

    Args:
        features: The dataset containing input and target features.

    Returns:
        The same dataset, but with empty examples removed.

    """

    def _dense_shape(tensor: Union[tf.Tensor, tf.RaggedTensor]) -> tf.Tensor:
        if isinstance(tensor, tf.Tensor):
            return tf.shape(tensor)
        else:
            return tensor.bounding_shape()

    def _is_not_empty(inputs: MaybeRaggedFeature, _) -> tf.Tensor:
        # Check both detections and tracklets, and eliminate examples where
        # either is empty.
        detections = inputs[ModelInputs.DETECTIONS.value]
        tracklets = inputs[ModelInputs.TRACKLETS.value]

        # Get the shape.
        detections_shape = _dense_shape(detections)
        tracklets_shape = _dense_shape(tracklets)

        return tf.logical_and(detections_shape[0] > 0, tracklets_shape[0] > 0)

    return features.filter(_is_not_empty)


def _ensure_ragged(
    features: tf.data.Dataset,
    input_keys: Iterable[str] = (),
    target_keys: Iterable[str] = (),
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


def _ensure_not_ragged(
    features: tf.data.Dataset,
    input_keys: Iterable[str] = (),
    target_keys: Iterable[str] = (),
) -> tf.data.Dataset:
    """
    `Dataset.map()` does not guarantee that the output will be a `RaggedTensor`.
    Some code can't handle `RaggedTensors`, so we convert to normal dense
    Tensors.

    Args:
        features: Dataset containing input and target feature dictionaries.
        input_keys: The keys in the input that we want to make dense.
        target_keys: The keys in the targets that we want to make dense.

    Returns:
        Dataset with the same features, but ensuring that the specified ones
        are ragged.

    """

    def _ensure_element_not_ragged(
        inputs: MaybeRaggedFeature, targets: MaybeRaggedFeature
    ) -> Tuple[MaybeRaggedFeature, MaybeRaggedFeature]:
        def _validate_feature(
            feature: Union[tf.Tensor, tf.RaggedTensor]
        ) -> tf.Tensor:
            if isinstance(feature, tf.Tensor):
                # Already not ragged.
                return feature
            # Otherwise, make it ragged.
            return feature.to_tensor()

        for key in input_keys:
            inputs[key] = _validate_feature(inputs[key])
        for key in target_keys:
            targets[key] = _validate_feature(targets[key])

        return inputs, targets

    return features.map(
        _ensure_element_not_ragged, num_parallel_calls=_NUM_THREADS
    )


def _inputs_and_targets_from_dataset(
    raw_dataset: tf.data.Dataset,
    *,
    config: ModelConfig,
    include_empty: bool = False,
    include_frame: bool = False,
) -> tf.data.Dataset:
    """
    Deserializes raw data from a `Dataset`, and coerces it into the form used by
    the model.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        config: Model configuration we are loading data for.
        include_empty: If true, will include examples with no detections
            or tracklets. Otherwise, it will filter them.
        include_frame: If true, will include the full frame image as well
            as the detection crops.

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
        deserialized, config=config, include_frame=include_frame
    )
    # Break into pairs.
    image_pairs = single_image_features.window(2, shift=1, drop_remainder=True)
    pair_features = _load_pair_features(image_pairs)

    # Remove empty examples.
    if not include_empty:
        pair_features = _filter_empty(pair_features)

    return pair_features


def _batch_and_prefetch(
    dataset: tf.data.Dataset,
    *,
    include_frame: bool = False,
    batch_size: int = 32,
    num_prefetch_batches: int = 5,
) -> tf.data.Dataset:
    """
    Batches and prefetches data from a dataset.

    Args:
        dataset: The dataset to process.
        include_frame: If true, will include the full frame image as well
            as the detection crops.
        batch_size: The batch size to use.
        num_prefetch_batches: The number of batches to prefetch.

    Returns:
        The batched dataset.

    """
    # Construct batches.
    batched = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size)
    )
    ragged = _ensure_ragged(
        batched,
        input_keys=[
            ModelInputs.DETECTIONS.value,
            ModelInputs.TRACKLETS.value,
            ModelInputs.DETECTION_GEOMETRY.value,
            ModelInputs.TRACKLET_GEOMETRY.value,
        ],
        target_keys=[],
    )

    input_keys_not_ragged = [ModelInputs.SEQUENCE_ID.value]
    if include_frame:
        input_keys_not_ragged.append(ModelInputs.FRAME.value)
    ragged = _ensure_not_ragged(
        ragged,
        input_keys=input_keys_not_ragged,
        target_keys=[
            ModelTargets.SINKHORN.value,
            ModelTargets.ASSIGNMENT.value,
        ],
    )

    return ragged.prefetch(num_prefetch_batches)


def inputs_and_targets_from_dataset(
    raw_dataset: tf.data.Dataset,
    *,
    config: ModelConfig,
    include_empty: bool = False,
    include_frame: bool = False,
    **kwargs: Any,
) -> tf.data.Dataset:
    """
    Deserializes raw data from a `Dataset`, and coerces it into the form used by
    the model, with batching and pre-fetching.

    Args:
        raw_dataset: The raw dataset, containing serialized data.
        config: Model configuration we are loading data for.
        include_empty: If true, will include examples with no detections
            or tracklets. Otherwise, it will filter them.
        include_frame: If true, will include the full frame image as well
            as the detection crops.
        kwargs: Will be forwarded to `_batch_and_prefetch`.

    Returns:
        A dataset that produces input images and target bounding boxes.

    """
    inputs_and_targets = _inputs_and_targets_from_dataset(
        raw_dataset,
        config=config,
        include_empty=include_empty,
        include_frame=include_frame,
    )
    return _batch_and_prefetch(
        inputs_and_targets, include_frame=include_frame, **kwargs
    )


def inputs_and_targets_from_datasets(
    raw_datasets: Iterable[tf.data.Dataset],
    *,
    config: ModelConfig,
    interleave: bool = True,
    include_empty: bool = False,
    include_frame: bool = False,
    **kwargs: Any,
) -> tf.data.Dataset:
    """
    Deserializes and interleaves data from multiple datasets, and coerces it
    into the form used by the model.

    Args:
        raw_datasets: The raw datasets to draw from.
        config: The model configuration to use.
        interleave: Allows frames from multiple datasets to be interleaved
            with each-other if true. Set to false if you want to keep
            individual clips intact.
        include_empty: If true, will include examples with no detections
            or tracklets. Otherwise, it will filter them.
        include_frame: If true, will include the full frame image as well
            as the detection crops.
        **kwargs: Will be forwarded to `_batch_and_prefetch`.

    Returns:
        A dataset that produces input images and target bounding boxes.

    """
    # Parse all the data.
    parsed_datasets = []
    for raw_dataset in raw_datasets:
        parsed_datasets.append(
            _inputs_and_targets_from_dataset(
                raw_dataset,
                config=config,
                include_empty=include_empty,
                include_frame=include_frame,
            )
        )

    if interleave:
        # Interleave the results.
        choices = tf.data.Dataset.range(len(parsed_datasets)).repeat()
        maybe_interleaved = tf.data.experimental.choose_from_datasets(
            parsed_datasets, choices
        )

    else:
        # Simply concatenate all of them together.
        maybe_interleaved = parsed_datasets[0]
        for dataset in parsed_datasets[1:]:
            maybe_interleaved = maybe_interleaved.concatenate(dataset)

    return _batch_and_prefetch(
        maybe_interleaved, include_frame=include_frame, **kwargs
    )
