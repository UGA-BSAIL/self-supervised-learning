import enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from pydantic.dataclasses import dataclass

from .assignment import construct_gt_sinkhorn_matrix
from .color_utils import rgb_to_hcl
from .config import ModelConfig
from .heat_maps import make_object_heat_map
from .schemas import ColorizationTargets, ModelInputs, ModelTargets
from .schemas import RotNetTargets
from .schemas import UnannotatedFeatures as Uf

_UF_FEATURE_DESCRIPTION = {
    Uf.IMAGE_ENCODED.value: tf.io.FixedLenFeature([1], tf.dtypes.string),
    Uf.IMAGE_FRAME_NUM.value: tf.io.RaggedFeature(tf.dtypes.int64),
    Uf.IMAGE_SEQUENCE_ID.value: tf.io.RaggedFeature(tf.dtypes.int64),
}
"""
Descriptions of the features found in the dataset containing unannotated images.
"""

Feature = Dict[str, tf.Tensor]
"""
Feature dictionary that contains only normal tensors.
"""
RaggedFeature = Dict[str, tf.RaggedTensor]
"""
Feature dictionary that contains only ragged tensors.
"""
MaybeRagged = Union[tf.Tensor, tf.RaggedTensor]
"""
Possibly a normal tensor or a ragged one.
"""
MaybeRaggedFeature = Dict[str, MaybeRagged]
"""
Feature dictionary that may contain normal or ragged tensors.
"""

_RANDOM_SEED = 2021
"""
Seed to use for random number generation.
"""

_RAGGED_INPUTS = {
    ModelInputs.DETECTION_GEOMETRY.value,
    ModelInputs.TRACKLET_GEOMETRY.value,
}
"""
Input features that the model expects to be `RaggedTensor`s.
"""
_NON_RAGGED_INPUTS = {
    ModelInputs.SEQUENCE_ID.value,
    ModelInputs.DETECTIONS_FRAME.value,
    ModelInputs.TRACKLETS_FRAME.value,
}
"""
Input features that the model expects to be normal tensors.
"""
_RAGGED_TARGETS = {
    ModelTargets.GEOMETRY_DENSE_PRED.value,
    ModelTargets.GEOMETRY_SPARSE_PRED.value,
}
"""
Target features that the model expects to be `RaggedTensor`s.
"""
_NON_RAGGED_TARGETS = {
    ModelTargets.SINKHORN.value,
    ModelTargets.ASSIGNMENT.value,
    ModelTargets.HEATMAP.value,
}
"""
Target features that the model expects to be normal tensors.
"""


@enum.unique
class FeatureName(enum.Enum):
    """
    Standard key names for processed features.
    """

    FRAME_IMAGE = "frame_image"
    """
    Full frame image.
    """
    HEAT_MAP = "heat_map"
    """
    Corresponding detection heatmap.
    """
    DETECTIONS_OFFSETS = "detections_offsets"
    """
    Pixel offsets for the detections in the heatmap.
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


_NUM_THREADS = 4
"""
Number of threads to use for multi-threaded operations.
"""


@dataclass(frozen=True)
class DataAugmentationConfig:
    """
    Configuration to use for data augmentation.

    Attributes:
        max_brightness_delta: Maximum amount to adjust the brightness by.
        max_hue_delta: Maximum amount to adjust the hue by.

        min_contrast: Minimum contrast to use.
        max_contrast: Maximum contrast to use.

        min_saturation: Minimum saturation factor to use.
        max_saturation: Maximum saturation factor to use.

        max_bbox_jitter: Maximum amount of bounding box jitter to add, in
            fractions of a frame.
        false_positive_rate: The (simulated) false-positive rate to use for
            tracking.

        flip: Whether to allow horizontal and vertical flipping.
    """

    max_brightness_delta: float = 0.0
    max_hue_delta: float = 0.0

    min_contrast: Optional[float] = None
    max_contrast: Optional[float] = None

    min_saturation: Optional[float] = None
    max_saturation: Optional[float] = None

    max_bbox_jitter: float = 0.0
    false_positive_rate: float = 0.0

    flip: bool = False


def _flip_geometry(
    geometry: tf.Tensor, *, left_right: tf.Tensor, up_down: tf.Tensor
) -> tf.Tensor:
    """
    Flips the bounding box geometry about the central axes of an image.

    Args:
        geometry: The bounding box geometry, in the form
            `[center_x, center_y, ...]`. No attributes beyond the center
            point will be changed.
        left_right: 0D bool tensor, whether to flip horizontally.
        up_down: 0D bool tensor, whether to flip vertically.

    Returns:
        The flipped geometry.

    """
    left_right = tf.ensure_shape(left_right, ())
    up_down = tf.ensure_shape(up_down, ())

    center_points = geometry[:, :2]
    other_attributes = geometry[:, 2:]

    def flip_up_down() -> tf.Tensor:
        center_distance = center_points - tf.constant([0.0, 0.5])
        flipped_distance = center_distance * tf.constant([1.0, -1.0])
        return flipped_distance + tf.constant([0.0, 0.5])

    def flip_left_right() -> tf.Tensor:
        center_distance = center_points - tf.constant([0.5, 0.0])
        flipped_distance = center_distance * tf.constant([-1.0, 1.0])
        return flipped_distance + tf.constant([0.5, 0.0])

    # Geometry should be normalized, so we can just flip about 0.5.
    center_points = tf.cond(up_down, flip_up_down, lambda: center_points)
    center_points = tf.cond(left_right, flip_left_right, lambda: center_points)
    return tf.concat((center_points, other_attributes), axis=1)


def _random_flip(
    *,
    images: Iterable[tf.Tensor],
    heatmaps: Optional[Iterable[tf.Tensor]],
    geometry: Iterable[tf.Tensor],
) -> Tuple[List[tf.Tensor], Optional[List[tf.Tensor]], List[tf.Tensor]]:
    """
    Randomly flips input images vertically and horizontally,
    also transforming the corresponding geometry. It will apply the same
    transformation to pairs of images, in order to avoid disturbing the
    geometric relationships that the tracker relies on.

    Args:
        images: The 3D images to possibly flip.
        heatmaps: The corresponding heatmaps, or None if there are no heatmaps.
        geometry: Bounding box geometry for the images, of the form
            `[center_x, center_y, width, height, offset_x, offset_y]`.

    Returns:
        The same images, heatmaps, and geometry, possibly flipped.

    """

    def _do_flip(_image: tf.Tensor) -> tf.Tensor:
        # Flip the image and heatmap.
        _image = tf.cond(
            should_flip_lr,
            lambda: tf.image.flip_left_right(_image),
            lambda: _image,
        )
        _image = tf.cond(
            should_flip_ud,
            lambda: tf.image.flip_up_down(_image),
            lambda: _image,
        )

        return _image

    # Determine if we should do the flipping.
    should_flip = tf.random.uniform((2,), maxval=2, dtype=tf.int32)
    should_flip = tf.cast(should_flip, tf.bool)
    should_flip_lr = should_flip[0]
    should_flip_ud = should_flip[1]

    images = [_do_flip(i) for i in images]
    if heatmaps is not None:
        heatmaps = [_do_flip(h) for h in heatmaps]

    # Flip the bounding boxes.
    _flip_geo_ = partial(
        _flip_geometry, left_right=should_flip_lr, up_down=should_flip_ud
    )
    geometry = [_flip_geo_(g) for g in geometry]

    return images, heatmaps, geometry


def _augment_images(
    images: tf.Tensor, config: DataAugmentationConfig
) -> tf.Tensor:
    """
    Applies data augmentation to images.
    Args:
        images: The images to augment.
        config: Configuration for data augmentation.
    Returns:
        The augmented images.
    """
    # Convert to floats once so we're not doing many redundant conversions.
    images = tf.cast(images, tf.float32)

    images = tf.image.random_brightness(images, config.max_brightness_delta)
    images = tf.image.random_hue(images, config.max_hue_delta)

    if config.min_contrast is not None and config.max_contrast is not None:
        images = tf.image.random_contrast(
            images, config.min_contrast, config.max_contrast
        )
    if config.min_saturation is not None and config.max_saturation is not None:
        images = tf.image.random_saturation(
            images, config.min_saturation, config.max_saturation
        )

    images = tf.clip_by_value(images, 0.0, 255.0)
    return tf.cast(images, tf.uint8)


def _extract_rotations(image: tf.Tensor) -> tf.Tensor:
    """
    Extracts all four rotations from an image.

    Args:
        image: The image to extract rotations from.

    Returns:
        The four rotations, in a single tensor, with a new dimension
        prepended. The order is always 0, 90, 180, 270.

    """
    # 90-degree rotation.
    deg_90 = tf.image.rot90(image)
    # 180-degree rotation, which is just flipping.
    deg_180 = tf.image.flip_up_down(image)
    # 270-degree rotation.
    deg_270 = tf.image.flip_left_right(deg_90)

    return tf.stack([image, deg_90, deg_180, deg_270])


def _load_rot_net(
    unannotated_features: Feature, *, config: ModelConfig
) -> Tuple[Feature, Feature]:
    """
    Extracts RotNet features from the unannotated TFRecords.

    Args:
        unannotated_features: The raw unannotated features.
        config: The model configuration to use.

    Returns:
        Equivalent RotNet features. Will return input and target features.

    """
    # Decode the image.
    image_encoded = unannotated_features[Uf.IMAGE_ENCODED.value]
    image = _decode_image(image_encoded, ratio=2)

    # Crop to the RotNet input shape.
    image = tf.image.random_crop(image, size=config.rot_net_input_shape)

    # Labels are just integers.
    labels = tf.range(4)
    rotations = _extract_rotations(image)
    return {ModelInputs.DETECTIONS_FRAME.value: rotations}, {
        RotNetTargets.ROTATION_CLASS.value: labels,
    }


def _compute_histograms(
    image_channel: tf.Tensor,
    window_size: int = 7,
    num_bins: int = 32,
) -> tf.Tensor:
    """
    Computes the histogram of an image, using color information
    from a window around each pixel.

    Args:
        image_channel: The single-channel image to compute histograms for.
            Should be rank 2.
        window_size: The size of the window to use, in pixels.
        num_bins: The number of bins to use for the histogram.

    Returns:
        The histograms it computed. Will be a 3D matrix of shape
        [num_rows, num_columns, num_bins], containing the histograms for each
        input pixel.

    """
    input_shape = tf.shape(image_channel)
    image_channel = tf.expand_dims(tf.expand_dims(image_channel, 0), -1)

    # First, pre-bin all the values in the image.
    binned = tf.histogram_fixed_width_bins(
        image_channel, value_range=[0.0, 1.0], nbins=num_bins, dtype=tf.uint8
    )

    # Extract patches.
    patches = tf.image.extract_patches(
        binned,
        sizes=[1, window_size, window_size, 1],
        strides=[1] * 4,
        rates=[1] * 4,
        padding="SAME",
    )
    # There will be only one batch dimension, so get rid of it. We also
    # flatten everything into a single matrix where each row is a patch.
    num_patches = tf.reduce_prod(patches.shape[:3])
    patches = tf.reshape(patches, tf.stack((num_patches, -1)))

    # Compute histograms of each patch.
    patches_sorted = tf.sort(patches, axis=1)
    histograms = tf.math.bincount(patches_sorted, axis=-1, minlength=num_bins)

    # Reshape to correspond with the image.
    return tf.reshape(histograms, tf.concat((input_shape, [-1]), axis=0))


def _load_colorization(
    unannotated_features: Feature, *, config: ModelConfig
) -> Tuple[Feature, Feature]:
    """
    Extracts colorization features from the unannotated TFRecords.

    Args:
        unannotated_features: The raw unannotated features.
        config: The model configuration to use.

    Returns:
        Equivalent Colorization features. Will return input and target features.

    """
    # Decode the image.
    image_encoded = unannotated_features[Uf.IMAGE_ENCODED.value]
    image = _decode_image(image_encoded, ratio=2)

    # Crop to the colorization input shape.
    input_shape_color = config.colorization_input_shape[:2] + (3,)
    image = tf.image.random_crop(image, size=input_shape_color)
    image_hcl = rgb_to_hcl(image)

    # Make sure the output has the correct shape.
    height, width, num_bins = config.colorization_output_shape
    image_hcl_small = tf.image.resize(image_hcl, (height, width))
    # Generate histograms.
    hue_histograms = _compute_histograms(
        image_hcl_small[:, :, 0], num_bins=num_bins
    )
    chroma_histograms = _compute_histograms(
        image_hcl_small[:, :, 1], num_bins=num_bins
    )

    lightness = image_hcl[:, :, 2]
    lightness = tf.cast(lightness * 255, tf.uint8)
    lightness = tf.expand_dims(lightness, axis=-1)
    return {ModelInputs.DETECTIONS_FRAME.value: lightness}, {
        ColorizationTargets.HUE_HIST.value: hue_histograms,
        ColorizationTargets.CHROMA_HIST.value: chroma_histograms,
    }


def _decode_image(encoded: tf.Tensor, ratio: int = 1) -> tf.Tensor:
    """
    Decodes an image from a feature dictionary.

    Args:
        encoded: The encoded image.
        ratio: Downsample ratio to use when decoding.

    Returns:
        The raw decoded image.

    """
    return tf.io.decode_jpeg(encoded[0], ratio=ratio)


def _window_to_nested(windowed_features: tf.data.Dataset) -> tf.data.Dataset:
    """
    Transforms a dataset that we have applied windowing to into one where
    each window is represented as a Tensor instead of a sub-dataset.

    Args:
        windowed_features: The windowed feature dataset.

    Returns:
        The same dataset, with windows represented as Tensors.

    """

    def _convert_element(features: MaybeRaggedFeature) -> MaybeRaggedFeature:
        # Windowing combines features into sub-datasets with two elements. To
        # access them, we will batch them into a single element and then
        # extract them.
        def _as_single_element(
            feature_key: str,
        ) -> Union[tf.Tensor, tf.RaggedTensor]:
            window_dataset = features[feature_key]
            window_dataset = window_dataset.apply(
                tf.data.experimental.dense_to_ragged_batch(2)
            )
            return tf.data.experimental.get_single_element(window_dataset)

        # Convert every feature.
        converted = {}
        for name in features:
            converted[name] = _as_single_element(name)

        return converted

    return windowed_features.map(_convert_element)


def _drop_mask(
    drop_probability: float = 0.5,
    repeat_probability: float = 0.9,
    width: int = 1,
) -> Iterable[np.ndarray]:
    """
    Generator that produces an infinite sequence of booleans indicating
    whether corresponding values should be dropped.

    Args:
        drop_probability: The probability of dropping the next item when we
            didn't drop the previous one.
        repeat_probability: The probability of dropping the next item when we
            did drop the previous one.
        width: The width of the mask to create.

    Yields:
        Boolean array of length _MASK_WIDTH where each element indicates
        whether a value should be kept.

    """
    generator = np.random.default_rng(_RANDOM_SEED)

    currently_dropping = np.zeros((width,), dtype=bool)
    while True:
        threshold = np.where(
            currently_dropping, repeat_probability, drop_probability
        )

        currently_dropping = generator.random(size=(width,)) < threshold
        yield np.logical_not(currently_dropping)


def _randomize_example_spacing(
    examples: tf.data.Dataset,
    *,
    drop_probability: float,
    repeats: int = 1,
) -> tf.data.Dataset:
    """
    Nominally, we create training examples from consecutive frames. However,
    better training results can sometimes be achieved by randomly skipping some
    frames. This transformation implements this using a geometric distribution
    to determine how many frames to skip.

    Args:
        examples: The dataset containing examples. This will not actually be
            interpreted, so it doesn't matter what it contains.
        drop_probability:
            The probability of dropping a particular frame. This is the p-value
            for the geometric distribution.
        repeats: Repeat the underlying dataset this many times. This allows us
            to make the length of the output similar to that of the input,
            despite dropping data.

    Returns:
        The transformed dataset, with some items dropped.

    """
    examples = examples.repeat(repeats)

    # Create the masks.
    drop_mask = partial(
        _drop_mask,
        drop_probability=drop_probability,
        repeat_probability=drop_probability,
    )
    drop_mask = tf.data.Dataset.from_generator(
        drop_mask, output_signature=tf.TensorSpec(shape=(1,), dtype=tf.bool)
    )
    # Combine the masks with the dataset.
    examples_with_masks = tf.data.Dataset.zip((examples, drop_mask))

    # Use the mask to perform filtering.
    filtered = examples_with_masks.filter(lambda _, m: tf.squeeze(m))

    # Remove the extraneous mask component.
    return filtered.map(lambda e, _: e, num_parallel_calls=_NUM_THREADS)


def _ensure_ragged(feature: MaybeRagged) -> tf.RaggedTensor:
    """
    Ensures that a tensor is ragged.

    Args:
        feature: The feature to check.

    Returns:
        The ragged feature.

    """
    if isinstance(feature, tf.RaggedTensor):
        # Already ragged.
        return feature
    # Otherwise, make it ragged.
    return tf.RaggedTensor.from_tensor(feature)


def _ensure_not_ragged(feature: MaybeRagged) -> tf.Tensor:
    """
    Ensures that a tensor is not ragged.

    Args:
        feature: The feature to check.

    Returns:
        The feature as a normal tensor, padded with zeros if necessary.

    """
    if isinstance(feature, tf.Tensor):
        # Already not ragged.
        return feature
    # Otherwise, make it ragged.
    return feature.to_tensor()


def _transform_features(
    features: tf.data.Dataset,
    *,
    transformer: Callable[[MaybeRagged], MaybeRagged],
    input_keys: Iterable[str] = (),
    target_keys: Iterable[str] = (),
) -> tf.data.Dataset:
    """
    Selectively transforms certain features in a dataset.

    Args:
        features: Dataset containing input and target feature dictionaries.
        transformer: Function that transforms a given tensor.
        input_keys: The keys in the input that we want to make ragged.
        target_keys: The keys in the targets that we want to make ragged.

    Returns:
        Dataset with the same features, but ensuring that the specified ones
        are ragged.

    """

    def _apply_transform(
        inputs: MaybeRaggedFeature, targets: MaybeRaggedFeature
    ) -> Tuple[MaybeRaggedFeature, MaybeRaggedFeature]:
        # Select only keys that exist in the data.
        existing_input_keys = frozenset(input_keys) & inputs.keys()
        existing_target_keys = frozenset(target_keys) & targets.keys()

        for key in existing_input_keys:
            inputs[key] = transformer(inputs[key])
        for key in existing_target_keys:
            targets[key] = transformer(targets[key])

        return inputs, targets

    return features.map(_apply_transform, num_parallel_calls=_NUM_THREADS)


def _batch_and_prefetch(
    dataset: tf.data.Dataset,
    *,
    batch_size: int = 32,
    num_prefetch_batches: int = 1,
    shuffle_buffer_size: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Batches and prefetches data from a dataset.

    Args:
        dataset: The dataset to process.
        batch_size: The batch size to use.
        num_prefetch_batches: The number of batches to prefetch.
        shuffle_buffer_size: The buffer size to use for shuffling. If set to
            None, it will not shuffle at all.

    Returns:
        The batched dataset.

    """
    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(batch_size * 20)

    # Construct batches.
    batched = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size)
    )

    # `Dataset.map()` doesn't always correctly figure out which features
    # should be ragged, and which shouldn't be, so we ensure ourselves that
    # they are correct.
    ragged = _transform_features(
        batched,
        input_keys=_RAGGED_INPUTS,
        target_keys=_RAGGED_TARGETS,
        transformer=_ensure_ragged,
    )
    ragged = _transform_features(
        ragged,
        input_keys=_NON_RAGGED_INPUTS,
        target_keys=_NON_RAGGED_TARGETS,
        transformer=_ensure_not_ragged,
    )

    prefetched = ragged.prefetch(num_prefetch_batches)

    options = tf.data.Options()
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_and_filter_fusion = True
    return prefetched.with_options(options)


def _shuffle_and_deserialize(
    dataset: tf.data.Dataset, num_shards: int = 100
) -> tf.data.Dataset:
    """
    Shuffles a dataset, and deserializes TFRecords.

    Args:
        dataset: The dataset to process.
        num_shards: The number of shards to use when randomizing. A larger
            number improves randomness, but may risk exhausting the maximum
            number of open files.

    Returns:
        The processed dataset.

    """
    # Shuffle all the clips together through sharding.
    sharded_datasets = []
    for i in range(0, num_shards):
        sharded_datasets.append(dataset.shard(num_shards, i))
    shuffled_dataset = tf.data.Dataset.sample_from_datasets(sharded_datasets)

    # Deserialize it.
    return shuffled_dataset.map(
        lambda s: tf.io.parse_single_example(s, _UF_FEATURE_DESCRIPTION),
        num_parallel_calls=_NUM_THREADS,
    )


def rot_net_inputs_and_targets_from_dataset(
    unannotated_dataset: tf.data.Dataset,
    *,
    config: ModelConfig,
    batch_size: int = 8,
    num_prefetch_batches: int = 1,
) -> tf.data.Dataset:
    """
    Loads RotNet input features from a dataset of unannotated images.

    Args:
        unannotated_dataset: The dataset of unannotated images.
        config: Model configuration we are loading data for.
        batch_size: The batch size to use. The actual batch size will be
            multiplied by four. because it will include all four rotations.
        num_prefetch_batches: The number of batches to prefetch.

    Returns:
        A dataset that produces input images and target classes.

    """
    deserialized = _shuffle_and_deserialize(unannotated_dataset)

    # Extract the rotations.
    rot_net_dataset = deserialized.map(
        partial(_load_rot_net, config=config), num_parallel_calls=_NUM_THREADS
    )
    # Unbatch to separate all the rotations.
    rot_net_dataset = rot_net_dataset.unbatch()

    # Batch and prefetch.
    batched = rot_net_dataset.batch(batch_size * 4)
    return batched.prefetch(num_prefetch_batches)


def colorization_inputs_and_targets_from_dataset(
    unannotated_dataset: tf.data.Dataset,
    *,
    config: ModelConfig,
    batch_size: int = 8,
    num_prefetch_batches: int = 1,
) -> tf.data.Dataset:
    """
    Loads colorization input features from a dataset of unannotated images.

    Args:
        unannotated_dataset: The dataset of unannotated images.
        config: Model configuration we are loading data for.
        batch_size: The batch size to use. The actual batch size will be
            multiplied by four. because it will include all four rotations.
        num_prefetch_batches: The number of batches to prefetch.

    Returns:
        A dataset that produces input images and target classes.

    """
    deserialized = _shuffle_and_deserialize(unannotated_dataset)

    # Extract the colorization features.
    rot_net_dataset = deserialized.map(
        partial(_load_colorization, config=config),
        num_parallel_calls=_NUM_THREADS,
    )

    # Batch and prefetch.
    batched = rot_net_dataset.batch(batch_size)
    return batched.prefetch(num_prefetch_batches)