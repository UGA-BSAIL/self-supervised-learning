"""
Encapsulates various schemas for the data.
"""


import enum
from typing import Iterable, Type


def _merge_enums(
    merged_name: str, enums: Iterable[Type[enum.Enum]]
) -> Type[enum.Enum]:
    """
    Merges two enums together into one that shares the same members.

    Args:
        merged_name: The name of the merged enum.
        enums: The enums to merge.

    Returns:
        The merged enum class.

    """
    members = []
    for sub_enum in enums:
        members.extend([(e.name, e.value) for e in sub_enum])

    return enum.Enum(value=merged_name, names=members)


@enum.unique
class ObjectDetectionFeatures(enum.Enum):
    """
    Standard feature names used by the TF Object Detection API.
    """

    IMAGE_HEIGHT = "image/height"
    """
    Height of the image in pixels.
    """
    IMAGE_WIDTH = "image/width"
    """
    Width of the image in pixels.
    """
    IMAGE_FILENAME = "image/filename"
    """
    Filename associated with the image.
    """
    IMAGE_SOURCE_ID = "image/source_id"
    """
    Numerical ID for the image source.
    """
    IMAGE_ENCODED = "image/encoded"
    """
    Encoded image data.
    """
    IMAGE_FORMAT = "image/format"
    """
    Image format.
    """
    OBJECT_BBOX_X_MIN = "image/object/bbox/xmin"
    """
    Min x-value in the detection bounding boxes.
    """
    OBJECT_BBOX_X_MAX = "image/object/bbox/xmax"
    """
    Max x-value in the detection bounding boxes.
    """
    OBJECT_BBOX_Y_MIN = "image/object/bbox/ymin"
    """
    Min y-value in the detection bounding boxes.
    """
    OBJECT_BBOX_Y_MAX = "image/object/bbox/ymax"
    """
    Max y-value in the detection bounding boxes.
    """
    OBJECT_CLASS_TEXT = "image/object/class/text"
    """
    Object class as a human-readable name.
    """
    OBJECT_CLASS_LABEL = "image/object/class/id"
    """
    Object class as a numerical label.
    """


@enum.unique
class _AdditionalTrackingFeatures(enum.Enum):
    """
    Additional features unique to the tracking problem. These are non-standard.
    """

    OBJECT_ID = "image/object/id"
    """
    Unique ID for an object that is used to identify it across multiple frames.
    """
    IMAGE_SEQUENCE_ID = "image/sequence_id"
    """
    Unique ID for identifying frames that belong to the same video clip.
    """
    IMAGE_FRAME_NUM = "image/frame_num"
    """
    The frame number within the clip.
    """


ObjectTrackingFeatures = enum.unique(
    _merge_enums(
        "ObjectTrackingFeatures",
        (ObjectDetectionFeatures, _AdditionalTrackingFeatures),
    )
)
"""
Extends `ObjectDetectionFeatures` with non-standard features unique to the
tracking problem.
"""


@enum.unique
class ModelInputs(enum.Enum):
    """
    Key names for the inputs to the model.
    """

    DETECTIONS = "detections"
    """
    Extracted detection crops.
    """
    TRACKLETS = "tracklets"
    """
    Extracted tracklet crops.
    """

    DETECTION_GEOMETRY = "detection_geometry"
    """
    Corresponding geometric features for the detection crops.
    """
    TRACKLET_GEOMETRY = "tracklet_geometry"
    """
    Corresponding geometric features for the tracklet crops.
    """


@enum.unique
class ModelTargets(enum.Enum):
    """
    Key names for the model targets.
    """

    SINKHORN = "sinkhorn"
    """
    The ground-truth Sinkhorn matrix.
    """
