"""
Utilities for creating heatmaps.
"""


from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.transforms.functional import gaussian_blur
from torchvision.utils import draw_keypoints

from .pooling import find_peaks


def heatmap_from_points(
    points: Tensor, *, output_size: Tuple[int, int], kernel_size: int = 5
) -> Tensor:
    """
    Creates a heatmap from a set of points. This is kind of a derpy
    implementation that doesn't have all the fancy features that CenterNet
    calls for, including dynamic sigmas. If it becomes a problem, I can fully
    port my more advanced TensorFlow implementation.

    Args:
        points: The points to put in the heatmap. Should be a 2D tensor with
            shape `(N, 2)`. The points should be in normalized form.
        output_size: The size of the output image (h, w).
        kernel_size: The size of the (square) kernel to use for gaussian blur.

    Returns:
        The heatmap that it generated.

    """
    # Convert from normalized to pixels.
    sizes_tiled = torch.as_tensor(output_size).tile(2)
    points_px = points * sizes_tiled

    # Add extra samples dimension.
    points_px = points_px[None, :, :]
    # We need an RGB image to draw keypoints.
    image_size = (3,) + output_size
    zeros = torch.zeros(image_size, dtype=torch.uint8)
    with_keypoints = draw_keypoints(
        zeros, points_px, radius=1, colors=(255, 255, 255)
    )

    heatmap = gaussian_blur(with_keypoints, [kernel_size, kernel_size])
    # Take just a single channel, since they all should be the same.
    heatmap = heatmap[:1, :, :]
    # Normalize it.
    heatmap = heatmap.to(torch.float)
    # Avoid dividing by zero if there are no points.
    heatmap /= torch.maximum(heatmap.max(), torch.as_tensor(0.01))

    return heatmap


def _filter_low_confidence(
    *, boxes: List[Tensor], confidence: List[Tensor], max_boxes: int
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Keeps only the boxes with the highest confidence values, and throws
    the rest away.

    Args:
        boxes: The initial boxes.
        confidence: The corresponding confidences.
        max_boxes: The maximum number of boxes to keep per example.

    Returns:
        The filtered boxes and confidences.

    """
    filtered_boxes = []
    filtered_confidences = []

    for example_boxes, example_confidence in zip(boxes, confidence):
        top_conf_indices = torch.argsort(example_confidence, descending=True)[
            :max_boxes
        ]

        filtered_boxes.append(example_boxes[top_conf_indices])
        filtered_confidences.append(example_confidence[top_conf_indices])

    return filtered_boxes, filtered_confidences


def boxes_from_heatmaps(
    *,
    heatmaps: Tensor,
    sizes: Tensor,
    offsets: Tensor,
    max_boxes: Optional[int] = None
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Computes bounding boxes from a set of heatmaps.

    Args:
        heatmaps: The heatmaps, as a 4D tensor.
        sizes: The corresponding size predictions at each pixel. Should have
            the same shape as `heatmaps`.
        offsets: The corresponding center offset predictions at each pixel.
            Should have the same shape as `heatmaps`.
        max_boxes: Limit the number of boxes returned to this number of boxes
            with the highest confidence values.

    Returns:
        - A list of detected bounding boxes for each heatmap. Each element of
        the list has a shape `[N, 4]`, where the columns are
        `[center_x, center_y, width, height]`.
        - A list of corresponding confidence scores for each box.

    """
    # Find the peaks in the heatmap.
    confidence_masks = find_peaks(heatmaps, with_confidence=True)[:, 0, :, :]
    point_masks = confidence_masks > 0.0

    mask_shape = confidence_masks.shape[2:]
    # Figure out the coordinates of the center points.
    sparse_centers = torch.nonzero(point_masks)[:, 1:]
    # Convert to normalized coordinates.
    sparse_centers = sparse_centers.to(torch.float) / torch.as_tensor(
        mask_shape, dtype=torch.float, device=sparse_centers.device
    )
    # Flip the two columns, because we want (x, y).
    sparse_centers = sparse_centers[:, [1, 0]]

    # Find the offsets and sizes for each point.
    sparse_sizes = sizes.permute((0, 2, 3, 1))[point_masks]
    sparse_offsets = offsets.permute((0, 2, 3, 1))[point_masks]
    sparse_confidence = confidence_masks[point_masks]

    # Nudge the centers by the offsets.
    sparse_centers += sparse_offsets
    # Combine into one array.
    all_boxes = torch.cat((sparse_centers, sparse_sizes), dim=1)
    # Centers and sizes should always be in [0, 1].
    all_boxes = torch.clamp(all_boxes, min=0.0, max=1.0)

    # Divide everything up by example.
    points_per_example = torch.count_nonzero(point_masks, dim=(1, 2))
    # Splitting requires indices on the CPU.
    example_point_indices = torch.cumsum(points_per_example, 0)[:-1].to(
        "cpu", non_blocking=True
    )
    per_example_boxes = torch.tensor_split(all_boxes, example_point_indices)
    per_example_confidence = torch.tensor_split(
        sparse_confidence, example_point_indices
    )

    per_example_boxes = list(per_example_boxes)
    per_example_confidence = list(per_example_confidence)
    if max_boxes is not None:
        return _filter_low_confidence(
            boxes=per_example_boxes,
            confidence=per_example_confidence,
            max_boxes=max_boxes,
        )
    return per_example_boxes, per_example_confidence
