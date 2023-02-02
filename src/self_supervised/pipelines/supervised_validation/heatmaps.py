"""
Utilities for creating heatmaps.
"""


from typing import Tuple

import torch
from torch import Tensor
from torchvision.transforms.functional import gaussian_blur
from torchvision.utils import draw_keypoints


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
