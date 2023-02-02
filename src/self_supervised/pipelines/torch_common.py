"""
Common torch utilities.
"""


import torch
from torch import Tensor
from torchvision.transforms import functional as F


def normalize(images: Tensor) -> Tensor:
    """
    Normalizes input images, converting to floats and giving them a
    mean of 1 and a standard deviation of 0.

    Args:
        images: The images to normalize.

    Returns:
        The normalized images.

    """
    images = images.to(torch.float)

    mean = images.mean(dim=(2, 3), keepdims=True)
    std = images.std(dim=(2, 3), keepdims=True)
    # Occasionally, due to weird augmentation, this can be zero, which we can't
    # divide by.
    std = std.clamp(min=0.01)
    return F.normalize(images, mean, std)
