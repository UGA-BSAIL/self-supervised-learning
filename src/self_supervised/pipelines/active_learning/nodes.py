from pathlib import Path

import torch
from torch import nn
from torchvision.io import read_image
from torchvision.transforms.functional import resize


def _get_image_rep(image_path: Path, *, model: nn.Module) -> torch.Tensor:
    """
    Gets the representation for a particular image based on the output of a
    model.

    Args:
        image_path: The path to the image.
        model: The model to use.

    Returns:
        The image representation.

    """
    image = read_image(image_path.as_posix())
    image.requires_grad = False
    image = resize(image, (512, 512))
    image = image.to(torch.float) / 255
    image_ex = image[None, :, :, :]

    return model(image_ex)[0].detach().reshape((1, -1))
