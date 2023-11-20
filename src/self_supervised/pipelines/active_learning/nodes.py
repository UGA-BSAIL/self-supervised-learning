from pathlib import Path
from typing import Callable, Dict
from functools import partial

import torch
from torch import nn
from torchvision.io import read_image
from torchvision.transforms.functional import resize
import pandas as pd

from ..frame_selector import FrameSelector


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


def save_image_reps(
    *, metadata: pd.DataFrame, root_path: Path, model: nn.Module
) -> Dict[str, Callable[[], pd.DataFrame]]:
    """
    Computes the image representations for all the images in a dataset.

    Args:
        metadata: The metadata for the MARS dataset.
        root_path: The root path of the dataset.
        model: The model to use for extracting representations.

    Returns:
        The dataset containing the image representations.

    """
    frame_selector = FrameSelector(mars_metadata=metadata)

    def _get_reps(index: int) -> pd.DataFrame:
        view_ids = frame_selector.get_all_views(index)
        # Get the image reps for all views.
        view_reps = [
            _get_image_rep(root_path / v, model=model) for v in view_ids
        ]

        return pd.DataFrame(data=view_reps)

    partitions = {}
    for i in range(frame_selector.num_frames):
        partitions[f"frame_{i}"] = partial(_get_reps, i)
    return partitions
