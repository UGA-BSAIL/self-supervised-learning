import random
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import nn
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from tqdm import trange

from ..frame_selector import FrameSelector

PartitionedVectorDataset = Dict[str, Callable[[], pd.DataFrame]]
"""
Type representing a partitioned dataset that stores vector image
representations.
"""


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
    image = read_image(image_path.as_posix()).cuda()
    image.requires_grad = False
    image = resize(image, (512, 512))
    image = image.to(torch.float) / 255
    image_ex = image[None, :, :, :]

    return model(image_ex)[0].detach().reshape((1, -1))


def save_image_reps(
    *, metadata: pd.DataFrame, root_path: Path | str, model: nn.Module
) -> PartitionedVectorDataset:
    """
    Computes the image representations for all the images in a dataset.

    Args:
        metadata: The metadata for the MARS dataset.
        root_path: The root path of the dataset.
        model: The model to use for extracting representations.

    Returns:
        The dataset containing the image representations.

    """
    root_path = Path(root_path)
    frame_selector = FrameSelector(mars_metadata=metadata)

    model = model.eval()

    def _get_reps(index: int) -> pd.DataFrame:
        logger.info("Saving reps for frame {}...", index)

        view_ids = frame_selector.get_all_views(index)
        # Get the image reps for all views.
        view_reps = [
            _get_image_rep(root_path / f"{v}.jpg", model=model)
            for v in view_ids
        ]
        view_reps = {
            id_: v.cpu().numpy().squeeze()
            for id_, v in zip(view_ids, view_reps)
        }

        return pd.DataFrame(data=view_reps)

    partitions = {}
    for i in range(frame_selector.num_frames):
        partitions[f"frame_{i}"] = partial(_get_reps, i)
    return partitions


def _choose_random_rep(
    reps: PartitionedVectorDataset,
    *,
    shuffled_indices: List[int],
    frames: FrameSelector,
) -> Tuple[str, np.array]:
    """
    Chooses a random representation from a dataset.

    Args:
        reps: The dataset containing the representations.
        shuffled_indices: The shuffled frame indices. The last index in this
            list will be popped and used to choose the representation.
        frames: The frame selector to use for getting file IDs.

    Returns:
        The file ID and the corresponding representation.

    """
    index = shuffled_indices.pop()
    frame_reps = reps[f"frame_{index}"]()
    file_ids = frames.get_all_views(index)

    # Choose randomly between the cameras.
    file_id = random.choice(file_ids)
    rep = frame_reps[file_id].values
    return file_id, rep


def _update_average(
    average: np.array, *, num_points: int, next_point: np.array
) -> np.array:
    """
    Updates the moving average of a set of points.

    Args:
        average: The current average.
        next_point: The next point to add.
        num_points: The number of points included in the average.

    Returns:
        The updated average.

    """
    return (average + next_point / num_points) * (
        num_points / (num_points + 1)
    )


def find_optimal_order(
    *,
    reps: PartitionedVectorDataset,
    metadata: pd.DataFrame,
    window_size: int = 2000,
) -> List[str]:
    """
    Finds the optimal order for annotating data based on the image
    representations. This uses furthest-point sampling on a subset of the
    dataset.

    Args:
        reps: The dataset containing the image representations.
        metadata: The metadata for the MARS dataset.
        window_size: The size of the subset to use for FPS.

    Returns:
        The list of image filenames, in the order that they should be annotated.

    """
    frames = FrameSelector(mars_metadata=metadata)

    # We will add new reps in this order.
    shuffled_indices = list(range(frames.num_frames))
    random.shuffle(shuffled_indices)
    # Choose a random subset of the dataset.
    _next_random_rep = partial(
        _choose_random_rep,
        reps,
        shuffled_indices=shuffled_indices,
        frames=frames,
    )
    subset = [_next_random_rep() for _ in range(window_size)]
    subset = dict(subset)

    # Choose the first point randomly.
    file_id, rep = _next_random_rep()
    # Initialize the average.
    average = rep.copy()
    # Initialize the order.
    order = [file_id]

    for _ in trange(frames.num_frames - 1):
        # Find the distances between the average point and the other points,
        # and choose the furthest.
        furthest_id = None
        furthest_rep = None
        furthest_distance = 0.0
        for file_id, rep in subset.items():
            distance = np.linalg.norm(average - rep)
            if distance > furthest_distance:
                furthest_id = file_id
                furthest_rep = rep
                furthest_distance = distance

        # Update the average.
        average = _update_average(
            average, num_points=len(order), next_point=furthest_rep
        )
        # Remove the furthest point from the subset, since we can only select
        # once.
        del subset[furthest_id]

        # Add a new random point to the subset.
        try:
            next_id, next_rep = _next_random_rep()
            subset[next_id] = next_rep
        except IndexError:
            # There are no more points. Just keep running until we exhaust
            # the subset.
            pass

        order.append(furthest_id)

    return order
