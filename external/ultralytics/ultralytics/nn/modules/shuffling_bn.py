"""
Implementation of shuffling batch norm as described in MoCo, but without
the dependence on multiple GPUs.
"""


from typing import Any, Tuple

import torch
from torch import Tensor


def shuffle_batch(batch: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Shuffles the batch by randomly permuting the order of the samples,
    in a reversible manner.

    Args:
        batch: The batch to shuffle.

    Returns:
        The shuffled batch, and the shuffle indices.

    """
    shuffle_indices = torch.randperm(len(batch))
    return batch[shuffle_indices], shuffle_indices


def unshuffle_batch(batch: Tensor, *, indices: Tensor) -> Tensor:
    """
    Reverses a previous shuffle operation on a batch.

    Args:
        batch: The 4D batch to unshuffle.
        indices: The shuffle indices from `shuffle_batch()`.

    Returns:
        The un-shuffled batch.

    """
    return batch.scatter(0, indices, batch)


class ShuffledModule(torch.nn.Module):
    """
    Shuffles input data before running a module, and un-shuffles it afterwards.
    """

    def __init__(self, module: torch.nn.Module):
        """
        Args:
            module: The module to wrap.

        """
        super().__init__()

        self.add_module("wrapped", module)

    def forward(self, batch: Tensor) -> Tensor:
        shuffled, indices = shuffle_batch(batch)
        module_out = self.wrapped(shuffled)
        return unshuffle_batch(module_out, indices=indices)

    def __getitem__(self, item):
        return self.wrapped[item]


class ShufflingBatchNorm2d(torch.nn.Module):
    """
    Implements 2D batch normalization that breaks the input batch into
    multiple samples and calculates the normalization parameters separately.
    """

    def __init__(self, num_features: int, *, num_slices: int = 4, **kwargs: Any):
        """
        Args:
            num_features: The size of the channel dimension we are normalizing.
            num_slices: Total number of slices to break the input batch into.
            **kwargs: Forwarded to the internal batch norm layers.

        """
        super().__init__()

        self._num_slices = num_slices

        # Create a separate normalization layer for each slice.
        self._slice_norm_layers = []
        for i in range(num_slices):
            slice_norm = torch.nn.BatchNorm2d(num_features, **kwargs)
            self.add_module(f"slice_norm_{i}", slice_norm)
            self._slice_norm_layers.append(slice_norm)

    def forward(self, in_tensor: Tensor) -> Tensor:
        """
        Args:
            in_tensor: The input tensor to normalize.

        Returns:
            The normalized tensor.

        """
        # Break the input batch into slices.
        slices = torch.tensor_split(in_tensor, self._num_slices, dim=0)

        # Normalize each slice.
        norm_slices = [
            slice_norm(s) for slice_norm, s in zip(self._slice_norm_layers, slices)
        ]

        # Restore the original batch order.
        return torch.concatenate(norm_slices, dim=0)
