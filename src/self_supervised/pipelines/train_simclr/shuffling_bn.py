"""
Implementation of shuffling batch norm as described in MoCo, but without
the dependence on multiple GPUs.
"""


from typing import Any

import torch
from torch import Tensor


class ShufflingBatchNorm2d(torch.nn.Module):
    """
    Implements 2D batch normalization that breaks the input batch into
    multiple samples and calculates the normalization parameters separately.
    """

    def __init__(
        self, num_features: int, *, num_slices: int = 4, **kwargs: Any
    ):
        """
        Args:
            num_features: The size of the channel dimension we are normalizing.
            num_slices: Total number of slices to break the input batch into.
            **kwargs: Forwarded to the internal batch norm layers.

        """
        super().__init__()

        self.__num_slices = num_slices

        # Create a separate normalization layer for each slice.
        self.__slice_norm_layers = []
        for i in range(num_slices):
            slice_norm = torch.nn.BatchNorm2d(num_features, **kwargs)
            self.add_module(f"slice_norm_{i}", slice_norm)
            self.__slice_norm_layers.append(slice_norm)

    def forward(self, in_tensor: Tensor) -> Tensor:
        """
        Args:
            in_tensor: The input tensor to normalize.

        Returns:
            The normalized tensor.

        """
        # Break the input batch into slices.
        shuffle_indices = torch.randperm(len(in_tensor))
        input_shuffled = in_tensor[shuffle_indices]
        slices = torch.tensor_split(input_shuffled, self.__num_slices, dim=0)

        # Normalize each slice.
        norm_slices = [
            slice_norm(s)
            for slice_norm, s in zip(self.__slice_norm_layers, slices)
        ]

        # Restore the original batch order.
        unsliced = torch.concatenate(norm_slices, dim=0)
        return unsliced.scatter(0, shuffle_indices, unsliced)
