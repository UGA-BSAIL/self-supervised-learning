"""
Implements the SimCLR-specific portions of the model.
"""


from torch import nn
from torch import Tensor
from .losses import NtXentLoss
from typing import Any
from torchvision.models import convnext_small


class ProjectionHead(nn.Module):
    """
    The projection head `g()` that gets applied to the representations.
    """

    def __init__(self, *, num_inputs: int, num_outputs: int = 256):
        """
        Args:
            num_inputs: The number of input features to the projection.
            num_outputs: The number of output features from the projection.

        """
        super().__init__()

        self.hidden = nn.Linear(num_inputs, num_outputs)
        self.act = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.act(self.hidden(inputs))


class SimClrModel(nn.Module):
    """
    Class that encapsulates the entire SimCLR model.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        loss: NtXentLoss,
        num_features: int = 2048,
        num_projected_outputs: int = 256
    ):
        """
        Args:
            encoder: This is `f()` in the paper. It will be applied to
                input images and used to generate representations.
            loss: The loss function to use.
            num_features: The number of features that we expect to be produced
                by the encoder.
            num_projected_outputs: Output size of the projection head.

        """
        super().__init__()

        self.encoder = encoder
        self.loss = loss
        self.projection = ProjectionHead(
            num_inputs=num_features, num_outputs=num_projected_outputs
        )

    def forward(self, left_inputs: Tensor, right_inputs: Tensor) -> Tensor:
        """
        Args:
            left_inputs: The images to apply the model to.
            right_inputs: The same images, but with different augmentations.

        Returns:
            The loss value.

        """
        # Get the representations.
        left_rep = self.encoder(left_inputs)
        right_rep = self.encoder(right_inputs)

        # Apply the projection.
        left_projected = self.projection(left_rep)
        right_projected = self.projection(right_rep)

        # Apply the loss.
        return self.loss(left_projected, right_projected)


class ConvNeXtSmallEncoder(nn.Module):
    """
    Encoder module that uses ConvNeXt-S.
    """

    def __init__(self, *args: Any, num_features: int = 2048, **kwargs: Any):
        """
        Args:
            *args: Will be forwarded to the ConvNeXt builder.
            num_features: Number of output features to use.
            **kwargs: Will be forwarded to the ConNeXt builder.

        """
        self.convnext = convnext_small()
        # Internal projection head used to get the right number of output
        # features for the representation.
        self.projection = nn.Conv2d(768, num_features, (1, 1), padding="same")

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(self.convnext(inputs))
