"""
Implements the SimCLR-specific portions of the model.
"""


from torch import nn
from torch import Tensor
from .losses import NtXentLoss


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

    def __init__(self, *, encoder: nn.Module, loss: NtXentLoss):
        """
        Args:
            encoder: This is `f()` in the paper. It will be applied to
                input images and used to generate representations.
            loss: The loss function to use.

        """
        super().__init__()

        self.encoder = encoder
        self.loss = loss

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
