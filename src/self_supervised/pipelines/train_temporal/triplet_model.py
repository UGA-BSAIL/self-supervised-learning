"""
Implements a model that generates representations for triplets.
"""


from torch import Tensor, nn

from ..representation_model import ProjectionHead


class TripletModel(nn.Module):
    """
    Class that encapsulates the entire representation learning model.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        num_features: int = 2048,
        num_projected_outputs: int = 256
    ):
        """
        Args:
            encoder: This will be applied to input images and used to
                generate representations.
            num_features: The number of features that we expect to be
                produced by the encoder.
            num_projected_outputs: Output size of the projection head.

        """
        super().__init__()

        self.encoder = encoder
        self.projection = ProjectionHead(
            num_inputs=num_features, num_outputs=num_projected_outputs
        )
