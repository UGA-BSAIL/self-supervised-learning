from typing import Tuple

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """
    Encapsulates configuration for the model.

    Attributes:
        image_input_shape: The shape of the detections and tracklets being
            input to the appearance feature extractor.
        num_appearance_features: The number of appearance features to extract
            from each image.

        num_gcn_channels: Number of output channels to use for the GCN blocks.

        sinkhorn_lambda: The lambda parameter to use for Sinkhorn normalization.

    """

    image_input_shape: Tuple[int, int, int]
    num_appearance_features: int

    num_gcn_channels: int

    sinkhorn_lambda: float
