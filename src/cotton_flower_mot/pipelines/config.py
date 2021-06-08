from typing import Tuple

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """
    Encapsulates configuration for the model.

    Attributes:
        image_input_shape: The shape of the detections and tracklets being
            input to the appearance feature extractor.
        num_reduction_stages: How many initial reduction stages to add to the
            detector. Every stage will reduce the size of the heatmap output
            by a factor of 2.
        detection_sigma: Sigma to use for detection heatmaps.

        num_appearance_features: The number of appearance features to extract
            from each image.
        num_gcn_channels: Number of output channels to use for the GCN blocks.

        sinkhorn_lambda: The lambda parameter to use for Sinkhorn normalization.

    """

    image_input_shape: Tuple[int, int, int]
    num_reduction_stages: int
    detection_sigma: float

    num_appearance_features: int
    num_gcn_channels: int

    sinkhorn_lambda: float
