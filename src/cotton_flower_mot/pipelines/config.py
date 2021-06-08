from typing import Tuple

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """
    Encapsulates configuration for the model.

    Attributes:
        image_input_shape: The shape of the detections and tracklets being
            input to the appearance feature extractor.
        detection_down_sample_factor: The factor by which the output heatmaps
            from the detection model will be down-sampled.
        detection_sigma: Sigma to use for detection heatmaps.

        num_appearance_features: The number of appearance features to extract
            from each image.
        num_gcn_channels: Number of output channels to use for the GCN blocks.

        sinkhorn_lambda: The lambda parameter to use for Sinkhorn normalization.

    """

    image_input_shape: Tuple[int, int, int]
    detection_down_sample_factor: int
    detection_sigma: float

    num_appearance_features: int
    num_gcn_channels: int

    sinkhorn_lambda: float
