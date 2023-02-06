"""
Implements the SimCLR-specific portions of the model.
"""


from typing import Any, Dict, Tuple

from torch import Tensor, nn
from torchvision.models import convnext_small, efficientnet_v2_s
from yolov5.models.yolo import DetectionModel


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

        # Global average pooling.
        self.average_pool = nn.AdaptiveAvgPool2d(1)

        self.hidden = nn.Linear(num_inputs, num_outputs)
        self.bn = nn.BatchNorm1d(num_inputs)
        self.act = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        # Perform global average pooling.
        pooled = self.average_pool(inputs)
        pooled = pooled.squeeze(2).squeeze(2)

        return self.hidden(self.act(self.bn(pooled)))


class SimClrModel(nn.Module):
    """
    Class that encapsulates the entire SimCLR model.
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
            encoder: This is `f()` in the paper. It will be applied to
                input images and used to generate representations.
            num_features: The number of features that we expect to be produced
                by the encoder.
            num_projected_outputs: Output size of the projection head.

        """
        super().__init__()

        self.encoder = encoder
        self.projection = ProjectionHead(
            num_inputs=num_features, num_outputs=num_projected_outputs
        )

    def forward(
        self, left_inputs: Tensor, right_inputs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            left_inputs: The images to apply the model to.
            right_inputs: The same images, but with different augmentations.

        Returns:
            The computed and projected encodings for each input.

        """
        # Get the representations.
        left_rep = self.encoder(left_inputs)
        right_rep = self.encoder(right_inputs)

        # Apply the projection.
        left_projected = self.projection(left_rep)
        right_projected = self.projection(right_rep)

        return left_projected, right_projected


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
        super().__init__()

        self.convnext = convnext_small(*args, **kwargs)
        # Internal projection head used to get the right number of output
        # features for the representation.
        self.projection = nn.Conv2d(768, num_features, (1, 1), padding="same")

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.convnext.features(inputs)
        return self.projection(features)


class EfficientNetSmallEncoder(nn.Module):
    """
    Encoder module that uses EfficientNetV2-S.
    """

    def __init__(self, *args: Any, num_features: int = 2048, **kwargs: Any):
        """
        Args:
            *args: Will be forwarded to the ConvNeXt builder.
            num_features: Number of output features to use.
            **kwargs: Will be forwarded to the ConNeXt builder.

        """
        super().__init__()

        self.efficient_net = efficientnet_v2_s(*args, **kwargs)
        # Internal projection head used to get the right number of output
        # features for the representation.
        self.projection = nn.Conv2d(1280, num_features, (1, 1), padding="same")

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.efficient_net.features(inputs)
        return self.projection(features)


class YoloEncoder(nn.Module):
    """
    Encoder module that uses the YOLOv5 backbone + pyramid.
    """

    def __init__(
        self, model_description: Dict[str, Any], num_features: int = 2048
    ):
        """
        Args:
            model_description: The description of the YOLO model we are basing
                this on, as a dictionary.
            num_features: The number of output features that we want.
        """
        super().__init__()

        # The last layer is going to be the head, so get rid of that.
        model_description["head"] = model_description["head"][:-1]
        self.yolo = DetectionModel(model_description)
        # Internal projection head used to get the right number of output
        # features for the representation.
        self.projection = nn.Conv2d(512, num_features, (1, 1), padding="same")

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.yolo(inputs)
        return self.projection(features)
