"""
Common components of models for representation learning.
"""


from typing import Any, Dict, Iterable, Tuple

from torch import Tensor, nn
from torchvision.models import convnext_small, efficientnet_v2_s
from ultralytics.models.yolo.detect.train import DetectionModel


class ProjectionHead(nn.Module):
    """
    The projection head `g()` that gets applied to the representations.
    """

    def __init__(self, *, num_inputs: int, num_outputs: int = 2048):
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
        self.__model_description = model_description.copy()
        self.__model_description["head"] = self.__model_description["head"][
            :-1
        ]
        self.yolo = DetectionModel(self.__model_description)

        # Internal projection head used to get the right number of output
        # features for the representation.
        self.projection = nn.LazyConv2d(num_features, (1, 1), padding="same")

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.yolo(inputs)
        return self.projection(features)

    def clone_some_layers(self, prune_head_layers: int) -> nn.Module:
        """
        Creates a clone of the YOLO model with the top layers missing.

        Args:
            prune_head_layers: The number of layers to prune from the head.

        Returns:
            The cloned model.

        """
        pruned_description = self.__model_description.copy()
        pruned_description["head"] = pruned_description["head"][
            :-prune_head_layers
        ]
        cloned_model = DetectionModel(pruned_description)

        # Copy the weights.
        state_dict = self.yolo.float().state_dict()
        cloned_model.load_state_dict(state_dict, strict=False)

        return cloned_model


class RepresentationModel(nn.Module):
    """
    Class that encapsulates a model for representation learning.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        num_features: int = 2048,
        num_outputs: int = 256
    ):
        """
        Args:
            encoder: This is `f()` in the paper. It will be applied to
                input images and used to generate representations.
            num_features: The number of features that we expect to be produced
                by the encoder.
            num_outputs: The number of outputs we want for the loss.

        """
        super().__init__()

        self.encoder = encoder
        self.projection = ProjectionHead(
            num_inputs=num_features, num_outputs=num_outputs
        )

    def forward(self, *images: Iterable[Tensor]) -> Tuple[Tensor, ...]:
        """
        Args:
            *images: The images to apply the model to.

        Returns:
            The computed and projected encodings for each input.

        """
        # Get the representations.
        reps = [self.encoder(i) for i in images]
        # Apply the projection.
        projected = [self.projection(r) for r in reps]

        return tuple(projected)
