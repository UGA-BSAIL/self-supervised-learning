"""
Implementation of a CenterNet detector that uses a feature pyramid
backbone.
"""
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torchvision.models import ConvNeXt


class BnReluConv(nn.Module):
    """
    Common pattern that implements a norm layer, followed by ReLU,
    followed by convolution.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Args:
            *args: Will be forwarded to `LazyConv2d`.
            **kwargs: Will be forwarded to `LazyConv2d`.

        """
        super().__init__()

        self.bn = nn.LazyBatchNorm2d()
        self.act = nn.ReLU()
        self.conv = nn.LazyConv2d(*args, **kwargs)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(self.act(self.bn(inputs)))


class PyramidUpStage(nn.Module):
    """
    Handles a single up-sampling stage in the feature pyramid.
    """

    def __init__(self, num_features: int):
        """
        Args:
            num_features: The number of output features from this stage.

        """
        super().__init__()

        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = BnReluConv(num_features, 1, padding="same", bias=False)
        self.conv2 = BnReluConv(num_features, 3, padding="same", bias=False)

    def forward(
        self, *, small_features: Tensor, big_features: Tensor
    ) -> Tensor:
        """
        Args:
            small_features: The features coming from the lower level in the
                pyramid, which need to be upsampled.
            big_features: The features coming from the corresponding level on
                the downsampling side of the pyramid.

        Returns:
            The output features.

        """
        up_sampled = self.conv1(self.up_sample(small_features))
        merged = torch.cat((big_features, up_sampled), dim=1)
        return self.conv2(merged)


class PyramidDecoder(nn.Module):
    """
    Implements the up-sampling side of the feature pyramid.
    """

    def __init__(self):
        super().__init__()

        self.up1 = PyramidUpStage(256)
        self.up2 = PyramidUpStage(128)
        self.up3 = PyramidUpStage(64)

        # Dropout to apply to the encoder features.
        self.scale4_drop = nn.Dropout(0.5)
        self.scale3_drop = nn.Dropout(0.4)
        self.scale2_drop = nn.Dropout(0.3)
        self.scale1_drop = nn.Dropout(0.2)

    def forward(
        self, multi_scale_features: Tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> Tensor:
        """
        Args:
            multi_scale_features: Features extracted from different layers of the
                encoder. Each set of features is expected to be half the spatial
                resolution of the previous one.

        Returns:
            The up-sampled features.

        """
        scale1, scale2, scale3, scale4 = multi_scale_features

        # Apply dropout.
        scale4 = self.scale4_drop(scale4)
        scale3 = self.scale3_drop(scale3)
        scale2 = self.scale2_drop(scale2)
        scale1 = self.scale1_drop(scale1)

        # Upsample
        scale3_merged = self.up1(small_features=scale4, big_features=scale3)
        scale2_merged = self.up2(
            small_features=scale3_merged, big_features=scale2
        )
        scale1_merged = self.up3(
            small_features=scale2_merged, big_features=scale1
        )

        return scale1_merged


class ConvNeXtSmallPyramid(nn.Module):
    """
    Implements a complete feature pyramid network using ConvNeXtS as the
    encoder.
    """

    def __init__(self, encoder: ConvNeXt):
        """
        Args:
            encoder: The encoder module to use.
        """
        super().__init__()

        # We have to do a little model surgery here to extract the right
        # features.
        features = encoder.features
        stem = features.get_submodule("0")
        stage1 = features.get_submodule("1")
        stage1_pool = features.get_submodule("2")
        stage2 = features.get_submodule("3")
        stage2_pool = features.get_submodule("4")
        stage3 = features.get_submodule("5")
        stage3_pool = features.get_submodule("6")
        stage4 = features.get_submodule("7")

        self.stage1 = nn.Sequential(stem, stage1)
        self.stage2 = nn.Sequential(stage1_pool, stage2)
        self.stage3 = nn.Sequential(stage2_pool, stage3)
        self.stage4 = nn.Sequential(stage3_pool, stage4)

        self.decoder = PyramidDecoder()

    def forward(self, inputs: Tensor) -> Tensor:
        # Do the down-sampling side.
        scale1 = self.stage1(inputs)
        scale2 = self.stage2(scale1)
        scale3 = self.stage3(scale2)
        scale4 = self.stage4(scale3)

        # Do the up-sampling side.
        return self.decoder((scale1, scale2, scale3, scale4))


class PredictionHead(nn.Module):
    """
    A prediction head for the CenterNet model.
    """

    def __init__(self, *, output_channels: int):
        """
        Args:
            output_channels: The number of output channels we want.

        """
        super().__init__()

        self.conv1_1 = BnReluConv(256, 3, padding="same")
        self.conv1_2 = BnReluConv(output_channels, 1, padding="same")

    def forward(self, features: Tensor) -> Tensor:
        return self.conv1_2(self.conv1_1(features))


class CenterNet(nn.Module):
    """
    A complete CenterNet model.
    """

    def __init__(self, *, encoder: ConvNeXt):
        """
        Args:
            encoder: The ConvNeXt encoder to use in this model.

        """
        super().__init__()

        self.features = ConvNeXtSmallPyramid(encoder)

        # Add prediction heads.
        self.heatmap = PredictionHead(output_channels=1)
        self.sizes = PredictionHead(output_channels=2)
        self.offsets = PredictionHead(output_channels=2)

        self.heatmap_act = nn.Sigmoid()
        self.ensure_positive = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            inputs: The input images.

        Returns:
            The heatmaps, the dense box sizes, and the dense center offsets.

        """
        # Apply the backbone.
        features = self.features(inputs)

        # Apply the heads.
        heatmaps = self.heatmap(features)
        sizes = self.sizes(features)
        offsets = self.offsets(features)

        # Heatmaps and sizes should never be <0.
        heatmaps = self.heatmap_act(heatmaps)
        sizes = self.ensure_positive(sizes)

        return heatmaps, sizes, offsets
