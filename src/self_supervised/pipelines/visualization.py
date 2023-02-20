import torch
from torch import Tensor
from torchvision.transforms.functional import resize


def _colorize_heat_maps(
    features: Tensor, *, max_color_threshold: float
) -> Tensor:
    """
    Converts 2D float matrices to a false-color heat map to make them easier to
    view by humans.

    Args:
        features: The 4D tensor of features to visualize.
        max_color_threshold: Threshold to use for colorization.
            Any pixel with this value or more will show up as the hottest color.

    Returns:
        The colorized heat maps. Note that these will be returned as uint8
        images, ready for display.

    """
    # Low values should show up as blue.
    blue_channel = max_color_threshold - features
    # Medium values should show up as yellow.
    max_green_point = max_color_threshold / 2.0
    green_channel = (
        -2.0 * torch.abs(features - max_green_point) + max_color_threshold
    )
    # High values should show up as red.
    red_channel = 2.0 * (features - max_green_point)

    # Combine into a color image.
    color_maps = torch.cat([red_channel, green_channel, blue_channel], dim=1)

    # Scale from 0-255 so they can be displayed easily.
    color_maps = color_maps / max_color_threshold * 255.0
    color_maps = torch.clip(color_maps, 0.0, 255.0)
    return color_maps.to(torch.uint8)


def visualize_heat_maps(
    *, images: Tensor, features: Tensor, max_color_threshold: float = 1.0
) -> Tensor:
    """
    Generates a nice visualization for density maps or feature maps that
    overlays the map on the actual image.

    Args:
        images: The raw input images, as a 4D tensor.
        features: The corresponding feature or density maps, as a 4D tensor.
        max_color_threshold: Threshold to use for colorization.
            Any pixel with this value or more will show up as the hottest color.

    Returns:
        A 4D tensor containing heatmap visualizations.

    """
    # Get the colorized density maps.
    colored_maps = _colorize_heat_maps(
        features, max_color_threshold=max_color_threshold
    )

    # Scale to the same size as the input.
    input_size = images.shape[2:]
    colored_maps = resize(colored_maps, size=input_size)
    colored_maps = colored_maps.to(torch.uint8)

    # Overlay the density maps on the images.
    combined = images / 2 + colored_maps / 2
    return combined.to(torch.uint8)
