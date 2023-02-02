import torch
import torch.nn.functional as F
from torch import Tensor, nn


def find_peaks(heatmaps: Tensor, with_confidence: bool = False) -> Tensor:
    """
    Detect the local maxima in an input image.

    Args:
        heatmaps: Input images we are finding maxima in. Should be a 4D
            tensor.
        with_confidence: If true, it will set the positive values in the
                resulting peak mask to the confidence scores for that peak,
                instead of merely one.

    Returns:
        A mask where local maximum pixels are 1 (or the corresponding
        confidence), and all other pixels are zero. A pixel
        is considered a local maximum when it is the maximum of all its
        neighbors in a 3x3 neighborhood.

    """
    # Extract all overlapping 3x3 patches from the images, as if we were
    # doing a 3x3 conv with stride 1.
    heatmap_patches = F.unfold(heatmaps, 3, padding=1)

    # Find patches where the largest value is the center one.
    patch_max_indices = torch.argmax(heatmap_patches, dim=1)
    # Since each patch is flattened, index 4 is the center point.
    is_peak = patch_max_indices == 4
    is_peak = torch.reshape(is_peak, heatmaps.shape)

    if with_confidence:
        # Get the actual confidence at those points.
        return heatmaps * is_peak.to(heatmaps.dtype)
    return is_peak


class FindPeaks(nn.Module):
    """
    A layer that detects local maxima in an input image. It outputs a mask
    where local maximum pixels are 1, and all other pixels are zero.
    """

    def __init__(self, with_confidence: bool = False):
        """
        Args:
            with_confidence: If true, it will set the positive values in the
                resulting peak mask to the confidence scores for that peak,
                instead of merely one.

        """
        super().__init__()

        __constants__ = ["with_confidence"]  # noqa: F841
        self.with_confidence = with_confidence

    def forward(self, heatmaps: Tensor) -> Tensor:
        return find_peaks(heatmaps, with_confidence=self.with_confidence)
