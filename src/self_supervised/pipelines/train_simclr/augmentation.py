import abc
import math
from typing import Any, List, Optional, Tuple

import torch
import torchvision.transforms.functional as F
from loguru import logger
from torch.distributions import Beta
from torch.utils import data
from torchvision.transforms import Compose, RandomResizedCrop

from ..representation_model import RepresentationModel


class MultiArgTransform(torch.nn.Module, abc.ABC):
    """
    Superclass for transforms that support more arguments than just the input image.
    """

    @abc.abstractmethod
    def forward(self, img: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Args:
            img: The image to transform.
            **kwargs: Additional image-dependent arguments.

        Returns:
            The transformed image.

        """


class MultiArgCompose(Compose):
    """
    A version of `Compose` that ignores extra arguments except when running
    `ContrastiveCrop` and similar. This makes it easier to add
    `ContrastiveCrop` to a larger augmentation pipeline.
    """

    def __call__(self, img: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        for transform in self.transforms:
            if isinstance(transform, MultiArgTransform):
                img = transform(img, **kwargs)
            else:
                img = transform(img)

        return img


class ContrastiveCrop(MultiArgTransform):
    """
    Implements the ContrastiveCrop algorithm:

    https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Crafting_Better_Contrastive_Views_for_Siamese_Representation_Learning_CVPR_2022_paper.pdf
    """

    def __init__(
        self,
        *args: Any,
        heatmap_threshold: float = 0.05,
        alpha: float = 1.0,
        layers_to_prune: int = 0,
        device: str = "cuda",
        **kwargs: Any,
    ):
        """
        Args:
            *args: Will be forwarded to `RandomResizedCrop`.
            heatmap_threshold: The threshold to use for the heatmap.
            alpha: The parameter to use for the beta distribution. Setting it
                1 is a uniform distribution, and setting it less than that puts
                more probability away from the center.
            layers_to_prune: The number of layers to prune from the
                representation model when generating activation maps.
                Increasing this will generate activations from lower layers.
            device: The device to process on.
            **kwargs: Will be forwarded to `RandomResizedCrop`.

        """
        super().__init__()

        self.__heatmap_thresh = heatmap_threshold
        self.__device = device
        self.__layers_to_prune = layers_to_prune
        self.cropper = RandomResizedCrop(*args, **kwargs)
        self.beta = Beta(alpha, alpha)

        # Saves the regions that we can crop from. Initially, it will crop
        # from the entire image.
        self.__regions = None

    def update_regions(
        self, model: RepresentationModel, data_loader: data.DataLoader
    ) -> None:
        """
        Updates the regions that it will crop from based on the learned
        heatmaps. This should be run periodically during training.

        Args:
            model: The partially-trained model.
            data_loader: The data loader that loads training data without
                augmentation.

        """
        logger.info("==> Start updating boxes...")

        # Get just the encoder portion to produce feature maps.
        encoder = model.encoder.eval()
        if self.__layers_to_prune > 0:
            # Prune layers from the encoder.
            encoder = (
                model.encoder.clone_some_layers(self.__layers_to_prune)
                .to(self.__device)
                .eval()
            )

        boxes = []
        for cur_iter, images in enumerate(data_loader):  # drop_last=False
            logger.debug("Updating batch {}", cur_iter)
            images = images.to(self.__device, non_blocking=True)
            images = images.to(torch.float) / 255
            with torch.no_grad():
                feat_map = encoder(images)  # (N, C, H, W)

            # Create the heatmap.
            N, Cf, Hf, Wf = feat_map.shape
            eval_train_map = feat_map.sum(1).view(N, -1)  # (N, Hf*Wf)
            eval_train_map = (
                eval_train_map - eval_train_map.min(1, keepdim=True)[0]
            )
            eval_train_map = (
                eval_train_map / eval_train_map.max(1, keepdim=True)[0]
            )
            eval_train_map = eval_train_map.view(N, 1, Hf, Wf)
            eval_train_map = torch.nn.functional.interpolate(
                eval_train_map, size=images.shape[-2:], mode="bilinear"
            )  # (N, 1, Hi, Wi)
            Hi, Wi = images.shape[-2:]

            for hmap in eval_train_map:
                hmap = hmap.squeeze(0)  # (Hi, Wi)

                h_filter = (hmap.max(1)[0] > self.__heatmap_thresh).int()
                w_filter = (hmap.max(0)[0] > self.__heatmap_thresh).int()

                h_filter_nonzero = torch.nonzero(h_filter).view(-1)
                w_filter_nonzero = torch.nonzero(w_filter).view(-1)
                if len(h_filter_nonzero) == 0 or len(w_filter_nonzero) == 0:
                    logger.warning(
                        "Got all-zero confidence map, not restricting crop area."
                    )
                    h_min, h_max = 0.0, 1.0
                    w_min, w_max = 0.0, 1.0

                else:
                    h_min, h_max = (
                        torch.nonzero(h_filter).view(-1)[[0, -1]] / Hi
                    )  # [h_min, h_max]; 0 <= h <= 1
                    w_min, w_max = (
                        torch.nonzero(w_filter).view(-1)[[0, -1]] / Wi
                    )  # [w_min, w_max]; 0 <= w <= 1
                boxes.append(torch.tensor([h_min, w_min, h_max, w_max]))

        self.__regions = torch.stack(boxes, dim=0)  # (num_iters, 4)
        regions_min = self.__regions.min(0)[0]
        regions_max = self.__regions.max(0)[0]
        logger.debug(
            "Region min: {} max: {}",
            regions_min.cpu().numpy(),
            regions_max.cpu().numpy(),
        )
        logger.info("Update boxes finished.")

    def __get_params(
        self,
        img: torch.Tensor,
        region: torch.Tensor,
        scale: Tuple[float],
        ratio: Tuple[float],
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img: Input image.
            region: The region we are allowed to crop from, in the form
                `[h_min, w_min, h_max, w_max]`.
            scale: range of scale of the origin size cropped
            ratio: range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        _, height, width = img.shape
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = (
                area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            )
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                h0, w0, h1, w1 = region
                ch0 = min(max(int(height * h0) - h // 2, 0), height - h)
                ch1 = min(max(int(height * h1) - h // 2, 0), height - h)
                cw0 = min(max(int(width * w0) - w // 2, 0), width - w)
                cw1 = min(max(int(width * w1) - w // 2, 0), width - w)

                i = ch0 + int((ch1 - ch0) * self.beta.sample())
                j = cw0 + int((cw1 - cw0) * self.beta.sample())
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(
        self,
        img: torch.Tensor,
        data_index: Optional[int] = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            img: Image to be cropped and resized.
            data_index: The index of this image in the original dataset.

        Returns:
            Randomly cropped and resized image.
        """
        if data_index is None:
            logger.warning(
                "No data index provided, using standard random crop."
            )

        if data_index is None or self.__regions is None:
            # We have no regions, so use the whole image.
            region = torch.Tensor([0.0, 0.0, 1.0, 1.0])
        else:
            region = self.__regions[data_index]

        i, j, h, w = self.__get_params(
            img, region, self.cropper.scale, self.cropper.ratio
        )
        return F.resized_crop(
            img, i, j, h, w, self.cropper.size, self.cropper.interpolation
        )
