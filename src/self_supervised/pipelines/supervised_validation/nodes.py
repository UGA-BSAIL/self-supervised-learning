"""
Node definitions for the `supervised_validation` pipeline.
"""
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import torch
import wandb
from loguru import logger
from pytorch_warmup import BaseWarmup, LinearWarmup
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torchvision.models import convnext_small

from ..torch_common import normalize
from .centernet import CenterNet
from .losses import HeatMapFocalLoss, SparseL1Loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using {} device.", DEVICE)


def _collate_fn(
    examples: List[Tuple[Tensor, Tensor, Tensor, Any, Any]]
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """
    Collation function for training CenterNet.

    Args:
        examples: The input examples, containing the images, heatmaps,
            and bounding boxes.

    Returns:
        - The images, as a batch.
        - The heatmaps, as a batch,
        - The bounding boxes, as a list of tensors, with one item
          for each image.

    """
    # Combine stuff into a batch.
    images, heatmaps, boxes, _, _ = zip(*examples)
    images = torch.stack(images)
    heatmaps = torch.stack(heatmaps)

    # The class type is useless for this problem.
    boxes = [b[:, 2:] for b in boxes]

    return images, heatmaps, boxes


class TrainingLoop:
    """
    Manages running a training loop.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        heatmap_loss_fn: nn.Module,
        geometry_loss_fn: SparseL1Loss,
        optimizer: Optimizer,
        scaler: GradScaler,
        warmup: BaseWarmup,
        checkpoint_dir: Path = Path("checkpoints"),
    ):
        """
        Args:
            model: The model to train.
            heatmap_loss_fn: The loss function to use for heatmaps.
            geometry_loss_fn: The loss function to use for geometry.
            optimizer: The optimizer to use.
            scaler: Gradient scaler to use.
            warmup: Warmup controller to use.
            checkpoint_dir: The directory to use for saving intermediate
                model checkpoints.

        """
        self.__model = model
        self.__heatmap_loss_fn = heatmap_loss_fn
        self.__geometry_loss_fn = geometry_loss_fn
        self.__optimizer = optimizer
        self.__scaler = scaler
        self.__warmup = warmup
        self.__checkpoint_dir = checkpoint_dir
        self.__checkpoint_dir.mkdir(exist_ok=True)

        # Keeps track of the current global training step.
        self.__global_step = 0

    @property
    def global_step(self) -> int:
        return self.__global_step

    def __log_first_batch(self, inputs: Tensor) -> None:
        """
        Logs data from the first batch of an epoch.

        Args:
            inputs: The input images.

        """
        # Log gradients. Needs to be done after the first forward pass,
        # since we are using `LazyModule`s.
        wandb.watch(self.__model, log_freq=500)

        def _to_wandb_image(image: Tensor) -> wandb.Image:
            # WandB doesn't seem to like initializing images from pure
            # tensors, so we have to do some fancy stuff.
            return wandb.Image(image.permute((1, 2, 0)).numpy())

        # Take at most 25 images from each batch.
        wandb.log(
            {
                "global_step": self.__global_step,
                "train/input_examples": [
                    _to_wandb_image(im) for im in inputs[:25]
                ],
            }
        )

    def __save_checkpoint(self) -> None:
        """
        Saves a model checkpoint.

        """
        checkpoint_name = f"checkpoint_{self.__global_step}.pt"
        logger.debug("Saving checkpoint '{}'...", checkpoint_name)

        checkpoint_path = self.__checkpoint_dir / checkpoint_name
        torch.save(self.__model, checkpoint_path.as_posix())

    def __compute_loss(
        self, images: Tensor, heatmaps: Tensor, boxes: List[Tensor]
    ) -> Tensor:
        """
        Applies the model and computes the loss value.

        Args:
            images: The input images.
            heatmaps: The target heatmaps.
            boxes: The target bounding boxes.

        Returns:
            The loss it computed.

        """
        pred_heatmaps, pred_sizes, pred_offsets = self.__model(images)

        heatmap_loss = self.__heatmap_loss_fn(heatmaps, pred_heatmaps)
        geometry_loss = self.__geometry_loss_fn(
            predicted_sizes=pred_sizes,
            predicted_offsets=pred_offsets,
            target_boxes=boxes,
        )

        wandb.log(
            {
                "global_step": self.__global_step,
                "train/heatmap_loss": heatmap_loss,
                "train/geometry_loss": geometry_loss,
            }
        )

        return heatmap_loss + geometry_loss

    @staticmethod
    def __iter_data_loader(
        data_loader: data.DataLoader,
    ) -> Iterable[Tuple[Tensor, Tensor, Tensor, List[Tensor]]]:
        """
        Iterates through the items in the `DataLoader`, and pre-processes them
        appropriately.

        Args:
            data_loader: The `DataLoader` to iterate.

        Yields:
            The raw images, pre-processed images, pre-processed heatmaps,
            and pre-processed bounding boxes.

        """
        for raw_images, heatmaps, boxes in data_loader:
            images = raw_images.to(DEVICE, non_blocking=True)
            heatmaps = heatmaps.to(DEVICE, non_blocking=True)
            boxes = [b.to(DEVICE, non_blocking=True) for b in boxes]
            images = normalize(images)

            yield raw_images, images, heatmaps, boxes

    def train_epoch(self, data_loader: data.DataLoader) -> float:
        """
        Trains the model for a single epoch on some data.

        Args:
            data_loader: The data to train on.

        Returns:
            The average loss for this epoch.

        """
        total_loss = 0.0
        batch_i = 0
        for batch_i, (raw_images, images, heatmaps, boxes) in enumerate(
            self.__iter_data_loader(data_loader)
        ):
            # Compute loss.
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                loss = self.__compute_loss(images, heatmaps, boxes)

            if batch_i == 0:
                self.__log_first_batch(raw_images)

            # Backward pass.
            self.__optimizer.zero_grad()
            self.__scaler.scale(loss).backward()
            self.__scaler.step(self.__optimizer)
            self.__scaler.update()

            logger.debug("batch {}: loss={}", batch_i, loss.item())
            wandb.log(
                {
                    "global_step": self.__global_step,
                    "lr": self.__optimizer.param_groups[0]["lr"],
                }
            )

            self.__global_step += 1
            with self.__warmup.dampening():
                pass
            total_loss += loss.item()

        self.__save_checkpoint()
        return total_loss / batch_i

    def test_model(self, data_loader: data.DataLoader) -> float:
        """
        Tests the model.

        Args:
            data_loader: The data to test on.

        Returns:
            THe average loss on the testing data.

        """
        total_loss = 0.0
        batch_i = 0
        for batch_i, (_, images, heatmaps, boxes) in enumerate(
            self.__iter_data_loader(data_loader)
        ):
            # Compute loss.
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                loss = self.__compute_loss(images, heatmaps, boxes)

                logger.debug("Testing batch {}: loss={}", batch_i, loss.item())
                total_loss += loss.item()

        return total_loss / batch_i


def build_model() -> CenterNet:
    """
    Builds the complete CenterNet model.

    Returns:
        The CenterNet model it created.

    """
    encoder = convnext_small()
    return CenterNet(encoder=encoder).to(DEVICE)


def train_model(
    model: nn.Module,
    *,
    training_data: data.Dataset,
    testing_data: data.Dataset,
    num_epochs: int,
    batch_size: int,
    learning_rate: float = 0.001,
    warmup_steps: int = 100,
    focal_alpha: float = 2.0,
    focal_beta: float = 4.0,
    size_loss_weight: float = 1.0,
    offset_loss_weight: float = 1.0,
) -> nn.Module:
    """
    Trains the model.

    Args:
        model: The model to train.
        training_data: The training dataset.
        testing_data: The testing dataset.
        num_epochs: The number of epochs to train for.
        batch_size: The batch size to use for training.
        learning_rate: The learning rate to use.
        warmup_steps: How many steps to perform LR warmup over.
        focal_alpha: The alpha parameter to use for focal loss.
        focal_beta: The beta parameter to use for focal loss.
        size_loss_weight: The weight to use for the object size loss.
        offset_loss_weight: The weight to use for the center offset loss.

    Returns:
        The trained model.

    """
    heatmap_loss_fn = HeatMapFocalLoss(alpha=focal_alpha, beta=focal_beta).to(
        DEVICE
    )
    geometry_loss_fn = SparseL1Loss(
        size_weight=size_loss_weight, offset_weight=offset_loss_weight
    ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=3)
    warmup = LinearWarmup(optimizer, warmup_period=warmup_steps)
    scaler = GradScaler()

    data_loader_params = dict(
        batch_size=batch_size,
        collate_fn=_collate_fn,
        pin_memory=True,
        num_workers=8,
    )
    training_data_loader = data.DataLoader(
        training_data,
        shuffle=True,
        **data_loader_params,
    )
    testing_data_loader = data.DataLoader(testing_data, **data_loader_params)

    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        heatmap_loss_fn=heatmap_loss_fn,
        geometry_loss_fn=geometry_loss_fn,
        warmup=warmup,
    )
    for i in range(num_epochs):
        logger.info("Starting epoch {}...", i)
        training_loss = training_loop.train_epoch(training_data_loader)
        testing_loss = training_loop.test_model(testing_data_loader)

        wandb.log(
            {
                "global_step": training_loop.global_step,
                "train/epoch_loss": training_loss,
                "test/epoch_loss": testing_loss,
            }
        )

        with warmup.dampening():
            scheduler.step(testing_loss)

    return model
