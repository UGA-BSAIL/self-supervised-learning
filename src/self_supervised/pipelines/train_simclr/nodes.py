"""
Nodes for the `train_simclr` pipeline.
"""


from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import wandb
from loguru import logger
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torchmetrics import Metric
from torchvision.transforms import CenterCrop, Compose, Lambda, RandAugment
from torchvision.transforms.functional import normalize

from .dataset_io import PairedAugmentedDataset, SingleFrameDataset
from .losses import NtXentLoss
from .metrics import ProxyClassAccuracy, binary_accuracy
from .simclr_model import ConvNeXtSmallEncoder, SimClrModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using {} device.", DEVICE)


def _collate_pairs(pairs: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """
    Collates a list of image pairs from a `PairedAugmentedDataset` into two
    separate batches of corresponding images with different augmentations.

    Args:
        pairs: The pairs of augmented images.

    Returns:
        The two corresponding batches.

    """
    # Combine into a single batch.
    unified_batch = torch.cat(pairs, 0)

    # Now, separate that into two batches.
    left_batch_indices = torch.arange(0, len(pairs), 2)
    right_batch_indices = torch.arange(1, len(pairs), 2)

    left_batch = unified_batch[left_batch_indices]
    right_batch = unified_batch[right_batch_indices]

    return left_batch, right_batch


def _normalize(images: Tensor) -> Tensor:
    """
    Normalizes input images, converting to floats and giving them a
    mean of 1 and a standard deviation of 0.

    Args:
        images: The images to normalize.

    Returns:
        The normalized images.

    """
    images = images.to(torch.float).to(DEVICE, non_blocking=True)

    mean = images.mean(dim=(2, 3), keepdims=True)
    std = images.std(dim=(2, 3), keepdims=True)
    # Occasionally, due to weird augmentation, this can be zero, which we can't
    # divide by.
    std = std.clamp(min=0.01)
    return normalize(images, mean, std)


class TrainingLoop:
    """
    Manages running a training loop.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scaler: GradScaler,
        accuracy: Metric,
        checkpoint_dir: Path = Path("checkpoints"),
    ):
        """
        Args:
            model: The model to train.
            loss_fn: The loss function to use.
            optimizer: The optimizer to use.
            scaler: Gradient scaler to use.
            accuracy: Metric to use for computing accuracy.
            checkpoint_dir: The directory to use for saving intermediate
                model checkpoints.

        """
        self.__model = model
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer
        self.__scaler = scaler
        self.__accuracy = accuracy
        self.__checkpoint_dir = checkpoint_dir
        self.__checkpoint_dir.mkdir(exist_ok=True)

        # Keeps track of the current global training step.
        self.__global_step = 0

    def __log_first_batch(
        self, *, left_inputs: Tensor, right_inputs: Tensor
    ) -> None:
        """
        Logs data from the first batch of an epoch.

        Args:
            left_inputs: The left side inputs.
            right_inputs: The right side inputs.

        """

        def _to_wandb_image(image: Tensor) -> wandb.Image:
            # WandB doesn't seem to like initializing images from pure
            # tensors, so we have to do some fancy stuff.
            return wandb.Image(image.permute((1, 2, 0)).numpy())

        # Take at most 25 images from each batch.
        wandb.log(
            {
                "global_step": self.__global_step,
                "train/left_input_examples": [
                    _to_wandb_image(im) for im in left_inputs[:25]
                ],
                "train/right_input_examples": [
                    _to_wandb_image(im) for im in right_inputs[:25]
                ],
            }
        )

    def __log_epoch_end(self) -> None:
        """
        Performs logging for the end of the epoch.

        """
        wandb.log(
            {
                "global_step": self.__global_step,
                "train/acc_epoch": self.__accuracy.compute(),
            }
        )

        # Reset this metric for next epoch.
        self.__accuracy.reset()

    def __save_checkpoint(self) -> None:
        """
        Saves a model checkpoint.

        """
        checkpoint_name = f"checkpoint_{self.__global_step}.pt"
        logger.debug("Saving checkpoint '{}'...", checkpoint_name)

        checkpoint_path = self.__checkpoint_dir / checkpoint_name
        torch.save(self.__model, checkpoint_path.as_posix())

    def train_epoch(self, data_loader: data.DataLoader) -> float:
        """
        Trains the model for a single epoch on some data.

        Args:
            data_loader: The data to train on.

        Returns:
            The average loss for this epoch.

        """
        losses = []
        for batch_i, (left_inputs, right_inputs) in enumerate(data_loader):
            if batch_i == 0:
                self.__log_first_batch(
                    left_inputs=left_inputs, right_inputs=right_inputs
                )
            left_inputs = _normalize(left_inputs)
            right_inputs = _normalize(right_inputs)

            # Compute loss.
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                left_pred, right_pred = self.__model(left_inputs, right_inputs)
                loss = self.__loss_fn(left_pred, right_pred)

            # Backward pass.
            self.__optimizer.zero_grad()
            self.__scaler.scale(loss).backward()
            self.__scaler.step(self.__optimizer)
            self.__scaler.update()

            # Compute accuracy.
            batch_acc = self.__accuracy(left_pred, right_pred)

            logger.debug("batch {}: loss={}", batch_i, loss.item())
            wandb.log(
                {
                    "global_step": self.__global_step,
                    "train/loss": loss.item(),
                    "train/acc_batch": batch_acc.item(),
                    "lr": self.__optimizer.param_groups[0]["lr"],
                }
            )

            self.__global_step += 1
            losses.append(loss.item())

        self.__log_epoch_end()
        self.__save_checkpoint()
        return np.mean(losses)


def build_model() -> nn.Module:
    """
    Builds the complete SimCLR model.

    Returns:
        The model that it built.

    """
    encoder = ConvNeXtSmallEncoder()
    return SimClrModel(encoder=encoder).to(DEVICE)


def load_dataset(
    *, image_folder: Union[Path, str], metadata: pd.DataFrame
) -> data.Dataset:
    """
    Loads the training dataset.

    Args:
        image_folder: The path to the training images.
        metadata: The metadata associated with this dataset.

    Returns:
        The dataset that it loaded.

    """
    image_folder = Path(image_folder)

    augmentation = Compose(
        [
            CenterCrop((240, 240)),
            # Apparently, crops sometimes produces non-contiguous views,
            # and RandAugment doesn't like that.
            Lambda(lambda t: t.contiguous()),
            RandAugment(),
        ]
    )

    single_frames = SingleFrameDataset(
        mars_metadata=metadata, image_folder=image_folder
    )
    paired_frames = PairedAugmentedDataset(
        image_dataset=single_frames, augmentation=augmentation
    )

    return paired_frames


def train_model(
    model: nn.Module,
    *,
    training_data: data.Dataset,
    num_epochs: int,
    batch_size: int,
    learning_rate: float = 0.001,
    temperature: float = 0.1,
) -> nn.Module:
    """
    Trains the model.

    Args:
        model: The model to train.
        training_data: The training dataset.
        num_epochs: The number of epochs to train for.
        batch_size: The batch size to use for training.
        learning_rate: The learning rate to use.
        temperature: The temperature parameter to use for the loss.

    Returns:
        The trained model.

    """
    loss_fn = NtXentLoss(temperature=temperature).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=2)
    scaler = GradScaler()
    accuracy = ProxyClassAccuracy().to(DEVICE)

    # Log gradients
    wandb.watch(model, log_freq=500)

    data_loader = data.DataLoader(
        training_data,
        batch_size=batch_size,
        collate_fn=_collate_pairs,
        pin_memory=True,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )

    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scaler=scaler,
        accuracy=accuracy,
    )
    for i in range(num_epochs):
        logger.info("Starting epoch {}...", i)
        average_loss = training_loop.train_epoch(data_loader)

        logger.info("Epoch {} loss: {}", i, average_loss)
        scheduler.step(average_loss)

    return model
