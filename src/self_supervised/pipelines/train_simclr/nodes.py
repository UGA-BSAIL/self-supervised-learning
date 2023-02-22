"""
Nodes for the `train_simclr` pipeline.
"""


from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

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
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Lambda,
    RandAugment,
    RandomResizedCrop,
)

from ..frame_selector import FrameSelector
from ..representation_model import RepresentationModel, YoloEncoder
from .dataset_io import MultiViewDataset
from .losses import FullGraphLoss, NtXentLoss
from .metrics import ProxyClassAccuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using {} device.", DEVICE)


def _collate_views(views: List[List[Tensor]]) -> List[Tensor]:
    """
    Collates a list of image pairs from a `MultiViewDataset` into
    separate batches of corresponding images with different views.

    Args:
        views: The views of each image.

    Returns:
        The two corresponding batches.

    """
    return [torch.stack(b, dim=0) for b in zip(*views)]


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
        augmentation: Callable[[Tensor], Tensor] = lambda x: x,
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
            augmentation: The data augmentation to apply.

        """
        self.__model = model
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer
        self.__scaler = scaler
        self.__accuracy = accuracy
        self.__checkpoint_dir = checkpoint_dir
        self.__checkpoint_dir.mkdir(exist_ok=True)
        self.__augmentation = augmentation

        # Keeps track of the current global training step.
        self.__global_step = 0
        # Whether we have already started gradient logging.
        self.__is_watching = False

    def __preprocess(self, images: Tensor) -> Tensor:
        """
        Normalizes input images, converting to floats and giving them a
        mean of 1 and a standard deviation of 0.

        Args:
            images: The images to normalize.

        Returns:
            The normalized images.

        """
        images = self.__augmentation(images.to(DEVICE, non_blocking=True))
        return images.to(torch.float) / 255

    def __log_first_batch(self, *, view_inputs: Iterable[Tensor]) -> None:
        """
        Logs data from the first batch of an epoch.

        Args:
            view_inputs: The corresponding inputs from all the views.

        """
        if not self.__is_watching:
            # Log gradients
            wandb.watch(self.__model, log_freq=2000)
            self.__is_watching = True

        def _to_wandb_image(image: Tensor) -> wandb.Image:
            # WandB doesn't seem to like initializing images from pure
            # tensors, so we have to do some fancy stuff.
            return wandb.Image(image.permute((1, 2, 0)).cpu().numpy())

        # Take at most 25 images from each batch.
        wandb_images = [
            [_to_wandb_image(im) for im in batch] for batch in view_inputs
        ]
        images_log = {
            f"view_{i}_input_examples": e[:15]
            for i, e in enumerate(wandb_images)
        }
        wandb.log(dict(global_step=self.__global_step, **images_log))

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
        for batch_i, view_inputs in enumerate(data_loader):
            view_inputs = [self.__preprocess(view) for view in view_inputs]

            # Compute loss.
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                view_preds = self.__model(*view_inputs)
                loss = self.__loss_fn(view_preds)

            if batch_i == 0:
                self.__log_first_batch(view_inputs=view_inputs)

            # Backward pass.
            self.__optimizer.zero_grad()
            self.__scaler.scale(loss).backward()
            self.__scaler.step(self.__optimizer)
            self.__scaler.update()

            # Compute accuracy.
            batch_acc = self.__accuracy(view_preds)

            logger.debug("batch {}: loss={}", batch_i, loss.item())
            if self.__global_step % 100 == 0:
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


def build_model(yolo_description: Dict[str, Any]) -> nn.Module:
    """
    Builds the complete SimCLR model.

    Args:
        yolo_description: The description of the YOLO model to use for the
            backbone.

    Returns:
        The model that it built.

    """
    encoder = YoloEncoder(yolo_description)
    return RepresentationModel(encoder=encoder).to(DEVICE)


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

    frame_selector = FrameSelector(mars_metadata=metadata)
    paired_frames = MultiViewDataset(
        frames=frame_selector,
        image_folder=image_folder,
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
    pair_loss = NtXentLoss(temperature=temperature)
    loss_fn = FullGraphLoss(pair_loss).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=2)
    scaler = GradScaler()
    accuracy = ProxyClassAccuracy().to(DEVICE)

    data_loader = data.DataLoader(
        training_data,
        batch_size=batch_size,
        collate_fn=_collate_views,
        pin_memory=True,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )

    augmentation = Compose(
        [
            RandomResizedCrop(
                410, scale=(0.5, 1.0), interpolation=InterpolationMode.NEAREST
            ),
            # Apparently, crops sometimes produce non-contiguous views,
            # and RandAugment doesn't like that.
            Lambda(lambda t: t.contiguous()),
            RandAugment(magnitude=9, interpolation=InterpolationMode.NEAREST),
        ]
    )
    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scaler=scaler,
        accuracy=accuracy,
        augmentation=augmentation,
    )
    for i in range(num_epochs):
        logger.info("Starting epoch {}...", i)
        average_loss = training_loop.train_epoch(data_loader)

        logger.info("Epoch {} loss: {}", i, average_loss)
        scheduler.step(average_loss)

    return model
