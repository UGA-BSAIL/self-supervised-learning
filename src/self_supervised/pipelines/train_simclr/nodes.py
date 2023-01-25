"""
Nodes for the `train_simclr` pipeline.
"""


from torch import nn
from .simclr_model import SimClrModel, ConvNeXtSmallEncoder
from .losses import NtXentLoss
from loguru import logger
import torch
from torch.utils import data
from torch.optim import Optimizer, AdamW
from typing import List, Tuple, Union
from torch import Tensor
from torchvision.transforms.functional import normalize
from torchvision.transforms import RandAugment, CenterCrop, Compose, Lambda
from torch.cuda.amp import GradScaler
from pathlib import Path
from .dataset_io import SingleFrameDataset, PairedAugmentedDataset
import pandas as pd


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
    return normalize(images, mean, std)


def _train_loop(
    *,
    dataloader: data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
) -> None:
    """
    Trains for a single epoch.

    Args:
        dataloader: The data to use for training.
        model: The model to train.
        loss_fn: The loss function to use.
        optimizer: The optimizer to use.
        scaler: Gradient scaler to use.

    """
    for batch_i, (left_inputs, right_inputs) in enumerate(dataloader):
        left_inputs = _normalize(left_inputs)
        right_inputs = _normalize(right_inputs)

        # Compute loss.
        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            left_pred, right_pred = model(left_inputs, right_inputs)
            loss = loss_fn(left_pred, right_pred)

        # Backward pass.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        logger.info("batch {}: loss={}", batch_i, loss.item())


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
) -> nn.Module:
    """
    Trains the model.

    Args:
        model: The model to train.
        training_data: The training dataset.
        num_epochs: The number of epochs to train for.
        batch_size: The batch size to use for training.
        learning_rate: The learning rate to use.

    Returns:
        The trained model.

    """
    loss_fn = NtXentLoss().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    data_loader = data.DataLoader(
        training_data,
        batch_size=batch_size,
        collate_fn=_collate_pairs,
        pin_memory=True,
        num_workers=8,
        shuffle=True,
    )

    for i in range(num_epochs):
        logger.info("Starting epoch {}...", i)
        _train_loop(
            dataloader=data_loader,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_fn,
            scaler=scaler,
        )

    return model
