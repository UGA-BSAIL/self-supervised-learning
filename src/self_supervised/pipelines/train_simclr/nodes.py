"""
Nodes for the `train_simclr` pipeline.
"""


from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torchmetrics import Metric
from torchvision.transforms import (
    InterpolationMode,
    Lambda,
    RandAugment,
    RandomResizedCrop,
    functional,
)

import wandb

from ..frame_selector import FrameSelector
from ..representation_model import RepresentationModel, YoloEncoder
from .augmentation import ContrastiveCrop, MultiArgCompose
from .dataset_io import (
    MultiViewDataset,
    PairedAugmentedDataset,
    SingleFrameDataset,
    TemporalMultiViewDataset,
)
from .losses import NtXentLoss
from .metrics import ProxyClassAccuracy
from .moco import MoCo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using {} device.", DEVICE)


def _collate_views(views: List[List[Tensor]]) -> List[List[Tensor]]:
    """
    Collates a list of image pairs from a `MultiViewDataset` into
    separate batches of corresponding images with different views.

    Args:
        views: The views of each image.

    Returns:
        The corresponding batches for each view.

    """
    return [list(b) for b in zip(*views)]


def _collate_different_sizes(
    batch: List[Tensor], *, output_size: Union[List[int], int]
) -> Tensor:
    """
    Collates a list of images from a source that can produce different image
    sizes.

    Args:
        batch: The batch of images.
        output_size: The output size to use for the images (h, w).

    Returns:
        The collated batch.

    """
    resized = []
    for image in batch:
        resized.append(
            functional.resize(
                image,
                output_size,
                interpolation=InterpolationMode.NEAREST,
            )
        )

    return torch.stack(resized, dim=0)


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
        accuracy: Metric | None = None,
        checkpoint_dir: Path = Path("checkpoints"),
        checkpoint_period: int = 1,
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
            checkpoint_period: How often to save a checkpoint, in epochs.
            augmentation: The data augmentation to apply.

        """
        self.__model = model
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer
        self.__scaler = scaler
        self.__accuracy = accuracy
        self.__checkpoint_dir = checkpoint_dir
        self.__checkpoint_dir.mkdir(exist_ok=True)
        self.__checkpoint_period = checkpoint_period
        self.__augmentation = augmentation

        # Keeps track of the current global training step.
        self.__global_step = 0
        # Keeps track of the current epoch.
        self.__epoch = 0
        # Whether we have already started gradient logging.
        self.__is_watching = False

    def __preprocess(self, images: List[Tensor]) -> Tensor:
        """
        Normalizes input images, converting to floats and giving them a
        mean of 1 and a standard deviation of 0.

        Args:
            images: The images to normalize.

        Returns:
            The normalized images.

        """
        for i in range(len(images)):
            images[i] = self.__augmentation(
                images[i].to(DEVICE, non_blocking=True)
            )

        images = torch.stack(images, dim=0)
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
        acc_epoch = {}
        if self.__accuracy is not None:
            acc_epoch = {"train/acc_epoch": self.__accuracy.compute()}
            # Reset this metric for next epoch.
            self.__accuracy.reset()

        wandb.log(
            {
                "global_step": self.__global_step,
                **acc_epoch,
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
                loss = self.__loss_fn(*view_preds)

            if batch_i == 0:
                self.__log_first_batch(view_inputs=view_inputs)

            # Backward pass.
            self.__optimizer.zero_grad()
            self.__scaler.scale(loss).backward()
            self.__scaler.step(self.__optimizer)
            self.__scaler.update()

            # Compute accuracy.
            batch_acc = None
            if self.__accuracy is not None:
                batch_acc = self.__accuracy(view_preds)

            logger.debug("batch {}: loss={}", batch_i, loss.item())
            if self.__global_step % 100 == 0:
                acc_log = {}
                if batch_acc is not None:
                    acc_log = {"train/acc_batch": batch_acc.item()}
                wandb.log(
                    {
                        "global_step": self.__global_step,
                        "train/loss": loss.item(),
                        "lr": self.__optimizer.param_groups[0]["lr"],
                        **acc_log,
                    }
                )

            self.__global_step += 1
            losses.append(loss.item())

        self.__log_epoch_end()
        self.__epoch += 1
        if self.__epoch % self.__checkpoint_period == 0:
            self.__save_checkpoint()

        return np.mean(losses)


def build_model(
    yolo_description: Dict[str, Any],
    *,
    moco: bool,
    rep_dims: int,
    queue_size: int,
    momentum_weight: float,
    temperature: float,
) -> nn.Module:
    """
    Builds the complete SimCLR model.

    Args:
        yolo_description: The description of the YOLO model to use for the
            backbone.
        moco: Whether to use MoCo.
        rep_dims: The size of the representation to use.
        queue_size: The size of the queue to use.
        momentum_weight: The weight to use for the momentum update.
        temperature: The temperature to use for the NT-Xent loss.

    Returns:
        The model that it built.

    """

    def _make_rep_model(num_outputs: int) -> RepresentationModel:
        encoder = YoloEncoder(yolo_description)
        return RepresentationModel(encoder=encoder, num_outputs=num_outputs)

    if moco:
        model = MoCo(
            _make_rep_model,
            dim=rep_dims,
            queue_size=queue_size,
            m=momentum_weight,
            temperature=temperature,
        )
    else:
        model = _make_rep_model(rep_dims)
    return model.to(DEVICE)


def load_dataset(
    *,
    image_folder: Union[Path, str],
    metadata: pd.DataFrame,
    max_frame_jitter: int = 0,
    enable_multi_view: bool = False,
    num_views: int = 3,
    samples_per_clip: Optional[int] = None,
) -> data.Dataset:
    """
    Loads the training dataset.

    Args:
        image_folder: The path to the training images.
        metadata: The metadata associated with this dataset.
        max_frame_jitter: Maximum amount of temporal jitter to apply when
            selecting frames.
        enable_multi_view: Whether to enable training with views from
            different cameras as positive pairs. Otherwise, it will use
            vanilla SimCLR.
        num_views: If multi-view training is enabled, how many views to use.
            If >3, it will use temporal augmentation.
        samples_per_clip: If specified, it will be the maximum number of
            examples to include in the dataset from each clip.

    Returns:
        The dataset that it loaded.

    """
    image_folder = Path(image_folder)

    # Specify 0 for the time ranges to use even short clips.
    frame_selector = FrameSelector(
        metadata,
        positive_time_range=(0.0, 0.0),
        negative_time_range=(0.0, 0.0),
    )
    if not enable_multi_view:
        # Use vanilla SimCLR.
        frame_dataset = SingleFrameDataset(
            mars_metadata=metadata,
            image_folder=image_folder,
            samples_per_clip=samples_per_clip,
        )
        paired_frames = PairedAugmentedDataset(
            image_dataset=frame_dataset,
            # We augment in the training loop.
            augmentation=lambda x: x,
        )
    elif num_views <= 3:
        # We don't need to add temporal augmentation.
        paired_frames = MultiViewDataset(
            frames=frame_selector,
            image_folder=image_folder,
            max_jitter=max_frame_jitter,
            all_views=(num_views > 2),
        )
    else:
        # We do need temporal augmentation.
        paired_frames = TemporalMultiViewDataset(
            frames=frame_selector,
            image_folder=image_folder,
            max_jitter=max_frame_jitter,
            all_views=True,
            num_extra_views=num_views - 3,
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
    contrastive_crop: bool = True,
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
        contrastive_crop: Whether to use ContrastiveCrop.

    Returns:
        The trained model.

    """
    is_moco = isinstance(model, MoCo)
    if is_moco:
        # MoCo expects us to apply normal loss here.
        loss_fn = torch.nn.CrossEntropyLoss()
        representation_model = model.encoder_q
    else:
        loss_fn = NtXentLoss(temperature=temperature)
        representation_model = model
    loss_fn = loss_fn.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=2, min_lr=1e-5)
    scaler = GradScaler()
    accuracy = ProxyClassAccuracy().to(DEVICE) if not is_moco else None

    crop_args = dict(
        size=410,
        scale=(0.08, 1.0),
        interpolation=InterpolationMode.NEAREST,
    )
    if contrastive_crop:
        crop = ContrastiveCrop(
            heatmap_threshold=0.1, alpha=0.6, device=DEVICE, **crop_args
        )
    else:
        crop = RandomResizedCrop(**crop_args)
    augmentation = MultiArgCompose(
        [
            crop,
            # Apparently, crops sometimes produce non-contiguous views,
            # and RandAugment doesn't like that.
            Lambda(lambda t: t.contiguous()),
            RandAugment(magnitude=2, interpolation=InterpolationMode.NEAREST),
        ]
    )
    # Update the dataset augmentation.
    training_data.augmentation = augmentation

    data_loader = data.DataLoader(
        training_data,
        batch_size=batch_size,
        collate_fn=_collate_views,
        pin_memory=True,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )
    # Create a secondary data loader for contrastive cropping.
    collate_resize = partial(_collate_different_sizes, output_size=(410, 410))
    single_frame_loader = data.DataLoader(
        training_data.single_frame_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=8,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_resize,
    )
    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scaler=scaler,
        accuracy=accuracy,
        checkpoint_period=5,
    )

    # Update contrastive crop 4 times during training.
    region_update_epochs = np.linspace(
        0, num_epochs, 5, endpoint=False
    ).astype(int)
    # Don't update on the first one.
    region_update_epochs = set(region_update_epochs[1:])
    if contrastive_crop:
        logger.debug("Updating regions on epochs {}", region_update_epochs)

    for i in range(num_epochs):
        logger.info("Starting epoch {}...", i)
        if contrastive_crop and i in region_update_epochs:
            # Update contrastive crop regions.
            crop.update_regions(representation_model, single_frame_loader)

        average_loss = training_loop.train_epoch(data_loader)

        logger.info("Epoch {} loss: {}", i, average_loss)
        scheduler.step(average_loss)

    return model
