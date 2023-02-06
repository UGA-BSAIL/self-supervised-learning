"""
Node definitions for the `supervised_validation` pipeline.
"""
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import wandb
from loguru import logger
from pytorch_warmup import BaseWarmup, LinearWarmup
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models import ConvNeXt, ConvNeXt_Small_Weights, convnext_small
from torchvision.transforms._presets import ImageClassification

from .centernet import CenterNet
from .heatmaps import boxes_from_heatmaps
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
        map_metric: MeanAveragePrecision,
        normalizer: Callable[[Tensor], Tensor],
        checkpoint_dir: Path = Path("checkpoints"),
        checkpoint_period: int = 10,
    ):
        """
        Args:
            model: The model to train.
            heatmap_loss_fn: The loss function to use for heatmaps.
            geometry_loss_fn: The loss function to use for geometry.
            optimizer: The optimizer to use.
            scaler: Gradient scaler to use.
            warmup: Warmup controller to use.
            map_metric: Metric to use for computing mean average precision.
                Should take bounding boxes in `[cx,cy,w,h]` format.
            normalizer: Function to use for normalizing image batches.
            checkpoint_dir: The directory to use for saving intermediate
                model checkpoints.
            checkpoint_period: How frequently to save checkpoint, in
                epochs.

        """
        self.__model = model
        self.__heatmap_loss_fn = heatmap_loss_fn
        self.__geometry_loss_fn = geometry_loss_fn
        self.__optimizer = optimizer
        self.__scaler = scaler
        self.__warmup = warmup
        self.__map_metric = map_metric
        self.__normalizer = normalizer
        self.__checkpoint_dir = checkpoint_dir
        self.__checkpoint_dir.mkdir(exist_ok=True)
        self.__checkpoint_period = checkpoint_period

        # Keeps track of the current global training step.
        self.__global_step = 0
        # Global epoch counter.
        self.__epoch = 0

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
        wandb.watch(self.__model, log_freq=100)

    def __log_epoch_end(self, prefix: str = "train") -> None:
        """
        Logs data at the end of an epoch.

        Args:
            prefix: The prefix to use for log messages.

        """
        # Compute metrics.
        all_metrics = self.__map_metric.compute()

        # Add the proper prefix.
        all_metrics = {f"{prefix}/{k}": v for k, v in all_metrics.items()}
        wandb.log({"global_step": self.__global_step, **all_metrics})

        self.__map_metric.reset()

    def __save_checkpoint(self) -> None:
        """
        Saves a model checkpoint.

        """
        if self.__epoch % self.__checkpoint_period != 0:
            # Don't save this time.
            return

        checkpoint_name = f"checkpoint_{self.__epoch}.pt"
        logger.debug("Saving checkpoint '{}'...", checkpoint_name)

        checkpoint_path = self.__checkpoint_dir / checkpoint_name
        torch.save(self.__model, checkpoint_path.as_posix())

    def __compute_loss(
        self,
        images: Tensor,
        heatmaps: Tensor,
        boxes: List[Tensor],
        log_boxes: bool = False,
        prefix: str = "train",
    ) -> Tensor:
        """
        Applies the model and computes the loss value, logging the
        loss and metrics.

        Args:
            images: The input images.
            heatmaps: The target heatmaps.
            boxes: The target bounding boxes.
            log_boxes: Whether to log bounding box visualizations.
            prefix: The prefix to use for logs.

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

        self.__update_box_metrics(
            images=images,
            pred_heatmaps=pred_heatmaps,
            pred_sizes=pred_sizes,
            pred_offsets=pred_offsets,
            target_boxes=boxes,
            log_boxes=log_boxes,
            prefix=prefix,
        )

        wandb.log(
            {
                "global_step": self.__global_step,
                f"{prefix}/heatmap_loss": heatmap_loss,
                f"{prefix}/geometry_loss": geometry_loss,
                f"{prefix}/max_confidence": pred_heatmaps.max(),
            }
        )

        return heatmap_loss + geometry_loss

    @staticmethod
    def __get_bounding_boxes(
        *,
        heatmaps: Tensor,
        sizes: Tensor,
        offsets: Tensor,
        input_shape: Optional[torch.Size] = None,
        max_boxes: Optional[int] = 100,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Gets the bounding boxes from a particular set of raw model predictions.

        Args:
            heatmaps: The predicted heatmaps.
            sizes: The predicted sizes.
            offsets: The predicted center offsets.
            input_shape: The shape of the input image. If provided, it will
                return boxes in pixel coordinates, based on this shape.
                Otherwise, it will return them in normalized coordinates.
            max_boxes: Maximum number of boxes to allow. If None, there is
                no maximum.

        Returns:
            - A list of bounding boxes in each image, in the form
                `[cx, cy, w, h]`.
            - A list of the corresponding confidences for each box in each
                image.

        """
        # Convert to bounding boxes.
        boxes, confidence = boxes_from_heatmaps(
            heatmaps=heatmaps,
            sizes=sizes,
            offsets=offsets,
            max_boxes=max_boxes,
        )

        if input_shape is not None:
            # Convert to pixel coordinates.
            input_width_height = torch.as_tensor(
                input_shape[-2:][::-1], device=DEVICE
            ).tile(2)
            boxes = [b * input_width_height for b in boxes]

        return boxes, confidence

    def __update_map(
        self,
        *,
        pred_boxes: List[Tensor],
        pred_confidence: List[Tensor],
        target_boxes: List[Tensor],
    ) -> None:
        """
        Updates the mAP metric with new bounding boxes.

        Args:
            pred_boxes: The predicted bounding boxes.
            pred_confidence: The corresponding predicted confidences.
            target_boxes: The GT bounding boxes.

        """
        predictions = [
            dict(
                boxes=b,
                scores=c,
                # There is only one class, so the labels are zero.
                labels=torch.zeros_like(c, dtype=torch.int),
            )
            for b, c in zip(pred_boxes, pred_confidence)
        ]
        targets = [
            dict(
                boxes=b,
                labels=torch.zeros(
                    (b.shape[0],), dtype=torch.int, device=b.device
                ),
            )
            for b in target_boxes
        ]
        self.__map_metric.update(predictions, targets)

    @staticmethod
    def __log_boxes(
        *,
        images: Tensor,
        pred_boxes: List[Tensor],
        pred_confidence: List[Tensor],
        prefix: str = "train",
    ) -> None:
        """
        Logs a visualization of the detected bounding boxes.

        Args:
            images: The input images.
            pred_boxes: The predicted bounding boxes.
            pred_confidence: The confidence of the predicted bounding boxes.

        """

        def _make_box_data_dict(
            _box: np.array, _confidence: float
        ) -> Dict[str, Any]:
            # Convert a single box to WandB format.
            return dict(
                position=dict(
                    middle=_box[:2].tolist(),
                    width=_box[2].tolist(),
                    height=_box[3].tolist(),
                ),
                class_id=0,
                scores=dict(confidence=float(_confidence)),
                domain="pixel",
            )

        images = images.cpu().permute((0, 2, 3, 1)).numpy()

        log_images = []
        for image, boxes, confidence in zip(
            images, pred_boxes, pred_confidence
        ):
            boxes = boxes.cpu().numpy()
            confidence = confidence.cpu().numpy()

            box_data = [
                _make_box_data_dict(b, c) for b, c in zip(boxes, confidence)
            ]
            log_images.append(
                wandb.Image(
                    image,
                    boxes=dict(
                        predictions=dict(
                            box_data=box_data, class_labels={0: "flower"}
                        )
                    ),
                )
            )

        wandb.log({f"{prefix}/boxes": log_images})

    def __update_box_metrics(
        self,
        *,
        images: Tensor,
        pred_heatmaps: Tensor,
        pred_sizes: Tensor,
        pred_offsets: Tensor,
        target_boxes: List[Tensor],
        log_boxes: bool = False,
        prefix: str = "train",
    ) -> None:
        """
        Updates metrics derived from the bounding boxes.

        Args:
            images: The input images.
            pred_heatmaps: The predicted heatmaps from the model.
            pred_sizes: The predicted sizes from the model.
            pred_offsets: The predicted offsets from the model.
            target_boxes: The target bounding boxes. Each list element contains
                boxes for one example, and should have a shape of
                `[N, 4]`, where the columns are
                `[center_x, center_y, width, height, offset_x, offset_y]`
            log_boxes: Whether to log bounding box visualizations.
            prefix: Prefix to use for logged data.

        """
        # Make sure everything is stripped of gradients before logging.
        pred_heatmaps = pred_heatmaps.detach()
        pred_sizes = pred_sizes.detach()
        pred_offsets = pred_offsets.detach()
        # Strip the offsets from the target bounding boxes.
        target_boxes = [b[:, :4] for b in target_boxes]

        # Convert to bounding boxes.
        boxes, confidence = self.__get_bounding_boxes(
            heatmaps=pred_heatmaps,
            sizes=pred_sizes,
            offsets=pred_offsets,
            input_shape=images.shape,
            max_boxes=30,
        )

        input_width_height = torch.as_tensor(
            images.shape[-2:][::-1], device=DEVICE
        ).tile(2)
        target_boxes = [b * input_width_height for b in target_boxes]

        # Update the metrics.
        self.__update_map(
            pred_boxes=boxes,
            pred_confidence=confidence,
            target_boxes=target_boxes,
        )
        if log_boxes:
            self.__log_boxes(
                images=images[:16],
                pred_boxes=boxes[:16],
                pred_confidence=confidence[:16],
                prefix=prefix,
            )

    def __iter_data_loader(
        self,
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
            images = self.__normalizer(images)

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
                loss = self.__compute_loss(
                    images, heatmaps, boxes, log_boxes=(batch_i == 0)
                )

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
        self.__epoch += 1
        return total_loss / batch_i

    def test_model(self, data_loader: data.DataLoader) -> float:
        """
        Tests the model.

        Args:
            data_loader: The data to test on.

        Returns:
            THe average loss on the testing data.

        """
        # Make sure we're only computing mAP on the validation.
        self.__map_metric.reset()

        total_loss = 0.0
        batch_i = 0
        with torch.no_grad():
            for batch_i, (_, images, heatmaps, boxes) in enumerate(
                self.__iter_data_loader(data_loader)
            ):
                # Compute loss.
                with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                    loss = self.__compute_loss(
                        images,
                        heatmaps,
                        boxes,
                        prefix="test",
                        log_boxes=(batch_i == 0),
                    )

                    logger.debug(
                        "Testing batch {}: loss={}", batch_i, loss.item()
                    )
                    total_loss += loss.item()

        self.__log_epoch_end(prefix="test")
        return total_loss / batch_i


def get_pretrained_encoder() -> ConvNeXt:
    """
    Creates an encoder model that's pre-trained on ImageNet.

    Returns:
        The model that it created.

    """
    return convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)


def build_model(encoder: Optional[nn.Module] = None) -> CenterNet:
    """
    Builds the complete CenterNet model.

    Args:
        encoder: If provided, will use this as a pre-trained encoder, instead
            of training a fresh encoder.

    Returns:
        The CenterNet model it created.

    """
    if encoder is not None:
        # Freeze all the layers.
        for param in encoder.parameters():
            param.requires_grad = False
    else:
        # Create a new one.
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
    map_metric = MeanAveragePrecision(box_format="cxcywh")

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
        map_metric=map_metric,
        normalizer=ImageClassification(crop_size=512, resize_size=512),
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
