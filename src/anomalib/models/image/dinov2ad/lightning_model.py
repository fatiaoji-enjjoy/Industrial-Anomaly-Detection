# Copyright (C) 2025 DINOv2AD Contributors
# SPDX-License-Identifier: Apache-2.0

"""DINOv2AD Lightning Module for anomaly detection.

DINOv2AD: DINOv2-based unsupervised industrial anomaly detection and localization.
This LightningModule fully adapts to Anomalib Engine, Trainer, test/eval standard
interfaces, with proper barebones mode support.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import DINOv2AD
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = DINOv2AD(
    ...     model_name='facebook/dinov2-small',
    ...     k=3,
    ...     aggregation='max',
    ...     inference_size=224,
    ...     coreset_sampling_ratio=0.1,
    ... )
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)
"""

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import DINOv2ADModel

logger = logging.getLogger(__name__)

__all__ = ["DINOv2AD"]


class DINOv2AD(MemoryBankMixin, AnomalibModule):
    """DINOv2AD Lightning Module for anomaly detection.

    Uses DINOv2 pretrained features, kNN distance scoring, and coreset
    subsampling for efficient anomaly detection and localization.

    Args:
        model_name: DINOv2 model name (e.g., 'facebook/dinov2-small').
        device: Computation device ('cuda' or 'cpu').
        pretrained_weights_path: Local pretrained weights path.
        k: Number of nearest neighbors for scoring.
        aggregation: Aggregation method ('max', 'mean', 'median').
        use_adaptive_mask: Whether to use PCA foreground masking.
        inference_size: Inference image size.
        coreset_sampling_ratio: Ratio for coreset subsampling.
        pre_processor: Pre-processor or bool flag.
        post_processor: Post-processor or bool flag.
        evaluator: Evaluator or bool flag.
        visualizer: Visualizer or bool flag.
    """

    def __init__(
        self,
        model_name: str = 'facebook/dinov2-small',
        device: str = 'cuda',
        pretrained_weights_path: str | None = None,
        k: int = 3,
        aggregation: str = 'max',
        use_adaptive_mask: bool = False,
        inference_size: int = 224,
        coreset_sampling_ratio: float = 0.1,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model: DINOv2ADModel = DINOv2ADModel(
            model_name=model_name,
            device=device,
            pretrained_weights_path=pretrained_weights_path,
            k=k,
            aggregation=aggregation,
            use_adaptive_mask=use_adaptive_mask,
            inference_size=inference_size,
            coreset_sampling_ratio=coreset_sampling_ratio,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio

    @staticmethod
    def configure_optimizers() -> None:
        """DINOv2AD requires no optimization (training-free memory bank)."""
        return

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:
        """Extract and store features from normal training images.

        Args:
            batch: Input batch containing images and metadata.
            *args: Unused.
            **kwargs: Unused.

        Returns:
            Dummy loss tensor for Lightning compatibility.
        """
        del args, kwargs
        _ = self.model(batch.image)
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Apply coreset subsampling to the memory bank after training."""
        logger.info("Applying coreset subsampling to memory bank (ratio=%.2f).", self.coreset_sampling_ratio)
        self.model.subsample_embedding(self.coreset_sampling_ratio)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Generate predictions for a batch of images.

        Args:
            batch: Input batch containing images and metadata.
            *args: Unused.
            **kwargs: Unused.

        Returns:
            Batch with added predictions (pred_score, anomaly_map).
        """
        del args, kwargs
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    def test_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Generate predictions for test batch.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
            *args: Unused.
            **kwargs: Unused.

        Returns:
            Batch with predictions.
        """
        del args, kwargs, batch_idx
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get default trainer arguments for DINOv2AD.

        Returns:
            Dict with trainer arguments:
                - max_epochs: 1 (single pass through training data)
                - num_sanity_val_steps: 0 (skip validation sanity checks)
                - devices: 1 (single GPU only)
        """
        return {"max_epochs": 1, "num_sanity_val_steps": 0, "devices": 1}

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type.

        Returns:
            LearningType.ONE_CLASS (only normal samples used for training).
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Configure the default post-processor for one-class models.

        Returns:
            PostProcessor instance.
        """
        return PostProcessor()

    @staticmethod
    def configure_pre_processor(image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the default pre-processor for DINOv2AD.

        Args:
            image_size: Target image size. Defaults to (224, 224).

        Returns:
            Configured PreProcessor instance.
        """
        from torchvision.transforms.v2 import Compose, Normalize, Resize

        image_size = image_size or (224, 224)
        return PreProcessor(
            transform=Compose([
                Resize(image_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        )