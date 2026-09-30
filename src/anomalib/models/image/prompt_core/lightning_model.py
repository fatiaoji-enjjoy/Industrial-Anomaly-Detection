"""PromptCore Lightning Module.

This module wraps the PromptCoreModel in a PyTorch Lightning module for training,
validation, and inference. It handles the training loop, optimizer configuration,
and metric logging.

The model is designed for single-model multi-category unsupervised anomaly
detection on industrial inspection datasets.
"""

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.nn.init import trunc_normal_
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from anomalib import LearningType, PrecisionType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.models.components.base import restore_frozen_encoder_weights
from anomalib.models.image.dinomaly.components import StableAdamW, WarmCosineScheduler
from anomalib.models.image.prompt_core.torch_model import PromptCoreModel
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

logger = logging.getLogger(__name__)

# Training constants
DEFAULT_IMAGE_SIZE = 448
DEFAULT_CROP_SIZE = 392
MAX_STEPS_DEFAULT = 5000

# Default training hyperparameters
TRAINING_CONFIG: dict[str, Any] = {
    "optimizer": {
        "lr": 2e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
        "amsgrad": True,
        "eps": 1e-8,
    },
    "scheduler": {
        "base_value": 2e-3,
        "final_value": 2e-4,
        "total_iters": MAX_STEPS_DEFAULT,
        "warmup_iters": 100,
    },
    "trainer": {
        "gradient_clip_val": 0.1,
        "num_sanity_val_steps": 0,
        "max_steps": MAX_STEPS_DEFAULT,
    },
}


class PromptCore(AnomalibModule):
    """PromptCore Lightning Module for Prompt-Guided Multi-Category Anomaly Detection.

    PromptCore uses a DINOv3 ViT encoder with category-specific prompt tokens and
    a prototype-guided Transformer decoder for unsupervised anomaly detection.
    The model is designed to handle multiple product categories with a single
    model, using learnable per-category prompt tokens.

    Args:
        encoder_name (str): ViT encoder model name.
            Defaults to ``"vit_base_patch14_reg4_dinov3"``.
        num_categories (int): Maximum number of product categories.
            Defaults to ``50``.
        num_prompts (int): Number of prompt tokens per category.
            Defaults to ``4``.
        num_inps (int): Number of INP tokens.
            Defaults to ``8``.
        decoder_depth (int): Number of decoder layers.
            Defaults to ``8``.
        bottleneck_dropout (float): Dropout rate for bottleneck.
            Defaults to ``0.2``.
        target_layers (list[int] | None): Encoder layer indices.
        fuse_layer_encoder (list[list[int]] | None): Encoder fusion groups.
        fuse_layer_decoder (list[list[int]] | None): Decoder fusion groups.
        remove_class_token (bool): Whether to remove class token.
            Defaults to ``False``.
        precision (str | PrecisionType): Numerical precision.
            Defaults to ``PrecisionType.FLOAT32``.
        pre_processor (PreProcessor | bool): Pre-processor configuration.
            Defaults to ``True``.
        post_processor (PostProcessor | bool): Post-processor configuration.
            Defaults to ``True``.
        evaluator (Evaluator | bool): Evaluator configuration.
            Defaults to ``True``.
        visualizer (Visualizer | bool): Visualizer configuration.
            Defaults to ``True``.
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch14_reg4_dinov3",
        num_categories: int = 50,
        num_prompts: int = 4,
        num_inps: int = 8,
        decoder_depth: int = 8,
        bottleneck_dropout: float = 0.2,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
        precision: str | PrecisionType = PrecisionType.FLOAT32,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model: PromptCoreModel = PromptCoreModel(
            encoder_name=encoder_name,
            num_categories=num_categories,
            num_prompts=num_prompts,
            num_inps=num_inps,
            decoder_depth=decoder_depth,
            bottleneck_dropout=bottleneck_dropout,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token,
        )

        if isinstance(precision, str):
            precision = PrecisionType(precision.lower())

        if precision == PrecisionType.FLOAT16:
            self.model = self.model.to(torch.bfloat16)
        elif precision == PrecisionType.FLOAT32:
            self.model = self.model.float()
        else:
            msg = (
                f"Unsupported precision type: {precision}. "
                f"Supported types are: {PrecisionType.FLOAT16}, {PrecisionType.FLOAT32}."
            )
            raise ValueError(msg)

        # Freeze encoder, train only prompt encoder, INP extractor, bottleneck, and decoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # Collect all trainable parameters
        self.trainable_modules = torch.nn.ModuleList([
            self.model.prompt_encoder,
            self.model.inp_extractor,
            self.model.bottleneck,
            self.model.decoder,
            self.model.anomaly_map_generator,
        ])
        self._initialize_trainable_modules(self.trainable_modules)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore frozen encoder weights from checkpoint.

        Args:
            checkpoint: The checkpoint dictionary being loaded.
        """
        restore_frozen_encoder_weights(self, checkpoint, encoder_key="encoder")

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        crop_size: int | None = None,
    ) -> PreProcessor:
        """Configure the default pre-processor for PromptCore.

        Args:
            image_size: Target resize size as ``(H, W)``.
            crop_size: Center crop size (square).

        Returns:
            Configured PreProcessor with DINOv3 normalization.
        """
        crop_size = crop_size or DEFAULT_CROP_SIZE
        image_size = image_size or (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

        if crop_size > min(image_size):
            msg = f"Crop size {crop_size} cannot be larger than image size {image_size}"
            raise ValueError(msg)

        data_transforms = Compose([
            Resize(image_size),
            CenterCrop(crop_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return PreProcessor(transform=data_transforms)

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Training step for PromptCore.

        Args:
            batch: Input batch containing images and category IDs.
            *args: Unused.
            **kwargs: Unused.

        Returns:
            Dictionary containing the combined loss.
        """
        del args, kwargs

        category_ids = batch.get("category_id", None)
        loss_dict = self.model(batch.image, category_ids=category_ids)

        # Log individual loss components
        for name, value in loss_dict.items():
            self.log(f"train_{name}", value, on_step=True, on_epoch=True, prog_bar=(name == "total_loss"))

        return {"loss": loss_dict["total_loss"]}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Validation step for PromptCore.

        Args:
            batch: Input batch containing images.
            *args: Unused.
            **kwargs: Unused.

        Returns:
            Updated batch with predictions.
        """
        del args, kwargs

        predictions = self.model(batch.image)
        return batch.update(
            pred_score=predictions.pred_score,
            anomaly_map=predictions.anomaly_map,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and learning rate scheduler.

        Uses StableAdamW with warm cosine scheduling. Only trainable modules
        (prompt encoder, INP extractor, bottleneck, decoder, anomaly map generator)
        are optimized.

        Returns:
            Tuple of optimizer and scheduler configurations.
        """
        max_epochs = getattr(self.trainer, "max_epochs", -1)
        max_steps = getattr(self.trainer, "max_steps", -1)

        if max_epochs is None:
            max_epochs = -1
        if max_steps is None:
            max_steps = -1

        if max_epochs < 0 and max_steps < 0:
            msg = "A finite number of steps or epochs must be defined"
            raise ValueError(msg)

        if max_epochs < 0:
            total_steps = max_steps
        elif max_steps < 0:
            total_steps = max_epochs * len(self.trainer.datamodule.train_dataloader())
        else:
            total_steps = min(max_steps, max_epochs * len(self.trainer.datamodule.train_dataloader()))

        optimizer_config = TRAINING_CONFIG["optimizer"]
        assert isinstance(optimizer_config, dict)
        optimizer = StableAdamW(
            [{"params": self.trainable_modules.parameters()}],
            **optimizer_config,
        )

        scheduler_config = TRAINING_CONFIG["scheduler"].copy()
        assert isinstance(scheduler_config, dict)
        scheduler_config["total_iters"] = total_steps
        lr_scheduler = WarmCosineScheduler(optimizer, **scheduler_config)

        return [optimizer], [lr_scheduler]

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type.

        PromptCore is an unsupervised anomaly detection model.

        Returns:
            ``LearningType.ONE_CLASS``.
        """
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return PromptCore-specific trainer arguments.

        Returns:
            Dictionary of trainer arguments.
        """
        trainer_config = TRAINING_CONFIG["trainer"].copy()
        assert isinstance(trainer_config, dict)
        trainer_config.pop("max_steps", None)
        return trainer_config

    @staticmethod
    def _initialize_trainable_modules(trainable_modules: torch.nn.ModuleList) -> None:
        """Initialize trainable modules with truncated normal initialization.

        Args:
            trainable_modules: ModuleList of modules to initialize.
        """
        for m in trainable_modules.modules():
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
