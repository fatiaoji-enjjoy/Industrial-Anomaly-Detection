"""PyTorch model for PromptCore: Prompt-Guided Multi-Category Transformer.

PromptCore is a Transformer-based unsupervised anomaly detection model designed for
multi-category industrial inspection. It combines:

1. **DINOv3 ViT Encoder** (frozen): Extracts universal visual features
2. **Category Prompt Encoder**: Learnable per-category prompt tokens for
   category-conditioned reconstruction
3. **INP Extractor**: Cross-attention based Intrinsic Normal Prototype
   extraction from patch features
4. **Prompt-Conditioned Decoder**: Transformer decoder that reconstructs
   normal features using the prototype bank as keys/values
5. **Adaptive Anomaly Map Generator**: Multi-scale anomaly map fusion with
   learned weights and category-conditioned normalization

Key innovations:
- Category-specific prompt tokens for single-model multi-category inference
- Hierarchical prototype bank (INPs + prompts) for richer feature conditioning
- INP-Guided cross-attention decoder preventing anomalous feature reconstruction
- Adaptive multi-scale fusion with learned scale weights
- Multi-category adaptive loss with contrastive regularization
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components.feature_extractors import TimmFeatureExtractor
from anomalib.models.image.dinomaly.components import DinomalyMLP

from .components.anomaly_map import (
    AdaptiveAnomalyMapGenerator,
    AnomalyMapPostProcessor,
)
from .components.decoder import PromptConditionedDecoder
from .components.loss import PromptCoreLoss
from .components.prompt_encoder import CategoryPromptEncoder
from .components.prototype_bank import INPExtractor

# DINOv3 architecture configurations
DINO_ARCHITECTURES = {
    "small": {"embed_dim": 384, "num_heads": 6, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "base": {"embed_dim": 768, "num_heads": 12, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "large": {"embed_dim": 1024, "num_heads": 16, "target_layers": [4, 6, 8, 10, 12, 14, 16, 18]},
}

# Default fusion layer configurations
DEFAULT_FUSE_LAYERS = [[0, 1, 2, 3], [4, 5, 6, 7]]

# Inference processing defaults
DEFAULT_RESIZE_SIZE = 256
DEFAULT_GAUSSIAN_KERNEL_SIZE = 5
DEFAULT_GAUSSIAN_SIGMA = 4
DEFAULT_MAX_RATIO = 0.01


class PromptCoreModel(nn.Module):
    """PromptCore: Prompt-Guided Multi-Category Transformer for Anomaly Detection.

    The model uses a DINOv3 ViT encoder to extract features, category-specific
    prompt tokens to condition the reconstruction, and a prototype-guided
    Transformer decoder to reconstruct normal features. Anomalies are detected
    by comparing encoder and decoder features.

    Args:
        encoder_name (str): Name of the ViT encoder model.
            Defaults to ``"vit_base_patch14_reg4_dinov3"``.
        num_categories (int): Maximum number of product categories.
            Defaults to ``50``.
        num_prompts (int): Number of prompt tokens per category.
            Defaults to ``4``.
        num_inps (int): Number of INP tokens to extract.
            Defaults to ``8``.
        decoder_depth (int): Number of decoder layers.
            Defaults to ``8``.
        bottleneck_dropout (float): Dropout rate for bottleneck MLP.
            Defaults to ``0.2``.
        target_layers (list[int] | None): Encoder layer indices for feature
            extraction. If ``None``, auto-detected from architecture.
        fuse_layer_encoder (list[list[int]] | None): Encoder layer fusion groups.
        fuse_layer_decoder (list[list[int]] | None): Decoder layer fusion groups.
        remove_class_token (bool): Whether to remove class token.
            Defaults to ``False``.
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
    ) -> None:
        super().__init__()

        # Extract architecture config
        arch_config = self._get_architecture_config(encoder_name, target_layers)
        embed_dim = arch_config["embed_dim"]
        num_heads = arch_config["num_heads"]

        if target_layers is None:
            self.target_layers = arch_config["target_layers"]
        else:
            self.target_layers = target_layers

        self.fuse_layer_encoder = fuse_layer_encoder or DEFAULT_FUSE_LAYERS
        self.fuse_layer_decoder = fuse_layer_decoder or DEFAULT_FUSE_LAYERS
        self.remove_class_token = remove_class_token
        self.embed_dim = embed_dim
        self.num_categories = num_categories

        # ---- Encoder: DINOv3 ViT (frozen) ----
        self.encoder = TimmFeatureExtractor(
            backbone=encoder_name,
            layers=[f"blocks.{i}" for i in self.target_layers],
            pre_trained=True,
            requires_grad=False,
            output_fmt="NLC",
            return_class_token=True,
            norm=False,
            dynamic_img_size=True,
        )

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0

        # ---- Category Prompt Encoder ----
        self.prompt_encoder = CategoryPromptEncoder(
            num_categories=num_categories,
            num_prompts=num_prompts,
            embed_dim=embed_dim,
        )

        # ---- INP Extractor ----
        self.inp_extractor = INPExtractor(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_inps=num_inps,
            num_extractor_layers=2,
        )

        # ---- Bottleneck MLP ----
        self.bottleneck = nn.ModuleList([
            DinomalyMLP(
                in_features=embed_dim,
                hidden_features=embed_dim * 4,
                out_features=embed_dim,
                act_layer=nn.GELU,
                drop=bottleneck_dropout,
                bias=False,
                apply_input_dropout=True,
            ),
        ])

        # ---- Prompt-Conditioned Decoder ----
        self.decoder = PromptConditionedDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=decoder_depth,
        )

        # ---- Adaptive Anomaly Map Generator ----
        self.anomaly_map_generator = AdaptiveAnomalyMapGenerator(
            num_scales=len(self.fuse_layer_encoder),
            embed_dim=embed_dim,
            num_categories=num_categories,
        )

        # ---- Post-processing ----
        self.post_processor = AnomalyMapPostProcessor(
            kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE,
            sigma=DEFAULT_GAUSSIAN_SIGMA,
            apply_morphology=False,
        )

        # ---- Loss ----
        self.loss_fn = PromptCoreLoss()

    def forward(
        self,
        batch: torch.Tensor,
        category_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | dict | InferenceBatch:
        """Forward pass of PromptCore.

        Training mode: Returns loss dictionary with reconstruction loss,
        INP coherence loss, category classification loss, and contrastive loss.
        Inference mode: Returns ``InferenceBatch`` with anomaly scores and maps.

        Args:
            batch (torch.Tensor): Input images of shape ``(B, C, H, W)``.
            category_ids (torch.Tensor | None): Category IDs of shape ``(B,)``.
                Required during training, optional during inference.

        Returns:
            Training: ``dict`` with loss components.
            Inference: ``InferenceBatch`` with ``pred_score`` and ``anomaly_map``.
        """
        dtype = next(self.encoder.parameters()).dtype
        batch = batch.type(dtype)
        image_size = (batch.shape[2], batch.shape[3])

        # Extract encoder features and class token
        en_features, de_features, cls_token, inp_loss, category_logits = (
            self._extract_and_reconstruct(batch, category_ids)
        )

        if self.training:
            if category_ids is None:
                msg = "category_ids must be provided during training"
                raise ValueError(msg)
            total_loss, loss_dict = self.loss_fn(
                encoder_features=en_features,
                decoder_features=de_features,
                inp_loss=inp_loss,
                category_logits=category_logits,
                category_ids=category_ids,
            )
            return loss_dict

        # ---- Inference: generate anomaly map ----
        anomaly_map = self.anomaly_map_generator(
            en_features, de_features, out_size=image_size,
            category_ids=category_ids,
        )

        # Store original anomaly map before post-processing
        anomaly_map_raw = anomaly_map.clone()

        # Apply post-processing pipeline
        anomaly_map = self.post_processor(anomaly_map)

        # Compute anomaly score (top-k mean of processed map)
        anomaly_map_flat = anomaly_map.flatten(1)
        k = max(1, int(anomaly_map_flat.shape[1] * DEFAULT_MAX_RATIO))
        pred_score = torch.topk(anomaly_map_flat, k=k, dim=1)[0].mean(dim=1)

        return InferenceBatch(
            pred_score=pred_score,
            anomaly_map=anomaly_map_raw,
        )

    def predict_with_tta(
        self,
        batch: torch.Tensor,
        category_ids: torch.Tensor | None = None,
        scales: tuple[float, ...] = (1.0, 0.75, 0.5),
    ) -> InferenceBatch:
        """Inference with Test-Time Augmentation (multi-scale).

        Runs inference at multiple scales and averages the anomaly maps
        for more robust predictions. This is optional and trades speed
        for accuracy.

        Args:
            batch: Input images of shape ``(B, C, H, W)``.
            category_ids: Optional category IDs.
            scales: Tuple of scale factors to use.

        Returns:
            ``InferenceBatch`` with averaged anomaly scores and maps.
        """
        image_size = (batch.shape[2], batch.shape[3])
        all_maps = []
        all_scores = []

        for scale in scales:
            if scale == 1.0:
                scaled_batch = batch
            else:
                scaled_size = (int(image_size[0] * scale), int(image_size[1] * scale))
                scaled_batch = F.interpolate(
                    batch, size=scaled_size, mode="bilinear", align_corners=False,
                )

            result = self.forward(scaled_batch, category_ids)
            if scale != 1.0:
                result = InferenceBatch(
                    pred_score=result.pred_score,
                    anomaly_map=F.interpolate(
                        result.anomaly_map, size=image_size,
                        mode="bilinear", align_corners=False,
                    ),
                )
            all_scores.append(result.pred_score)
            all_maps.append(result.anomaly_map)

        # Average across scales
        avg_score = torch.stack(all_scores).mean(dim=0)
        avg_map = torch.stack(all_maps).mean(dim=0)

        return InferenceBatch(pred_score=avg_score, anomaly_map=avg_map)

    def estimate_inference_time(
        self,
        input_shape: tuple[int, int, int, int] = (1, 3, 392, 392),
        num_warmup: int = 10,
        num_runs: int = 100,
        device: str = "cuda",
    ) -> dict[str, float]:
        """Estimate per-component inference time.

        Utility for benchmarking inference speed. Returns timing breakdown
        for each stage of the pipeline.

        Args:
            input_shape: Input tensor shape ``(B, C, H, W)``.
            num_warmup: Number of warmup iterations.
            num_runs: Number of timed iterations.
            device: Device to run on.

        Returns:
            Dictionary with timing breakdown in milliseconds.
        """
        dummy_input = torch.randn(input_shape, device=device)
        self.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.forward(dummy_input)

        # Timed run
        import time

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.forward(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize()
        total_time = (time.perf_counter() - start) / num_runs * 1000

        return {"total_inference_ms": total_time}

    def _extract_and_reconstruct(
        self,
        x: torch.Tensor,
        category_ids: torch.Tensor | None = None,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Extract features and reconstruct through the full pipeline.

        Args:
            x: Input images of shape ``(B, C, H, W)``.
            category_ids: Optional category IDs.

        Returns:
            Tuple of (encoder_features, decoder_features, cls_token, inp_loss,
            category_logits).
        """
        h_patches = x.shape[2] // self.encoder.patch_size
        w_patches = x.shape[3] // self.encoder.patch_size

        # 1. Extract features from encoder
        features = self.encoder(x)
        encoder_features = [features[f"blocks.{i}"] for i in self.target_layers]

        # 2. Extract class token for prompt generation
        cls_token = encoder_features[0][:, 0, :]  # (B, D)

        # 3. Generate category prompts
        category_prompts, category_logits = self.prompt_encoder(cls_token, category_ids)

        # 4. Remove class token and register tokens from patch features
        if self.remove_class_token:
            patch_features = [
                f[:, 1 + self.encoder.num_register_tokens:, :]
                for f in encoder_features
            ]
        else:
            patch_features = encoder_features

        # 5. Fuse multi-layer encoder features
        fused_patches = self._fuse_feature(patch_features)

        # 6. Extract INPs and build prototype bank
        prototype_bank, inp_loss = self.inp_extractor(fused_patches, category_prompts)

        # 7. Bottleneck compression
        x_bottleneck = fused_patches
        for block in self.bottleneck:
            x_bottleneck = block(x_bottleneck)

        # 8. Decoder reconstruction with prototype guidance
        decoder_outputs = self.decoder(x_bottleneck, prototype_bank)

        # 9. Fuse multi-scale encoder and decoder features
        en_fused = [
            self._fuse_feature([encoder_features[idx] for idx in idxs])
            for idxs in self.fuse_layer_encoder
        ]
        de_fused = [
            self._fuse_feature([decoder_outputs[idx] for idx in idxs])
            for idxs in self.fuse_layer_decoder
        ]

        # 10. Reshape to spatial format
        en_fused = self._to_spatial(en_fused, h_patches, w_patches)
        de_fused = self._to_spatial(de_fused, h_patches, w_patches)

        return en_fused, de_fused, cls_token, inp_loss, category_logits

    def _to_spatial(
        self,
        features: list[torch.Tensor],
        h_patches: int,
        w_patches: int,
    ) -> list[torch.Tensor]:
        """Convert token sequences to spatial feature maps.

        Args:
            features: List of token tensors of shape ``(B, N, D)``.
            h_patches: Number of patches in height.
            w_patches: Number of patches in width.

        Returns:
            List of spatial tensors of shape ``(B, D, H, W)``.
        """
        if not self.remove_class_token:
            features = [
                f[:, 1 + self.encoder.num_register_tokens:, :]
                for f in features
            ]
        batch_size = features[0].shape[0]
        return [
            f.permute(0, 2, 1)
            .reshape(batch_size, -1, h_patches, w_patches)
            .contiguous()
            for f in features
        ]

    @staticmethod
    def _fuse_feature(feat_list: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple feature tensors by averaging.

        Args:
            feat_list: List of feature tensors to fuse.

        Returns:
            Averaged feature tensor.
        """
        return torch.stack(feat_list, dim=1).mean(dim=1)

    @staticmethod
    def _get_architecture_config(
        encoder_name: str,
        target_layers: list[int] | None,
    ) -> dict:
        """Get architecture configuration from encoder name.

        Args:
            encoder_name: Name of the encoder model.
            target_layers: Optional override for target layers.

        Returns:
            Dictionary with ``embed_dim``, ``num_heads``, and ``target_layers``.

        Raises:
            ValueError: If architecture is not supported.
        """
        for arch_name, config in DINO_ARCHITECTURES.items():
            if arch_name in encoder_name:
                result = config.copy()
                if target_layers is not None:
                    result["target_layers"] = target_layers
                return result

        msg = (
            f"Architecture not supported. Encoder name must contain one of "
            f"{list(DINO_ARCHITECTURES.keys())}"
        )
        raise ValueError(msg)
