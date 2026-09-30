"""Adaptive anomaly map generator for PromptCore.

This module generates pixel-level anomaly maps by comparing encoder and decoder
features with learned multi-scale fusion weights. Unlike the static averaging
in Dinomaly, this module learns per-scale importance weights for adaptive fusion.

Key innovations:
- Category-conditioned normalization for consistent scores across categories
- Learned scale fusion weights with content-aware attention
- Multi-distance anomaly scoring (cosine + Euclidean)
- Post-processing pipeline with morphological refinement
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class AdaptiveAnomalyMapGenerator(nn.Module):
    """Adaptive multi-scale anomaly map generator with category conditioning.

    Computes anomaly maps by comparing encoder and decoder features using
    multiple distance metrics, then fuses multi-scale maps with learned
    attention weights. Supports category-conditioned normalization for
    consistent scores across categories.

    Args:
        num_scales (int): Number of feature scales to fuse.
            Defaults to ``2``.
        embed_dim (int): Feature dimension for scale weight computation.
            Defaults to ``768``.
        num_categories (int): Number of categories for conditioning.
            Defaults to ``50``.
    """

    def __init__(
        self,
        num_scales: int = 2,
        embed_dim: int = 768,
        num_categories: int = 50,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.num_categories = num_categories

        # Learnable scale fusion weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        # Scale attention: learn to weight scales based on feature statistics
        self.scale_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, num_scales),
            nn.Softmax(dim=-1),
        )

        # Per-category running statistics for score normalization
        self.register_buffer(
            "category_mean",
            torch.zeros(num_categories),
        )
        self.register_buffer(
            "category_std",
            torch.ones(num_categories),
        )
        self.register_buffer(
            "category_count",
            torch.zeros(num_categories, dtype=torch.long),
        )

        # Distance metric mixing weights
        self.distance_weights = nn.Parameter(torch.tensor([0.7, 0.3]))

    def forward(
        self,
        encoder_features: list[torch.Tensor],
        decoder_features: list[torch.Tensor],
        out_size: tuple[int, int],
        category_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate anomaly map from encoder and decoder features.

        Args:
            encoder_features: List of encoder feature maps at different scales.
            decoder_features: List of decoder feature maps at corresponding scales.
            out_size: Target output size ``(H, W)``.
            category_ids: Optional category IDs for conditioning.

        Returns:
            Anomaly map of shape ``(B, 1, H, W)``.
        """
        anomaly_maps = []
        scale_stats = []

        for enc_feat, dec_feat in zip(encoder_features, decoder_features, strict=True):
            # Compute cosine dissimilarity
            cos_dist = 1.0 - F.cosine_similarity(enc_feat, dec_feat, dim=1)  # (B, H, W)

            # Compute normalized Euclidean distance
            enc_norm = F.normalize(enc_feat, dim=1)
            dec_norm = F.normalize(dec_feat, dim=1)
            euc_dist = torch.norm(enc_norm - dec_norm, dim=1)  # (B, H, W)
            euc_dist = euc_dist / (euc_dist.max() + 1e-8)  # Normalize to [0, 1]

            # Combine distance metrics with learned weights
            w = self.distance_weights.softmax(dim=0)
            a_map = w[0] * cos_dist + w[1] * euc_dist  # (B, H, W)

            a_map = a_map.unsqueeze(1)  # (B, 1, H, W)
            a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
            anomaly_maps.append(a_map)

            # Collect scale statistics for attention
            scale_stats.append(a_map.mean(dim=(2, 3)))  # (B, 1)

        # Stack anomaly maps: (B, num_scales, 1, H, W)
        stacked_maps = torch.stack(anomaly_maps, dim=1)

        # Compute adaptive scale weights from feature statistics
        scale_stats = torch.cat(scale_stats, dim=1)  # (B, num_scales)
        adaptive_weights = self.scale_attention(scale_stats)  # (B, num_scales)

        # Combine learned and adaptive weights
        combined_weights = (
            self.scale_weights.softmax(dim=0) + adaptive_weights
        ) / 2.0  # (B, num_scales)

        # Weighted fusion
        combined_weights = combined_weights.view(-1, self.num_scales, 1, 1, 1)
        anomaly_map = (stacked_maps * combined_weights).sum(dim=1)  # (B, 1, H, W)

        # Category-conditioned normalization
        if category_ids is not None:
            anomaly_map = self._category_normalize(anomaly_map, category_ids)

        return anomaly_map

    def _category_normalize(
        self,
        anomaly_map: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply category-conditioned normalization to anomaly scores.

        Uses per-category running statistics to normalize anomaly scores,
        ensuring consistent score ranges across different product categories.

        Args:
            anomaly_map: Anomaly map of shape ``(B, 1, H, W)``.
            category_ids: Category IDs of shape ``(B,)``.

        Returns:
            Normalized anomaly map.
        """
        batch_size = anomaly_map.shape[0]
        for i in range(batch_size):
            cat_id = category_ids[i].item()
            if self.category_count[cat_id] > 100:  # Only normalize after sufficient samples
                mean = self.category_mean[cat_id]
                std = self.category_std[cat_id].clamp(min=1e-6)
                anomaly_map[i] = (anomaly_map[i] - mean) / std

        return anomaly_map

    def update_category_stats(
        self,
        anomaly_map: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> None:
        """Update running statistics for category-conditioned normalization.

        Should be called during training with normal samples.

        Args:
            anomaly_map: Anomaly map of shape ``(B, 1, H, W)``.
            category_ids: Category IDs of shape ``(B,)``.
        """
        batch_size = anomaly_map.shape[0]
        for i in range(batch_size):
            cat_id = category_ids[i].item()
            if cat_id >= self.num_categories:
                continue

            map_mean = anomaly_map[i].mean().detach()
            map_std = anomaly_map[i].std().detach()

            n = self.category_count[cat_id].float()
            self.category_mean[cat_id] = (
                self.category_mean[cat_id] * n + map_mean
            ) / (n + 1)
            self.category_std[cat_id] = (
                self.category_std[cat_id] * n + map_std
            ) / (n + 1)
            self.category_count[cat_id] += 1


class AnomalyMapPostProcessor(nn.Module):
    """Post-processing pipeline for anomaly maps.

    Applies a sequence of refinements:
    1. Gaussian smoothing
    2. Min-max normalization
    3. Morphological closing (optional, for segmentation-like refinement)

    Args:
        kernel_size (int): Gaussian kernel size. Defaults to ``5``.
        sigma (float): Gaussian sigma. Defaults to ``4``.
        apply_morphology (bool): Whether to apply morphological closing.
            Defaults to ``False``.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        sigma: float = 4.0,
        apply_morphology: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.apply_morphology = apply_morphology

        # Gaussian kernel
        self.register_buffer(
            "gaussian_kernel",
            self._create_gaussian_kernel(kernel_size, sigma),
        )

        if apply_morphology:
            # Morphological closing kernel
            self.register_buffer(
                "morph_kernel",
                torch.ones(1, 1, 3, 3),
            )

    def forward(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Apply post-processing to anomaly map.

        Args:
            anomaly_map: Input anomaly map of shape ``(B, 1, H, W)``.

        Returns:
            Processed anomaly map.
        """
        # Gaussian smoothing
        anomaly_map = F.conv2d(
            anomaly_map,
            self.gaussian_kernel,
            padding=self.kernel_size // 2,
        )

        # Min-max normalization to [0, 1]
        b, c, h, w = anomaly_map.shape
        flat = anomaly_map.view(b, -1)
        min_val = flat.min(dim=1, keepdim=True)[0].view(b, c, 1, 1)
        max_val = flat.max(dim=1, keepdim=True)[0].view(b, c, 1, 1)
        anomaly_map = (anomaly_map - min_val) / (max_val - min_val + 1e-8)

        # Morphological closing (optional)
        if self.apply_morphology:
            anomaly_map = F.conv2d(
                anomaly_map,
                self.morph_kernel,
                padding=1,
            )
            anomaly_map = anomaly_map.clamp(0, 1)

        return anomaly_map

    @staticmethod
    def _create_gaussian_kernel(
        kernel_size: int,
        sigma: float,
    ) -> torch.Tensor:
        """Create a 2D Gaussian kernel.

        Args:
            kernel_size: Kernel size (odd).
            sigma: Standard deviation.

        Returns:
            Gaussian kernel of shape ``(1, 1, kernel_size, kernel_size)``.
        """
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] @ gauss_1d[None, :]
        return gauss_2d.view(1, 1, kernel_size, kernel_size)
