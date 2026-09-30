"""Multi-category adaptive loss for PromptCore.

This module implements a combined loss function that integrates:
1. Adaptive cosine similarity loss with gradient re-weighting
2. INP coherence loss for prototype regularization
3. Category contrastive loss for multi-category feature separation
4. Category classification loss for prompt learning

The loss is designed to work in a single-model multi-category setting,
encouraging both accurate reconstruction and category-aware feature learning.
"""

from functools import partial

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class PromptCoreLoss(nn.Module):
    """Multi-category adaptive loss for PromptCore.

    Combines reconstruction loss, INP coherence loss, category contrastive loss,
    and category classification loss into a unified training objective.

    Args:
        rec_weight (float): Weight for reconstruction loss.
            Defaults to ``1.0``.
        inp_weight (float): Weight for INP coherence loss.
            Defaults to ``0.2``.
        contrast_weight (float): Weight for category contrastive loss.
            Defaults to ``0.1``.
        cls_weight (float): Weight for category classification loss.
            Defaults to ``0.05``.
        adaptive_power (int): Power for adaptive gradient weighting.
            Higher values amplify the difference between easy and hard points.
            Defaults to ``3``.
    """

    def __init__(
        self,
        rec_weight: float = 1.0,
        inp_weight: float = 0.2,
        contrast_weight: float = 0.1,
        cls_weight: float = 0.05,
        adaptive_power: int = 3,
    ) -> None:
        super().__init__()
        self.rec_weight = rec_weight
        self.inp_weight = inp_weight
        self.contrast_weight = contrast_weight
        self.cls_weight = cls_weight
        self.adaptive_power = adaptive_power

    def forward(
        self,
        encoder_features: list[torch.Tensor],
        decoder_features: list[torch.Tensor],
        inp_loss: torch.Tensor,
        category_logits: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the combined loss.

        Args:
            encoder_features (list[torch.Tensor]): Encoder feature maps at
                multiple scales. Each of shape ``(B, C, H, W)``.
            decoder_features (list[torch.Tensor]): Decoder feature maps at
                corresponding scales.
            inp_loss (torch.Tensor): INP coherence loss scalar.
            category_logits (torch.Tensor): Category prediction logits of shape
                ``(B, num_categories)``.
            category_ids (torch.Tensor): Ground-truth category IDs of shape
                ``(B,)``.

        Returns:
            tuple[torch.Tensor, dict]: Tuple containing:
                - total_loss: Combined scalar loss.
                - loss_dict: Dictionary with individual loss components.
        """
        # 1. Adaptive reconstruction loss
        rec_loss = self._adaptive_cosine_loss(encoder_features, decoder_features)

        # 2. Category classification loss
        cls_loss = F.cross_entropy(category_logits, category_ids)

        # 3. Category contrastive loss (encourage category separation)
        contrast_loss = self._category_contrastive_loss(
            encoder_features, decoder_features, category_ids,
        )

        # Combined loss
        total_loss = (
            self.rec_weight * rec_loss
            + self.inp_weight * inp_loss
            + self.cls_weight * cls_loss
            + self.contrast_weight * contrast_loss
        )

        loss_dict = {
            "rec_loss": rec_loss,
            "inp_loss": inp_loss,
            "cls_loss": cls_loss,
            "contrast_loss": contrast_loss,
            "total_loss": total_loss,
        }

        return total_loss, loss_dict

    def _adaptive_cosine_loss(
        self,
        encoder_features: list[torch.Tensor],
        decoder_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Adaptive cosine similarity loss with gradient re-weighting.

        Computes cosine similarity loss and applies adaptive gradient weights
        based on per-point reconstruction difficulty. Hard-to-reconstruct points
        receive higher gradients, focusing training on challenging regions.

        Args:
            encoder_features: Encoder feature maps.
            decoder_features: Decoder feature maps.

        Returns:
            Scalar reconstruction loss.
        """
        cos_loss_fn = nn.CosineSimilarity(dim=1)
        loss = torch.tensor(0.0, device=encoder_features[0].device)

        for enc_feat, dec_feat in zip(encoder_features, decoder_features, strict=True):
            # Detach encoder features to prevent gradient flow
            enc_feat = enc_feat.detach()

            # Compute per-point cosine distance
            with torch.no_grad():
                point_dist = 1.0 - cos_loss_fn(enc_feat, dec_feat).unsqueeze(1)

            # Adaptive weights: amplify hard points
            mean_dist = point_dist.mean()
            adaptive_weights = (point_dist / (mean_dist + 1e-8)) ** self.adaptive_power

            # Base cosine loss
            cos_sim = cos_loss_fn(
                enc_feat.reshape(enc_feat.shape[0], enc_feat.shape[1], -1),
                dec_feat.reshape(dec_feat.shape[0], dec_feat.shape[1], -1),
            )
            loss += (1.0 - cos_sim).mean()

            # Register gradient hook for adaptive weighting
            partial_func = partial(self._modify_grad, adaptive_weights=adaptive_weights)
            dec_feat.register_hook(partial_func)

        return loss / len(encoder_features)

    @staticmethod
    def _modify_grad(
        x: torch.Tensor,
        adaptive_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Modify gradients based on adaptive weights.

        Args:
            x: Gradient tensor.
            adaptive_weights: Weight multipliers for each spatial position.

        Returns:
            Modified gradient tensor.
        """
        adaptive_weights = adaptive_weights.expand_as(x)
        return x.clone() * adaptive_weights

    def _category_contrastive_loss(
        self,
        encoder_features: list[torch.Tensor],
        decoder_features: list[torch.Tensor],
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Category contrastive loss for multi-category feature separation.

        Encourages features from different categories to be well-separated
        in the embedding space, improving multi-category discrimination.

        Args:
            encoder_features: Encoder feature maps.
            decoder_features: Decoder feature maps.
            category_ids: Category IDs for each sample.

        Returns:
            Scalar contrastive loss.
        """
        batch_size = category_ids.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=category_ids.device)

        # Use the last decoder feature map for contrastive learning
        dec_feat = decoder_features[-1]  # (B, C, H, W)
        # Global average pooling
        feat = dec_feat.mean(dim=(2, 3))  # (B, C)
        feat = F.normalize(feat, dim=-1)  # L2 normalize

        # Compute pairwise similarity matrix
        sim_matrix = feat @ feat.T  # (B, B)

        # Create positive/negative mask based on category
        category_ids = category_ids.view(-1, 1)
        pos_mask = (category_ids == category_ids.T).float()  # (B, B)
        # Remove self-similarity from positive mask
        pos_mask = pos_mask - torch.eye(batch_size, device=pos_mask.device)
        neg_mask = 1.0 - pos_mask - torch.eye(batch_size, device=pos_mask.device)

        # InfoNCE-style contrastive loss
        temperature = 0.07
        sim_matrix = sim_matrix / temperature

        # Positive logits
        pos_sum = (sim_matrix.exp() * pos_mask).sum(dim=1)
        # Total logits (excluding self)
        total_sum = (sim_matrix.exp() * (pos_mask + neg_mask)).sum(dim=1)

        # Avoid division by zero
        valid = pos_mask.sum(dim=1) > 0
        if valid.any():
            contrast_loss = -torch.log((pos_sum[valid] + 1e-8) / (total_sum[valid] + 1e-8)).mean()
        else:
            contrast_loss = torch.tensor(0.0, device=feat.device)

        return contrast_loss
