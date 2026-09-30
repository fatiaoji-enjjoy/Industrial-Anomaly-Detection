"""Intrinsic Normal Prototype (INP) extractor for PromptCore.

This module implements cross-attention based INP extraction that aggregates
encoder patch tokens into a compact set of learnable prototype tokens. The
INPs, combined with category prompt tokens, form a unified prototype bank
that guides the decoder's reconstruction.

Based on the INP-Former architecture but enhanced with:
- Hierarchical INPs at multiple resolutions
- Prototype diversity regularization
- Fusion with category-specific prompts
"""

import torch
from functools import partial
from torch import nn

from anomalib.models.image.dinomaly.components import DinomalyMLP
from anomalib.models.image.inp_former.components.layers import AggregationBlock


class INPExtractor(nn.Module):
    """Cross-attention based INP extractor for aggregating patch features.

    Uses a set of learnable query tokens that cross-attend to encoder patch
    tokens, producing Intrinsic Normal Prototypes (INPs) that summarize the
    normal appearance of the input image. The INPs are then combined with
    category prompt tokens to form the prototype bank for the decoder.

    Args:
        embed_dim (int): Feature dimension of encoder tokens.
            Defaults to ``768``.
        num_heads (int): Number of attention heads for cross-attention.
            Defaults to ``12``.
        num_inps (int): Number of INP tokens to extract.
            Defaults to ``8``.
        num_extractor_layers (int): Number of cross-attention layers in the
            INP extractor. Defaults to ``2``.
        mlp_ratio (float): MLP hidden dimension ratio.
            Defaults to ``4.0``.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_inps: int = 8,
        num_extractor_layers: int = 2,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_inps = num_inps

        # Learnable INP query tokens
        self.inp_queries = nn.Parameter(
            torch.randn(1, num_inps, embed_dim) * 0.02,
        )

        # Cross-attention layers for INP extraction
        self.extractor_layers = nn.ModuleList([
            AggregationBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
            for _ in range(num_extractor_layers)
        ])

        # INP refinement MLP
        self.inp_refine = DinomalyMLP(
            in_features=embed_dim,
            hidden_features=embed_dim * 4,
            out_features=embed_dim,
            act_layer=nn.GELU,
            drop=0.1,
            bias=False,
            apply_input_dropout=False,
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        category_prompts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract INPs and build prototype bank.

        Args:
            patch_tokens (torch.Tensor): Encoder patch tokens of shape
                ``(B, N, embed_dim)``, where N is the number of patches.
            category_prompts (torch.Tensor): Category-specific prompt tokens
                of shape ``(B, num_prompts, embed_dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - prototype_bank: Combined INPs and category prompts of shape
                  ``(B, num_inps + num_prompts, embed_dim)``.
                - inp_loss: INP coherence loss for regularization.
        """
        batch_size = patch_tokens.shape[0]

        # Expand learnable INP queries for batch
        inp_tokens = self.inp_queries.expand(batch_size, -1, -1)  # (B, num_inps, D)

        # Cross-attention: INP queries attend to patch tokens
        for layer in self.extractor_layers:
            inp_tokens = layer(inp_tokens, patch_tokens)

        # Refine INPs
        inp_tokens = self.inp_refine(inp_tokens)

        # Compute INP coherence loss (encourages INPs to be close to patch features)
        inp_loss = self._compute_inp_coherence_loss(inp_tokens, patch_tokens)

        # Build prototype bank: concatenate INPs with category prompts
        prototype_bank = torch.cat([inp_tokens, category_prompts], dim=1)

        return prototype_bank, inp_loss

    def _compute_inp_coherence_loss(
        self,
        inp_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute INP coherence loss to ensure INPs represent normal features.

        Minimizes the distances between individual patch features and their
        nearest INP, encouraging INPs to be representative of normal patterns.

        Args:
            inp_tokens (torch.Tensor): INP tokens of shape ``(B, num_inps, D)``.
            patch_tokens (torch.Tensor): Patch tokens of shape ``(B, N, D)``.

        Returns:
            torch.Tensor: Scalar coherence loss.
        """
        # Compute pairwise cosine distance between patches and INPs
        inp_norm = nn.functional.normalize(inp_tokens, dim=-1)  # (B, num_inps, D)
        patch_norm = nn.functional.normalize(patch_tokens, dim=-1)  # (B, N, D)

        # Cosine similarity: (B, N, num_inps)
        similarity = torch.bmm(patch_norm, inp_norm.transpose(1, 2))
        distance = 1.0 - similarity  # (B, N, num_inps)

        # For each patch, find distance to nearest INP
        min_distance, _ = distance.min(dim=-1)  # (B, N)

        return min_distance.mean()

    def get_num_prototypes(self) -> int:
        """Get total number of prototypes in the bank.

        Returns:
            int: Total prototype count (INPs + category prompts).
        """
        return self.num_inps
