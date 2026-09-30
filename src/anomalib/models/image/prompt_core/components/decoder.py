"""Prompt-Conditioned Transformer Decoder for PromptCore.

This module implements the decoder that reconstructs normal features using a
prototype bank (INPs + category prompts) as keys and values. The decoder uses
INP-Guided cross-attention from INP-Former, where patch features query the
prototype bank. This design ensures reconstructed features stay within the
normal feature manifold, suppressing anomalous patterns.

Key innovation: The decoder is conditioned on both INPs (image-specific normal
prototypes) and category prompts (category-specific context), providing a
richer conditioning signal than either approach alone.
"""

from functools import partial

import torch
from torch import nn

from anomalib.models.image.inp_former.components.layers import PrototypeBlock


class PromptConditionedDecoder(nn.Module):
    """Transformer decoder conditioned on prototype bank.

    Instead of self-attention, each decoder layer uses INP-Guided cross-attention
    where patch features query the prototype bank. The first residual connection
    is removed (following INP-Former) to prevent anomalous features from bypassing
    the prototype-guided attention.

    Args:
        embed_dim (int): Feature dimension. Defaults to ``768``.
        num_heads (int): Number of attention heads. Defaults to ``12``.
        num_layers (int): Number of decoder layers. Defaults to ``8``.
        mlp_ratio (float): MLP hidden dimension ratio. Defaults to ``4.0``.
        dropout (float): Dropout rate. Defaults to ``0.0``.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            PrototypeBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=dropout,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        patch_features: torch.Tensor,
        prototype_bank: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Forward pass through the prompt-conditioned decoder.

        Args:
            patch_features (torch.Tensor): Bottleneck-processed patch features
                of shape ``(B, N, embed_dim)``.
            prototype_bank (torch.Tensor): Combined INPs and category prompts
                of shape ``(B, num_prototypes, embed_dim)``.

        Returns:
            list[torch.Tensor]: List of decoder layer outputs, each of shape
                ``(B, N, embed_dim)``. The list is reversed so that later
                layers (deeper features) appear first.
        """
        decoder_outputs = []
        x = patch_features

        for layer in self.layers:
            x = layer(x, prototype_bank)
            decoder_outputs.append(x)

        # Reverse so that deeper features come first (matching encoder layer order)
        return decoder_outputs[::-1]
