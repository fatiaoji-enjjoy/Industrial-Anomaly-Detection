"""Category-specific prompt token encoder for PromptCore.

This module implements learnable prompt tokens that encode category-specific
context for guiding the Transformer decoder's reconstruction. Each product
category has its own set of learnable prompt tokens, enabling single-model
multi-category generalization.

Key innovation: Instead of using a simple class token subtraction (as in
Dinomaly2's context recentering), we use a learnable prompt bank that is
selected and refined per category, providing more expressive and flexible
category conditioning.
"""

import torch
from torch import nn


class CategoryPromptEncoder(nn.Module):
    """Learnable category-specific prompt token encoder.

    Maintains a prompt bank of shape ``(num_categories, num_prompts, embed_dim)``
    where each category has its own set of learnable prompt tokens. The prompts
    are selected by category ID and fused with the encoder's class token to
    produce category-conditioned prompt features.

    During inference, the category ID can be specified or inferred from the
    class token's nearest neighbor in the prompt bank.

    Args:
        num_categories (int): Maximum number of product categories supported.
            Defaults to ``50``.
        num_prompts (int): Number of prompt tokens per category.
            Defaults to ``4``.
        embed_dim (int): Feature dimension of the prompt tokens.
            Defaults to ``768``.
        prompt_dropout (float): Dropout rate applied to prompts during training
            for regularization. Defaults to ``0.1``.
    """

    def __init__(
        self,
        num_categories: int = 50,
        num_prompts: int = 4,
        embed_dim: int = 768,
        prompt_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim

        # Learnable prompt bank: (num_categories, num_prompts, embed_dim)
        self.prompt_bank = nn.Parameter(
            torch.randn(num_categories, num_prompts, embed_dim) * 0.02,
        )

        # Lightweight MLP to fuse class token with selected prompts
        self.prompt_fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(prompt_dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # Category classifier for inference-time category prediction
        self.category_classifier = nn.Linear(embed_dim, num_categories)

    def forward(
        self,
        cls_token: torch.Tensor,
        category_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate category-conditioned prompts from class token.

        Args:
            cls_token (torch.Tensor): Class token from ViT encoder with shape
                ``(B, embed_dim)``.
            category_ids (torch.Tensor | None): Ground-truth category IDs with
                shape ``(B,)``. If ``None`` during inference, the category is
                predicted from the prompt bank.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - category_prompts: Category-conditioned prompts of shape
                  ``(B, num_prompts, embed_dim)``.
                - category_logits: Category prediction logits of shape
                  ``(B, num_categories)`` for loss computation.
        """
        batch_size = cls_token.shape[0]

        if category_ids is not None:
            # Training: select prompts by category ID
            category_prompts = self.prompt_bank[category_ids]  # (B, num_prompts, D)
        else:
            # Inference: predict category from prompt bank
            category_logits = self.category_classifier(cls_token)
            pred_category = category_logits.argmax(dim=-1)
            category_prompts = self.prompt_bank[pred_category]

        # Compute category logits for loss
        # Compare class token with mean prompt of each category
        mean_prompts = self.prompt_bank.mean(dim=1)  # (num_categories, D)
        category_logits = cls_token @ mean_prompts.T  # (B, num_categories)

        # Fuse class token with selected prompts
        # Expand cls_token for each prompt: (B, num_prompts, D)
        cls_expanded = cls_token.unsqueeze(1).expand(-1, self.num_prompts, -1)
        fused = torch.cat([cls_expanded, category_prompts], dim=-1)  # (B, num_prompts, 2D)
        fused_prompts = self.prompt_fusion(fused)  # (B, num_prompts, D)

        return fused_prompts, category_logits

    def get_prompt_by_category(self, category_id: int) -> torch.Tensor:
        """Retrieve prompt tokens for a specific category.

        Args:
            category_id (int): Category index.

        Returns:
            torch.Tensor: Prompt tokens of shape ``(num_prompts, embed_dim)``.
        """
        return self.prompt_bank[category_id]
