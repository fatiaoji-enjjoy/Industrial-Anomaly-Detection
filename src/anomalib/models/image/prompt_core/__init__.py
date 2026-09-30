"""PromptCore: Prompt-Guided Multi-Category Transformer for Anomaly Detection.

PromptCore is a Transformer-based unsupervised anomaly detection model designed for
single-model multi-category industrial inspection. It combines a frozen DINOv3
ViT encoder with category-specific prompt tokens and a prototype-guided decoder.

Key features:
    - Category-specific learnable prompt tokens for multi-category generalization
    - INP (Intrinsic Normal Prototype) extraction via cross-attention
    - Prompt-conditioned Transformer decoder with prototype-guided attention
    - Adaptive multi-scale anomaly map fusion
    - Multi-category contrastive loss for feature separation
"""

from .lightning_model import PromptCore
from .torch_model import PromptCoreModel

__all__ = [
    "PromptCore",
    "PromptCoreModel",
]
