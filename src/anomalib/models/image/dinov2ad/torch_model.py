# Copyright (C) 2025 DINOv2AD Contributors
# SPDX-License-Identifier: Apache-2.0

"""DINOv2AD PyTorch model for anomaly detection.

DINOv2AD: DINOv2-based unsupervised industrial anomaly detection and localization.
Uses pretrained DINOv2 features, PCA foreground masking, kNN distance scoring,
coreset subsampling, and multiple aggregation strategies.

Example:
    >>> from anomalib.models.image.dinov2ad.torch_model import DINOv2ADModel
    >>> model = DINOv2ADModel(
    ...     model_name='facebook/dinov2-small',
    ...     k=3,
    ...     aggregation='max',
    ...     inference_size=224,
    ...     coreset_sampling_ratio=0.1,
    ... )
    >>> input_tensor = torch.randn(32, 3, 224, 224)
    >>> output = model(input_tensor)
"""

import logging
import os
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from PIL import Image

from anomalib.data import InferenceBatch
from anomalib.models.components import DynamicBufferMixin, KCenterGreedy

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed, using brute-force search instead")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2 normalize rows of a numpy array in-place (equivalent to faiss.normalize_L2)."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    x /= norm
    return x

try:
    from sklearn.decomposition import PCA
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False
    logger.warning("sklearn not installed, disabling adaptive masking")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Adaptive Foreground Mask
# ---------------------------------------------------------------------------

class AdaptiveForegroundMask:
    """PCA-based foreground/background separation for anomaly detection.

    Excludes background noise to focus on object features.

    Args:
        threshold: Center region foreground ratio threshold.
        min_foreground_ratio: Minimum foreground ratio to activate masking.
    """

    def __init__(self, threshold: float = 0.35, min_foreground_ratio: float = 0.1):
        self.threshold = threshold
        self.min_foreground_ratio = min_foreground_ratio

    def __call__(self, feature_map: np.ndarray, apply_morphology: bool = True) -> np.ndarray:
        """Generate adaptive foreground mask.

        Args:
            feature_map: Patch-level feature map (H, W, D).
            apply_morphology: Whether to apply morphological post-processing.

        Returns:
            Foreground mask (H, W), True for foreground.
        """
        if not PCA_AVAILABLE:
            return np.ones(feature_map.shape[:2], dtype=bool)

        H, W, D = feature_map.shape
        flattened = feature_map.reshape(-1, D)

        pca = PCA(n_components=1)
        projection = pca.fit_transform(flattened)
        normalized = (projection - projection.min()) / (projection.max() - projection.min() + 1e-8)
        normalized = normalized.reshape(H, W)

        mask = normalized > 0.5

        center_region = mask[H // 4:3 * H // 4, W // 4:3 * W // 4]
        foreground_ratio = center_region.mean()
        if foreground_ratio < self.threshold:
            mask = ~mask

        foreground_ratio_total = mask.mean()
        if foreground_ratio_total < self.min_foreground_ratio:
            mask = np.ones((H, W), dtype=bool)

        if apply_morphology and mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_uint8 = mask.astype(np.uint8) * 255
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=2)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            mask = (mask_uint8 > 0).astype(bool)

        return mask


# ---------------------------------------------------------------------------
# DINOv2 Feature Extractor
# ---------------------------------------------------------------------------

class DINOv2FeatureExtractor(nn.Module):
    """DINOv2 feature extractor using timm pretrained models.

    Supports:
        - facebook/dinov2-small (384-dim, fastest)
        - facebook/dinov2-base (768-dim, balanced)
        - facebook/dinov2-large (1024-dim, high precision)
        - facebook/dinov2-giant (1536-dim, highest precision)

    Args:
        model_name: HuggingFace model name.
        device: Computation device.
        pretrained_weights_path: Local pretrained weights path (offline).
        inference_size: Inference image size (must be divisible by 14).
    """

    def __init__(
        self,
        model_name: str = 'facebook/dinov2-small',
        device: str = 'cuda',
        pretrained_weights_path: str | None = None,
        inference_size: int = 224,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained_weights_path = pretrained_weights_path
        self.inference_size = inference_size
        self._use_timm = True

        # Adjust inference_size to be divisible by 14
        target_size = self.inference_size
        while target_size % 14 != 0:
            target_size += 1
        self.inference_size = target_size

        # 初始设备（仅 __init__ 内使用，用于首次 .to()）
        _init_device = torch.device(device if torch.cuda.is_available() else 'cpu')

        try:
            import timm
            model_name_map = {
                'facebook/dinov2-base': 'vit_base_patch14_dinov2',
                'facebook/dinov2-small': 'vit_small_patch14_dinov2',
                'facebook/dinov2-tiny': 'vit_tiny_patch14_dinov2',
                'facebook/dinov2-large': 'vit_large_patch14_dinov2',
                'facebook/dinov2-giant': 'vit_giant_patch14_dinov2',
            }
            timm_model_name = model_name_map.get(self.model_name, 'vit_small_patch14_dinov2')

            if self.pretrained_weights_path and os.path.exists(self.pretrained_weights_path):
                self.model = timm.create_model(
                    timm_model_name, pretrained=False, num_classes=0, img_size=target_size,
                ).to(_init_device)
                state_dict = torch.load(self.pretrained_weights_path, map_location=_init_device)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                self.model = timm.create_model(
                    timm_model_name, pretrained=True, num_classes=0, img_size=target_size,
                ).to(_init_device)

            self.model.eval()
            self.feature_dim = self.model.num_features
            self._preprocess = transforms.Compose([
                transforms.Resize((self.inference_size, self.inference_size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            logger.info("DINOv2 model loaded via timm: %s, feature_dim: %d", timm_model_name, self.feature_dim)

        except Exception as e:
            logger.error("Failed to load DINOv2 via timm: %s", e)
            raise RuntimeError(f"Failed to load DINOv2 model: {e}")

    @property
    def device(self) -> torch.device:
        """返回 timm 模型权重的实际所在设备，跟随 .to() 自动更新。"""
        return next(self.model.parameters()).device

    @torch.no_grad()
    def extract_features(self, image: Image.Image | torch.Tensor) -> dict:
        """Extract image features.

        Args:
            image: PIL.Image or torch.Tensor (C, H, W) or (B, C, H, W).

        Returns:
            Dict with feature_map (H,W,D) or (B,H,W,D), patch_size, grid_size, original_size, processing_time_ms.
        """
        start_time = time.time()

        if isinstance(image, Image.Image):
            self.original_size = image.size
            pixel_values = self._preprocess(image).unsqueeze(0).to(self.device)
            is_batch = False
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                self.original_size = (image.shape[-1], image.shape[-2])
                pixel_values = image.unsqueeze(0).to(self.device)
                is_batch = False
            elif image.dim() == 4:
                self.original_size = (image.shape[-1], image.shape[-2])
                pixel_values = image.to(self.device)
                is_batch = True
            else:
                raise ValueError(f"Unsupported tensor shape: {image.shape}")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        with torch.inference_mode():
            if self._use_timm:
                features = self.model.forward_features(pixel_values)
                if isinstance(features, tuple):
                    features = features[0]
                patch_size = self.model.patch_embed.patch_size[0]
            else:
                outputs = self.model(pixel_values=pixel_values)
                features = outputs.last_hidden_state
                patch_size = self.model.config.patch_size

        processing_time_ms = (time.time() - start_time) * 1000

        H_original, W_original = pixel_values.shape[-2:]
        H_patch = H_original // patch_size
        W_patch = W_original // patch_size

        patch_tokens = features[:, 1:, :].cpu().numpy()  # (B, N_patches, D)
        B = patch_tokens.shape[0]
        feature_map = patch_tokens.reshape(B, H_patch, W_patch, self.feature_dim)
        if not is_batch:
            feature_map = feature_map[0]  # (H, W, D)

        return {
            'feature_map': feature_map,
            'patch_size': patch_size,
            'grid_size': (H_patch, W_patch),
            'processing_time_ms': processing_time_ms,
            'original_size': self.original_size,
        }


# ---------------------------------------------------------------------------
# Memory Bank
# ---------------------------------------------------------------------------

class MemoryBank:
    """Feature memory bank with per-category FAISS GPU indices and category routing.

    Stores normal sample features organized by category. During inference,
    queries are routed to the matching category's FAISS index, eliminating
    cross-category feature interference and reducing search latency.

    Uses FAISS GpuIndexFlatL2 for GPU-accelerated L2 distance search,
    falls back to brute-force numpy search.

    Args:
        feature_dim: Feature dimension.
        use_gpu: Whether to use GPU acceleration.
    """

    def __init__(self, feature_dim: int, use_gpu: bool = True):
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self._features_by_cat: dict[str, list[np.ndarray]] = {}
        self._indices_by_cat: dict[str, "faiss.Index"] = {}
        self._all_features_cache: np.ndarray | None = None
        self._categories: list[str] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_features(self) -> int:
        """Total number of features across all categories."""
        if self._all_features_cache is not None:
            return self._all_features_cache.shape[0]
        total = 0
        for chunks in self._features_by_cat.values():
            total += sum(c.shape[0] for c in chunks)
        return total

    @property
    def categories(self) -> list[str]:
        """List of categories with stored features."""
        return list(self._features_by_cat.keys())

    # ------------------------------------------------------------------
    # Feature accumulation (no FAISS index building yet)
    # ------------------------------------------------------------------

    def add_features(
        self,
        features: np.ndarray,
        mask: np.ndarray | None = None,
        category: str | None = None,
    ) -> None:
        """Add features to the memory bank under a category.

        Features are accumulated without building FAISS indices.
        Call ``build_indices()`` after all features are added.

        Args:
            features: Patch-level features (H, W, D) or (N, D).
            mask: Foreground boolean mask (H, W), optional.
            category: Category name for routing. Defaults to ``'__global__'``.
        """
        cat = category or '__global__'

        if len(features.shape) == 3:
            H, W, D = features.shape
            features_flat = features.reshape(-1, D)
            if mask is not None and mask.any():
                mask_flat = mask.reshape(-1)
                features_flat = features_flat[mask_flat]
        else:
            features_flat = features

        if features_flat.size == 0:
            return

        features_float = features_flat.astype(np.float32)
        if cat not in self._features_by_cat:
            self._features_by_cat[cat] = []
        self._features_by_cat[cat].append(features_float)
        self._all_features_cache = None  # invalidate cache

    # ------------------------------------------------------------------
    # FAISS index building (call after all features are accumulated)
    # ------------------------------------------------------------------

    def build_indices(self) -> None:
        """Build per-category FAISS GPU indices from accumulated features.

        Must be called after all ``add_features()`` calls and after any
        coreset/filtering operations that modify the feature store.
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, indices not built.")
            return

        self._indices_by_cat.clear()
        self._all_features_cache = None
        all_chunks: list[np.ndarray] = []

        for cat, chunks in self._features_by_cat.items():
            cat_features = np.concatenate(chunks, axis=0).astype(np.float32)
            # L2 normalize features for cosine-distance search
            _l2_normalize(cat_features)

            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    idx = faiss.GpuIndexFlatL2(res, self.feature_dim)
                    idx.add(cat_features)
                    self._indices_by_cat[cat] = idx
                except Exception:
                    idx = faiss.IndexFlatL2(self.feature_dim)
                    idx.add(cat_features)
                    self._indices_by_cat[cat] = idx
            else:
                idx = faiss.IndexFlatL2(self.feature_dim)
                idx.add(cat_features)
                self._indices_by_cat[cat] = idx

            all_chunks.append(cat_features)

        if all_chunks:
            self._all_features_cache = np.concatenate(all_chunks, axis=0)
        logger.info(
            "FAISS indices built: %d categories, %d total features.",
            len(self._indices_by_cat), self.num_features,
        )

    def rebuild_from_features(self) -> None:
        """Rebuild FAISS indices from the current feature store.

        Use after coreset subsampling or feature filtering to refresh indices.
        """
        self.build_indices()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_features: np.ndarray,
        k: int = 5,
        category: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search k nearest neighbors, optionally routed to a category.

        Args:
            query_features: Query features (N, D).
            k: Number of nearest neighbors.
            category: Category name for routing. If provided, only searches
                within that category's FAISS index. If None, searches all.

        Returns:
            Tuple of (distances (N, k), indices (N, k)).
        """
        query_float = query_features.astype(np.float32)

        # --- Category-routed FAISS search (GPU) ---
        if category is not None and category in self._indices_by_cat:
            _l2_normalize(query_float)
            distances, indices = self._indices_by_cat[category].search(query_float, k)
            return distances.astype(np.float32), indices.astype(np.int64)

        # --- Category-routed brute-force fallback ---
        if category is not None and category in self._features_by_cat:
            cat_features = np.concatenate(self._features_by_cat[category], axis=0).astype(np.float32)
            _l2_normalize(cat_features)
            _l2_normalize(query_float)
            sim = query_float @ cat_features.T
            dist = 1.0 - sim
            idx = np.argsort(dist, axis=1)[:, :k]
            dist = np.take_along_axis(dist, idx, axis=1)
            return dist.astype(np.float32), idx.astype(np.int64)

        # --- Global search (FAISS or brute-force) ---
        if FAISS_AVAILABLE and self._indices_by_cat:
            # Search across all per-category indices and merge results
            _l2_normalize(query_float)
            all_dists = []
            all_ids = []
            offset = 0
            for cat_idx in self._indices_by_cat.values():
                d, ids = cat_idx.search(query_float, k)
                all_dists.append(d)
                all_ids.append(ids + offset)
                offset += cat_idx.ntotal
            all_dists = np.concatenate(all_dists, axis=1)
            all_ids = np.concatenate(all_ids, axis=1)
            # Re-rank top-k across all categories
            top_order = np.argsort(all_dists, axis=1)[:, :k]
            final_dists = np.take_along_axis(all_dists, top_order, axis=1)
            final_ids = np.take_along_axis(all_ids, top_order, axis=1)
            return final_dists.astype(np.float32), final_ids.astype(np.int64)

        # --- Pure brute-force global fallback ---
        all_features = self.get_all_features()
        if all_features.size == 0:
            raise RuntimeError("Memory bank is empty")

        _l2_normalize(query_float)
        _l2_normalize(all_features)
        sim = query_float @ all_features.T
        dist = 1.0 - sim
        idx = np.argsort(dist, axis=1)[:, :k]
        dist = np.take_along_axis(dist, idx, axis=1)
        return dist.astype(np.float32), idx.astype(np.int64)

    # ------------------------------------------------------------------
    # Feature access (for coreset / filtering)
    # ------------------------------------------------------------------

    def get_all_features(self) -> np.ndarray:
        """Get all stored features as a single numpy array (across all categories)."""
        if self._all_features_cache is not None:
            return self._all_features_cache
        all_chunks = []
        for chunks in self._features_by_cat.values():
            all_chunks.extend(chunks)
        if not all_chunks:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        result = np.concatenate(all_chunks, axis=0)
        self._all_features_cache = result
        return result

    def replace_all_features(self, features: np.ndarray) -> None:
        """Replace the entire feature store with new features.

        Args:
            features: New features (N, D). All assigned to ``'__global__'``.
        """
        self._features_by_cat = {'__global__': [features.astype(np.float32)]}
        self._all_features_cache = None
        self._indices_by_cat.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_path: str):
        """Save memory bank to file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        npy_path = save_path.replace('.index', '.npy')
        all_features = self.get_all_features()
        if all_features.size > 0:
            np.save(npy_path, all_features)

        if FAISS_AVAILABLE and self._indices_by_cat:
            for cat, idx in self._indices_by_cat.items():
                cat_path = save_path.replace('.index', f'_{cat}.index')
                try:
                    faiss.write_index(
                        faiss.index_gpu_to_cpu(idx) if self.use_gpu else idx,
                        cat_path,
                    )
                except Exception:
                    pass

    @classmethod
    def load(cls, load_path: str, feature_dim: int, use_gpu: bool = True) -> "MemoryBank":
        """Load memory bank from file."""
        bank = cls(feature_dim, use_gpu)
        npy_path = load_path.replace('.index', '.npy')
        if os.path.exists(npy_path):
            features = np.load(npy_path).astype(np.float32)
            bank._features_by_cat['__global__'] = [features]
            bank._all_features_cache = features
            bank.build_indices()

        return bank


# ---------------------------------------------------------------------------
# Anomaly Scorer
# ---------------------------------------------------------------------------

class AnomalyScorer:
    """Anomaly score computation based on kNN distances.

    Supports multiple aggregation strategies: max, mean, median.

    Args:
        k: Number of nearest neighbors.
        aggregation: Aggregation method ('max', 'mean', 'median').
        blur_kernel_size: Gaussian blur kernel size (must be odd). Default 5.
        blur_sigma: Gaussian blur sigma. Default 1.0.
    """

    def __init__(self, k: int = 5, aggregation: str = 'max',
                 blur_kernel_size: int = 5, blur_sigma: float = 1.0):
        self.k = k
        self.aggregation = aggregation
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma

    def compute_anomaly_map(
        self,
        query_feature_map: np.ndarray,
        memory_bank: MemoryBank,
        mask: np.ndarray | None = None,
        category: str | None = None,
    ) -> dict:
        """Compute anomaly score map.

        Args:
            query_feature_map: Query feature map (H, W, D).
            memory_bank: MemoryBank instance.
            mask: Foreground mask (H, W), optional.

        Returns:
            Dict with anomaly_map (H, W), image_score, patch_scores.
        """
        H, W, D = query_feature_map.shape

        if mask is not None:
            query_flat = query_feature_map[mask]
        else:
            query_flat = query_feature_map.reshape(-1, D)

        if query_flat.size == 0:
            return {
                'anomaly_map': np.zeros((H, W), dtype=np.float64),
                'image_score': 0.0,
                'patch_scores': np.array([]),
            }

        distances, _ = memory_bank.search(query_flat, k=self.k, category=category)

        if self.aggregation == 'max':
            patch_scores = distances.max(axis=1)
        elif self.aggregation == 'mean':
            patch_scores = distances.mean(axis=1)
        elif self.aggregation == 'median':
            patch_scores = np.median(distances, axis=1)
        else:
            patch_scores = distances.mean(axis=1)

        patch_scores = np.nan_to_num(patch_scores, nan=0.0, posinf=1e6, neginf=0.0)

        anomaly_map = np.zeros((H, W), dtype=np.float64)
        if mask is not None and mask.any():
            anomaly_map[mask] = patch_scores.astype(np.float64)
            # image_score: top-1% max pooling for better defect detection
            valid = anomaly_map[mask]
            k_top = max(1, int(len(valid) * 0.01))
            top_indices = np.argpartition(valid, -k_top)[-k_top:]
            image_score = float(valid[top_indices].mean())
        else:
            anomaly_map = patch_scores.reshape(H, W).astype(np.float64)
            # image_score: top-1% max pooling
            flat = anomaly_map.flatten()
            k_top = max(1, int(len(flat) * 0.01))
            top_indices = np.argpartition(flat, -k_top)[-k_top:]
            image_score = float(flat[top_indices].mean())

        # Gaussian smoothing for sharper anomaly boundaries (pixel F1)
        ksize = self.blur_kernel_size if self.blur_kernel_size % 2 == 1 else self.blur_kernel_size + 1
        anomaly_map_smoothed = cv2.GaussianBlur(anomaly_map.astype(np.float32), (ksize, ksize), self.blur_sigma)

        return {
            'anomaly_map': anomaly_map_smoothed,
            'image_score': image_score,
            'patch_scores': patch_scores,
        }


# ---------------------------------------------------------------------------
# DINOv2AD Model
# ---------------------------------------------------------------------------

class DINOv2ADModel(DynamicBufferMixin, nn.Module):
    """DINOv2AD PyTorch model for anomaly detection.

    The model uses DINOv2 pretrained features stored in a memory bank,
    with kNN distance scoring for anomaly detection and localization.

    Args:
        model_name: DINOv2 model name (e.g., 'facebook/dinov2-small').
        device: Computation device.
        pretrained_weights_path: Local pretrained weights path.
        k: Number of nearest neighbors for scoring.
        aggregation: Aggregation method ('max', 'mean', 'median').
        use_adaptive_mask: Whether to use PCA foreground masking.
        inference_size: Inference image size.
        coreset_sampling_ratio: Ratio for coreset subsampling.
        blur_kernel_size: Gaussian blur kernel size for anomaly map smoothing.
        blur_sigma: Gaussian blur sigma for anomaly map smoothing.
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
    ) -> None:
        super().__init__()

        self.feature_extractor = DINOv2FeatureExtractor(
            model_name=model_name,
            device=device,
            pretrained_weights_path=pretrained_weights_path,
            inference_size=inference_size,
        )
        self.foreground_mask = AdaptiveForegroundMask() if use_adaptive_mask else None
        self.anomaly_scorer = AnomalyScorer(k=k, aggregation=aggregation,
                                            blur_kernel_size=blur_kernel_size,
                                            blur_sigma=blur_sigma)
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.feature_dim = self.feature_extractor.feature_dim

        self.memory_bank = MemoryBank(self.feature_dim, use_gpu=(device == 'cuda' and torch.cuda.is_available()))
        self.embedding_store: list[np.ndarray] = []
        self.current_category: str | None = None  # set before engine.test() for category routing

        _init_device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.to(_init_device)

    @property
    def _device(self) -> torch.device:
        """返回特征提取器权重的实际所在设备，跟随 .to() 自动更新。"""
        return next(self.feature_extractor.parameters()).device

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Process input tensor through the model.

        During training, returns embeddings extracted from the input.
        During inference, returns anomaly maps and scores.

        Args:
            input_tensor: Input images (B, C, H, W).

        Returns:
            During training: feature embeddings.
            During inference: InferenceBatch with pred_score and anomaly_map.
        """
        output_size = input_tensor.shape[-2:]
        batch_size = input_tensor.shape[0]

        if self.training:
            # Training mode: batch-extract and store features for speed
            features = self.feature_extractor.extract_features(input_tensor)
            feature_map = features['feature_map']  # (B, H, W, D) or (H_patch, W_patch, D)
            if feature_map.ndim == 4:
                for b in range(feature_map.shape[0]):
                    mask = self.foreground_mask(feature_map[b]) if self.foreground_mask else None
                    self.memory_bank.add_features(feature_map[b], mask, category=self.current_category)
            else:
                mask = self.foreground_mask(feature_map) if self.foreground_mask else None
                self.memory_bank.add_features(feature_map, mask, category=self.current_category)
            return torch.tensor(feature_map.mean(), device=input_tensor.device, requires_grad=True)
        else:
            # Inference mode: compute anomaly maps
            pred_scores = []
            anomaly_maps = []

            for i in range(batch_size):
                img = input_tensor[i]
                features = self.feature_extractor.extract_features(img)
                feature_map = features['feature_map']
                original_size = features.get('original_size', output_size)

                mask = self.foreground_mask(feature_map) if self.foreground_mask else None
                result = self.anomaly_scorer.compute_anomaly_map(
                    feature_map, self.memory_bank, mask, category=self.current_category,
                )

                anomaly_map = result['anomaly_map']
                H_patch, W_patch = anomaly_map.shape
                H_original, W_original = original_size

                # Resize anomaly map to original image size
                if H_patch != H_original or W_patch != W_original:
                    anomaly_map_resized = cv2.resize(
                        anomaly_map.astype(np.float32),
                        (W_original, H_original),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    anomaly_map_resized = anomaly_map

                pred_scores.append(result['image_score'])
                anomaly_maps.append(anomaly_map_resized)

            # Convert to tensors
            pred_score_tensor = torch.tensor(pred_scores, device=input_tensor.device)
            # Stack anomaly maps and add channel dimension
            anomaly_map_tensor = torch.tensor(
                np.stack(anomaly_maps, axis=0), device=input_tensor.device, dtype=torch.float32,
            ).unsqueeze(1)

            return InferenceBatch(pred_score=pred_score_tensor, anomaly_map=anomaly_map_tensor)

    def subsample_embedding(self, sampling_ratio: float | None = None) -> None:
        """Apply coreset subsampling per-category to preserve category routing.

        Each category's features are sampled independently via KCenterGreedy,
        so per-category FAISS indices remain valid after subsampling.

        Args:
            sampling_ratio: Fraction of embeddings to keep. Uses model default if None.
        """
        if sampling_ratio is None:
            sampling_ratio = self.coreset_sampling_ratio

        if sampling_ratio >= 1.0:
            return

        categories = self.memory_bank.categories
        if not categories:
            logger.warning("Memory bank has no categories, skipping coreset subsampling.")
            return

        total_before = self.memory_bank.num_features
        if total_before == 0:
            logger.warning("Memory bank is empty, skipping coreset subsampling.")
            return

        device = self._device
        total_after = 0

        logger.info(
            "Applying coreset subsampling per-category (GPU): ratio=%.2f, %d categories",
            sampling_ratio, len(categories),
        )

        for cat in categories:
            chunks = self.memory_bank._features_by_cat[cat]
            cat_features = np.concatenate(chunks, axis=0).astype(np.float32)
            n_cat = cat_features.shape[0]
            n_keep = int(n_cat * sampling_ratio)

            if n_keep < 10:
                logger.info("  %s: %d features too few, keeping all", cat, n_cat)
                total_after += n_cat
                continue

            features_tensor = torch.tensor(cat_features, dtype=torch.float32, device=device)
            sampler = KCenterGreedy(embedding=features_tensor, sampling_ratio=sampling_ratio)
            coreset = sampler.sample_coreset()
            self.memory_bank._features_by_cat[cat] = [coreset.cpu().numpy().astype(np.float32)]
            total_after += coreset.shape[0]
            logger.info("  %s: %d -> %d features", cat, n_cat, coreset.shape[0])

        self.memory_bank._all_features_cache = None
        self.memory_bank.build_indices()

        logger.info(
            "Coreset subsampling complete: %d -> %d features retained across %d categories.",
            total_before, total_after, len(categories),
        )

    def filter_features_by_distance(
        self,
        intra_class_percentile: float = 90.0,
        top_k_neighbors: int = 5,
    ) -> None:
        """Filter redundant/conflicting features per-category (GPU-accelerated).

        Each category is filtered independently to preserve per-category FAISS
        indices and category routing. Removes features that are too close to
        each other (high redundancy) and outliers (too far from cluster centers).

        All distance computations run on GPU. For large banks (>5000 features),
        uses batched random sampling to stay within GPU memory limits.

        Args:
            intra_class_percentile: Keep features whose mean distance to neighbors
                is below this percentile (default 90%).
            top_k_neighbors: Number of neighbors for distance computation.
        """
        categories = self.memory_bank.categories
        if not categories:
            return

        device = self._device
        total_before = self.memory_bank.num_features
        total_after = 0

        logger.info(
            "Filtering features per-category (GPU): intra_class_percentile=%.1f, top_k=%d, %d categories",
            intra_class_percentile, top_k_neighbors, len(categories),
        )

        for cat in categories:
            chunks = self.memory_bank._features_by_cat[cat]
            cat_features = np.concatenate(chunks, axis=0).astype(np.float32)
            n_cat = cat_features.shape[0]

            if n_cat < 100:
                logger.info("  %s: %d features, skipping filter", cat, n_cat)
                total_after += n_cat
                continue

            # 迁移到 GPU 张量
            features_t = torch.tensor(cat_features, dtype=torch.float32, device=device)
            features_norm = F.normalize(features_t, p=2, dim=1)

            if n_cat <= 5000:
                # 小规模：全量 GPU 距离矩阵
                sim = features_norm @ features_norm.T  # (N, N)
                dist = 1.0 - sim
                dist.fill_diagonal_(float('inf'))
                sorted_dist, _ = torch.sort(dist, dim=1)
                sorted_dist = sorted_dist[:, :top_k_neighbors]  # (N, k)
            else:
                # 大规模：GPU 批量采样避免 OOM
                batch_size = min(5000, n_cat)
                sample_indices = torch.randperm(n_cat, device=device)[:batch_size]
                batch = features_norm[sample_indices]  # (B, D)
                sim = batch @ features_norm.T  # (B, N)
                dist = 1.0 - sim
                sorted_dist, _ = torch.sort(dist, dim=1)
                sorted_dist = sorted_dist[:, 1:top_k_neighbors + 1]  # (B, k), skip self

            # 到 k 近邻的平均距离
            mean_dist = sorted_dist.mean(dim=1)  # (N,) or (B,)

            # GPU 百分位数过滤
            p_low = torch.quantile(mean_dist, 0.05)
            p_high = torch.quantile(mean_dist, intra_class_percentile / 100.0)
            keep_mask = (mean_dist >= p_low) & (mean_dist <= p_high)

            if n_cat > 5000:
                # 大规模：仅对采样批次应用过滤，其余保留
                keep_mask_cpu = keep_mask.cpu().numpy()
                filtered = cat_features[sample_indices.cpu().numpy()][keep_mask_cpu]
                all_indices_np = np.arange(n_cat)
                unselected_idx = np.delete(all_indices_np, sample_indices.cpu().numpy())
                unselected = cat_features[unselected_idx] if len(unselected_idx) > 0 else np.empty((0, cat_features.shape[1]))
                filtered = np.concatenate([filtered, unselected], axis=0) if unselected.size > 0 else filtered
            else:
                keep_mask_cpu = keep_mask.cpu().numpy()
                filtered = cat_features[keep_mask_cpu]

            n_kept = filtered.shape[0]
            if n_kept >= 10:
                self.memory_bank._features_by_cat[cat] = [filtered.astype(np.float32)]
                total_after += n_kept
                logger.info("  %s: %d -> %d features", cat, n_cat, n_kept)
            else:
                logger.info("  %s: %d features, keeping all (filter would remove too many)", cat, n_cat)
                total_after += n_cat

        self.memory_bank._all_features_cache = None
        self.memory_bank.build_indices()
        logger.info(
            "Feature filtering complete: %d -> %d features retained across %d categories.",
            total_before, total_after, len(categories),
        )

    def predict(self, image: Image.Image, category: str | None = None) -> dict | None:
        """Run full anomaly detection on a single PIL image.

        Args:
            image: PIL Image in RGB mode.
            category: Category name (not used for single-bank models, kept for API compatibility).

        Returns:
            Dict with keys: anomaly_map, anomaly_map_resized, image_score,
            is_anomaly, predicted_category, e2e_time_ms, or None on failure.
        """
        e2e_start = time.time()
        try:
            # 自动同步模型到 GPU（防止 engine.test() 后回到 CPU）
            if torch.cuda.is_available():
                self.to('cuda')
                self.eval()

            features = self.feature_extractor.extract_features(image)
            feature_map = features['feature_map']
            original_size = features.get('original_size', (512, 512))

            if self.memory_bank.num_features == 0:
                logger.warning("Memory bank is empty, cannot predict.")
                return None

            mask = self.foreground_mask(feature_map) if self.foreground_mask else None
            result = self.anomaly_scorer.compute_anomaly_map(
                feature_map, self.memory_bank, mask, category=category,
            )

            anomaly_map = result['anomaly_map']
            H_patch, W_patch = anomaly_map.shape
            H_original, W_original = original_size

            if H_patch != H_original or W_patch != W_original:
                anomaly_map_resized = cv2.resize(
                    anomaly_map.astype(np.float32),
                    (W_original, H_original),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                anomaly_map_resized = anomaly_map

            result['anomaly_map_resized'] = anomaly_map_resized
            result['predicted_category'] = category or 'unknown'
            result['is_anomaly'] = result['image_score'] > 0.5
            result['e2e_time_ms'] = (time.time() - e2e_start) * 1000
            return result
        except Exception as e:
            logger.error("Predict failed: %s", e)
            return None

    def save_model(self, save_dir: str):
        """Save model weights and memory bank.

        Args:
            save_dir: Directory to save model files.
        """
        os.makedirs(save_dir, exist_ok=True)
        bank_path = os.path.join(save_dir, "memory_bank.index")
        self.memory_bank.save(bank_path)
        logger.info("Model saved to %s", save_dir)

    @classmethod
    def load_model(
        cls,
        load_dir: str,
        model_name: str = 'facebook/dinov2-small',
        device: str = 'cuda',
        pretrained_weights_path: str | None = None,
        **kwargs,
    ) -> "DINOv2ADModel":
        """Load model from saved directory.

        Args:
            load_dir: Directory containing saved model files.
            model_name: DINOv2 model name.
            device: Computation device.
            pretrained_weights_path: Local pretrained weights path.
            **kwargs: Additional model arguments.

        Returns:
            Loaded DINOv2ADModel instance.
        """
        model = cls(
            model_name=model_name,
            device=device,
            pretrained_weights_path=pretrained_weights_path,
            **kwargs,
        )

        bank_path = os.path.join(load_dir, "memory_bank.index")
        if os.path.exists(bank_path.replace('.index', '.npy')):
            model.memory_bank = MemoryBank.load(
                bank_path, model.feature_dim, use_gpu=(device == 'cuda' and torch.cuda.is_available()),
            )
            logger.info("Loaded memory bank from %s: %d features", load_dir, model.memory_bank.num_features)

        return model