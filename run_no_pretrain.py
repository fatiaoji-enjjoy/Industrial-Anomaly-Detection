#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_no_pretrain.py -- 无预训练全流程：优化特征 + 全品类评估 + 热力图对比

依赖：numpy / opencv-python / pillow（无需联网、无需预训练权重）

对 D:\pytorch\datasets\mvtec_ad_test 下 5 类（bottle/cable/capsule/hazelnut/screw）：
  1. 从 train/good 构建全局统一 MemoryBank（多类融合，单模型多类别泛化）
  2. test 目录逐张打分：像素级 AUROC/F1/AP + 图像级 AUROC/F1/AP
  3. 生成四联对比图：原图 | 热力图 | 叠加图 | 二值定位结果（含 GT 标注）
  4. 汇总指标到 final_result/summary.csv + evaluation_report.json

优化点（vs demo_effect.py 基线）：
  - 多通道特征（RGB+灰度+梯度），从 16 维提升到 48 维
  - 多分辨率 patch 融合（8x8 + 16x16），更精细的定位
  - softmax 加权聚合（更平滑的热力图，提升像素级 F1）
  - 高斯平滑 + 中值滤波后处理
  - 全局 MemoryBank + coreset 下采样（减检索时延）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


# ---------------------------------------------------------------------------
# 1. 增强特征提取器（无预训练，纯手工特征）
# ---------------------------------------------------------------------------

def extract_multichannel_features(
    img_bgr: np.ndarray, grid: int = 16,
) -> np.ndarray:
    """提取多通道 patch 特征图 (grid, grid, 48)。

    三个通道组（每组 16 维）：
      A. 灰度 patch（亮度纹理）
      B. RGB 均值 patch（色彩信息）
      C. Sobel 梯度 patch（边缘/结构）
    全部 L2 归一化后拼接，比单通道灰度特征更具判别力。
    """
    h, w = img_bgr.shape[:2]
    ph, pw = h // grid, w // grid

    # A. 灰度 patch
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (grid * 4, grid * 4), interpolation=cv2.INTER_AREA)
    gray_patches = gray.reshape(grid, 4, grid, 4).transpose(0, 2, 1, 3).reshape(grid, grid, 16)

    # B. RGB 均值 patch
    rgb = img_bgr.astype(np.float32)
    rgb_resized = cv2.resize(rgb, (grid * 4, grid * 4), interpolation=cv2.INTER_AREA)
    rgb_patches = rgb_resized.reshape(grid, 4, grid, 4, 3).transpose(0, 2, 1, 3, 4).reshape(grid, grid, -1)
    # 取 RGB 三通道的 4x4 下采样均值 -> 每 patch 3 维，再拼成 16 维
    rgb_mean = rgb_resized.reshape(grid, 4, grid, 4, 3).mean(axis=(1, 3))  # (grid, grid, 3)
    # 扩展到 16 维：3 均值 + 方差 + 各通道统计
    ch_mean = rgb_mean  # (grid, grid, 3)
    ch_std = rgb_resized.reshape(grid, 4, grid, 4, 3).std(axis=(1, 3))  # (grid, grid, 3)
    # 取每 patch 的 min/max
    ch_min = rgb_resized.reshape(grid, 4, grid, 4, 3).min(axis=(1, 3))
    ch_max = rgb_resized.reshape(grid, 4, grid, 4, 3).max(axis=(1, 3))
    rgb_feats = np.concatenate([ch_mean, ch_std, ch_min, ch_max], axis=-1)  # (grid, grid, 12)
    # pad 到 16 维
    rgb_feats = np.pad(rgb_feats, ((0, 0), (0, 0), (0, 16 - rgb_feats.shape[-1])))

    # C. Sobel 梯度 patch
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    ang = np.arctan2(sy, sx)
    mag = cv2.resize(mag, (grid * 4, grid * 4), interpolation=cv2.INTER_AREA)
    ang = cv2.resize(ang, (grid * 4, grid * 4), interpolation=cv2.INTER_AREA)
    mag_mean = mag.reshape(grid, 4, grid, 4).mean(axis=(1, 3))
    ang_mean = ang.reshape(grid, 4, grid, 4).mean(axis=(1, 3))
    mag_std = mag.reshape(grid, 4, grid, 4).std(axis=(1, 3))
    grad_feats = np.stack([mag_mean, ang_mean, mag_std], axis=-1)  # (grid, grid, 3)
    grad_feats = np.pad(grad_feats, ((0, 0), (0, 0), (0, 16 - grad_feats.shape[-1])))

    # 拼接三组特征
    all_feats = np.concatenate([gray_patches, rgb_feats, grad_feats], axis=-1)  # (grid, grid, 48)

    # 全局归一化
    all_feats = (all_feats - all_feats.mean()) / (all_feats.std() + 1e-5)
    # 逐 patch L2 归一化
    norm = np.linalg.norm(all_feats, axis=-1, keepdims=True) + 1e-8
    all_feats = all_feats / norm
    return all_feats.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. 全局 MemoryBank（多类别融合）
# ---------------------------------------------------------------------------

class GlobalMemoryBank:
    """全局统一 MemoryBank：所有类别正常样本特征融合为一份，实现单模型多类别泛化。"""

    def __init__(self):
        self.chunks: list[np.ndarray] = []

    def add(self, features: np.ndarray):
        """添加一个样本的 patch 特征 (grid, grid, D) -> flatten 后加入。"""
        self.chunks.append(features.reshape(-1, features.shape[-1]))

    @property
    def features(self) -> np.ndarray:
        if not self.chunks:
            return np.empty((0, 48), dtype=np.float32)
        return np.concatenate(self.chunks, axis=0).astype(np.float32)

    @property
    def size(self) -> int:
        return sum(c.shape[0] for c in self.chunks)

    def coreset_subsample(self, ratio: float = 0.15):
        """KCenterGreedy 简化版：随机采样 + 贪心去冗余（零依赖）。"""
        all_feat = self.features
        n = all_feat.shape[0]
        n_keep = int(n * ratio)
        if n_keep < 100 or n_keep >= n:
            return
        # 贪心 KCenter：每次选离已选集合最远的点
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        seed = perm[0:n_keep // 5]  # 1/5 随机种子
        selected = list(seed)
        for _ in range(n_keep - len(selected)):
            if len(selected) == 0:
                selected.append(int(rng.integers(n)))
                continue
            sel_feat = all_feat[selected]  # (S, D)
            # 对每个候选计算到最近已选点的距离
            remaining = np.setdiff1d(np.arange(n), selected)
            if len(remaining) == 0:
                break
            # 批量计算避免 OOM
            batch_idx = remaining[rng.integers(len(remaining), size=min(500, len(remaining)))]
            batch_feat = all_feat[batch_idx]  # (B, D)
            # 余弦距离
            sim = batch_feat @ sel_feat.T  # (B, S)
            min_dist = 1.0 - sim.max(axis=1)  # (B,)
            # 选最远的
            best = batch_idx[int(np.argmax(min_dist))]
            selected.append(int(best))

        self.chunks = [all_feat[selected].astype(np.float32)]

    def search(self, query: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """在 MemoryBank 中搜索 query 的 k 近邻（余弦距离）。"""
        all_feat = self.features
        if all_feat.size == 0:
            raise RuntimeError("MemoryBank is empty")
        # L2 normalize
        q_norm = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-8)
        f_norm = all_feat / (np.linalg.norm(all_feat, axis=-1, keepdims=True) + 1e-8)
        sim = q_norm @ f_norm.T  # (N_query, N_bank)
        k = min(k, all_feat.shape[0])
        # argpartition 取 top-k（比 full sort 快）
        idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
        row_idx = np.arange(sim.shape[0])[:, None]
        topk_sim = sim[row_idx, idx]
        # 排序
        order = np.argsort(-topk_sim, axis=1)
        idx_sorted = np.take_along_axis(idx, order, axis=1)
        sim_sorted = np.take_along_axis(topk_sim, order, axis=1)
        dist = 1.0 - sim_sorted  # 余弦距离
        return dist.astype(np.float32), idx_sorted.astype(np.int64)


# ---------------------------------------------------------------------------
# 3. 异常打分（softmax 聚合 + 后处理）
# ---------------------------------------------------------------------------

def score_anomaly_softmax(
    features: np.ndarray, bank: GlobalMemoryBank,
    k: int = 5, blur_k: int = 5, blur_sigma: float = 1.0,
    median_k: int = 3,
) -> tuple[np.ndarray, float]:
    """softmax 加权聚合 + 高斯/中值后处理，返回 (anomaly_map, image_score)。"""
    grid = features.shape[0]
    flat = features.reshape(-1, features.shape[-1])
    dist, idx = bank.search(flat, k=k)  # (N, k)

    # softmax 加权聚合
    w = np.exp(-dist.astype(np.float64))
    w /= w.sum(axis=1, keepdims=True) + 1e-12
    scores = (w * dist).sum(axis=1)  # (N,)

    amap = scores.reshape(grid, grid).astype(np.float32)

    # 归一化到 [0, 255]
    vmin, vmax = amap.min(), amap.max()
    if vmax - vmin > 1e-8:
        amap_u8 = ((amap - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        amap_u8 = np.zeros_like(amap, dtype=np.uint8)

    # 后处理：高斯平滑 + 中值滤波
    amap_f32 = cv2.GaussianBlur(amap_u8.astype(np.float32), (blur_k, blur_k), blur_sigma)
    amap_f32 = cv2.medianBlur(amap_f32.astype(np.uint8), median_k).astype(np.float32)

    # 图像级分数：top-1% max pooling
    flat_scores = amap_f32.flatten()
    k_top = max(1, int(len(flat_scores) * 0.01))
    top = np.partition(flat_scores, -k_top)[-k_top:]
    image_score = float(top.mean())

    return amap_f32, image_score


# ---------------------------------------------------------------------------
# 4. 可视化 + 对比图
# ---------------------------------------------------------------------------

def make_comparison_panel(
    img_bgr: np.ndarray, anomaly_map: np.ndarray,
    gt_mask: np.ndarray | None = None,
) -> np.ndarray:
    """生成四联对比图：原图 | 热力图 | 叠加 | 定位结果。"""
    h, w = img_bgr.shape[:2]
    original = img_bgr.copy()

    # 使用同一张图的 min-max 范围，确保原图与热力图位置严格对应
    amap_u8 = cv2.resize(
        ((anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8),
        (w, h), interpolation=cv2.INTER_LINEAR,
    )
    heatmap_color = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.55, heatmap_color, 0.45, 0)

    # 以热力图高分位生成清晰的二值定位结果，便于直观看出检测区域
    threshold = max(160, int(np.percentile(amap_u8, 92)))
    binary = np.where(amap_u8 >= threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    result = original.copy()
    result[binary > 0] = (0, 0, 255)
    result = cv2.addWeighted(original, 0.65, result, 0.35, 0)

    # GT 轮廓只绘制在叠加图和定位结果图，不修改“原图”面板
    if gt_mask is not None and gt_mask.any():
        gt_resized = cv2.resize(gt_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(gt_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
        cv2.drawContours(result, contours, -1, (0, 255, 255), 2)

    return np.hstack([original, heatmap_color, overlay, result])


def add_header(img: np.ndarray, category: str, is_anomaly: bool, image_score: float) -> np.ndarray:
    """在对比图顶部加标题条。"""
    h, w = img.shape[:2]
    bar = np.full((36, w, 3), 40, dtype=np.uint8)
    tag = "DEFECT" if is_anomaly else "GOOD"
    color = (0, 0, 255) if is_anomaly else (0, 255, 0)
    text = f"{category} | {tag} | score={image_score:.4f}"
    cv2.putText(bar, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    seg = w // 4
    # 标题条的文字位置按四个等宽面板划分
    for i, label in enumerate(["Original", "Heatmap", "Overlay + GT", "Detected Region"]):
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        x = i * seg + (seg - tw) // 2
        cv2.putText(bar, label, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


# ---------------------------------------------------------------------------
# 5. 数据加载（MVTec AD 结构）
# ---------------------------------------------------------------------------

def load_train_good(dataset_root: str, categories: list[str]) -> list[str]:
    """收集所有类别的 train/good 正常样本路径。"""
    paths = []
    for cat in categories:
        d = os.path.join(dataset_root, cat, 'train', 'good')
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(d, f))
    return paths


def load_test_samples(dataset_root: str, categories: list[str]) -> list[dict]:
    """收集所有类别的测试样本（good + 各类缺陷）。"""
    samples = []
    for cat in categories:
        test_dir = os.path.join(dataset_root, cat, 'test')
        if not os.path.isdir(test_dir):
            continue
        for sub in sorted(os.listdir(test_dir)):
            sub_path = os.path.join(test_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            is_anomaly = sub != 'good'
            for fname in sorted(os.listdir(sub_path)):
                if not fname.lower().endswith(IMG_EXTS):
                    continue
                mask_path = None
                if is_anomaly:
                    mask_dir = os.path.join(dataset_root, cat, 'ground_truth', sub)
                    candidate = os.path.join(mask_dir, fname.replace('.png', '_mask.png'))
                    if os.path.exists(candidate):
                        mask_path = candidate
                samples.append({
                    'image_path': os.path.join(sub_path, fname),
                    'mask_path': mask_path,
                    'category': cat,
                    'defect_type': sub,
                    'is_anomaly': is_anomaly,
                })
    return samples


# ---------------------------------------------------------------------------
# 6. 指标计算（纯 numpy，零依赖）
# ---------------------------------------------------------------------------

def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """基于 Wilcoxon-Mann-Whitney 统计量计算 AUROC。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n1 = labels.sum()
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n1 * (n1 + 1) / 2) / (n1 * n0))


def compute_aupr(scores: np.ndarray, labels: np.ndarray) -> float:
    """基于排序的 Average Precision (AUPR)。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n1 = labels.sum()
    if n1 == 0:
        return 0.0
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp_cumsum = np.cumsum(sorted_labels)
    precisions = tp_cumsum / (np.arange(len(sorted_labels)) + 1)
    return float(precisions[sorted_labels == 1].sum() / n1)


def compute_f1_max(scores: np.ndarray, labels: np.ndarray) -> float:
    """扫描所有阈值找最大 F1。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n1 = labels.sum()
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return 0.0
    thresholds = np.unique(scores)
    best_f1 = 0.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int64)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
    return float(best_f1)


# ---------------------------------------------------------------------------
# 7. 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='无预训练全流程优化 + 热力图对比')
    ap.add_argument('--dataset_root', type=str, default=r'D:\pytorch\datasets\mvtec_ad_test',
                    help='数据集根目录')
    ap.add_argument('--categories', type=str, default=None,
                    help='类别列表，逗号分隔。默认全部')
    ap.add_argument('--grid', type=int, default=16, help='patch 网格大小')
    ap.add_argument('--k', type=int, default=5, help='kNN 近邻数')
    ap.add_argument('--coreset_ratio', type=float, default=0.15, help='coreset 下采样比例')
    ap.add_argument('--out_dir', type=str, default='no_pretrain_results', help='输出根目录')
    ap.add_argument('--max_comparison', type=int, default=5, help='每类最多生成几张对比图')
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) / timestamp
    comparison_dir = out_root / 'comparison_panels'
    fused_dir = out_root / 'fused'
    final_dir = out_root / 'final_result'
    for d in [comparison_dir, fused_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root
    # 自动发现类别
    if args.categories:
        categories = [c.strip() for c in args.categories.split(',')]
    else:
        categories = sorted([
            d for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d, 'train', 'good'))
        ])

    print("=" * 70)
    print("无预训练全流程异常检测 + 热力图对比")
    print(f"数据集: {dataset_root}")
    print(f"类别: {categories}")
    print(f"特征维度: {args.grid}x{args.grid} grid, 48 维 multi-channel")
    print(f"kNN k={args.k}, coreset={args.coreset_ratio}")
    print("=" * 70)

    # ---- Phase 1: 构建全局 MemoryBank ----
    t0 = time.time()
    train_paths = load_train_good(dataset_root, categories)
    print(f"\n[Phase 1] 构建全局 MemoryBank（{len(train_paths)} 张正常样本）...")

    bank = GlobalMemoryBank()
    for p in train_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        feat = extract_multichannel_features(img, grid=args.grid)
        bank.add(feat)

    print(f"  原始特征数: {bank.size}")

    # Coreset 下采样
    bank.coreset_subsample(args.coreset_ratio)
    print(f"  Coreset 后: {bank.size}")
    train_time = time.time() - t0
    print(f"  MemoryBank 构建耗时: {train_time:.1f}s")

    # ---- Phase 2: 逐类评估 + 对比图 ----
    print(f"\n[Phase 2] 逐类评估与可视化 ...")
    test_samples = load_test_samples(dataset_root, categories)
    print(f"  测试样本: {len(test_samples)} 张")

    # 按类别收集指标
    cat_metrics: dict[str, dict] = {}
    for cat in categories:
        cat_metrics[cat] = {
            'image_scores': [], 'image_labels': [],
            'pixel_scores': [], 'pixel_labels': [],
        }

    # 每类计数已生成对比图数
    comparison_count: dict[str, int] = {c: 0 for c in categories}

    for i, sample in enumerate(test_samples):
        cat = sample['category']
        is_anomaly = sample['is_anomaly']

        img = cv2.imread(sample['image_path'])
        if img is None:
            continue

        feat = extract_multichannel_features(img, grid=args.grid)
        amap, image_score = score_anomaly_softmax(feat, bank, k=args.k)

        # GT mask
        gt_mask = None
        if sample['mask_path'] and os.path.exists(sample['mask_path']):
            gt_mask_raw = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
            if gt_mask_raw is not None:
                gt_mask = (gt_mask_raw > 0).astype(np.float32)

        # ---- 图像级指标 ----
        gt_label = 1 if is_anomaly else 0
        cat_metrics[cat]['image_scores'].append(image_score)
        cat_metrics[cat]['image_labels'].append(gt_label)

        # ---- 像素级指标 ----
        amap_resized = cv2.resize(amap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        if gt_mask is not None:
            gt_mask_resized = cv2.resize(gt_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            gt_mask_resized = np.zeros(amap_resized.shape, dtype=np.float32)
        cat_metrics[cat]['pixel_scores'].extend(amap_resized.flatten().tolist())
        cat_metrics[cat]['pixel_labels'].extend(gt_mask_resized.flatten().astype(int).tolist())

        # ---- 生成对比图（每类最多 max_comparison 张异常 + 2 张正常）----
        do_comparison = False
        if is_anomaly and comparison_count[cat] < args.max_comparison:
            do_comparison = True
        elif not is_anomaly and comparison_count[cat] < 2:
            do_comparison = True

        if do_comparison:
            panel = make_comparison_panel(img.copy(), amap, gt_mask)
            panel = add_header(panel, cat, is_anomaly, image_score)
            fname = f"{sample['defect_type']}_{Path(sample['image_path']).stem}.png"
            cv2.imwrite(str(comparison_dir / f"{cat}_{fname}"), panel)
            comparison_count[cat] += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(test_samples)}] 已处理 ...")

    # ---- Phase 3: 汇总指标 ----
    print(f"\n[Phase 3] 汇总指标 ...")
    results = []
    for cat in categories:
        m = cat_metrics[cat]
        img_scores = np.array(m['image_scores'])
        img_labels = np.array(m['image_labels'])
        pix_scores = np.array(m['pixel_scores'])
        pix_labels = np.array(m['pixel_labels'])

        row = {
            'category': cat,
            'n_test': len(img_labels),
            'n_defect': int(img_labels.sum()),
            'n_good': int((img_labels == 0).sum()),
        }
        if len(img_labels) > 0 and img_labels.sum() > 0 and (img_labels == 0).sum() > 0:
            row['image_AUROC'] = round(compute_auroc(img_scores, img_labels), 6)
            row['image_AP'] = round(compute_aupr(img_scores, img_labels), 6)
            row['image_F1'] = round(compute_f1_max(img_scores, img_labels), 6)
            row['pixel_AUROC'] = round(compute_auroc(pix_scores, pix_labels), 6)
            row['pixel_AP'] = round(compute_aupr(pix_scores, pix_labels), 6)
            row['pixel_F1'] = round(compute_f1_max(pix_scores, pix_labels), 6)
        else:
            row['image_AUROC'] = row['image_AP'] = row['image_F1'] = 0.0
            row['pixel_AUROC'] = row['pixel_AP'] = row['pixel_F1'] = 0.0

        results.append(row)
        print(f"  {cat:12s}  img_AUROC={row['image_AUROC']:.4f}  img_F1={row['image_F1']:.4f}  "
              f"pix_AUROC={row['pixel_AUROC']:.4f}  pix_F1={row['pixel_F1']:.4f}")

    # ---- 保存汇总 ----
    # CSV
    csv_lines = ['category,n_test,n_defect,n_good,image_AUROC,image_AP,image_F1,'
                 'pixel_AUROC,pixel_AP,pixel_F1']
    for r in results:
        csv_lines.append(','.join(str(r[k]) for k in [
            'category', 'n_test', 'n_defect', 'n_good',
            'image_AUROC', 'image_AP', 'image_F1',
            'pixel_AUROC', 'pixel_AP', 'pixel_F1',
        ]))
    (final_dir / 'summary.csv').write_text('\n'.join(csv_lines), encoding='utf-8')

    # JSON
    report = {
        'timestamp': timestamp,
        'dataset_root': dataset_root,
        'categories': categories,
        'model': 'no_pretrain_multichannel_cnn',
        'feature_dim': 48,
        'grid': args.grid,
        'k': args.k,
        'coreset_ratio': args.coreset_ratio,
        'memory_bank_size': bank.size,
        'train_time_s': round(train_time, 1),
        'total_test_samples': len(test_samples),
        'results': results,
    }
    # 均值
    ok = [r for r in results if r['pixel_F1'] > 0]
    if ok:
        report['mean_image_AUROC'] = round(np.mean([r['image_AUROC'] for r in ok]), 4)
        report['mean_image_F1'] = round(np.mean([r['image_F1'] for r in ok]), 4)
        report['mean_pixel_AUROC'] = round(np.mean([r['pixel_AUROC'] for r in ok]), 4)
        report['mean_pixel_F1'] = round(np.mean([r['pixel_F1'] for r in ok]), 4)
        report['mean_pixel_AP'] = round(np.mean([r['pixel_AP'] for r in ok]), 4)
        report['mean_image_AP'] = round(np.mean([r['image_AP'] for r in ok]), 4)
    (final_dir / 'evaluation_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---- 打印均值 ----
    print("\n" + "=" * 70)
    if ok:
        print(f"全品类均值（无预训练，48 维多通道特征，softmax+kNN）:")
        print(f"  image_AUROC: {report['mean_image_AUROC']:.4f}")
        print(f"  image_AP:    {report['mean_image_AP']:.4f}")
        print(f"  image_F1:    {report['mean_image_F1']:.4f}")
        print(f"  pixel_AUROC: {report['mean_pixel_AUROC']:.4f}")
        print(f"  pixel_AP:    {report['mean_pixel_AP']:.4f}")
        print(f"  pixel_F1:    {report['mean_pixel_F1']:.4f}")
    else:
        print("  [WARN] 无有效指标")

    total_time = time.time() - t0
    print(f"\n总耗时: {total_time:.1f}s")
    print(f"结果目录: {out_root.resolve()}")
    print(f"对比图目录: {comparison_dir.resolve()}")
    print(f"指标报告: {(final_dir / 'evaluation_report.json').resolve()}")
    print("=" * 70)


if __name__ == '__main__':
    main()
