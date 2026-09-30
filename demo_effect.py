#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_effect.py —— 无预训练 / 无 GPU 的简单异常检测「效果图」演示

目标：在不加载任何预训练模型（DINOv2/ImageNet 等）、不依赖 GPU/FAISS 的前提下，
用「patch 特征 + 近邻距离」的最小实现，演示工业异常检测的完整可视化闭环：

    正常样本 -> 构建内存库 -> 测试图逐 patch 近邻打分 -> 异常热力图 + 原图叠加

依赖（仅 3 个）：numpy / opencv-python / pillow —— 均无需联网、无需预训练权重。

用法：
  1) 零配置，直接生成合成示例（推荐先跑这个看效果）：
       python demo_effect.py
  2) 用自己的正常样本目录 + 测试图：
       python demo_effect.py --normal_dir "D:/MVTec AD/bottle/train/good" \
                             --test_dir  "D:/MVTec AD/bottle/test/broken_large"
  3) 单张测试图：
       python demo_effect.py --normal_dir path/to/good --test_image path/to/a.png

输出（demo_output/，与 DINOv2AD_pipeline 的 fused 目录同款结构）：
    color/       热力图（JET 伪彩）
    combined/    原图 + 热力图叠加（0.5/0.5）
    crack/       缺陷样本的灰度异常图
    good/        正常样本的灰度异常图
    demo_report.txt   每张图的图像级异常分数
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------------------------
# 1. 最简单的「特征提取器」：把灰度图分成小块，每块下采样成 4x4=16 维特征
#    完全替代 DINOv2 骨干，零预训练。真实方案里这里换成 DINOv2 patch token 即可。
# ---------------------------------------------------------------------------
GRID = 12          # 图像被切成 GRID x GRID 个 patch
PATCH_FEAT = 16    # 每个 patch 的 4x4 灰度下采样特征


def extract_patch_features(img_bgr: np.ndarray) -> np.ndarray:
    """提取 (GRID, GRID, PATCH_FEAT) 的 patch 特征图。

    Args:
        img_bgr: BGR 图像 (H, W, 3)。

    Returns:
        feature_map: (GRID, GRID, PATCH_FEAT) float32，逐 patch L2 归一化。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 归一化到固定尺寸，保证任意输入大小都可处理
    side = GRID * 4
    gray = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)
    # (GRID, 4, GRID, 4) -> (GRID, GRID, 16)
    patches = gray.reshape(GRID, 4, GRID, 4).transpose(0, 2, 1, 3).reshape(GRID, GRID, 16)
    patches = patches.astype(np.float32)
    # 全局归一化（减整图均值 / 除整图标准差）：保留 patch 之间的相对明暗对比，
    # 让「暗色划痕 patch」明显区别于「亮色正常 patch」，且对噪声不敏感。
    patches = (patches - patches.mean()) / (patches.std() + 1e-5)
    # 逐 patch L2 归一化，用余弦距离度量
    norm = np.linalg.norm(patches, axis=-1, keepdims=True) + 1e-8
    patches = patches / norm
    return patches


# ---------------------------------------------------------------------------
# 2. 内存库：所有正常 patch 特征堆成一个矩阵
# ---------------------------------------------------------------------------
def build_memory_bank(normal_paths: list[str]) -> np.ndarray:
    """把正常样本的 patch 特征堆叠为内存库 (N, PATCH_FEAT)。"""
    chunks = []
    for p in normal_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        feats = extract_patch_features(img)          # (GRID, GRID, 16)
        chunks.append(feats.reshape(-1, PATCH_FEAT))  # (GRID*GRID, 16)
    if not chunks:
        raise RuntimeError("未读取到任何正常样本图像")
    return np.concatenate(chunks, axis=0).astype(np.float32)


def score_anomaly_map(features: np.ndarray, bank: np.ndarray, k: int = 3) -> np.ndarray:
    """逐 patch 计算到内存库 k 近邻的平均距离 -> (GRID, GRID) 异常图。

    Args:
        features: (GRID, GRID, PATCH_FEAT) 测试图特征。
        bank: (N, PATCH_FEAT) 正常特征内存库。
        k: 近邻数量。

    Returns:
        anomaly_map: (GRID, GRID) float32 距离分数（越大越异常）。
    """
    h, w, d = features.shape
    q = features.reshape(-1, d)
    # 余弦距离 = 1 - q @ bank.T（特征已 L2 归一化）
    sim = q @ bank.T
    dist = 1.0 - sim
    # 取最近的 k 个平均
    k = min(k, bank.shape[0])
    dist.sort(axis=1)
    score = dist[:, :k].mean(axis=1)
    return score.reshape(h, w).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. 可视化（与 DINOv2AD_pipeline 的 save_anomaly_visualization 一致）
# ---------------------------------------------------------------------------
def render_visualization(img_bgr: np.ndarray, anomaly_map: np.ndarray,
                         out_dir: Path, name: str, is_anomaly: bool):
    """保存热力图 color、叠加 combined、灰度 crack/good 三套图。"""
    amap = anomaly_map
    vmin, vmax = amap.min(), amap.max()
    if vmax - vmin > 1e-8:
        amap_norm = ((amap - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        amap_norm = np.zeros_like(amap, dtype=np.uint8)

    # 放大到原图尺寸
    amap_norm = cv2.resize(amap_norm, (img_bgr.shape[1], img_bgr.shape[0]),
                           interpolation=cv2.INTER_LINEAR)

    # JET 伪彩热力图
    color = cv2.applyColorMap(amap_norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_dir / 'color' / f"{name}.png"), color)

    # 原图 + 热力图叠加
    combined = cv2.addWeighted(img_bgr, 0.5, color, 0.5, 0)
    cv2.imwrite(str(out_dir / 'combined' / f"{name}.png"), combined)

    # 灰度异常图
    sub = 'crack' if is_anomaly else 'good'
    cv2.imwrite(str(out_dir / sub / f"{name}.png"), amap_norm)


# ---------------------------------------------------------------------------
# 4. 合成示例数据（零配置时用）
# ---------------------------------------------------------------------------
def _draw_normal(seed: int) -> np.ndarray:
    """画一张「正常」工件图：带渐变纹理的圆盘 + 轻微扰动。"""
    rng = np.random.default_rng(seed)
    img = np.full((256, 256, 3), 96, dtype=np.uint8)  # 背景深灰
    cx, cy, r = 128, 128, 80
    yy, xx = np.mgrid[0:256, 0:256]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # 圆盘：中心亮、边缘暗的径向纹理
    inside = d <= r
    val = (np.clip(200 - 0.9 * d, 60, 200)).astype(np.uint8)
    # 轻微环状纹理（模拟加工痕迹）—— 固定相位，保证「正常之间相似」
    ripple = (np.sin(d * 0.6) * 8).astype(np.float32)
    img[inside, 0] = np.clip(val[inside] + ripple[inside], 0, 255)
    img[inside, 1] = np.clip(val[inside] + ripple[inside] * 0.5, 0, 255)
    img[inside, 2] = np.clip(val[inside], 0, 255)
    # 轻微高斯噪声
    noise = rng.normal(0, 1.5, img.shape[:2]).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
    return img


def _draw_defect(kind: str, seed: int = 0) -> np.ndarray:
    """在正常图基础上叠加一种缺陷：scratch 划痕 / blob 污渍。"""
    img = _draw_normal(seed)
    cx, cy, r = 128, 128, 80
    if kind == 'scratch':
        # 一条贯穿圆盘的深色划痕（足够粗，确保降采样后仍可见）
        cv2.line(img, (70, 128), (186, 128), (20, 20, 20), 12)
    else:
        # 一个高对比的亮斑污渍
        cv2.circle(img, (150, 90), 26, (240, 240, 240), -1)
        cv2.circle(img, (150, 90), 26, (210, 210, 210), 2)
    return img


def make_synthetic_dataset() -> tuple[list[str], list[tuple[str, bool]]]:
    """生成 8 张正常 + 2 张缺陷样本，返回 (normal_paths, [(test_path, is_anomaly)])。"""
    root = Path(__file__).parent / 'demo_input'
    normal_dir = root / 'normal'
    test_dir = root / 'test'
    normal_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    normals = []
    for i in range(8):
        p = normal_dir / f"normal_{i:02d}.png"
        cv2.imwrite(str(p), _draw_normal(seed=i))
        normals.append(str(p))

    tests = []
    for kind, i in [('scratch', 0), ('blob', 1)]:
        p = test_dir / f"{kind}_{i:02d}.png"
        cv2.imwrite(str(p), _draw_defect(kind, seed=i))
        tests.append((str(p), True))
    # 再放一张正常图当测试，验证正常样本分数低
    p = test_dir / "normal_test.png"
    cv2.imwrite(str(p), _draw_normal(seed=99))
    tests.append((str(p), False))
    return normals, tests


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='无预训练异常检测效果图演示')
    ap.add_argument('--normal_dir', type=str, default=None, help='正常样本目录')
    ap.add_argument('--test_dir', type=str, default=None, help='测试样本目录（每张都打分）')
    ap.add_argument('--test_image', type=str, default=None, help='单张测试图')
    ap.add_argument('--k', type=int, default=3, help='近邻数量')
    ap.add_argument('--out_dir', type=str, default='demo_output', help='输出目录')
    args = ap.parse_args()

    # 收集输入
    if args.normal_dir:
        normal_paths = [str(p) for p in Path(args.normal_dir).glob('*')
                        if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')]
        tests = []
        if args.test_dir:
            tests = [(str(p), p.parent.name != 'good') for p in Path(args.test_dir).glob('*')
                     if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')]
        elif args.test_image:
            tests = [(args.test_image, True)]
    else:
        print("[demo] 未提供 --normal_dir，生成合成示例数据 ...")
        normal_paths, tests = make_synthetic_dataset()

    if not normal_paths:
        print("[ERROR] 正常样本为空，请检查 --normal_dir")
        sys.exit(1)
    if not tests:
        print("[ERROR] 测试样本为空，请提供 --test_dir 或 --test_image")
        sys.exit(1)

    # 输出目录
    out_dir = Path(args.out_dir)
    for sub in ['color', 'combined', 'crack', 'good']:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # 构建内存库（等价于 DINOv2AD 的训练阶段）
    print(f"[demo] 用 {len(normal_paths)} 张正常图构建内存库 ...")
    bank = build_memory_bank(normal_paths)
    print(f"[demo] 内存库特征数: {bank.shape[0]} (dim={bank.shape[1]})")

    # 逐张打分 + 可视化
    report = []
    print(f"[demo] 处理 {len(tests)} 张测试图 ...")
    for path, is_anomaly in tests:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] 无法读取 {path}，跳过")
            continue
        feats = extract_patch_features(img)
        amap = score_anomaly_map(feats, bank, k=args.k)

        # 图像级分数：top-1% patch 的均值（与 DINOv2AD 一致）
        flat = amap.flatten()
        ktop = max(1, int(len(flat) * 0.01))
        top = np.partition(flat, -ktop)[-ktop:]
        img_score = float(top.mean())

        name = Path(path).stem
        render_visualization(img, amap, out_dir, name, is_anomaly)
        tag = '异常' if is_anomaly else '正常'
        report.append(f"{name}\t{tag}\timage_score={img_score:.4f}")
        print(f"  {name:20s} {tag}  image_score={img_score:.4f}")

    # 写报告
    report_path = out_dir / 'demo_report.txt'
    report_path.write_text('\n'.join(report), encoding='utf-8')
    print(f"\n[demo] 完成！效果图已输出到 {out_dir.resolve()}")
    print(f"  - {out_dir}/color/      热力图（JET 伪彩）")
    print(f"  - {out_dir}/combined/   原图 + 热力图叠加")
    print(f"  - {out_dir}/crack/ good/ 灰度异常图")
    print(f"  - {report_path.name}")


if __name__ == '__main__':
    main()
