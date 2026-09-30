#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_comparison.py —— 把 demo_effect.py 已生成的热力图与 MVTec 原图
拼成「原图 | 热力图 | 叠加定位」三联并排对比图。

不改 demo_effect.py，仅读取其输出目录与原始测试图。

用法：
  py -3.13 make_comparison.py
  py -3.13 make_comparison.py --demo_dir demo_output_mvtec_bottle_broken_large \
                              --src_dir "D:/datasets/mvtec-ad/bottle/test/broken_large"
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def make_panel(img_bgr: np.ndarray, heatmap_color: np.ndarray) -> np.ndarray:
    """原图 | 热力图 | 叠加 三联并排。

    Args:
        img_bgr: 原图 BGR (H, W, 3)。
        heatmap_color: JET 伪彩热力图 BGR，同尺寸。

    Returns:
        panel: 三张图横向拼接 (H, 3*W, 3)。
    """
    h, w = img_bgr.shape[:2]
    # 叠加图（与 demo_effect.py combined 一致：0.5/0.5）
    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)
    panel = np.hstack([img_bgr, heatmap_color, overlay])
    return panel


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """在三联图顶部加标题条，标注每段含义。"""
    h, w = img.shape[:2]
    bar = np.full((40, w, 3), 30, dtype=np.uint8)
    seg = w // 3
    labels = ["Original", "Heatmap (JET)", "Overlay (0.5/0.5)"]
    for i, t in enumerate(labels):
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x = i * seg + (seg - tw) // 2
        y = 27
        cv2.putText(bar, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser(description="三联并排对比图生成器")
    ap.add_argument("--demo_dir", type=str,
                    default="demo_output_mvtec_bottle_broken_large",
                    help="demo_effect.py 输出目录（含 color/）")
    ap.add_argument("--src_dir", type=str,
                    default="D:/datasets/mvtec-ad/bottle/test/broken_large",
                    help="MVTec 测试图目录（原图）")
    ap.add_argument("--out_dir", type=str, default="demo_comparison",
                    help="对比图输出目录")
    args = ap.parse_args()

    demo_dir = Path(args.demo_dir)
    color_dir = demo_dir / "color"
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    color_files = sorted(color_dir.glob("*.png"))
    if not color_files:
        raise RuntimeError(f"未在 {color_dir} 找到热力图，请先运行 demo_effect.py")

    print(f"[cmp] 处理 {len(color_files)} 张图，输出到 {out_dir}")
    for cf in color_files:
        # 找对应原图（同名 png）
        src = src_dir / cf.name
        if not src.exists():
            print(f"  [WARN] 未找到原图 {src}，跳过")
            continue
        img = cv2.imread(str(src))
        heat = cv2.imread(str(cf))
        if img is None or heat is None:
            print(f"  [WARN] 读取失败 {cf.name}，跳过")
            continue
        # 尺寸对齐
        if img.shape[:2] != heat.shape[:2]:
            heat = cv2.resize(heat, (img.shape[1], img.shape[0]))
        panel = make_panel(img, heat)
        panel = add_label(panel, cf.stem)
        cv2.imwrite(str(out_dir / cf.name), panel)
        print(f"  {cf.name} done")
    print(f"[cmp] 完成！对比图在 {out_dir.resolve()}")


if __name__ == "__main__":
    main()
