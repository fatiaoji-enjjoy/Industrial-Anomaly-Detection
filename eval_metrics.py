#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_metrics.py —— 基于 demo_effect.py 输出的 demo_report.txt 评估图像级指标。

不改 demo_effect.py，只读取已生成的报告。
"""
import re
from pathlib import Path
import numpy as np


def load_report(path: Path, is_anomaly: bool):
    scores = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        m = re.search(r'image_score=([\d.]+)', line)
        if m:
            scores.append(float(m.group(1)))
    return scores, [1 if is_anomaly else 0] * len(scores)


def main():
    defect_scores, defect_labels = load_report(
        Path('demo_output_mvtec_bottle_broken_large/demo_report.txt'), True)
    normal_scores, normal_labels = load_report(
        Path('demo_output_mvtec_bottle_good/demo_report.txt'), False)
    scores = np.array(defect_scores + normal_scores)
    labels = np.array(defect_labels + normal_labels)

    # AUROC（基于 rank，升序：rank=1 是最小分数）
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = labels.sum()
    n0 = len(labels) - n1
    sum_ranks_pos = ranks[labels == 1].sum()
    auroc = (sum_ranks_pos - n1 * (n1 + 1) / 2) / (n1 * n0)

    # 最佳阈值准确率（Youden's J + accuracy）
    thr = np.linspace(scores.min(), scores.max(), 1000)
    best_acc, best_t, best_j, best_j_t = 0, 0, -1, 0
    for t in thr:
        pred = (scores >= t).astype(int)
        acc = (pred == labels).sum() / len(labels)
        if acc > best_acc:
            best_acc, best_t = acc, t
        tp = ((scores >= t) & (labels == 1)).sum()
        fp = ((scores >= t) & (labels == 0)).sum()
        fn = ((scores < t) & (labels == 1)).sum()
        tpr = tp / n1 if n1 else 0
        fpr = fp / n0 if n0 else 0
        j = tpr - fpr
        if j > best_j:
            best_j, best_j_t = j, t

    # 混淆矩阵用最佳准确率阈值
    tp = ((scores >= best_t) & (labels == 1)).sum()
    fp = ((scores >= best_t) & (labels == 0)).sum()
    tn = ((scores < best_t) & (labels == 0)).sum()
    fn = ((scores < best_t) & (labels == 1)).sum()
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

    print("=" * 50)
    print("Bottle 图像级异常检测评估")
    print("=" * 50)
    print(f"样本数: {len(labels)} (异常 {n1} + 正常 {n0})")
    print(f"\n异常分数统计:")
    print(f"  缺陷 mean={np.mean(defect_scores):.4f} std={np.std(defect_scores):.4f} "
          f"min={min(defect_scores):.4f} max={max(defect_scores):.4f}")
    print(f"  正常 mean={np.mean(normal_scores):.4f} std={np.std(normal_scores):.4f} "
          f"min={min(normal_scores):.4f} max={max(normal_scores):.4f}")
    print(f"\n指标:")
    print(f"  AUROC       = {auroc:.4f}")
    print(f"  最佳准确率  = {best_acc*100:.2f}% (阈值 {best_t:.4f})")
    print(f"  Youden 最佳阈值 = {best_j_t:.4f} (J={best_j:.3f})")
    print(f"\n混淆矩阵 (阈值={best_t:.4f}):")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")


if __name__ == "__main__":
    main()
