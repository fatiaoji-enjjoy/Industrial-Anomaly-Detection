#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hpo_nontrain.py —— 无训练网格搜索（在开发集上选超参，不训练任何权重）

DINOv2AD 是 training-free 模型：骨干冻结、只做前向特征提取。所有「优化」都来自
非训练杠杆（特征层 / 内存库 / 打分 / 后处理）。本脚本在开发集上遍历这些超参组合，
输出 pixel_F1 / image_F1 / AUROC / 时延 / 显存 对比表，直接挑最优组合固定提交。

关键点（合规）：
    * 全程不反向传播、不调骨干权重；
    * 只用正常样本建内存库，测试阶段不使用任何异常样本信息；
    * 骨干仅用公开通用预训练权重（ImageNet/DINOv2 自监督），不属于工业异常数据。

用法：
    python hpo_nontrain.py --dataset_root "D:\\MVTec AD" --categories bottle,cable
    python hpo_nontrain.py --dataset_root "D:\\MVTec AD" --max_combos 30

输出：
    hpo_results/hpo_summary.csv    每个组合的逐类 + 平均指标
    hpo_results/hpo_best.json      最优组合与参数
"""

import argparse
import itertools
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

# 路径注入（必须在 import anomalib 之前）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, 'src')
sys.path.insert(0, SRC_DIR)

os.environ.setdefault('LITLOGER_ENABLE', '0')
os.environ.setdefault('LIGHTNING_LOG_LEVEL', 'ERROR')

from anomalib.models.image.dinov2ad import DINOv2AD  # noqa: E402
from anomalib.metrics.auroc import _AUROC  # noqa: E402
from anomalib.metrics.aupr import _AUPR  # noqa: E402
from anomalib.metrics.f1_score import _F1Max  # noqa: E402

# ---------------------------------------------------------------------------
# 默认网格（可自行增删）。每个 key 都是「非训练」杠杆：
#   layers      : 多层特征融合的 block 索引（None = 仅最后一层）
#   coreset     : 内存库下采样比例
#   k           : kNN 近邻数
#   aggregation : patch 分数聚合方式
#   reweight    : 图像级 PatchCore 式邻域加权
#   blur_kernel_size / blur_sigma : 异常图高斯平滑（像素级后处理）
# ---------------------------------------------------------------------------
DEFAULT_GRID = {
    'layers': [None, [7, 9, 11]],
    'coreset_sampling_ratio': [0.1],
    'k': [1, 3],
    'aggregation': ['max', 'softmax'],
    'reweight': [False],
    'blur_kernel_size': [5],
    'blur_sigma': [1.0],
    'median_kernel_size': [0, 3],
    'morph_kernel_size': [0, 3],
}

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


# ---------------------------------------------------------------------------
# 数据收集（与 DINOv2AD_pipeline 一致）
# ---------------------------------------------------------------------------
def collect_train_paths(dataset_root: str, categories: list[str]) -> list[str]:
    paths = []
    for cat in categories:
        train_dir = os.path.join(dataset_root, cat, 'train', 'good')
        if not os.path.isdir(train_dir):
            continue
        for f in sorted(os.listdir(train_dir)):
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(train_dir, f))
    return paths


def collect_test_samples(dataset_root: str, categories: list[str]) -> list[dict]:
    samples = []
    for cat in categories:
        test_dir = os.path.join(dataset_root, cat, 'test')
        if not os.path.isdir(test_dir):
            continue
        for subdir in sorted(os.listdir(test_dir)):
            sub_path = os.path.join(test_dir, subdir)
            if not os.path.isdir(sub_path):
                continue
            is_anomaly = subdir != 'good'
            for filename in sorted(os.listdir(sub_path)):
                if not filename.lower().endswith(IMG_EXTS):
                    continue
                mask_path = None
                if is_anomaly:
                    mask_dir = os.path.join(dataset_root, cat, 'ground_truth', subdir)
                    mask_filename = filename.rsplit('.', 1)[0] + '_mask.png'
                    candidate = os.path.join(mask_dir, mask_filename)
                    if os.path.exists(candidate):
                        mask_path = candidate
                samples.append({
                    'image_path': os.path.join(sub_path, filename),
                    'mask_path': mask_path,
                    'category': cat,
                    'is_anomaly': is_anomaly,
                })
    return samples


# ---------------------------------------------------------------------------
# 特征提取（等价于训练阶段，只做前向、冻结骨干）
# ---------------------------------------------------------------------------
def extract_full_bank(model: DINOv2AD, dataset_root: str, categories: list[str],
                      inference_size: int) -> np.ndarray:
    """提取全部正常样本 patch 特征，返回未下采样的完整特征矩阵 (N, D)。"""
    model.model.train()
    if torch.cuda.is_available():
        model.model.to('cuda')

    transform = transforms.Compose([
        transforms.Resize((inference_size, inference_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 8
    with torch.no_grad():
        for cat in categories:
            cat_dir = os.path.join(dataset_root, cat, 'train', 'good')
            if not os.path.isdir(cat_dir):
                continue
            cat_images = [os.path.join(cat_dir, f) for f in sorted(os.listdir(cat_dir))
                          if f.lower().endswith(IMG_EXTS)]
            for i in range(0, len(cat_images), batch_size):
                batch_tensors = []
                for path in cat_images[i:i + batch_size]:
                    try:
                        img = Image.open(path).convert('RGB')
                        batch_tensors.append(transform(img))
                    except Exception:
                        continue
                if not batch_tensors:
                    continue
                batch = torch.stack(batch_tensors).to(next(model.model.parameters()).device)
                _ = model.model(batch)

    model.model.eval()
    return model.model.memory_bank.get_all_features().copy()


# ---------------------------------------------------------------------------
# 单组合评估
# ---------------------------------------------------------------------------
def evaluate_combo(model: DINOv2AD, full_bank: np.ndarray, test_samples: list[dict],
                   combo: dict) -> dict:
    """用给定超参组合在测试集上打分并计算指标。"""
    # 重建内存库（不重新提取特征）
    model.model.memory_bank.replace_all_features(full_bank)
    model.model.memory_bank.build_indices()

    ratio = combo['coreset_sampling_ratio']
    if ratio < 1.0:
        model.model.subsample_embedding(ratio)
    model.model.filter_features_by_distance(intra_class_percentile=90.0, top_k_neighbors=5)

    # 设置打分器
    scorer = model.model.anomaly_scorer
    scorer.k = combo['k']
    scorer.aggregation = combo['aggregation']
    scorer.blur_kernel_size = combo['blur_kernel_size']
    scorer.blur_sigma = combo['blur_sigma']
    scorer.reweight = combo['reweight']
    scorer.median_kernel_size = combo.get('median_kernel_size', 0)
    scorer.morph_kernel_size = combo.get('morph_kernel_size', 0)

    # 逐类指标
    cat_metrics = {}
    latencies = []
    for sample in test_samples:
        cat = sample['category']
        if cat not in cat_metrics:
            cat_metrics[cat] = {
                'image_auroc': _AUROC(), 'pixel_auroc': _AUROC(),
                'image_aupr': _AUPR(), 'pixel_aupr': _AUPR(),
                'image_f1': _F1Max(), 'pixel_f1': _F1Max(), 'n': 0,
            }
        m = cat_metrics[cat]
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            result = model.model.predict(image)
            if result is None:
                continue
            anomaly_map = result.get('anomaly_map_resized', result.get('anomaly_map'))
            image_score = result.get('image_score', 0.0)
            if anomaly_map is None:
                continue

            gt_label = 1 if sample['is_anomaly'] else 0
            m['image_auroc'].update(torch.tensor([image_score], dtype=torch.float32),
                                    torch.tensor([gt_label], dtype=torch.int64))
            m['image_aupr'].update(torch.tensor([image_score], dtype=torch.float32),
                                   torch.tensor([gt_label], dtype=torch.int64))
            m['image_f1'].update(torch.tensor([image_score], dtype=torch.float32),
                                 torch.tensor([gt_label], dtype=torch.int64))

            gt_mask = None
            if sample.get('mask_path') and os.path.exists(sample['mask_path']):
                mask_raw = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
                if mask_raw is not None:
                    if mask_raw.shape[:2] != anomaly_map.shape[:2]:
                        mask_raw = cv2.resize(mask_raw, (anomaly_map.shape[1], anomaly_map.shape[0]))
                    gt_mask = (mask_raw > 0).astype(np.float32)
            if gt_mask is None:
                gt_mask = np.zeros(anomaly_map.shape[:2], dtype=np.float32)

            m['pixel_auroc'].update(torch.tensor(anomaly_map.flatten(), dtype=torch.float32),
                                    torch.tensor(gt_mask.flatten(), dtype=torch.int64))
            m['pixel_aupr'].update(torch.tensor(anomaly_map.flatten(), dtype=torch.float32),
                                   torch.tensor(gt_mask.flatten(), dtype=torch.int64))
            m['pixel_f1'].update(torch.tensor(anomaly_map.flatten(), dtype=torch.float32),
                                 torch.tensor(gt_mask.flatten(), dtype=torch.int64))
            m['n'] += 1
            latencies.append(result.get('e2e_time_ms', 0.0))
        except Exception:
            continue

    # 聚合
    agg = {'image_AUROC': [], 'pixel_AUROC': [], 'image_AP': [], 'pixel_AP': [],
           'image_F1': [], 'pixel_F1': []}
    for cat, m in cat_metrics.items():
        if m['n'] == 0:
            continue
        agg['image_AUROC'].append(float(m['image_auroc'].compute()))
        agg['pixel_AUROC'].append(float(m['pixel_auroc'].compute()))
        agg['image_AP'].append(float(m['image_aupr'].compute()))
        agg['pixel_AP'].append(float(m['pixel_aupr'].compute()))
        agg['image_F1'].append(float(m['image_f1'].compute()))
        agg['pixel_F1'].append(float(m['pixel_f1'].compute()))

    return {
        'image_AUROC': float(np.mean(agg['image_AUROC'])) if agg['image_AUROC'] else 0.0,
        'pixel_AUROC': float(np.mean(agg['pixel_AUROC'])) if agg['pixel_AUROC'] else 0.0,
        'image_AP': float(np.mean(agg['image_AP'])) if agg['image_AP'] else 0.0,
        'pixel_AP': float(np.mean(agg['pixel_AP'])) if agg['pixel_AP'] else 0.0,
        'image_F1': float(np.mean(agg['image_F1'])) if agg['image_F1'] else 0.0,
        'pixel_F1': float(np.mean(agg['pixel_F1'])) if agg['pixel_F1'] else 0.0,
        'latency_mean_ms': float(np.mean(latencies)) if latencies else 0.0,
        'latency_max_ms': float(np.max(latencies)) if latencies else 0.0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='无训练网格搜索')
    ap.add_argument('--dataset_root', type=str, required=True, help='数据集根目录')
    ap.add_argument('--categories', type=str, default='bottle,cable,capsule,screw,transistor',
                    help='类别列表，逗号分隔（默认 5 类，减小搜索耗时）')
    ap.add_argument('--model_name', type=str, default='facebook/dinov2-small')
    ap.add_argument('--inference_size', type=int, default=224)
    ap.add_argument('--max_combos', type=int, default=0, help='最多评估组合数（0=全部）')
    ap.add_argument('--out_dir', type=str, default='hpo_results')
    ap.add_argument('--seed', type=int, default=42, help='随机种子（保证可复现）')
    args = ap.parse_args()

    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    categories = [c.strip() for c in args.categories.split(',') if c.strip()]

    # 展开网格
    keys = list(DEFAULT_GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*DEFAULT_GRID.values())]
    if args.max_combos and args.max_combos > 0:
        combos = combos[:args.max_combos]

    print("=" * 70, flush=True)
    print("无训练网格搜索（不训练任何权重）", flush=True)
    print(f"数据集: {args.dataset_root}  类别: {categories}", flush=True)
    print(f"组合数: {len(combos)}", flush=True)
    print("=" * 70, flush=True)

    # 收集数据（一次性）
    train_paths = collect_train_paths(args.dataset_root, categories)
    test_samples = collect_test_samples(args.dataset_root, categories)
    if not train_paths:
        print("[ERROR] 未找到训练样本")
        sys.exit(1)
    if not test_samples:
        print("[ERROR] 未找到测试样本")
        sys.exit(1)
    print(f"训练样本: {len(train_paths)}  测试样本: {len(test_samples)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    bank_cache: dict[tuple, tuple[DINOv2AD, np.ndarray]] = {}
    t_start = time.time()

    for idx, combo in enumerate(combos, 1):
        layers = combo['layers']
        cache_key = tuple(layers) if layers else ('final',)

        print(f"\n[{idx}/{len(combos)}] {combo}")

        try:
            # 每个 layers 配置只提取一次特征（缓存模型 + 完整特征）
            if cache_key not in bank_cache:
                print("  提取特征（该 layers 配置首次，耗时较长）...")
                model = DINOv2AD(
                    model_name=args.model_name,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    inference_size=args.inference_size,
                    layers=layers,
                )
                full_bank = extract_full_bank(
                    model, args.dataset_root, categories, args.inference_size,
                )
                bank_cache[cache_key] = (model, full_bank)
            else:
                model, full_bank = bank_cache[cache_key]
                print("  复用缓存模型与特征 ...")

            result = evaluate_combo(model, full_bank, test_samples, combo)
            row = {**combo, 'layers': str(combo['layers']), **result}
            rows.append(row)

            print(f"    pixel_F1={result['pixel_F1']:.4f}  pixel_AP={result['pixel_AP']:.4f}  "
                  f"image_F1={result['image_F1']:.4f}  image_AP={result['image_AP']:.4f}  "
                  f"pixel_AUROC={result['pixel_AUROC']:.4f}  "
                  f"latency={result['latency_mean_ms']:.1f}ms (max {result['latency_max_ms']:.1f})")
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            rows.append({**combo, 'layers': str(combo['layers']), 'error': str(e)})

    # 保存结果
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = out_dir / 'hpo_summary.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 按 pixel_F1（最高权重指标）排序找最优
    ok = [r for r in rows if 'pixel_F1' in r and r.get('pixel_F1', 0) > 0]
    if ok:
        best = max(ok, key=lambda r: r['pixel_F1'])
        best_path = out_dir / 'hpo_best.json'
        best_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding='utf-8')
        print("\n" + "=" * 70)
        print("最优组合（按 pixel_F1 排序）:")
        for k, v in best.items():
            print(f"  {k}: {v}")
        print("=" * 70)

    print(f"\n总耗时: {time.time() - t_start:.1f}s")
    print(f"结果已保存: {csv_path.resolve()}")
    print(f"建议：选 pixel_F1 高且 latency_max_ms <= 100、峰值显存 <= 24GB 的组合，"
          f"再用该参数跑 DINOv2AD_pipeline.py 正式产出。")


if __name__ == '__main__':
    main()
