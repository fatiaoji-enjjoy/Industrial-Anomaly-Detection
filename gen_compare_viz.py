# -*- coding: utf-8 -*-
"""生成「原图 + GT缺陷标注 + 热力图 + 检测叠加」四联对照图（V7 配置，全局库模式）。

用法：
    python gen_compare_viz.py --dataset_root "D:\pytorch\datasets\mvtec_ad_test" \
        --categories bottle,cable,capsule,hazelnut,screw --max_per_category 6
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import DINOv2AD_pipeline as P
from DINOv2AD_pipeline import DINOv2AD, set_seed, setup_gpu_optimizations_once


def build_v7_model(dataset_root, categories, device='cuda'):
    """复用 pipeline 建库流程，构建 V7 配置（560 / [7,9,11] / softmax / k3 / coreset0.3）模型。"""
    set_seed(42)
    setup_gpu_optimizations_once()

    pipeline = P.DINOv2ADPipeline(
        categories=categories,
        dataset_root=dataset_root,
        model_name='facebook/dinov2-small',
        device=device,
        k=3,
        aggregation='softmax',
        inference_size=560,
        coreset_sampling_ratio=0.3,
        blur_kernel_size=5,
        blur_sigma=1.0,
        layers=[7, 9, 11],
        seed=42,
    )
    model = DINOv2AD(
        model_name=pipeline.model_name,
        device=pipeline.device,
        pretrained_weights_path=None,
        k=pipeline.k,
        aggregation=pipeline.aggregation,
        use_adaptive_mask=False,
        inference_size=pipeline.inference_size,
        coreset_sampling_ratio=pipeline.coreset_sampling_ratio,
        blur_kernel_size=pipeline.blur_kernel_size,
        blur_sigma=pipeline.blur_sigma,
        reweight=False,
        layers=pipeline.layers,
        median_kernel_size=0,
        morph_kernel_size=0,
    )
    t0 = time.time()
    train_time, total_features = pipeline._build_global_bank(model)
    model.model.subsample_embedding(pipeline.coreset_sampling_ratio)
    model.model.filter_features_by_distance(intra_class_percentile=90.0, top_k_neighbors=5)
    # 自适应阈值：按类别路由采正常样本分数的 99 百分位（与评估口径一致）
    scores = []
    for cat in categories:
        train_dir = os.path.join(dataset_root, cat, 'train', 'good')
        if not os.path.isdir(train_dir):
            continue
        files = [f for f in sorted(os.listdir(train_dir))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:20]
        for fname in files:
            try:
                img = Image.open(os.path.join(train_dir, fname)).convert('RGB')
                r = model.model.predict(img, category=cat)
                if r is not None:
                    scores.append(float(r.get('image_score', 0.0)))
            except Exception:
                continue
    threshold = float(np.percentile(scores, 99.0)) if scores else 0.5
    model.model.image_threshold = threshold
    if device == 'cuda' and __import__('torch').cuda.is_available():
        model.model.to('cuda')
    model.model.eval()
    print(f'[build] bank={model.model.memory_bank.num_features} feat, '
          f'train={time.time()-t0:.1f}s, threshold={model.model.image_threshold:.5f}')
    return model, pipeline


def load_gt_mask(mask_path, target_size):
    mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        return None
    if mask_raw.shape[:2] != target_size:
        mask_raw = cv2.resize(mask_raw, (target_size[1], target_size[0]),
                              interpolation=cv2.INTER_NEAREST)
    return (mask_raw > 0).astype(np.uint8)


def make_panel(img_np, gt_mask, anomaly_map, score, sample_name, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    H, W = img_np.shape[:2]
    # 热力图归一化 + JET
    am = anomaly_map.astype(np.float32)
    vmin, vmax = float(am.min()), float(am.max())
    if vmax - vmin > 1e-8:
        norm = (am - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(am)
    heat = (norm * 255).astype(np.uint8)
    heat_color = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    # GT 红色叠加
    img_gt = img_np.copy()
    if gt_mask is not None and gt_mask.any():
        overlay = img_gt.copy()
        overlay[gt_mask > 0] = (255, 60, 60)
        img_gt = cv2.addWeighted(img_gt, 0.5, overlay, 0.5, 0)

    # 检测叠加：热力图半透明 + GT 轮廓
    img_det = cv2.addWeighted(img_np, 0.55, heat_color, 0.45, 0)
    if gt_mask is not None and gt_mask.any():
        gt_uint8 = (gt_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_det, contours, -1, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    titles = ['Original Image', 'Ground Truth (defect)', 'Anomaly Heatmap', 'Detection Overlay']
    panels = [img_np, img_gt, heat_color, img_det]
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    fig.suptitle(f'{sample_name}   |   image_score={score:.4f}', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default=r'D:\pytorch\datasets\mvtec_ad_test')
    parser.add_argument('--categories', type=str, default='bottle,cable,capsule,hazelnut,screw')
    parser.add_argument('--max_per_category', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(',')]
    model, pipeline = build_v7_model(args.dataset_root, categories, args.device)

    out_root = os.path.join('DINOv2AD_results', 'compare_viz')
    os.makedirs(out_root, exist_ok=True)

    for cat in categories:
        samples = [s for s in pipeline._get_test_samples(cat) if s['is_anomaly'] and s['mask_path']]
        samples = samples[:args.max_per_category]
        if not samples:
            print(f'[{cat}] no anomalous samples with mask, skip')
            continue
        cat_dir = os.path.join(out_root, cat)
        os.makedirs(cat_dir, exist_ok=True)
        for s in samples:
            img = Image.open(s['image_path']).convert('RGB')
            img_np = np.array(img)
            result = model.model.predict(img, category=cat)
            if result is None:
                print(f'[{cat}] predict failed: {os.path.basename(s["image_path"])}')
                continue
            am = result['anomaly_map_resized']
            if am.shape[:2] != img_np.shape[:2]:
                am = cv2.resize(am, (img_np.shape[1], img_np.shape[0]))
            gt = load_gt_mask(s['mask_path'], img_np.shape[:2])
            name = os.path.basename(s['image_path']).replace('.png', '').replace('.jpg', '')
            out = os.path.join(cat_dir, f'{name}.png')
            make_panel(img_np, gt, am, result['image_score'], name, out)
            print(f'[{cat}] saved {os.path.basename(out)}  score={result["image_score"]:.4f}')
    print('done ->', out_root)


if __name__ == '__main__':
    main()