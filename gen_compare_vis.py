# -*- coding: utf-8 -*-
"""生成「原图 + 缺陷真值 + 预测热力图 + 叠加图」四联对比可视化。

从训练集建库（与主流水线一致），对每个类别抽取典型样本（各缺陷类型 + 正常），
输出单张大图，直观展示缺陷位置与模型预测的对齐情况。
"""
import os
import sys
import time
import random

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from anomalib.models.image.dinov2ad import DINOv2AD  # noqa: E402

DATASET_ROOT = r'D:\pytorch\datasets\mvtec_ad_test'
OUT_DIR = r'D:\pytorch\anomalib_test\DINOv2AD_results\visualization_compare'
CATEGORIES = ['bottle', 'cable', 'capsule', 'hazelnut', 'screw']
SAMPLES_PER_CAT = 4  # 每类取 3 张缺陷 + 1 张正常

MODEL_KWARGS = dict(
    model_name='facebook/dinov2-small',
    device='cuda',
    k=3,
    aggregation='softmax',
    inference_size=560,
    coreset_sampling_ratio=0.3,
    blur_kernel_size=5,
    blur_sigma=1.0,
    layers=[7, 9, 11],
)

IMG_SIZE = 560


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_bank(model, categories):
    """与主流水线一致的建库流程：按类提特征 → coreset → 距离过滤 → 建索引。"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model.model.train()
    model.model.to('cuda')

    model.model.category_memory_banks = {}
    model.model.owner_category_memory_banks = {
        cat: type(model.model.memory_bank)(model.model.feature_dim, use_gpu=True)
        for cat in categories
    }

    with torch.no_grad():
        for cat in categories:
            model.model.current_category = cat
            cat_dir = os.path.join(DATASET_ROOT, cat, 'train', 'good')
            imgs = [os.path.join(cat_dir, f) for f in sorted(os.listdir(cat_dir))
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for i in range(0, len(imgs), 8):
                batch = []
                for p in imgs[i:i + 8]:
                    batch.append(transform(Image.open(p).convert('RGB')))
                if batch:
                    model.model(torch.stack(batch).to('cuda'))

    model.model.eval()
    model.model.memory_bank.build_indices()
    for b in model.model.owner_category_memory_banks.values():
        b.build_indices()
    model.model.current_category = None

    model.model.subsample_embedding(0.3)
    model.model.filter_features_by_distance(90.0, 5)
    return model.model.memory_bank.num_features


def pick_samples(category):
    """每类抽 3 张缺陷（不同缺陷类型）+ 1 张正常，带真值掩膜路径。"""
    rng = random.Random(42)
    test_dir = os.path.join(DATASET_ROOT, category, 'test')
    gt_dir = os.path.join(DATASET_ROOT, category, 'ground_truth')
    defect_types = [d for d in sorted(os.listdir(test_dir))
                    if d != 'good' and os.path.isdir(os.path.join(test_dir, d))]
    chosen = []
    picked_types = defect_types[:3] if len(defect_types) >= 3 else defect_types
    for d in picked_types:
        files = sorted(os.listdir(os.path.join(test_dir, d)))
        f = rng.choice(files)
        mask = os.path.join(gt_dir, d, f.replace('.png', '_mask.png'))
        chosen.append({
            'name': f'{d}_{f}', 'img': os.path.join(test_dir, d, f),
            'mask': mask if os.path.exists(mask) else None, 'anomaly': True,
        })
    good_dir = os.path.join(test_dir, 'good')
    if os.path.isdir(good_dir):
        f = rng.choice(sorted(os.listdir(good_dir)))
        chosen.append({'name': f'good_{f}', 'img': os.path.join(good_dir, f),
                       'mask': None, 'anomaly': False})
    return chosen


def make_panel(image, anomaly_map, mask_path, title_prefix):
    """四联面板：原图 / 真值掩膜 / 预测热力图 / 原图叠加热力图。"""
    img = np.array(image.convert('RGB'))
    amap = cv2.resize(anomaly_map.astype(np.float32), (img.shape[1], img.shape[0]))
    vmin, vmax = amap.min(), amap.max()
    norm = np.zeros_like(amap, dtype=np.uint8) if vmax - vmin < 1e-8 else (
        (amap - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.45, heat, 0.55, 0)

    if mask_path and os.path.exists(mask_path):
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_vis = np.zeros_like(img)
        gt_vis[gt > 0] = [255, 60, 60]
        gt_blend = cv2.addWeighted(img, 0.6, gt_vis, 0.4, 0)
        gt_title = 'Ground Truth'
    else:
        gt_blend = img.copy()
        gt_title = 'Ground Truth (none)'

    panels = [img, gt_blend, heat, overlay]
    titles = ['Original', gt_title, 'Predicted Heatmap', 'Overlay']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    for ax, p, t in zip(axes, panels, titles):
        ax.imshow(p)
        ax.set_title(f'{title_prefix} | {t}', fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    return fig


def main():
    set_seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    print('creating model ...')
    model = DINOv2AD(**MODEL_KWARGS)
    t0 = time.time()
    n = build_bank(model, CATEGORIES)
    print(f'memory bank: {n} features, {time.time() - t0:.1f}s')

    for cat in CATEGORIES:
        samples = pick_samples(cat)
        for s in samples:
            img = Image.open(s['img']).convert('RGB')
            result = model.model.predict(img, category=cat)
            if result is None:
                print(f'  [SKIP] {cat}/{s["name"]}')
                continue
            amap = result.get('anomaly_map_resized', result.get('anomaly_map'))
            fig = make_panel(img, amap, s['mask'], f'{cat}/{s["name"]}')
            out = os.path.join(OUT_DIR, cat, f'{s["name"].replace(".png", "")}.png')
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fig.savefig(out, dpi=110, bbox_inches='tight')
            plt.close(fig)
            print(f'  saved {out}  (score={result.get("image_score", 0):.4f})')

    print('all done ->', OUT_DIR)


if __name__ == '__main__':
    main()
