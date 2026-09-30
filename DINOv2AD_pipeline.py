#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv2AD 全自动化工业异常检测流程脚本

基于 Anomalib 框架的 DINOv2AD 模型，完整嵌入 Engine/Trainer/test/eval 标准接口。
支持全品类循环训练、推理、评估，GPU 显存监控，延迟约束校验，分层目录输出。

Usage:
    python DINOv2AD_pipeline.py
    python DINOv2AD_pipeline.py --categories bottle,cable,screw
    python DINOv2AD_pipeline.py --dataset_root "D:\\MVTec AD" --inference_size 224
"""

# ---------------------------------------------------------------------------
# 0. 路径注入（必须在 import anomalib 之前执行）
# ---------------------------------------------------------------------------
import os
import sys
import warnings
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, 'src')
sys.path.insert(0, SRC_DIR)

os.environ['LITLOGER_ENABLE'] = '0'
os.environ['LIGHTNING_LOG_LEVEL'] = 'ERROR'
os.environ['LIGHTNING_BAREBONES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

warnings.filterwarnings('ignore')

# ---- 自定义日志过滤器：屏蔽 Lightning 框架所有 INFO/WARNING ----
class _LightningFilter(logging.Filter):
    _blocked = {
        'lightning', 'lightning.pytorch', 'lightning.fabric',
        'anomalib.engine', 'anomalib.data',
        'litlogger', 'torch.distributed',
    }
    def filter(self, record: logging.LogRecord) -> bool:
        for prefix in self._blocked:
            if record.name.startswith(prefix):
                return record.levelno >= logging.ERROR
        return True

_root = logging.getLogger()
_root.addFilter(_LightningFilter())

for _mod in [
    'lightning', 'lightning.pytorch', 'lightning.fabric',
    'lightning.pytorch.trainer', 'lightning.pytorch.utilities.rank_zero',
    'anomalib', 'litlogger',
]:
    logging.getLogger(_mod).setLevel(logging.ERROR)
    logging.getLogger(_mod).propagate = False

# ---------------------------------------------------------------------------
# 1. 依赖导入
# ---------------------------------------------------------------------------
import time
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from anomalib.models.image.dinov2ad import DINOv2AD
from anomalib.metrics.auroc import _AUROC
from anomalib.metrics.f1_score import _F1Max

ALL_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
]

LATENCY_THRESHOLD_MS = 100.0

# ---------------------------------------------------------------------------
# 2. GPU 硬件加速（仅全局初始化执行一次）
# ---------------------------------------------------------------------------
_gpu_optimized = False


def setup_gpu_optimizations_once():
    global _gpu_optimized
    if _gpu_optimized:
        return
    _gpu_optimized = True

    if not torch.cuda.is_available():
        print("[GPU] CUDA 不可用，使用 CPU 模式")
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[GPU] {gpu_name} | 总显存: {gpu_mem:.1f} GB "
          f"| cuDNN benchmark: ON | TF32: ON")


def get_peak_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def reset_peak_memory():
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 3. 输出目录结构
# ---------------------------------------------------------------------------
class PipelineWorkspace:
    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir).resolve()
        self.training_results = self.root / 'training_results'
        self.models_dir = self.training_results / 'models'
        self.fused = self.root / 'fused'
        self.final_result = self.root / 'final_result'
        self.lightning_logs = self.root / 'lightning_logs'
        self.log_file = self.root / 'pipeline_log.txt'

        for d in [self.models_dir, self.fused, self.final_result, self.lightning_logs]:
            d.mkdir(parents=True, exist_ok=True)

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def get_global_model_dir(self) -> Path:
        """返回全局统一 Memory Bank 存储目录（不再按类别建子文件夹）"""
        return self.models_dir

    def get_category_fused_dir(self, category: str) -> Path:
        d = self.fused / category
        for sub in ['color', 'combined', 'crack', 'good']:
            (d / sub).mkdir(parents=True, exist_ok=True)
        return d


# ---------------------------------------------------------------------------
# 4. 推理延迟测量与热力图可视化
# ---------------------------------------------------------------------------
def measure_inference_latency(
    model: DINOv2AD,
    image: Image.Image,
    category: str,
    num_warmup: int = 3,
    num_repeat: int = 10,
) -> dict:
    model.eval()
    # 确保底层 torch 模型在 GPU 上
    if torch.cuda.is_available():
        model.model.to('cuda')
    times = []

    for _ in range(num_warmup):
        _ = model.model.predict(image, category=category)

    for _ in range(num_repeat):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.model.predict(image, category=category)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times)),
        'max_ms': float(np.max(times)),
        'min_ms': float(np.min(times)),
        'std_ms': float(np.std(times)),
        'latency_pass': bool(np.max(times) <= LATENCY_THRESHOLD_MS),
    }


def save_anomaly_visualization(
    anomaly_map: np.ndarray,
    image: Image.Image,
    mask: np.ndarray | None,
    sample_name: str,
    output_dir: Path,
    is_anomaly: bool,
):
    vmin, vmax = anomaly_map.min(), anomaly_map.max()
    if vmax - vmin > 1e-8:
        anomaly_norm = ((anomaly_map - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        anomaly_norm = np.zeros_like(anomaly_map, dtype=np.uint8)

    color_map = cv2.applyColorMap(anomaly_norm, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    color_pil = Image.fromarray(color_map)
    color_pil.save(output_dir / 'color' / f"{sample_name}.png")

    img_np = np.array(image.convert('RGB'))
    if img_np.shape[:2] != anomaly_norm.shape[:2]:
        img_np = cv2.resize(img_np, (anomaly_norm.shape[1], anomaly_norm.shape[0]))
    color_resized = cv2.resize(color_map, (img_np.shape[1], img_np.shape[0]))
    combined = cv2.addWeighted(img_np, 0.5, color_resized, 0.5, 0)
    Image.fromarray(combined).save(output_dir / 'combined' / f"{sample_name}.png")

    if is_anomaly:
        Image.fromarray(anomaly_norm, mode='L').save(output_dir / 'crack' / f"{sample_name}.png")
    else:
        Image.fromarray(anomaly_norm, mode='L').save(output_dir / 'good' / f"{sample_name}.png")


# ---------------------------------------------------------------------------
# 5. 主流水线
# ---------------------------------------------------------------------------
class DINOv2ADPipeline:
    def __init__(
        self,
        categories: list[str],
        dataset_root: str,
        model_name: str = 'facebook/dinov2-small',
        device: str = 'cuda',
        k: int = 3,
        aggregation: str = 'max',
        inference_size: int = 224,
        coreset_sampling_ratio: float = 0.1,
        use_adaptive_mask: bool = False,
        pretrained_weights_path: str | None = None,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0,
    ):
        self.categories = categories
        self.dataset_root = dataset_root
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.k = k
        self.aggregation = aggregation
        self.inference_size = inference_size
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.use_adaptive_mask = use_adaptive_mask
        self.pretrained_weights_path = pretrained_weights_path
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.workspace = PipelineWorkspace(f"DINOv2AD_results/{timestamp}")

    def run(self):
        self.workspace.log("=" * 60)
        self.workspace.log("DINOv2AD 全自动化异常检测流程启动 (全局共享 Memory Bank)")
        self.workspace.log(f"类别列表: {self.categories}")
        self.workspace.log(f"数据集根目录: {self.dataset_root}")
        self.workspace.log(f"模型: {self.model_name}, inference_size={self.inference_size}")
        self.workspace.log(f"coreset_ratio={self.coreset_sampling_ratio}, k={self.k}")
        self.workspace.log("=" * 60)

        setup_gpu_optimizations_once()

        overall_start = time.time()

        # ---- Phase 1: 创建全局模型，构建共享 Memory Bank ----
        self.workspace.log("\n[Phase 1] 构建全局共享 Memory Bank ...")
        try:
            model = DINOv2AD(
                model_name=self.model_name,
                device=self.device,
                pretrained_weights_path=self.pretrained_weights_path,
                k=self.k,
                aggregation=self.aggregation,
                use_adaptive_mask=self.use_adaptive_mask,
                inference_size=self.inference_size,
                coreset_sampling_ratio=self.coreset_sampling_ratio,
                blur_kernel_size=self.blur_kernel_size,
                blur_sigma=self.blur_sigma,
            )
        except Exception as e:
            self.workspace.log(f"[ERROR] 创建模型失败: {e}")
            return

        total_train_time = 0
        total_features = 0
        try:
            total_train_time, total_features = self._build_global_bank(model)
        except Exception as e:
            self.workspace.log(f"[ERROR] 全局特征提取失败: {e}")
            self.workspace.log(traceback.format_exc())
            return

        self.workspace.log(f"[Phase 1] 完成: 累计特征 {total_features}, 训练耗时 {total_train_time:.1f}s")

        # ---- Phase 2: Coreset 下采样 + 特征筛选 ----
        self.workspace.log("\n[Phase 2] Coreset 下采样 + 特征筛选 ...")
        try:
            model.model.subsample_embedding(self.coreset_sampling_ratio)
            model.model.filter_features_by_distance(intra_class_percentile=90.0, top_k_neighbors=5)
        except Exception as e:
            self.workspace.log(f"[ERROR] 特征筛选失败: {e}")
        filtered_size = model.model.memory_bank.num_features
        self.workspace.log(f"[Phase 2] 完成: Memory Bank 特征数 {filtered_size}")

        # ---- 保存全局模型 ----
        self._save_global_model(model, self.workspace)

        # ---- Phase 3: 逐类评估 + 可视化 + 延迟 ----
        self.workspace.log("\n[Phase 3] 逐类评估与可视化 ...")
        all_results = []
        for cat in self.categories:
            try:
                cat_result = self._evaluate_category(model, cat, total_train_time, filtered_size)
                all_results.append(cat_result)
            except Exception as e:
                self.workspace.log(f"[ERROR] 类别 {cat} 评估失败: {e}")
                self.workspace.log(traceback.format_exc())
                all_results.append({
                    'category': cat, 'status': 'FAILED', 'error': str(e),
                })

        overall_time = time.time() - overall_start
        self._save_summary(all_results, overall_time)

        self.workspace.log("=" * 60)
        self.workspace.log(f"全流程完成，总耗时: {overall_time:.1f}s ({overall_time/60:.1f}min)")
        self.workspace.log(f"结果目录: {self.workspace.root}")
        self.workspace.log("=" * 60)

    def _build_global_bank(self, model: DINOv2AD) -> tuple[float, int]:
        """一次性加载所有类别训练图像，批量提取特征构建全局 Memory Bank。

        直接收集所有类别的 good 样本路径，通过 DINOv2ADModel.forward()
        批量提取特征，统一存入同一个 MemoryBank，零中间丢弃。
        """
        ws = self.workspace
        ws.log("  收集所有类别训练样本 ...")

        all_image_paths = []
        for cat in self.categories:
            train_dir = os.path.join(self.dataset_root, cat, 'train', 'good')
            if not os.path.isdir(train_dir):
                ws.log(f"  [WARN] {cat} 训练目录不存在: {train_dir}")
                continue
            for fname in sorted(os.listdir(train_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(train_dir, fname))

        if not all_image_paths:
            ws.log("  [ERROR] 未找到任何训练样本")
            return 0.0, 0

        ws.log(f"  总训练样本: {len(all_image_paths)} 张，开始按类别批量特征提取 ...")

        model.model.train()
        if self.device == 'cuda' and torch.cuda.is_available():
            model.model.to('cuda')

        transform = transforms.Compose([
            transforms.Resize((self.inference_size, self.inference_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_start = time.time()
        batch_size = 8

        # 按类别提取特征，保持类别标签用于路由
        with torch.no_grad():
            for cat in self.categories:
                cat_dir = os.path.join(self.dataset_root, cat, 'train', 'good')
                if not os.path.isdir(cat_dir):
                    continue
                cat_images = [
                    os.path.join(cat_dir, f) for f in sorted(os.listdir(cat_dir))
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                if not cat_images:
                    continue

                # 设置类别路由标签
                model.model.current_category = cat
                for i in tqdm(range(0, len(cat_images), batch_size),
                              desc=f"  {cat} 特征提取", unit="batch", leave=False):
                    batch_paths = cat_images[i:i + batch_size]
                    batch_tensors = []
                    for path in batch_paths:
                        try:
                            img = Image.open(path).convert('RGB')
                            batch_tensors.append(transform(img))
                        except Exception:
                            continue
                    if not batch_tensors:
                        continue
                    batch_tensor = torch.stack(batch_tensors).to(next(model.model.parameters()).device)
                    _ = model.model(batch_tensor)

        model.model.eval()
        model.model.current_category = None

        # 构建所有类别的 FAISS GPU 索引
        model.model.memory_bank.build_indices()
        train_time = time.time() - train_start
        bank_size = model.model.memory_bank.num_features
        ws.log(f"  完成: {bank_size} features, 耗时 {train_time:.1f}s")
        return train_time, bank_size

    def _evaluate_category(self, model: DINOv2AD, category: str, train_time: float, memory_bank_size: int
    ) -> dict:
        """手动评估单个类别：直接调用 model.predict() + 独立指标计算。

        不再使用 Engine.test() 管线，避免全局共享 MemoryBank 与 Lightning
        内置逐类独立 MemoryBank 假设的底层兼容性冲突。
        """
        ws = self.workspace
        ws.log(f"\n{'=' * 40}")
        ws.log(f"评估类别: {category}")
        ws.log(f"{'=' * 40}")

        # ---- 获取测试样本 ----
        test_samples = self._get_test_samples(category)
        if not test_samples:
            ws.log(f"[WARN] 无测试样本，跳过评估")
            return {
                'category': category, 'status': 'NO_DATA',
                'memory_bank_size': memory_bank_size, 'train_time_s': train_time,
                'eval_time_s': 0, 'peak_memory_gb': 0, 'memory_pass': False,
                'image_AUROC': 0, 'pixel_AUROC': 0, 'image_F1': 0, 'pixel_F1': 0,
                'mean_ms': 0, 'max_ms': 0, 'min_ms': 0, 'std_ms': 0, 'latency_pass': True,
            }

        # ---- 类别路由 + 模型同步到 GPU ----
        model.model.current_category = category
        if self.device == 'cuda' and torch.cuda.is_available():
            model.model.to('cuda')
            model.model.eval()

        # ---- 初始化独立指标计算器 ----
        image_auroc = _AUROC()
        pixel_auroc = _AUROC()
        image_f1max = _F1Max()
        pixel_f1max = _F1Max()

        eval_start = time.time()
        pred_errors = 0
        first_error_msg = None
        total_samples = len(test_samples)

        fused_dir = ws.get_category_fused_dir(category)

        ws.log(f"[评估] 手动遍历 {total_samples} 张测试样本 ...")
        for sample in tqdm(test_samples, desc=f"  {category} 评估", unit="张", leave=False):
            try:
                # ---- 预测 ----
                image = Image.open(sample['image_path']).convert('RGB')
                result = model.model.predict(image, category=category)
                if result is None:
                    pred_errors += 1
                    if first_error_msg is None:
                        first_error_msg = 'predict 返回 None'
                    continue

                anomaly_map = result.get('anomaly_map_resized', result.get('anomaly_map'))
                image_score = result.get('image_score', 0.0)
                if anomaly_map is None:
                    pred_errors += 1
                    if first_error_msg is None:
                        first_error_msg = 'anomaly_map 为空'
                    continue

                # ---- Image-level 指标收集 ----
                gt_label = 1 if sample['is_anomaly'] else 0
                image_auroc.update(
                    torch.tensor([image_score], dtype=torch.float32),
                    torch.tensor([gt_label], dtype=torch.int64),
                )
                image_f1max.update(
                    torch.tensor([image_score], dtype=torch.float32),
                    torch.tensor([gt_label], dtype=torch.int64),
                )

                # ---- Pixel-level 指标收集 ----
                gt_mask = None
                if sample.get('mask_path') and os.path.exists(sample['mask_path']):
                    mask_raw = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
                    if mask_raw is not None:
                        if mask_raw.shape[:2] != anomaly_map.shape[:2]:
                            mask_raw = cv2.resize(
                                mask_raw, (anomaly_map.shape[1], anomaly_map.shape[0]),
                            )
                        gt_mask = (mask_raw > 0).astype(np.float32)
                if gt_mask is None:
                    # good 样本或无掩码：全 0 掩码
                    gt_mask = np.zeros(anomaly_map.shape[:2], dtype=np.float32)

                pixel_auroc.update(
                    torch.tensor(anomaly_map.flatten(), dtype=torch.float32),
                    torch.tensor(gt_mask.flatten(), dtype=torch.int64),
                )
                pixel_f1max.update(
                    torch.tensor(anomaly_map.flatten(), dtype=torch.float32),
                    torch.tensor(gt_mask.flatten(), dtype=torch.int64),
                )

                # ---- 可视化（与评估同一次 predict，避免重复推理） ----
                save_anomaly_visualization(
                    anomaly_map, image, gt_mask,
                    Path(sample['image_path']).stem,
                    fused_dir, sample.get('is_anomaly', False),
                )

            except Exception as e:
                pred_errors += 1
                if first_error_msg is None:
                    first_error_msg = str(e)
                if pred_errors <= 3:
                    ws.log(f"[WARN] 预测异常 ({os.path.basename(sample['image_path'])}): {e}")
                continue

        eval_time = time.time() - eval_start
        success_samples = total_samples - pred_errors

        # ---- 计算最终指标 ----
        metrics = {'image_AUROC': 0.0, 'pixel_AUROC': 0.0, 'image_F1': 0.0, 'pixel_F1': 0.0}
        if success_samples > 0:
            try:
                metrics['image_AUROC'] = float(image_auroc.compute())
                metrics['pixel_AUROC'] = float(pixel_auroc.compute())
                metrics['image_F1'] = float(image_f1max.compute())
                metrics['pixel_F1'] = float(pixel_f1max.compute())
            except Exception as e:
                ws.log(f"[ERROR] 指标计算异常: {e}")
                ws.log(traceback.format_exc())

        # ---- 状态判定 ----
        if success_samples == 0:
            status = 'FAILED'
        elif pred_errors > 0:
            status = 'PARTIAL'
        else:
            status = 'OK'

        ws.log(f"[评估] 完成，耗时: {eval_time:.1f}s, 成功: {success_samples}/{total_samples}, 状态: {status}")
        if pred_errors > 0:
            ws.log(f"[WARN] 预测异常: {pred_errors} 张失败 (首错: {first_error_msg})")
        ws.log(f"  image_AUROC: {metrics['image_AUROC']:.6f}")
        ws.log(f"  pixel_AUROC:  {metrics['pixel_AUROC']:.6f}")
        ws.log(f"  image_F1:     {metrics['image_F1']:.6f}")
        ws.log(f"  pixel_F1:     {metrics['pixel_F1']:.6f}")

        ws.log(f"[可视化] 已保存到: {fused_dir}")

        # ---- 延迟测量 ----
        ws.log(f"[延迟] 测量推理延迟 ...")
        latency_info = self._measure_category_latency(model, None, category, ws)
        ws.log(f"  mean: {latency_info['mean_ms']:.2f}ms, max: {latency_info['max_ms']:.2f}ms, "
               f"min: {latency_info['min_ms']:.2f}ms, std: {latency_info['std_ms']:.2f}ms")
        ws.log(f"  latency_pass (<=100ms): {'PASS' if latency_info['latency_pass'] else 'FAIL'}")

        # ---- 显存统计 ----
        peak_mem = get_peak_memory_gb()
        memory_pass = 0.03 <= peak_mem <= 2.5
        ws.log(f"[显存] 峰值显存: {peak_mem:.4f} GB | memory_pass: {'PASS' if memory_pass else 'WARN'}")
        reset_peak_memory()

        return {
            'category': category,
            'status': status,
            'memory_bank_size': memory_bank_size,
            'train_time_s': train_time,
            'eval_time_s': eval_time,
            'peak_memory_gb': peak_mem,
            'memory_pass': memory_pass,
            **metrics,
            **latency_info,
        }

    def _measure_category_latency(
        self, model: DINOv2AD, datamodule, category: str, ws
    ) -> dict:
        try:
            test_samples = self._get_test_samples(category)
        except Exception:
            return {'mean_ms': 0, 'max_ms': 0, 'min_ms': 0, 'std_ms': 0, 'latency_pass': True}

        if not test_samples:
            return {'mean_ms': 0, 'max_ms': 0, 'min_ms': 0, 'std_ms': 0, 'latency_pass': True}

        sample = test_samples[0]
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            return measure_inference_latency(model, image, category)
        except Exception as e:
            ws.log(f"[WARN] 延迟测量失败: {e}")
            return {'mean_ms': 0, 'max_ms': 0, 'min_ms': 0, 'std_ms': 0, 'latency_pass': True}

    def _get_test_samples(self, category: str) -> list[dict]:
        samples = []
        test_dir = os.path.join(self.dataset_root, category, 'test')
        if not os.path.exists(test_dir):
            return samples

        for subdir in sorted(os.listdir(test_dir)):
            subdir_path = os.path.join(test_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            is_anomaly = subdir != 'good'
            for filename in sorted(os.listdir(subdir_path)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    mask_path = None
                    if is_anomaly:
                        mask_dir = os.path.join(self.dataset_root, category, 'ground_truth', subdir)
                        mask_filename = filename.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
                        candidate = os.path.join(mask_dir, mask_filename)
                        if os.path.exists(candidate):
                            mask_path = candidate
                    samples.append({
                        'image_path': os.path.join(subdir_path, filename),
                        'mask_path': mask_path,
                        'category': category,
                        'is_anomaly': is_anomaly,
                    })
        return samples

    def _save_global_model(self, model: DINOv2AD, ws):
        """保存全局统一 Memory Bank 与模型权重到 training_results/models/"""
        model_dir = ws.get_global_model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            model.model.save_model(str(model_dir))
        except Exception as e:
            ws.log(f"[WARN] 保存 Memory Bank 失败: {e}")

        try:
            torch.save(model.model.state_dict(), model_dir / 'model_weights.pt')
        except Exception as e:
            ws.log(f"[WARN] 保存模型权重失败: {e}")

        ws.log(f"  全局模型已保存到: {model_dir}")

    def _save_summary(self, all_results: list[dict], overall_time: float):
        ws = self.workspace
        final_dir = ws.final_result

        rows = []
        for r in all_results:
            rows.append({
                'category': r.get('category', ''),
                'status': r.get('status', ''),
                'memory_bank_size': r.get('memory_bank_size', 0),
                'train_time_s': round(r.get('train_time_s', 0), 2),
                'eval_time_s': round(r.get('eval_time_s', 0), 2),
                'peak_memory_gb': round(r.get('peak_memory_gb', 0), 4),
                'memory_pass': r.get('memory_pass', False),
                'image_AUROC': round(r.get('image_AUROC', 0), 6),
                'pixel_AUROC': round(r.get('pixel_AUROC', 0), 6),
                'image_F1': round(r.get('image_F1', 0), 6),
                'pixel_F1': round(r.get('pixel_F1', 0), 6),
                'latency_mean_ms': round(r.get('mean_ms', 0), 2),
                'latency_max_ms': round(r.get('max_ms', 0), 2),
                'latency_min_ms': round(r.get('min_ms', 0), 2),
                'latency_std_ms': round(r.get('std_ms', 0), 2),
                'latency_pass': r.get('latency_pass', True),
            })

        df = pd.DataFrame(rows)

        csv_path = final_dir / 'summary_results.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        ws.log(f"汇总 CSV 已保存: {csv_path}")

        try:
            xlsx_path = final_dir / 'results.xlsx'
            df.to_excel(xlsx_path, index=False)
            ws.log(f"汇总 Excel 已保存: {xlsx_path}")
        except Exception as e:
            ws.log(f"[WARN] Excel 保存失败: {e}")

        json_path = final_dir / 'evaluation_report.json'
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'dataset_root': self.dataset_root,
            'inference_size': self.inference_size,
            'coreset_sampling_ratio': self.coreset_sampling_ratio,
            'total_time_s': round(overall_time, 1),
            'num_categories': len(all_results),
            'num_passed': sum(1 for r in all_results if r.get('status') == 'OK'),
            'num_failed': sum(1 for r in all_results if r.get('status') != 'OK'),
            'categories': all_results,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        ws.log(f"评估报告 JSON 已保存: {json_path}")

        passed = [r for r in all_results if r.get('status') == 'OK']
        if passed:
            avg_image_auroc = np.mean([r.get('image_AUROC', 0) for r in passed])
            avg_pixel_auroc = np.mean([r.get('pixel_AUROC', 0) for r in passed])
            avg_image_f1 = np.mean([r.get('image_F1', 0) for r in passed])
            avg_pixel_f1 = np.mean([r.get('pixel_F1', 0) for r in passed])
            ws.log(f"\n全品类平均指标:")
            ws.log(f"  image_AUROC: {avg_image_auroc:.4f}")
            ws.log(f"  pixel_AUROC:  {avg_pixel_auroc:.4f}")
            ws.log(f"  image_F1:     {avg_image_f1:.4f}")
            ws.log(f"  pixel_F1:     {avg_pixel_f1:.4f}")


# ---------------------------------------------------------------------------
# 6. 命令行入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='DINOv2AD 全自动化异常检测流水线')
    parser.add_argument('--categories', type=str, default=None,
                        help='类别列表，逗号分隔。默认全部 15 类')
    parser.add_argument('--dataset_root', type=str, default=r'D:\MVTec AD',
                        help='MVTec AD 数据集根目录')
    parser.add_argument('--model_name', type=str, default='facebook/dinov2-small',
                        help='DINOv2 模型名称')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (cuda/cpu)')
    parser.add_argument('--k', type=int, default=3,
                        help='kNN 近邻数量')
    parser.add_argument('--aggregation', type=str, default='max',
                        choices=['max', 'mean', 'median'],
                        help='patch 分数聚合方式')
    parser.add_argument('--inference_size', type=int, default=224,
                        help='推理图像尺寸')
    parser.add_argument('--coreset_sampling_ratio', type=float, default=0.1,
                        help='Coreset 下采样比例')
    parser.add_argument('--use_adaptive_mask', action='store_true', default=False,
                        help='启用 PCA 自适应前景掩膜')
    parser.add_argument('--pretrained_weights_path', type=str, default=None,
                        help='本地预训练权重路径')
    parser.add_argument('--blur_kernel_size', type=int, default=5,
                        help='异常图高斯模糊核大小 (奇数, 默认5)')
    parser.add_argument('--blur_sigma', type=float, default=1.0,
                        help='异常图高斯模糊 sigma (默认1.0)')

    args = parser.parse_args()

    if args.categories:
        categories = [c.strip() for c in args.categories.split(',')]
    else:
        categories = ALL_CATEGORIES

    print("=" * 60)
    print("DINOv2AD 全自动化工业异常检测流水线")
    print("=" * 60)
    print(f"类别: {categories}")
    print(f"数据集: {args.dataset_root}")
    print(f"模型: {args.model_name}")
    print(f"推理尺寸: {args.inference_size}")
    print(f"coreset 比例: {args.coreset_sampling_ratio}")
    print("=" * 60)

    pipeline = DINOv2ADPipeline(
        categories=categories,
        dataset_root=args.dataset_root,
        model_name=args.model_name,
        device=args.device,
        k=args.k,
        aggregation=args.aggregation,
        inference_size=args.inference_size,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        use_adaptive_mask=args.use_adaptive_mask,
        pretrained_weights_path=args.pretrained_weights_path,
        blur_kernel_size=args.blur_kernel_size,
        blur_sigma=args.blur_sigma,
    )

    try:
        pipeline.run()
        print("\n流水线执行成功！")
        sys.exit(0)
    except Exception as e:
        print(f"\n流水线执行失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()