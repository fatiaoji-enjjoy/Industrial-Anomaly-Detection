# DINOv2AD for Anomalib

基于 [Anomalib](https://github.com/open-edge-platform/anomalib) 扩展的工业图像异常检测与定位项目。项目加入了 DINOv2AD 与 PromptCore 模型，并提供从正常样本特征建库、异常评分、像素级定位到评估和可视化的完整流程。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

## 功能特点

- 使用预训练 DINOv2 视觉 Transformer 提取 patch 特征，主干网络保持冻结。
- 仅使用正常样本构建特征内存库，支持图像级异常检测与像素级异常定位。
- 支持按类别路由的 FAISS 近邻检索，降低多类别特征之间的干扰。
- 支持 coreset 下采样、PCA 前景掩膜以及 `max`、`mean`、`median` 三种近邻分数聚合方式。
- 提供训练、评估、指标汇总、热力图生成和推理延迟统计的一体化脚本。
- 包含实验性的 PromptCore 多类别异常检测模型。

## 项目结构

```text
.
├── DINOv2AD_pipeline.py                   # DINOv2AD 完整运行流程
├── hpo_nontrain.py                        # 无训练超参数搜索
├── run_no_pretrain.py                     # 无预训练基线与可视化
├── demo_effect.py                         # 单图/目录推理演示
├── eval_metrics.py                        # 指标计算工具
├── examples/configs/model/prompt_core.yaml
├── src/anomalib/models/image/dinov2ad/    # DINOv2AD 模型实现
├── src/anomalib/models/image/prompt_core/ # PromptCore 模型实现
├── Dockerfile
└── pyproject.toml
```

数据集、模型权重和运行结果不会提交到仓库，请在本地准备。

## 环境要求

- Python 3.10 或更高版本
- PyTorch 2.6 或更高版本
- NVIDIA GPU（推荐；CPU 也可运行，但速度较慢）
- FAISS（GPU 环境推荐 `faiss-gpu`，CPU 环境可使用 `faiss-cpu`）

## 安装

推荐使用独立的虚拟环境：

```bash
git clone https://github.com/fatiaoji-enjjoy/Industrial-Anomaly-Detection.git
cd Industrial-Anomaly-Detection

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
pip install faiss-cpu openpyxl
```

如需使用 CUDA，请根据本机 CUDA 版本安装对应的 PyTorch 和 FAISS 构建版本。

## 数据目录

流水线使用与 MVTec AD 兼容的目录结构：

```text
dataset_root/
└── category_name/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   └── defect_type/
    └── ground_truth/
        └── defect_type/
```

训练阶段只读取 `train/good` 中的正常图像。

## 快速开始

运行完整训练与评估流程：

```bash
python DINOv2AD_pipeline.py \
  --dataset_root "/path/to/dataset" \
  --categories bottle,cable,capsule \
  --model_name facebook/dinov2-small \
  --device cuda \
  --inference_size 224 \
  --k 3 \
  --aggregation max \
  --coreset_sampling_ratio 0.1
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset_root` | `D:\MVTec AD` | 数据集根目录，建议显式指定 |
| `--categories` | 全部支持类别 | 以英文逗号分隔的类别列表 |
| `--model_name` | `facebook/dinov2-small` | DINOv2 模型名称 |
| `--device` | `cuda` | 运行设备：`cuda` 或 `cpu` |
| `--inference_size` | `224` | 输入图像尺寸 |
| `--k` | `3` | kNN 近邻数量 |
| `--aggregation` | `max` | 近邻分数聚合方式 |
| `--coreset_sampling_ratio` | `0.1` | 内存库 coreset 采样比例 |
| `--use_adaptive_mask` | 关闭 | 启用 PCA 自适应前景掩膜 |
| `--pretrained_weights_path` | 空 | 离线预训练权重路径 |

查看全部参数：

```bash
python DINOv2AD_pipeline.py --help
```

首次使用在线预训练权重时，`timm` 会下载对应模型文件。离线环境可以通过 `--pretrained_weights_path` 指定本地权重。

## Docker

```bash
docker build -t dinov2ad .
docker run --rm --gpus all \
  -v /path/to/dataset:/data:ro \
  -v /path/to/output:/workspace/DINOv2AD_results \
  dinov2ad --dataset_root /data --categories bottle,cable,capsule
```

## 输出说明

默认输出位于 `DINOv2AD_results/<timestamp>/`，包含模型内存库、逐类别指标、异常热力图和汇总结果。输出目录和模型权重已加入 `.gitignore`，不会被意外提交。

## 致谢与许可

本项目基于 Anomalib 开发，并保留其 Apache License 2.0 许可与第三方声明。使用或分发前请阅读 [LICENSE](LICENSE) 和 [third-party-programs.txt](third-party-programs.txt)。
