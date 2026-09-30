# syntax=docker/dockerfile:1
# =============================================================================
# DINOv2AD — 工业图像异常检测与定位
# 基于 Anomalib + DINOv2 (Transformer) 的无监督异常检测与定位
#
# 说明：
#   * 使用通用 CUDA 12.x + PyTorch 2.x 运行环境。
#   * DINOv2 作为冻结的 Transformer 特征提取器。
#   * 支持全局内存库与按类别路由的特征检索。
#   * 无监督范式：训练仅使用正常样本，不引入任何异常样本信息。
#
# 构建：
#   docker build -t dinov2ad .
#
# 运行（训练 + 全流程评估）：
#   docker run --gpus all -v /path/to/dataset:/data dinov2ad \
#       --dataset_root /data --categories bottle,cable,screw
#
# 结果输出目录（挂载到宿主机）：
#   /workspace/DINOv2AD_results/<时间戳>/
# =============================================================================

# ---- 基础镜像（可按实际 CUDA/PyTorch 环境调整）----
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # 屏蔽 Lightning / litlogger 冗余日志，保证输出可读
    LITLOGER_ENABLE=0

WORKDIR /workspace

# ---- 1. 系统依赖（OpenCV headless 运行所需的 GL/线程库）----
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- 2. 安装项目（含 anomalib 及 timm/opencv/scikit-learn/pandas 核心依赖）----
# 先拷贝整个工程，再 pip install -e .
COPY . /workspace

# faiss-gpu：GPU 加速近邻检索（DINOv2AD 内存库索引）
# openpyxl：Excel 结果导出（pandas to_excel）
RUN pip install --no-cache-dir "faiss-gpu" "openpyxl" \
    && pip install --no-cache-dir -e .

# ---- 3. 数据与结果目录 ----
# MVTec AD 兼容数据集挂载到 /data
# 结果写入 /workspace/DINOv2AD_results
VOLUME ["/data", "/workspace/DINOv2AD_results"]

# ---- 4. 入口：全自动化流水线 ----
# 默认运行全部类别；通过 CMD 覆盖指定类别等参数
ENTRYPOINT ["python", "DINOv2AD_pipeline.py"]
CMD ["--dataset_root", "/data"]
