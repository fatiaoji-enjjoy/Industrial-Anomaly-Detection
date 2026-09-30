# Copyright (C) 2025 DINOv2AD Contributors
# SPDX-License-Identifier: Apache-2.0

"""DINOv2AD: DINOv2-based Unsupervised Industrial Anomaly Detection.

This module implements the DINOv2AD model for anomaly detection and localization
using pretrained DINOv2 features, kNN distance scoring, and coreset subsampling.

Example:
    >>> from anomalib.models.image.dinov2ad import DINOv2AD
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = DINOv2AD(model_name='facebook/dinov2-small')
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)
"""

from .lightning_model import DINOv2AD

__all__ = ["DINOv2AD"]