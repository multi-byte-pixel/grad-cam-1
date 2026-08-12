"""Grad-CAM / Grad-CAM++ explanation toolkit.

Pure, importable functions for computing gradient-based class-activation
heatmaps, deriving attention masks, and scoring explanation quality. Model
training lives in the runnable scripts under ``scripts/`` so importing this
package has no side effects.
"""

from .data import (
    CIFAR10_CLASSES,
    STL10_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transform,
    unnormalize_image,
)
from .explain import (
    compute_gradcam_heatmap,
    gradcam_binary_mask,
    normalize_heatmap,
    overlay_heatmap,
)
from .metrics import (
    calculate_fidelity,
    calculate_spread,
    is_degenerate,
    mean_intensity,
    review_status,
    robustness,
)
from .model import ImageCNN, build_resnet

__all__ = [
    "CIFAR10_CLASSES",
    "STL10_CLASSES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_transform",
    "unnormalize_image",
    "compute_gradcam_heatmap",
    "gradcam_binary_mask",
    "normalize_heatmap",
    "overlay_heatmap",
    "calculate_fidelity",
    "calculate_spread",
    "is_degenerate",
    "mean_intensity",
    "review_status",
    "robustness",
    "ImageCNN",
    "build_resnet",
]
