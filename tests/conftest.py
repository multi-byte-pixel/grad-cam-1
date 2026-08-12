"""Shared fixtures for the offline quality-validation suite.

All fixtures are deterministic and CPU-only: no dataset downloads, no training,
no GPU, no live cloud access.
"""

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gradcam.model import ImageCNN

TINY_INPUT_SIZE = 96
METRICS_CSVS = [
    os.path.join("results", "tinycnn", "metrics.csv"),
    os.path.join("results", "resnet18", "metrics.csv"),
]


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def sample_heatmap(rng):
    """A non-constant heatmap already normalized to [0, 1]."""
    heatmap = rng.random((TINY_INPUT_SIZE, TINY_INPUT_SIZE)).astype(np.float64)
    heatmap[20:40, 20:40] = 1.0
    return heatmap


@pytest.fixture
def degenerate_heatmap():
    """The documented all-zero small-model edge case."""
    return np.zeros((TINY_INPUT_SIZE, TINY_INPUT_SIZE), dtype=np.float64)


@pytest.fixture
def sample_image(rng):
    """A synthetic RGB image in HWC [0, 1]."""
    return rng.random((TINY_INPUT_SIZE, TINY_INPUT_SIZE, 3)).astype(np.float64)


@pytest.fixture
def tiny_model():
    """An untrained ImageCNN in eval mode — enough to exercise the Grad-CAM path."""
    torch.manual_seed(0)
    model = ImageCNN(num_classes=10)
    model.eval()
    return model


@pytest.fixture
def image_tensor():
    """A deterministic normalized CHW tensor matching the tiny model input."""
    torch.manual_seed(0)
    return torch.randn(3, TINY_INPUT_SIZE, TINY_INPUT_SIZE)
