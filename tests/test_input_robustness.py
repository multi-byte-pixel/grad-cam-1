"""Limited metamorphic input-robustness checks.

Proof-of-concept regression checks that the Grad-CAM path produces a valid,
finite explanation under mild input perturbations. These are not field-certified
tolerance requirements; they assert valid execution and finite output only.
"""

import numpy as np
import pytest
import torch

from gradcam.explain import compute_gradcam_heatmap

TINY_INPUT_SIZE = 96


def _heatmap_for(model, tensor):
    heatmap, predicted_class = compute_gradcam_heatmap(
        model, tensor, model.conv3, TINY_INPUT_SIZE, target_class=None, method="Grad-CAM"
    )
    return heatmap, predicted_class


def test_baseline_heatmap_is_valid(tiny_model, image_tensor):
    heatmap, predicted_class = _heatmap_for(tiny_model, image_tensor)
    assert heatmap.shape == (TINY_INPUT_SIZE, TINY_INPUT_SIZE)
    assert np.all(np.isfinite(heatmap))
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    assert 0 <= predicted_class < 10


@pytest.mark.parametrize("factor", [0.5, 1.5])
def test_brightness_perturbation_produces_valid_finite_result(tiny_model, image_tensor, factor):
    perturbed = image_tensor * factor  # degraded lighting condition
    heatmap, _ = _heatmap_for(tiny_model, perturbed)
    assert heatmap.shape == (TINY_INPUT_SIZE, TINY_INPUT_SIZE)
    assert np.all(np.isfinite(heatmap))


def test_small_translation_preserves_output_dimensions(tiny_model, image_tensor):
    shifted = torch.roll(image_tensor, shifts=(3, 3), dims=(1, 2))  # degraded alignment
    heatmap, _ = _heatmap_for(tiny_model, shifted)
    assert heatmap.shape == (TINY_INPUT_SIZE, TINY_INPUT_SIZE)
    assert np.all(np.isfinite(heatmap))


def test_additive_noise_does_not_crash_explanation(tiny_model, image_tensor):
    torch.manual_seed(7)
    noisy = image_tensor + 0.1 * torch.randn_like(image_tensor)  # degraded signal
    heatmap, _ = _heatmap_for(tiny_model, noisy)
    assert np.all(np.isfinite(heatmap))
    assert heatmap.max() <= 1.0
