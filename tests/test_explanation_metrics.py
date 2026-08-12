"""Deterministic unit tests for explanation metric and heatmap helpers."""

import numpy as np
import pytest

from gradcam.explain import gradcam_binary_mask, normalize_heatmap
from gradcam.metrics import (
    calculate_fidelity,
    calculate_spread,
    mean_intensity,
    robustness,
)


def _constant_predict_fn(prob_original, prob_masked):
    """Return a predict_fn whose confidence depends on whether pixels were masked."""
    calls = {"n": 0}

    def predict(images):
        # First call is the original image, second is the masked image.
        value = prob_original if calls["n"] == 0 else prob_masked
        calls["n"] += 1
        return np.array([[1.0 - value, value]])

    return predict


def test_fidelity_is_original_minus_masked_confidence():
    image = np.ones((8, 8, 3), dtype=np.float64)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    predict_fn = _constant_predict_fn(prob_original=0.9, prob_masked=0.4)

    fidelity = calculate_fidelity(image, mask, predicted_class=1, predict_fn=predict_fn)

    assert fidelity == pytest.approx(0.5)


def test_spread_is_bounded_between_zero_and_one():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:2, :] = 1  # 20% coverage
    spread = calculate_spread(mask)
    assert 0.0 <= spread <= 1.0
    assert spread == pytest.approx(0.2)


def test_normalize_heatmap_returns_finite_values_in_unit_range():
    raw = np.array([[-5.0, 0.0], [2.5, 10.0]])
    normalized = normalize_heatmap(raw)
    assert np.all(np.isfinite(normalized))
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized.max() == pytest.approx(1.0)


def test_constant_heatmap_normalizes_without_nan_or_divide_by_zero():
    constant = np.full((16, 16), 3.3, dtype=np.float64)
    normalized = normalize_heatmap(constant)
    assert np.all(np.isfinite(normalized))
    assert np.all(normalized == 0.0)


def test_fidelity_raises_on_mask_image_shape_mismatch():
    image = np.ones((8, 8, 3), dtype=np.float64)
    bad_mask = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        calculate_fidelity(image, bad_mask, predicted_class=0, predict_fn=lambda x: np.array([[1.0]]))


def test_spread_raises_on_non_2d_mask():
    with pytest.raises(ValueError):
        calculate_spread(np.zeros((4, 4, 3), dtype=np.uint8))


@pytest.mark.parametrize("percentile,expected_fraction", [(80, 0.2), (50, 0.5)])
def test_binary_mask_keeps_expected_fraction(percentile, expected_fraction):
    heatmap = np.linspace(0, 1, 100, dtype=np.float64).reshape(10, 10)
    mask = gradcam_binary_mask(heatmap, percentile=percentile)
    assert set(np.unique(mask)).issubset({0, 1})
    assert mask.mean() == pytest.approx(expected_fraction, abs=0.02)


def test_mean_intensity_and_robustness_are_finite(sample_heatmap):
    assert np.isfinite(mean_intensity(sample_heatmap))
    assert np.isfinite(robustness(sample_heatmap))
    assert robustness(sample_heatmap) >= 0.0
