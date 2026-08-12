"""Anomaly / failure-mode tests.

Unusual explanation behavior is treated as an observable quality-review signal,
not automatically as a software defect.
"""

import numpy as np

from gradcam.explain import gradcam_binary_mask, normalize_heatmap, overlay_heatmap
from gradcam.metrics import is_degenerate, review_status


def test_all_zero_heatmap_is_flagged_degenerate(degenerate_heatmap):
    assert is_degenerate(degenerate_heatmap) is True


def test_healthy_heatmap_is_not_degenerate(sample_heatmap):
    assert is_degenerate(sample_heatmap) is False


def test_degenerate_heatmap_does_not_crash_downstream(degenerate_heatmap, sample_image):
    # Normalization, masking, and overlay must all remain finite and non-crashing.
    normalized = normalize_heatmap(degenerate_heatmap)
    mask = gradcam_binary_mask(normalized, percentile=80)
    overlay = overlay_heatmap(sample_image, normalized)

    assert np.all(np.isfinite(normalized))
    assert set(np.unique(mask)).issubset({0, 1})
    assert np.all(np.isfinite(overlay))


def test_non_positive_fidelity_is_flagged_for_review(sample_heatmap):
    assert review_status(fidelity=-0.1, heatmap=sample_heatmap) == "review_required"
    assert review_status(fidelity=0.0, heatmap=sample_heatmap) == "review_required"


def test_degenerate_heatmap_is_flagged_for_review(degenerate_heatmap):
    assert review_status(fidelity=0.8, heatmap=degenerate_heatmap) == "review_required"


def test_healthy_explanation_passes_review(sample_heatmap):
    assert review_status(fidelity=0.5, heatmap=sample_heatmap) == "ok"
