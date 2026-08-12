"""Explanation-quality metrics and anomaly/review classification.

These functions are deterministic and operate on plain NumPy arrays so they can
be unit-tested offline without a GPU, a trained model, or dataset downloads.
"""

import numpy as np


def _check_mask_matches_image(image, mask):
    """Raise ``ValueError`` if a 2D mask does not match the image height/width."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H, W); got shape {mask.shape}")
    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image spatial shape {image.shape[:2]}"
        )


def calculate_fidelity(original_image, mask, predicted_class, predict_fn):
    """Confidence drop for ``predicted_class`` when masked pixels are blacked out.

    A higher value means the highlighted region mattered more to the prediction.
    The value can be negative, which is a legitimate signal to review rather than
    an impossible outcome.
    """
    mask = np.asarray(mask)
    original_image = np.asarray(original_image)
    _check_mask_matches_image(original_image, mask)

    masked_image = original_image.copy()
    masked_image[mask > 0] = 0
    original_probability = predict_fn(np.array([original_image]))[0][predicted_class]
    masked_probability = predict_fn(np.array([masked_image]))[0][predicted_class]
    return float(original_probability - masked_probability)


def calculate_spread(mask):
    """Fraction of the image flagged as important (mask coverage in [0, 1])."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H, W); got shape {mask.shape}")
    total_pixels = mask.shape[0] * mask.shape[1]
    return float(np.sum(mask > 0) / total_pixels)


def mean_intensity(heatmap):
    """Average activation magnitude across the heatmap."""
    return float(np.mean(heatmap))


def robustness(heatmap):
    """Spatial spread of activation, measured as the heatmap standard deviation."""
    return float(np.std(heatmap))


def is_degenerate(heatmap, tol=1e-8):
    """True when a heatmap carries essentially no activation (all-zero / flat).

    This is the documented small-model edge case where every gradient-weighted
    activation was negative before ReLU.
    """
    heatmap = np.asarray(heatmap, dtype=np.float64)
    return bool(float(np.max(heatmap)) <= tol)


def review_status(fidelity, heatmap):
    """Classify an explanation as ``"ok"`` or ``"review_required"``.

    A degenerate heatmap or a non-positive fidelity is a quality signal worth a
    human review; it is not automatically a software defect.
    """
    if is_degenerate(heatmap) or fidelity <= 0:
        return "review_required"
    return "ok"
