"""Grad-CAM / Grad-CAM++ heatmap computation and visualization helpers."""

import cv2
import numpy as np
import torch


def normalize_heatmap(cam):
    """Shift-and-scale a raw activation map to [0, 1].

    A constant or all-zero map normalizes to all zeros without producing NaN,
    infinity, or a divide-by-zero. This is the degenerate case that Grad-CAM can
    legitimately produce for a weak model.
    """
    cam = np.asarray(cam, dtype=np.float64)
    cam = cam - cam.min()
    max_val = cam.max()
    if max_val > 0:
        cam = cam / max_val
    return cam


def gradcam_binary_mask(heatmap, percentile=80):
    """Keep the top ``(100 - percentile)``% most-activated pixels as a 0/1 mask."""
    threshold = np.percentile(heatmap, percentile)
    return (heatmap >= threshold).astype(np.uint8)


def overlay_heatmap(original_image, heatmap):
    """Blend a JET-colored heatmap over the original RGB image, clipped to [0, 1]."""
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip(0.5 * original_image + 0.5 * heatmap_color, 0, 1)


def compute_gradcam_heatmap(
    model,
    image_tensor,
    target_layer,
    input_size,
    target_class=None,
    method="Grad-CAM",
    device=None,
):
    """Compute a Grad-CAM or Grad-CAM++ heatmap for a single image.

    Registers forward/backward hooks on ``target_layer``, backpropagates the
    target class score, and returns ``(heatmap, target_class)`` where ``heatmap``
    is a ``(input_size, input_size)`` array normalized to [0, 1].
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    activations, gradients = {}, {}

    def forward_hook(_module, _inp, out):
        activations["value"] = out.detach()

    def backward_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        input_batch = image_tensor.unsqueeze(0).to(device)
        output = model(input_batch)
        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())
        model.zero_grad()
        output[0, target_class].backward()

        feature_maps = activations["value"][0]
        grad_maps = gradients["value"][0]

        if method == "Grad-CAM++":
            grads_2 = grad_maps ** 2
            grads_3 = grad_maps ** 3
            sum_activations = feature_maps.sum(dim=(1, 2), keepdim=True)
            eps = 1e-8
            alpha = grads_2 / (2 * grads_2 + sum_activations * grads_3 + eps)
            alpha = torch.where(grad_maps != 0, alpha, torch.zeros_like(alpha))
            weights = (alpha * torch.relu(grad_maps)).sum(dim=(1, 2))
        elif method == "Grad-CAM":
            weights = grad_maps.mean(dim=(1, 2))
        else:
            raise ValueError(f"Unknown method: {method!r}; expected 'Grad-CAM' or 'Grad-CAM++'")

        cam = torch.relu((weights[:, None, None] * feature_maps).sum(dim=0))
        cam = normalize_heatmap(cam.cpu().numpy())
        cam = cv2.resize(cam.astype(np.float32), (input_size, input_size))
        return cam, target_class
    finally:
        forward_handle.remove()
        backward_handle.remove()
