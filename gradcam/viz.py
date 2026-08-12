"""Visualization helpers and the model prediction closure used for fidelity."""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from .data import IMAGENET_MEAN, IMAGENET_STD


def create_predict_function(model, device=None, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Return a closure mapping a batch of HWC [0, 1] images to class probabilities."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    normalize = transforms.Normalize(mean=mean, std=std)

    def predict(images):
        batch = []
        for image in images:
            tensor = torch.tensor(np.asarray(image), dtype=torch.float32).permute(2, 0, 1)
            batch.append(normalize(tensor))
        batch = torch.stack(batch).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1)
        return probs.cpu().numpy()

    return predict


def enhance_with_blur(original_image, mask):
    """Keep masked regions sharp, blur the background, and outline the region."""
    image_uint8 = (original_image * 255).astype(np.uint8)
    binary_mask = np.where(mask.astype(np.uint8) > 0, 255, 0).astype(np.uint8)
    blurred = cv2.GaussianBlur(image_uint8, ksize=(17, 17), sigmaX=0)
    mask_3 = np.stack([binary_mask, binary_mask, binary_mask], axis=-1)
    enhanced = np.where(mask_3 == 255, image_uint8, blurred)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(enhanced, contours, -1, color=(0, 255, 0), thickness=2)
    return enhanced / 255.0
