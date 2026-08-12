"""Dataset constants and image (de)normalization helpers."""

import numpy as np
import torchvision.transforms as transforms

# ImageNet normalization statistics, shared by the from-scratch CNN and the
# pretrained ResNet-18 backbone.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]


def build_transform(size):
    """Resize -> tensor -> ImageNet-normalize transform for a square input."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def unnormalize_image(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Convert a normalized CHW tensor back to a viewable HWC image in [0, 1]."""
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * np.array(std) + np.array(mean)
    return np.clip(image, 0, 1)
