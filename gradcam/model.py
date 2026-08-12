"""Model definitions: a small from-scratch CNN and a pretrained ResNet-18."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ImageCNN(nn.Module):
    """Compact 3-conv classifier used as the from-scratch baseline.

    The final convolution (``conv3``) is the Grad-CAM target layer.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.30)
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def build_resnet(num_classes=10):
    """ImageNet-pretrained ResNet-18 with a fresh classifier head.

    The Grad-CAM target layer is ``model.layer4``.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
