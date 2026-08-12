#!/usr/bin/env python3
"""Fine-tune a pretrained ResNet-18 and export Grad-CAM / Grad-CAM++ figures.

Run:  python scripts/run_resnet.py
Outputs figures to results/resnet18/<DATASET>/ and results/resnet18/metrics.csv.
The ImageNet backbone yields semantically richer features, so heat lands on the
object and the degenerate all-zero heatmaps of the tiny CNN disappear.
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradcam.data import CIFAR10_CLASSES, STL10_CLASSES, build_transform, unnormalize_image
from gradcam.explain import compute_gradcam_heatmap, gradcam_binary_mask, overlay_heatmap
from gradcam.metrics import calculate_fidelity, calculate_spread, mean_intensity, robustness
from gradcam.model import build_resnet
from gradcam.viz import create_predict_function, enhance_with_blur

INPUT_SIZE = 160
EPOCHS_CIFAR = 12
EPOCHS_STL = 20
NUM_IMAGES = 10
OUTPUT_FOLDER = os.path.join("results", "resnet18")
CHECKPOINT_FOLDER = "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, dataset_name, epochs, lr=3e-4):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"{dataset_name} | Epoch {epoch + 1}/{epochs} | Loss: {running_loss / len(train_loader):.4f}")
    return model


def test_model(model, test_loader, dataset_name):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"{dataset_name} Test Accuracy: {accuracy:.2f}%")
    return accuracy


def explain_one_image(model, test_dataset, class_names, dataset_name, image_index, method):
    image_tensor, true_label = test_dataset[image_index]
    original_image = unnormalize_image(image_tensor)
    predict_fn = create_predict_function(model, DEVICE)

    start = time.time()
    heatmap, predicted_class = compute_gradcam_heatmap(
        model, image_tensor, model.layer4, INPUT_SIZE, target_class=None, method=method, device=DEVICE
    )
    runtime = time.time() - start

    mask = gradcam_binary_mask(heatmap, percentile=80)
    overlay = overlay_heatmap(original_image, heatmap)
    enhanced = enhance_with_blur(original_image, mask)

    fidelity = calculate_fidelity(original_image, mask, predicted_class, predict_fn)
    spread = calculate_spread(mask)

    method_slug = "gradcampp" if method == "Grad-CAM++" else "gradcam"
    dataset_folder = os.path.join(OUTPUT_FOLDER, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)
    save_path = os.path.join(dataset_folder, f"{dataset_name}_{method_slug}_image_{image_index}.png")

    plt.figure(figsize=(12, 4))
    for pos, (img, title) in enumerate((
        (original_image, f"{dataset_name} Original\nTrue: {class_names[true_label]}"),
        (overlay, f"{method} Heatmap\nPred: {class_names[predicted_class]}"),
        (enhanced, "Attention Mask\nHighlight + Blur"),
    ), start=1):
        plt.subplot(1, 3, pos)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"{dataset_name} img {image_index} {method}: true={class_names[true_label]} "
          f"pred={class_names[predicted_class]} fid={fidelity:.3f} spread={spread:.3f}")

    return {
        "dataset": dataset_name, "image_index": image_index, "method": method,
        "true_class": class_names[true_label], "predicted_class": class_names[predicted_class],
        "fidelity_threshold": float(fidelity), "spread_threshold": float(spread),
        "mean_intensity": mean_intensity(heatmap), "robustness": robustness(heatmap),
        "runtime": float(runtime), "save_path": save_path.replace(os.sep, "/"),
    }


def gradcam_images(model, test_dataset, class_names, dataset_name):
    results = []
    for image_index in range(NUM_IMAGES):
        for method in ("Grad-CAM", "Grad-CAM++"):
            results.append(explain_one_image(model, test_dataset, class_names, dataset_name, image_index, method))
    return results


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
    transform = build_transform(INPUT_SIZE)

    cifar_train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    cifar_test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    stl_train = torchvision.datasets.STL10("./data", split="train", download=True, transform=transform)
    stl_test = torchvision.datasets.STL10("./data", split="test", download=True, transform=transform)

    cifar_model = train_model(build_resnet(10), DataLoader(cifar_train, batch_size=128, shuffle=True, num_workers=2), "CIFAR-10", EPOCHS_CIFAR)
    test_model(cifar_model, DataLoader(cifar_test, batch_size=128, num_workers=2), "CIFAR-10")
    torch.save(cifar_model.state_dict(), os.path.join(CHECKPOINT_FOLDER, "cifar10_resnet18.pt"))

    stl_model = train_model(build_resnet(10), DataLoader(stl_train, batch_size=128, shuffle=True, num_workers=2), "STL-10", EPOCHS_STL)
    test_model(stl_model, DataLoader(stl_test, batch_size=128, num_workers=2), "STL-10")
    torch.save(stl_model.state_dict(), os.path.join(CHECKPOINT_FOLDER, "stl10_resnet18.pt"))

    results = gradcam_images(cifar_model, cifar_test, CIFAR10_CLASSES, "CIFAR10")
    results += gradcam_images(stl_model, stl_test, STL10_CLASSES, "STL10")

    df = pd.DataFrame(results)
    print("\nResNet-18 Grad-CAM averages by dataset and method:")
    print(df.groupby(["dataset", "method"])[
        ["fidelity_threshold", "spread_threshold", "mean_intensity", "robustness", "runtime"]
    ].mean().round(4))
    df.to_csv(os.path.join(OUTPUT_FOLDER, "metrics.csv"), index=False)
    print("\nSaved metrics CSV:", os.path.join(OUTPUT_FOLDER, "metrics.csv"))


if __name__ == "__main__":
    main()
