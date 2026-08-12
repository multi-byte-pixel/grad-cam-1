#!/usr/bin/env python3
"""Stack the tiny-CNN and ResNet-18 figures for the same image into one PNG.

Run:  python scripts/generate_comparison.py
Reads results/tinycnn and results/resnet18, writes results/comparisons.
"""

import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "results", "tinycnn")
RESNET = os.path.join(ROOT, "results", "resnet18")
OUT = os.path.join(ROOT, "results", "comparisons")

METHOD_LABEL = {"gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++"}


def main():
    os.makedirs(OUT, exist_ok=True)
    for dataset in ("CIFAR10", "STL10"):
        for idx in range(10):
            for slug in ("gradcam", "gradcampp"):
                fname = f"{dataset}_{slug}_image_{idx}.png"
                tiny_png = os.path.join(TINY, dataset, fname)
                resnet_png = os.path.join(RESNET, dataset, fname)
                if not (os.path.exists(tiny_png) and os.path.exists(resnet_png)):
                    print("skip (missing):", fname)
                    continue

                caption = f"{dataset} image {idx} ({METHOD_LABEL[slug]})"
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                axes[0].imshow(mpimg.imread(tiny_png))
                axes[0].set_ylabel("Tiny CNN\n(from scratch)", fontsize=12, fontweight="bold",
                                   rotation=0, ha="right", va="center", labelpad=40)
                axes[1].imshow(mpimg.imread(resnet_png))
                axes[1].set_ylabel("ResNet-18\n(pretrained)", fontsize=12, fontweight="bold",
                                   rotation=0, ha="right", va="center", labelpad=40)
                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                fig.suptitle(f"{caption}: from-scratch CNN vs pretrained ResNet-18", fontsize=13)
                fig.tight_layout(rect=(0, 0, 1, 0.97))
                out_path = os.path.join(OUT, f"compare_{dataset}_{slug}_image_{idx}.png")
                fig.savefig(out_path, dpi=200)
                plt.close(fig)
                print("wrote", out_path)


if __name__ == "__main__":
    main()
