# Grad-CAM Explainability for Visual QA

**Author:** J. Canady (JCanady)

Gradient-based visual explanations (Grad-CAM and Grad-CAM++) for image
classifiers, paired with an offline quality-validation test suite. Both
explanation methods are implemented from scratch (forward/backward hooks on the
target conv layer) and compared across a from-scratch CNN and a fine-tuned
ResNet-18 on the public CIFAR-10 and STL-10 datasets.

This is a proof of concept toward an automotive / IoT computer-vision **QA**
imaging pipeline: explain a classifier's decision, then treat unusual
explanations as an inspectable quality signal. It uses public image datasets
only and does not represent performance on any production inspection system.

## Layout

```
gradcam/                 # importable library (no import-time side effects)
  data.py                # dataset constants, transforms, (de)normalization
  model.py               # ImageCNN + pretrained ResNet-18 builder
  explain.py             # Grad-CAM / Grad-CAM++ heatmaps, normalization, masks
  metrics.py             # fidelity, spread, intensity, degenerate/review checks
  viz.py                 # overlay + highlight-blur rendering, predict closure
  artifacts.py           # metrics-CSV schema + artifact-integrity validation
scripts/                 # runnable pipelines (train + explain)
  run_tinycnn.py         # from-scratch ImageCNN (96px)
  run_resnet.py          # fine-tuned ResNet-18 (160px)
  generate_comparison.py # side-by-side tiny-CNN vs ResNet figures
results/                 # committed figures + metrics.csv per model
  tinycnn/  resnet18/  comparisons/
tests/                   # offline pytest quality-validation suite
aws/                     # EC2 GPU training toolkit + downstream handoff stub
notes/                   # Grad-CAM / Grad-CAM++ paper map + terminology
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_tinycnn.py        # trains + explains (downloads datasets)
python scripts/run_resnet.py
python scripts/generate_comparison.py
```

Trained weights are written under `checkpoints/` and are intentionally not
committed (regenerate them with the scripts above).

## Tests

Offline, deterministic, CPU-only — no dataset download, GPU, training run, or
cloud access required.

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers explanation-metric correctness, degenerate/anomaly handling
(flagged for review rather than silently ignored), metrics-CSV schema and
artifact integrity, limited synthetic input-perturbation checks, and the
downstream handoff interface contract.

## Results (mean per method)

| Model | Method | Fidelity | Spread | Mean intensity | Degenerate heatmaps |
|---|---|---|---|---|---|
| Tiny-CNN | Grad-CAM | 0.583 | 0.640 | 0.043 | 4 / 20 |
| Tiny-CNN | Grad-CAM++ | 0.361 | 0.200 | 0.223 | 0 |
| ResNet-18 | Grad-CAM | 0.663 | 0.200 | 0.410 | 0 |
| ResNet-18 | Grad-CAM++ | 0.618 | 0.200 | 0.425 | 0 |

The fine-tuned ResNet-18 (CIFAR-10 ~91.8%, STL-10 ~90.2%) eliminates the
all-zero degenerate heatmaps the tiny CNN produced and concentrates attention on
the object rather than the background.

## Future work: edge-to-cloud QA pipeline

The intended production flow is an IoT computer-vision QA loop: an edge camera
runs the classifier + Grad-CAM explainer, and each frame's QA verdict flows to a
cloud big-data / ML pipeline. Those downstream components are defined as interface stubs
in [aws/pipeline_handoff.py](aws/pipeline_handoff.py) (no live AWS calls yet):

```
edge camera + QA model
  -> IoT Core / Greengrass   (publish per-frame QA verdict)
  -> Kinesis                 (stream inference records)
  -> S3                      (raw + flagged-defect frames + heatmaps)
  -> Glue / Athena           (ETL + queryable QA analytics)
  -> SageMaker               (retraining / batch re-inference)
```

The existing `aws/` scripts provide a boto3 + paramiko EC2 GPU training toolkit
used to produce the ResNet-18 results above.

## References

See [notes/paper_map.md](notes/paper_map.md) and
[notes/terminology.md](notes/terminology.md) for the Grad-CAM (ICCV 2017) and
Grad-CAM++ (WACV 2018) method summaries.
