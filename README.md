# Grad-CAM / Grad-CAM++ Visual QA Proof of Concept

**Author:** Jonathan Canady (JCanady)

A sanitized personal portfolio implementation based on an earlier visual-quality-assurance proof of concept. This repository demonstrates explainable computer vision for quality-validation workflows: it produces Grad-CAM and Grad-CAM++ visual explanations, validates outputs through an offline test suite, and includes an AWS-oriented handoff interface for a future IoT vision pipeline.

> Scope note: This is a standalone, sanitized portfolio project. It contains no proprietary code, data, configurations, customer information, or production credentials. The AWS pipeline components are interface-level tooling, not a claim of a deployed production system.

## What I built

- From-scratch Grad-CAM and Grad-CAM++ explainability implementations
- Offline visual-QA validation and automated tests
- Reproducible Python environment and test configuration
- AWS-oriented lifecycle and pipeline-handoff tooling for a computer-vision QA workflow

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
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
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

## Evidence of engineering quality

- Automated tests validate core explainability and quality-validation behavior
- Public datasets are used for experimentation
- AWS scripts are separated from local validation logic and include preflight, deployment, monitoring, and teardown paths
- The repository is intentionally scoped as a proof of concept; documented boundaries are part of its design

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

