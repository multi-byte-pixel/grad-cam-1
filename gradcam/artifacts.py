"""Validation of the evidence package produced by an experiment run.

Used by the offline artifact-integrity tests to confirm that a committed metrics
CSV is well-formed and that every referenced figure exists on disk.
"""

import os

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "dataset",
    "image_index",
    "method",
    "true_class",
    "predicted_class",
    "fidelity_threshold",
    "spread_threshold",
    "mean_intensity",
    "robustness",
    "runtime",
    "save_path",
]

RECOGNIZED_METHODS = {"Grad-CAM", "Grad-CAM++"}
RECOGNIZED_DATASETS = {"CIFAR10", "STL10"}
FINITE_COLUMNS = [
    "fidelity_threshold",
    "spread_threshold",
    "mean_intensity",
    "robustness",
    "runtime",
]


def load_metrics(csv_path):
    """Read a metrics CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def schema_problems(df):
    """Return a list of schema violations; empty means the schema is valid."""
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        return problems

    unknown_methods = set(df["method"]) - RECOGNIZED_METHODS
    if unknown_methods:
        problems.append(f"unrecognized methods: {sorted(unknown_methods)}")

    unknown_datasets = set(df["dataset"]) - RECOGNIZED_DATASETS
    if unknown_datasets:
        problems.append(f"unrecognized datasets: {sorted(unknown_datasets)}")

    for column in FINITE_COLUMNS:
        if not np.all(np.isfinite(df[column].to_numpy(dtype=float))):
            problems.append(f"non-finite values in column: {column}")
    return problems


def missing_artifacts(df, base_dir):
    """Return ``save_path`` entries that are absent or empty on disk."""
    missing = []
    for save_path in df["save_path"]:
        full_path = os.path.join(base_dir, save_path)
        if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
            missing.append(save_path)
    return missing
