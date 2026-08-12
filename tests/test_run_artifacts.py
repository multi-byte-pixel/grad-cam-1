"""Run-artifact integrity tests for the committed metrics CSVs and figures."""

import os

import pytest

from gradcam.artifacts import load_metrics, missing_artifacts, schema_problems

METRICS_CSVS = [
    os.path.join("results", "tinycnn", "metrics.csv"),
    os.path.join("results", "resnet18", "metrics.csv"),
]


def _csv_path(repo_root, rel):
    return os.path.join(repo_root, rel)


@pytest.mark.parametrize("rel_csv", METRICS_CSVS)
def test_metrics_csv_exists_and_is_nonempty(repo_root, rel_csv):
    path = _csv_path(repo_root, rel_csv)
    if not os.path.exists(path):
        pytest.skip(f"metrics CSV not present: {rel_csv}")
    assert os.path.getsize(path) > 0


@pytest.mark.parametrize("rel_csv", METRICS_CSVS)
def test_metrics_csv_schema_is_valid(repo_root, rel_csv):
    path = _csv_path(repo_root, rel_csv)
    if not os.path.exists(path):
        pytest.skip(f"metrics CSV not present: {rel_csv}")
    df = load_metrics(path)
    assert len(df) > 0
    problems = schema_problems(df)
    assert problems == [], f"schema problems in {rel_csv}: {problems}"


@pytest.mark.parametrize("rel_csv", METRICS_CSVS)
def test_every_referenced_artifact_exists_and_is_nonempty(repo_root, rel_csv):
    path = _csv_path(repo_root, rel_csv)
    if not os.path.exists(path):
        pytest.skip(f"metrics CSV not present: {rel_csv}")
    df = load_metrics(path)
    missing = missing_artifacts(df, repo_root)
    assert missing == [], f"missing/empty artifacts referenced by {rel_csv}: {missing[:5]}"


def test_missing_artifact_is_reported(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"save_path": ["does/not/exist.png"]})
    assert missing_artifacts(df, str(tmp_path)) == ["does/not/exist.png"]
