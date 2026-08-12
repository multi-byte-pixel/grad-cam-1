"""Downstream handoff interface: IoT edge CV QA -> AWS big-data / ML pipeline.

This module is an **interface stub only**. It defines the shape of the handoff
between an edge inspection device (running the Grad-CAM QA explainer in this
repo) and a cloud big-data / retraining pipeline. Nothing here calls AWS; every
function raises :class:`NotImplementedError`. It exists to document the intended
seam for a future automotive / IoT computer-vision QA deployment.

Intended flow (see the README "Future work" section):

    edge camera + QA model
        -> IoT Core / Greengrass   (publish per-frame QA verdict)
        -> Kinesis                 (stream inference records)
        -> S3                      (land raw + flagged-defect frames + heatmaps)
        -> Glue / Athena           (ETL + queryable QA analytics)
        -> SageMaker               (retraining / batch re-inference)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class QAFrameRecord:
    """One frame's QA result emitted at the edge and carried downstream.

    Attributes:
        frame_id: Unique id for the inspected frame.
        device_id: Edge device / camera identifier.
        timestamp_utc: ISO-8601 capture time in UTC.
        predicted_class: Model's predicted label.
        confidence: Softmax confidence for ``predicted_class`` in [0, 1].
        qa_verdict: Downstream QA decision, e.g. "pass" | "flag" | "review".
        attention_summary: Compact Grad-CAM summary (e.g. spread, fidelity).
        heatmap_uri: Optional pointer (e.g. S3 URI) to the stored heatmap.
    """

    frame_id: str
    device_id: str
    timestamp_utc: str
    predicted_class: str
    confidence: float
    qa_verdict: str
    attention_summary: Mapping[str, float] = field(default_factory=dict)
    heatmap_uri: Optional[str] = None


def publish_frame_verdict(record: QAFrameRecord, *, topic: str) -> None:
    """Publish a per-frame QA verdict from the edge via IoT Core / Greengrass."""
    raise NotImplementedError("IoT Core / Greengrass publish is not implemented (stub).")


def stream_inference_record(record: QAFrameRecord, *, stream_name: str) -> None:
    """Put a single inference record onto a Kinesis stream for ingestion."""
    raise NotImplementedError("Kinesis stream put is not implemented (stub).")


def upload_flagged_frame(
    record: QAFrameRecord, *, bucket: str, frame_bytes: bytes
) -> str:
    """Land a flagged/defect frame (+ heatmap) in the S3 QA zone; return its URI."""
    raise NotImplementedError("S3 upload is not implemented (stub).")


def register_qa_batch_for_etl(
    records: Sequence[QAFrameRecord], *, database: str, table: str
) -> None:
    """Register a batch of QA records for Glue/Athena ETL + analytics."""
    raise NotImplementedError("Glue/Athena registration is not implemented (stub).")


def trigger_retraining_job(*, dataset_uri: str, base_model_uri: str) -> str:
    """Kick off a SageMaker retraining / batch-inference job; return its job id."""
    raise NotImplementedError("SageMaker job trigger is not implemented (stub).")
