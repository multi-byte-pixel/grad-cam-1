"""Contract tests for the downstream AWS handoff interface stubs.

These assert the *shape* of the future IoT-CV QA -> AWS pipeline seam: the
record payload is well-formed and every transport function is an explicit,
not-yet-implemented stub (so the boundary is documented and discoverable).
"""

import pytest

from aws.pipeline_handoff import (
    QAFrameRecord,
    publish_frame_verdict,
    register_qa_batch_for_etl,
    stream_inference_record,
    trigger_retraining_job,
    upload_flagged_frame,
)


def _record():
    return QAFrameRecord(
        frame_id="f-001",
        device_id="cam-A",
        timestamp_utc="2026-08-12T00:00:00Z",
        predicted_class="ship",
        confidence=0.92,
        qa_verdict="flag",
        attention_summary={"spread": 0.2, "fidelity": 0.66},
        heatmap_uri=None,
    )


def test_record_is_populated_and_immutable():
    record = _record()
    assert record.frame_id == "f-001"
    assert 0.0 <= record.confidence <= 1.0
    assert record.attention_summary["fidelity"] == pytest.approx(0.66)
    with pytest.raises(Exception):
        record.confidence = 0.1  # frozen dataclass must reject mutation


def test_record_defaults_are_safe():
    record = QAFrameRecord(
        frame_id="f-002",
        device_id="cam-B",
        timestamp_utc="2026-08-12T00:00:01Z",
        predicted_class="cat",
        confidence=0.5,
        qa_verdict="pass",
    )
    assert record.attention_summary == {}
    assert record.heatmap_uri is None


def test_transport_functions_are_explicit_stubs():
    record = _record()
    with pytest.raises(NotImplementedError):
        publish_frame_verdict(record, topic="qa/verdicts")
    with pytest.raises(NotImplementedError):
        stream_inference_record(record, stream_name="qa-inference")
    with pytest.raises(NotImplementedError):
        upload_flagged_frame(record, bucket="qa-zone", frame_bytes=b"")
    with pytest.raises(NotImplementedError):
        register_qa_batch_for_etl([record], database="qa", table="frames")
    with pytest.raises(NotImplementedError):
        trigger_retraining_job(dataset_uri="s3://qa/ds", base_model_uri="s3://qa/m")
