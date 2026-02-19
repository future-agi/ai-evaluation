"""Tests for fi.evals.framework.types module."""

import pytest
from datetime import datetime, timezone
from fi.evals.framework.types import (
    ExecutionMode,
    EvalStatus,
    EvalResult,
    BatchEvalResult,
)


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_blocking_mode(self):
        """Test blocking mode value."""
        assert ExecutionMode.BLOCKING.value == "blocking"

    def test_non_blocking_mode(self):
        """Test non-blocking mode value."""
        assert ExecutionMode.NON_BLOCKING.value == "non_blocking"

    def test_distributed_mode(self):
        """Test distributed mode value."""
        assert ExecutionMode.DISTRIBUTED.value == "distributed"

    def test_all_modes_unique(self):
        """Test all modes have unique values."""
        values = [m.value for m in ExecutionMode]
        assert len(values) == len(set(values))


class TestEvalStatus:
    """Tests for EvalStatus enum."""

    def test_all_statuses(self):
        """Test all status values exist."""
        statuses = [
            EvalStatus.PENDING,
            EvalStatus.RUNNING,
            EvalStatus.COMPLETED,
            EvalStatus.FAILED,
            EvalStatus.CANCELLED,
            EvalStatus.TIMEOUT,
        ]
        assert len(statuses) == 6

    def test_status_values(self):
        """Test status string values."""
        assert EvalStatus.COMPLETED.value == "completed"
        assert EvalStatus.FAILED.value == "failed"


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_create_basic_result(self):
        """Test creating a basic result."""
        result = EvalResult(
            value={"score": 0.95},
            eval_name="faithfulness",
            eval_version="1.0.0",
            latency_ms=150.5,
        )

        assert result.value == {"score": 0.95}
        assert result.eval_name == "faithfulness"
        assert result.eval_version == "1.0.0"
        assert result.latency_ms == 150.5
        assert result.status == EvalStatus.COMPLETED
        assert result.error is None

    def test_succeeded_property(self):
        """Test succeeded property."""
        result = EvalResult(
            value={},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10,
            status=EvalStatus.COMPLETED,
        )
        assert result.succeeded is True
        assert result.failed is False

    def test_failed_property(self):
        """Test failed property."""
        result = EvalResult(
            value=None,
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10,
            status=EvalStatus.FAILED,
            error="Something went wrong",
        )
        assert result.succeeded is False
        assert result.failed is True

    def test_to_dict(self):
        """Test serialization to dict."""
        result = EvalResult(
            value={"score": 0.8},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=100,
        )
        data = result.to_dict()

        assert data["value"] == {"score": 0.8}
        assert data["eval_name"] == "test"
        assert data["eval_version"] == "1.0.0"
        assert data["latency_ms"] == 100
        assert data["status"] == "completed"
        assert "timestamp" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "value": {"score": 0.9},
            "eval_name": "test",
            "eval_version": "2.0.0",
            "latency_ms": 200,
            "status": "failed",
            "error": "Test error",
            "timestamp": "2024-01-01T00:00:00",
        }
        result = EvalResult.from_dict(data)

        assert result.value == {"score": 0.9}
        assert result.eval_name == "test"
        assert result.eval_version == "2.0.0"
        assert result.status == EvalStatus.FAILED
        assert result.error == "Test error"

    def test_roundtrip_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        original = EvalResult(
            value={"nested": {"data": [1, 2, 3]}},
            eval_name="complex",
            eval_version="1.0.0",
            latency_ms=50.5,
            metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = EvalResult.from_dict(data)

        assert restored.value == original.value
        assert restored.eval_name == original.eval_name
        assert restored.latency_ms == original.latency_ms

    def test_to_span_attributes(self):
        """Test conversion to span attributes."""
        result = EvalResult(
            value={},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=100,
            status=EvalStatus.FAILED,
            error="Error message",
        )
        attrs = result.to_span_attributes()

        assert attrs["eval_name"] == "test"
        assert attrs["latency_ms"] == 100
        assert attrs["status"] == "failed"
        assert attrs["error"] == "Error message"

    def test_failure_factory(self):
        """Test failure factory method."""
        result = EvalResult.failure(
            eval_name="test",
            eval_version="1.0.0",
            error="Something broke",
            latency_ms=5.0,
        )

        assert result.value is None
        assert result.status == EvalStatus.FAILED
        assert result.error == "Something broke"
        assert result.latency_ms == 5.0

    def test_metadata_default(self):
        """Test that metadata defaults to empty dict."""
        result = EvalResult(
            value={},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10,
        )
        assert result.metadata == {}
        # Ensure it's a new dict each time
        result2 = EvalResult(
            value={},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10,
        )
        assert result.metadata is not result2.metadata

    def test_timestamp_auto_set(self):
        """Test that timestamp is auto-set."""
        before = datetime.now(timezone.utc)
        result = EvalResult(
            value={},
            eval_name="test",
            eval_version="1.0.0",
            latency_ms=10,
        )
        after = datetime.now(timezone.utc)

        assert before <= result.timestamp <= after


class TestBatchEvalResult:
    """Tests for BatchEvalResult dataclass."""

    def test_create_from_results(self):
        """Test creating batch result from list."""
        results = [
            EvalResult(value={}, eval_name="a", eval_version="1.0.0", latency_ms=100),
            EvalResult(value={}, eval_name="b", eval_version="1.0.0", latency_ms=200),
            EvalResult(
                value=None,
                eval_name="c",
                eval_version="1.0.0",
                latency_ms=50,
                status=EvalStatus.FAILED,
            ),
        ]
        batch = BatchEvalResult.from_results(results)

        assert batch.total_count == 3
        assert batch.success_count == 2
        assert batch.failure_count == 1
        assert batch.total_latency_ms == 350

    def test_success_rate(self):
        """Test success rate calculation."""
        results = [
            EvalResult(value={}, eval_name="a", eval_version="1.0.0", latency_ms=100),
            EvalResult(
                value=None,
                eval_name="b",
                eval_version="1.0.0",
                latency_ms=100,
                status=EvalStatus.FAILED,
            ),
        ]
        batch = BatchEvalResult.from_results(results)

        assert batch.success_rate == 0.5

    def test_success_rate_empty(self):
        """Test success rate with no results."""
        batch = BatchEvalResult.from_results([])
        assert batch.success_rate == 0.0

    def test_avg_latency(self):
        """Test average latency calculation."""
        results = [
            EvalResult(value={}, eval_name="a", eval_version="1.0.0", latency_ms=100),
            EvalResult(value={}, eval_name="b", eval_version="1.0.0", latency_ms=200),
        ]
        batch = BatchEvalResult.from_results(results)

        assert batch.avg_latency_ms == 150.0

    def test_avg_latency_empty(self):
        """Test average latency with no results."""
        batch = BatchEvalResult.from_results([])
        assert batch.avg_latency_ms == 0.0

    def test_get_by_name(self):
        """Test filtering by eval name."""
        results = [
            EvalResult(value={"v": 1}, eval_name="a", eval_version="1.0.0", latency_ms=100),
            EvalResult(value={"v": 2}, eval_name="b", eval_version="1.0.0", latency_ms=100),
            EvalResult(value={"v": 3}, eval_name="a", eval_version="1.0.0", latency_ms=100),
        ]
        batch = BatchEvalResult.from_results(results)

        a_results = batch.get_by_name("a")
        assert len(a_results) == 2
        assert all(r.eval_name == "a" for r in a_results)

    def test_get_failures(self):
        """Test getting failed results."""
        results = [
            EvalResult(value={}, eval_name="a", eval_version="1.0.0", latency_ms=100),
            EvalResult(
                value=None,
                eval_name="b",
                eval_version="1.0.0",
                latency_ms=100,
                status=EvalStatus.FAILED,
                error="Error 1",
            ),
            EvalResult(
                value=None,
                eval_name="c",
                eval_version="1.0.0",
                latency_ms=100,
                status=EvalStatus.FAILED,
                error="Error 2",
            ),
        ]
        batch = BatchEvalResult.from_results(results)

        failures = batch.get_failures()
        assert len(failures) == 2
        assert all(f.status == EvalStatus.FAILED for f in failures)

    def test_to_dict(self):
        """Test serialization to dict."""
        results = [
            EvalResult(value={"x": 1}, eval_name="a", eval_version="1.0.0", latency_ms=100),
        ]
        batch = BatchEvalResult.from_results(results, source="test")

        data = batch.to_dict()
        assert data["total_count"] == 1
        assert data["success_count"] == 1
        assert len(data["results"]) == 1
        assert data["metadata"]["source"] == "test"

    def test_metadata_passthrough(self):
        """Test that metadata is passed through from_results."""
        batch = BatchEvalResult.from_results(
            [],
            run_id="abc123",
            environment="test",
        )
        assert batch.metadata["run_id"] == "abc123"
        assert batch.metadata["environment"] == "test"
