"""Tests for streaming evaluation types."""

import pytest
from datetime import datetime, timezone

from fi.evals.streaming.types import (
    ChunkResult,
    EarlyStopCondition,
    EarlyStopReason,
    StreamingConfig,
    StreamingEvalResult,
    StreamingState,
)


class TestEarlyStopReason:
    """Tests for EarlyStopReason enum."""

    def test_all_reasons_have_values(self):
        """All reasons should have string values."""
        for reason in EarlyStopReason:
            assert isinstance(reason.value, str)
            assert len(reason.value) > 0

    def test_none_reason_exists(self):
        """NONE reason should exist for no early stop."""
        assert EarlyStopReason.NONE.value == "none"

    def test_safety_reasons_exist(self):
        """Safety-related reasons should exist."""
        assert EarlyStopReason.TOXICITY.value == "toxicity"
        assert EarlyStopReason.SAFETY.value == "safety"
        assert EarlyStopReason.PII.value == "pii"
        assert EarlyStopReason.JAILBREAK.value == "jailbreak"


class TestStreamingState:
    """Tests for StreamingState enum."""

    def test_all_states_have_values(self):
        """All states should have string values."""
        for state in StreamingState:
            assert isinstance(state.value, str)
            assert len(state.value) > 0

    def test_lifecycle_states_exist(self):
        """Lifecycle states should exist."""
        assert StreamingState.IDLE.value == "idle"
        assert StreamingState.STREAMING.value == "streaming"
        assert StreamingState.COMPLETED.value == "completed"
        assert StreamingState.STOPPED.value == "stopped"
        assert StreamingState.ERROR.value == "error"


class TestChunkResult:
    """Tests for ChunkResult dataclass."""

    def test_create_basic(self):
        """Should create with basic parameters."""
        result = ChunkResult(
            chunk_index=0,
            chunk_text="Hello",
            cumulative_text="Hello",
            scores={"toxicity": 0.1},
            flags={"toxicity": True},
        )
        assert result.chunk_index == 0
        assert result.chunk_text == "Hello"
        assert result.scores["toxicity"] == 0.1
        assert result.flags["toxicity"] is True

    def test_default_values(self):
        """Should have correct default values."""
        result = ChunkResult(
            chunk_index=0,
            chunk_text="",
            cumulative_text="",
            scores={},
            flags={},
        )
        assert result.should_stop is False
        assert result.stop_reason == EarlyStopReason.NONE
        assert result.latency_ms == 0.0
        assert isinstance(result.timestamp, datetime)
        assert result.metadata == {}

    def test_all_passed_property_true(self):
        """all_passed should be True when all flags are True."""
        result = ChunkResult(
            chunk_index=0,
            chunk_text="",
            cumulative_text="",
            scores={"a": 0.9, "b": 0.8},
            flags={"a": True, "b": True},
        )
        assert result.all_passed is True

    def test_all_passed_property_false(self):
        """all_passed should be False when any flag is False."""
        result = ChunkResult(
            chunk_index=0,
            chunk_text="",
            cumulative_text="",
            scores={"a": 0.9, "b": 0.3},
            flags={"a": True, "b": False},
        )
        assert result.all_passed is False

    def test_all_passed_empty_flags(self):
        """all_passed should be True with empty flags."""
        result = ChunkResult(
            chunk_index=0,
            chunk_text="",
            cumulative_text="",
            scores={},
            flags={},
        )
        assert result.all_passed is True

    def test_min_score_property(self):
        """min_score should return minimum score value."""
        result = ChunkResult(
            chunk_index=0,
            chunk_text="",
            cumulative_text="",
            scores={"a": 0.9, "b": 0.3, "c": 0.7},
            flags={},
        )
        assert result.min_score == 0.3

    def test_min_score_empty_scores(self):
        """min_score should return 1.0 with empty scores."""
        result = ChunkResult(
            chunk_index=0,
            chunk_text="",
            cumulative_text="",
            scores={},
            flags={},
        )
        assert result.min_score == 1.0

    def test_to_dict(self):
        """to_dict should serialize properly."""
        result = ChunkResult(
            chunk_index=0,
            chunk_text="test",
            cumulative_text="test",
            scores={"a": 0.5},
            flags={"a": True},
            should_stop=True,
            stop_reason=EarlyStopReason.TOXICITY,
        )
        d = result.to_dict()
        assert d["chunk_index"] == 0
        assert d["chunk_text"] == "test"
        assert d["scores"] == {"a": 0.5}
        assert d["should_stop"] is True
        assert d["stop_reason"] == "toxicity"


class TestStreamingEvalResult:
    """Tests for StreamingEvalResult dataclass."""

    def test_create_basic(self):
        """Should create with basic parameters."""
        result = StreamingEvalResult(
            passed=True,
            final_text="Hello world",
            total_chunks=2,
            chunk_results=[],
            final_scores={"toxicity": 0.1},
        )
        assert result.passed is True
        assert result.final_text == "Hello world"
        assert result.total_chunks == 2

    def test_default_values(self):
        """Should have correct default values."""
        result = StreamingEvalResult(
            passed=True,
            final_text="",
            total_chunks=0,
            chunk_results=[],
            final_scores={},
        )
        assert result.early_stopped is False
        assert result.stop_reason == EarlyStopReason.NONE
        assert result.stopped_at_chunk is None
        assert result.total_latency_ms == 0.0
        assert result.state == StreamingState.COMPLETED

    def test_average_chunk_latency(self):
        """average_chunk_latency_ms should calculate correctly."""
        chunks = [
            ChunkResult(0, "", "", {}, {}, latency_ms=10),
            ChunkResult(1, "", "", {}, {}, latency_ms=20),
            ChunkResult(2, "", "", {}, {}, latency_ms=30),
        ]
        result = StreamingEvalResult(
            passed=True,
            final_text="",
            total_chunks=3,
            chunk_results=chunks,
            final_scores={},
        )
        assert result.average_chunk_latency_ms == 20.0

    def test_average_chunk_latency_empty(self):
        """average_chunk_latency_ms should be 0 with no chunks."""
        result = StreamingEvalResult(
            passed=True,
            final_text="",
            total_chunks=0,
            chunk_results=[],
            final_scores={},
        )
        assert result.average_chunk_latency_ms == 0.0

    def test_min_score_history(self):
        """min_score_history should return list of min scores."""
        chunks = [
            ChunkResult(0, "", "", {"a": 0.9, "b": 0.8}, {}),
            ChunkResult(1, "", "", {"a": 0.7, "b": 0.6}, {}),
            ChunkResult(2, "", "", {"a": 0.5, "b": 0.9}, {}),
        ]
        result = StreamingEvalResult(
            passed=True,
            final_text="",
            total_chunks=3,
            chunk_results=chunks,
            final_scores={},
        )
        assert result.min_score_history == [0.8, 0.6, 0.5]

    def test_score_by_eval(self):
        """score_by_eval should group scores by evaluation."""
        chunks = [
            ChunkResult(0, "", "", {"a": 0.9, "b": 0.8}, {}),
            ChunkResult(1, "", "", {"a": 0.7, "b": 0.6}, {}),
        ]
        result = StreamingEvalResult(
            passed=True,
            final_text="",
            total_chunks=2,
            chunk_results=chunks,
            final_scores={},
        )
        scores = result.score_by_eval
        assert scores["a"] == [0.9, 0.7]
        assert scores["b"] == [0.8, 0.6]

    def test_summary(self):
        """summary should generate readable output."""
        result = StreamingEvalResult(
            passed=True,
            final_text="Hello world",
            total_chunks=2,
            chunk_results=[],
            final_scores={"toxicity": 0.1},
            total_latency_ms=100.0,
        )
        summary = result.summary()
        assert "PASSED" in summary
        assert "Total Chunks: 2" in summary
        assert "toxicity: 0.100" in summary

    def test_summary_with_early_stop(self):
        """summary should include early stop info."""
        result = StreamingEvalResult(
            passed=False,
            final_text="Hello",
            total_chunks=1,
            chunk_results=[],
            final_scores={},
            early_stopped=True,
            stop_reason=EarlyStopReason.TOXICITY,
            stopped_at_chunk=0,
        )
        summary = result.summary()
        assert "FAILED" in summary
        assert "Early Stopped: Yes" in summary
        assert "toxicity" in summary

    def test_to_dict(self):
        """to_dict should serialize properly."""
        result = StreamingEvalResult(
            passed=True,
            final_text="test",
            total_chunks=1,
            chunk_results=[],
            final_scores={"a": 0.5},
            early_stopped=True,
            stop_reason=EarlyStopReason.SAFETY,
        )
        d = result.to_dict()
        assert d["passed"] is True
        assert d["final_text"] == "test"
        assert d["early_stopped"] is True
        assert d["stop_reason"] == "safety"


class TestStreamingConfig:
    """Tests for StreamingConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = StreamingConfig()
        assert config.min_chunk_size == 1
        assert config.max_chunk_size == 100
        assert config.eval_interval_ms == 100
        assert config.enable_early_stop is True
        assert config.stop_on_first_failure is False

    def test_custom_values(self):
        """Should accept custom values."""
        config = StreamingConfig(
            min_chunk_size=10,
            max_chunk_size=500,
            eval_interval_ms=50,
            max_tokens=1000,
            enable_early_stop=False,
        )
        assert config.min_chunk_size == 10
        assert config.max_chunk_size == 500
        assert config.max_tokens == 1000
        assert config.enable_early_stop is False

    def test_callbacks(self):
        """Should accept callback functions."""
        callback_called = []

        def on_chunk(result):
            callback_called.append(result)

        config = StreamingConfig(on_chunk_callback=on_chunk)
        assert config.on_chunk_callback is not None

    def test_to_dict_excludes_callbacks(self):
        """to_dict should exclude callbacks."""
        config = StreamingConfig(
            on_chunk_callback=lambda x: None,
            on_stop_callback=lambda x, y: None,
        )
        d = config.to_dict()
        assert "on_chunk_callback" not in d
        assert "on_stop_callback" not in d
        assert "min_chunk_size" in d


class TestEarlyStopCondition:
    """Tests for EarlyStopCondition dataclass."""

    def test_create_basic(self):
        """Should create with basic parameters."""
        condition = EarlyStopCondition(
            name="toxicity_stop",
            eval_name="toxicity",
            threshold=0.7,
        )
        assert condition.name == "toxicity_stop"
        assert condition.eval_name == "toxicity"
        assert condition.threshold == 0.7
        assert condition.comparison == "below"
        assert condition.consecutive_chunks == 1
        assert condition.enabled is True

    def test_check_below_threshold(self):
        """check should trigger when below threshold."""
        condition = EarlyStopCondition(
            name="test",
            eval_name="score",
            threshold=0.5,
            comparison="below",
            consecutive_chunks=1,
        )
        # Score 0.3 is below 0.5, should trigger
        assert condition.check(0.3, 1) is True
        # Score 0.7 is above 0.5, should not trigger
        assert condition.check(0.7, 1) is False

    def test_check_above_threshold(self):
        """check should trigger when above threshold."""
        condition = EarlyStopCondition(
            name="test",
            eval_name="score",
            threshold=0.5,
            comparison="above",
            consecutive_chunks=1,
        )
        # Score 0.7 is above 0.5, should trigger
        assert condition.check(0.7, 1) is True
        # Score 0.3 is below 0.5, should not trigger
        assert condition.check(0.3, 1) is False

    def test_check_consecutive_chunks(self):
        """check should require consecutive chunks."""
        condition = EarlyStopCondition(
            name="test",
            eval_name="score",
            threshold=0.5,
            comparison="below",
            consecutive_chunks=3,
        )
        # Not enough consecutive chunks
        assert condition.check(0.3, 1) is False
        assert condition.check(0.3, 2) is False
        # Enough consecutive chunks
        assert condition.check(0.3, 3) is True

    def test_check_disabled(self):
        """check should return False when disabled."""
        condition = EarlyStopCondition(
            name="test",
            eval_name="score",
            threshold=0.5,
            comparison="below",
            enabled=False,
        )
        assert condition.check(0.1, 1) is False

    def test_to_dict(self):
        """to_dict should serialize properly."""
        condition = EarlyStopCondition(
            name="toxicity",
            eval_name="toxicity_score",
            threshold=0.7,
            comparison="above",
            consecutive_chunks=2,
        )
        d = condition.to_dict()
        assert d["name"] == "toxicity"
        assert d["eval_name"] == "toxicity_score"
        assert d["threshold"] == 0.7
        assert d["comparison"] == "above"
        assert d["consecutive_chunks"] == 2
