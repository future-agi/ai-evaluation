"""Tests for StreamingEvaluator."""

import pytest
import asyncio

from fi.evals.streaming.evaluator import StreamingEvaluator, EvalSpec
from fi.evals.streaming.types import (
    ChunkResult,
    EarlyStopReason,
    StreamingConfig,
    StreamingEvalResult,
    StreamingState,
)
from fi.evals.streaming.policy import EarlyStopPolicy


# Simple test scorers
def always_pass_scorer(chunk: str, cumulative: str) -> float:
    """Always returns high score."""
    return 0.9


def always_fail_scorer(chunk: str, cumulative: str) -> float:
    """Always returns low score."""
    return 0.1


def word_count_scorer(chunk: str, cumulative: str) -> float:
    """Returns score based on word count."""
    words = len(cumulative.split())
    return min(1.0, words / 10)


def toxic_word_scorer(chunk: str, cumulative: str) -> float:
    """Returns 1.0 if 'toxic' in text, else 0.0."""
    return 1.0 if "toxic" in cumulative.lower() else 0.0


class TestEvalSpec:
    """Tests for EvalSpec dataclass."""

    def test_create_basic(self):
        """Should create with basic parameters."""
        spec = EvalSpec(
            name="test",
            eval_fn=always_pass_scorer,
        )
        assert spec.name == "test"
        assert spec.threshold == 0.7
        assert spec.weight == 1.0
        assert spec.pass_above is True

    def test_create_custom(self):
        """Should accept custom parameters."""
        spec = EvalSpec(
            name="toxicity",
            eval_fn=always_fail_scorer,
            threshold=0.3,
            weight=2.0,
            pass_above=False,
        )
        assert spec.threshold == 0.3
        assert spec.weight == 2.0
        assert spec.pass_above is False


class TestStreamingEvaluator:
    """Tests for StreamingEvaluator."""

    def test_create_default(self):
        """Should create with default config."""
        evaluator = StreamingEvaluator()
        assert evaluator.config is not None
        assert evaluator.state == StreamingState.IDLE

    def test_create_with_config(self):
        """Should create with custom config."""
        config = StreamingConfig(min_chunk_size=50)
        evaluator = StreamingEvaluator(config=config)
        assert evaluator.config.min_chunk_size == 50

    def test_create_with_policy(self):
        """Should create with custom policy."""
        policy = EarlyStopPolicy.strict()
        evaluator = StreamingEvaluator(policy=policy)
        # Policy is set internally

    def test_add_eval(self):
        """Should add evaluation functions."""
        evaluator = StreamingEvaluator()
        result = evaluator.add_eval("test", always_pass_scorer, threshold=0.5)
        assert result is evaluator  # Chaining
        assert len(evaluator._evals) == 1

    def test_set_policy(self):
        """Should set policy."""
        evaluator = StreamingEvaluator()
        policy = EarlyStopPolicy.strict()
        result = evaluator.set_policy(policy)
        assert result is evaluator  # Chaining

    def test_process_token_starts_streaming(self):
        """process_token should start streaming state."""
        evaluator = StreamingEvaluator()
        assert evaluator.state == StreamingState.IDLE
        evaluator.process_token("Hello")
        assert evaluator.state == StreamingState.STREAMING

    def test_process_token_accumulates(self):
        """process_token should accumulate text."""
        config = StreamingConfig(min_chunk_size=100, eval_interval_ms=0)
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_pass_scorer)

        evaluator.process_token("Hello")
        evaluator.process_token(" ")
        evaluator.process_token("world")

        result = evaluator.finalize()
        assert result.final_text == "Hello world"

    def test_process_token_triggers_eval(self):
        """process_token should trigger evaluation when conditions met."""
        config = StreamingConfig(min_chunk_size=5, eval_interval_ms=0)
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_pass_scorer, threshold=0.5)

        # Not enough characters yet
        result = evaluator.process_token("Hi")
        assert result is None

        # Now enough
        result = evaluator.process_token("Hello")
        assert result is not None
        assert isinstance(result, ChunkResult)
        assert "test" in result.scores

    def test_process_chunk(self):
        """process_chunk should handle larger text."""
        config = StreamingConfig(min_chunk_size=1, eval_interval_ms=0)
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", word_count_scorer)

        result = evaluator.process_chunk("Hello world, this is a test.")
        assert result is not None
        assert result.chunk_text == "Hello world, this is a test."

    def test_finalize_returns_result(self):
        """finalize should return StreamingEvalResult."""
        config = StreamingConfig(min_chunk_size=1, eval_interval_ms=0)
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_pass_scorer)

        evaluator.process_token("Hello")
        result = evaluator.finalize()

        assert isinstance(result, StreamingEvalResult)
        assert result.final_text == "Hello"
        assert result.state == StreamingState.COMPLETED

    def test_finalize_evaluates_pending(self):
        """finalize should evaluate any pending content."""
        config = StreamingConfig(min_chunk_size=100, eval_interval_ms=0)
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_pass_scorer)

        # Add content but don't trigger eval
        evaluator.process_token("Hello")
        assert evaluator.chunk_count == 0

        # Finalize should evaluate remaining
        result = evaluator.finalize()
        assert result.total_chunks == 1

    def test_reset(self):
        """reset should clear state for new stream."""
        config = StreamingConfig(min_chunk_size=1, eval_interval_ms=0)
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_pass_scorer)

        evaluator.process_token("First stream")
        evaluator.finalize()

        evaluator.reset()
        assert evaluator.state == StreamingState.IDLE
        assert evaluator.chunk_count == 0

        evaluator.process_token("New stream")
        result = evaluator.finalize()
        assert result.final_text == "New stream"

    def test_is_stopped_property(self):
        """is_stopped should reflect state."""
        evaluator = StreamingEvaluator()
        assert evaluator.is_stopped is False

        evaluator.process_token("test")
        assert evaluator.is_stopped is False

        evaluator.finalize()
        assert evaluator.is_stopped is True

    def test_chunk_count_property(self):
        """chunk_count should track evaluations."""
        config = StreamingConfig(min_chunk_size=1, eval_interval_ms=0)
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_pass_scorer)

        assert evaluator.chunk_count == 0

        evaluator.process_token("chunk1")
        assert evaluator.chunk_count == 1

        evaluator.process_token("chunk2")
        assert evaluator.chunk_count == 2


class TestStreamingEvaluatorEarlyStop:
    """Tests for early stopping functionality."""

    def test_early_stop_on_toxicity(self):
        """Should stop early on toxic content."""
        config = StreamingConfig(min_chunk_size=1, eval_interval_ms=0, enable_early_stop=True)
        policy = EarlyStopPolicy()
        policy.add_toxicity_stop(threshold=0.5)

        evaluator = StreamingEvaluator(config=config, policy=policy)
        evaluator.add_eval("toxicity", toxic_word_scorer, threshold=0.5, pass_above=False)

        # Process safe content
        result = evaluator.process_token("Hello ")
        assert result is None or result.should_stop is False

        # Process toxic content
        result = evaluator.process_chunk("this is toxic content")
        assert result is not None
        assert result.should_stop is True
        assert result.stop_reason == EarlyStopReason.TOXICITY

    def test_stop_on_first_failure(self):
        """stop_on_first_failure should trigger immediate stop."""
        config = StreamingConfig(
            min_chunk_size=1,
            eval_interval_ms=0,
            enable_early_stop=True,
            stop_on_first_failure=True,
        )
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("quality", always_fail_scorer, threshold=0.5)

        result = evaluator.process_token("test")
        assert result.should_stop is True
        assert result.stop_reason == EarlyStopReason.THRESHOLD

    def test_early_stop_preserves_state(self):
        """Early stopped evaluation should preserve state in result."""
        config = StreamingConfig(
            min_chunk_size=1,
            eval_interval_ms=0,
            enable_early_stop=True,
            stop_on_first_failure=True,
        )
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_fail_scorer, threshold=0.5)

        evaluator.process_token("chunk1 ")
        evaluator.process_token("chunk2")

        result = evaluator.finalize()
        assert result.early_stopped is True
        assert result.state == StreamingState.STOPPED
        assert result.stopped_at_chunk is not None

    def test_disable_early_stop(self):
        """enable_early_stop=False should disable stopping."""
        config = StreamingConfig(
            min_chunk_size=1,
            eval_interval_ms=0,
            enable_early_stop=False,
        )
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_fail_scorer, threshold=0.5)

        result = evaluator.process_token("test content that would normally fail")
        assert result.should_stop is False

    def test_no_processing_after_stop(self):
        """Should not process tokens after stopping."""
        config = StreamingConfig(
            min_chunk_size=1,
            eval_interval_ms=0,
            enable_early_stop=True,
            stop_on_first_failure=True,
        )
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_fail_scorer, threshold=0.5)

        evaluator.process_token("fail")  # Triggers stop
        result = evaluator.process_token("more text")  # Should be ignored

        assert result is None


class TestStreamingEvaluatorScoring:
    """Tests for score calculation."""

    def test_final_scores_average(self):
        """Final scores should be averaged across chunks."""
        config = StreamingConfig(min_chunk_size=1, eval_interval_ms=0)
        evaluator = StreamingEvaluator(config=config)

        # Scorer that returns different values based on length
        def length_scorer(chunk: str, cumulative: str) -> float:
            return min(1.0, len(cumulative) / 20)

        evaluator.add_eval("length", length_scorer)

        evaluator.process_token("12345")  # Score ~0.25
        evaluator.process_token("12345")  # Score ~0.5
        evaluator.process_token("12345")  # Score ~0.75

        result = evaluator.finalize()
        # Average should be somewhere in between
        assert 0.3 < result.final_scores["length"] < 0.8

    def test_passed_based_on_final_scores(self):
        """passed should be based on final scores vs thresholds."""
        config = StreamingConfig(min_chunk_size=1, eval_interval_ms=0, enable_early_stop=False)
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_pass_scorer, threshold=0.5)

        evaluator.process_token("test")
        result = evaluator.finalize()
        assert result.passed is True

    def test_failed_based_on_final_scores(self):
        """passed should be False when scores below threshold."""
        config = StreamingConfig(min_chunk_size=1, eval_interval_ms=0, enable_early_stop=False)
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_fail_scorer, threshold=0.5)

        evaluator.process_token("test")
        result = evaluator.finalize()
        assert result.passed is False

    def test_pass_above_false(self):
        """Should handle pass_above=False (lower is better)."""
        config = StreamingConfig(min_chunk_size=1, eval_interval_ms=0, enable_early_stop=False)
        evaluator = StreamingEvaluator(config=config)
        # For toxicity, lower is better
        evaluator.add_eval("toxicity", always_fail_scorer, threshold=0.5, pass_above=False)

        evaluator.process_token("safe content")
        result = evaluator.finalize()
        # Score is 0.1, threshold is 0.5, pass_above=False means score <= threshold passes
        assert result.passed is True


class TestStreamingEvaluatorCallbacks:
    """Tests for callback functionality."""

    def test_on_chunk_callback(self):
        """on_chunk_callback should be called for each evaluation."""
        chunks_received = []

        def on_chunk(chunk_result):
            chunks_received.append(chunk_result)

        config = StreamingConfig(
            min_chunk_size=1,
            eval_interval_ms=0,
            on_chunk_callback=on_chunk,
        )
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_pass_scorer)

        evaluator.process_token("chunk1")
        evaluator.process_token("chunk2")

        assert len(chunks_received) == 2

    def test_on_stop_callback(self):
        """on_stop_callback should be called on early stop."""
        stop_reasons = []

        def on_stop(reason, text):
            stop_reasons.append((reason, text))

        config = StreamingConfig(
            min_chunk_size=1,
            eval_interval_ms=0,
            enable_early_stop=True,
            stop_on_first_failure=True,
            on_stop_callback=on_stop,
        )
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_fail_scorer, threshold=0.5)

        evaluator.process_token("fail")

        assert len(stop_reasons) == 1
        assert stop_reasons[0][0] == EarlyStopReason.THRESHOLD


class TestStreamingEvaluatorStream:
    """Tests for evaluate_stream methods."""

    def test_evaluate_stream(self):
        """evaluate_stream should process entire stream."""
        config = StreamingConfig(min_chunk_size=1, eval_interval_ms=0)
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_pass_scorer)

        tokens = ["Hello", " ", "world", "!"]
        result = evaluator.evaluate_stream(iter(tokens))

        assert result.final_text == "Hello world!"
        assert result.passed is True

    def test_evaluate_stream_early_stop(self):
        """evaluate_stream should respect early stopping."""
        config = StreamingConfig(
            min_chunk_size=1,
            eval_interval_ms=0,
            enable_early_stop=True,
        )
        policy = EarlyStopPolicy()
        policy.add_toxicity_stop(threshold=0.5)

        evaluator = StreamingEvaluator(config=config, policy=policy)
        evaluator.add_eval("toxicity", toxic_word_scorer, threshold=0.5, pass_above=False)

        tokens = ["Hello", " ", "toxic", " ", "world"]
        result = evaluator.evaluate_stream(iter(tokens))

        # Should have stopped at "toxic"
        assert result.early_stopped is True
        assert "toxic" in result.final_text
        assert "world" not in result.final_text

    def test_evaluate_stream_async(self):
        """evaluate_stream_async should process async stream."""
        config = StreamingConfig(min_chunk_size=1, eval_interval_ms=0)
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_pass_scorer)

        async def async_tokens():
            for token in ["Hello", " ", "async", "!"]:
                yield token

        async def run_test():
            return await evaluator.evaluate_stream_async(async_tokens())

        result = asyncio.run(run_test())

        assert result.final_text == "Hello async!"
        assert result.passed is True


class TestStreamingEvaluatorFactoryMethods:
    """Tests for factory class methods."""

    def test_with_defaults(self):
        """with_defaults should create standard evaluator."""
        evaluator = StreamingEvaluator.with_defaults()
        assert isinstance(evaluator, StreamingEvaluator)
        assert evaluator.config is not None

    def test_for_safety(self):
        """for_safety should create safety-focused evaluator."""
        evaluator = StreamingEvaluator.for_safety(
            toxicity_threshold=0.3,
            safety_threshold=0.6,
        )
        assert evaluator.config.enable_early_stop is True
        assert evaluator.config.stop_on_first_failure is True
        assert evaluator.config.toxicity_threshold == 0.3

    def test_for_quality(self):
        """for_quality should create quality-focused evaluator."""
        evaluator = StreamingEvaluator.for_quality(
            min_chunk_size=100,
            eval_interval_ms=200,
        )
        assert evaluator.config.min_chunk_size == 100
        assert evaluator.config.eval_interval_ms == 200
        assert evaluator.config.enable_early_stop is False


class TestStreamingEvaluatorMetadata:
    """Tests for metadata in results."""

    def test_buffer_stats_in_metadata(self):
        """Result metadata should include buffer stats."""
        config = StreamingConfig(min_chunk_size=1, eval_interval_ms=0)
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_pass_scorer)

        evaluator.process_token("test content")
        result = evaluator.finalize()

        assert "buffer_stats" in result.metadata
        assert "total_chars" in result.metadata["buffer_stats"]

    def test_policy_stats_in_metadata(self):
        """Result metadata should include policy stats."""
        config = StreamingConfig(min_chunk_size=1, eval_interval_ms=0)
        evaluator = StreamingEvaluator(config=config)
        evaluator.add_eval("test", always_pass_scorer)

        evaluator.process_token("test")
        result = evaluator.finalize()

        assert "policy_stats" in result.metadata
