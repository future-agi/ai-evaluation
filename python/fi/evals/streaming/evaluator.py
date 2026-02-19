"""Streaming Evaluator for real-time LLM output evaluation.

Evaluates LLM outputs in real-time as tokens stream in, with support for
early stopping based on configurable policies.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from .types import (
    ChunkResult,
    EarlyStopReason,
    StreamingConfig,
    StreamingEvalResult,
    StreamingState,
)
from .buffer import ChunkBuffer
from .policy import EarlyStopPolicy


@dataclass
class EvalSpec:
    """Specification for a streaming evaluation."""

    name: str
    eval_fn: Callable[[str, str], float]  # (chunk, cumulative) -> score
    threshold: float = 0.7
    weight: float = 1.0
    pass_above: bool = True  # True if higher scores are better


class StreamingEvaluator:
    """
    Evaluates LLM outputs in real-time as tokens stream in.

    Supports early stopping based on configurable policies and provides
    detailed per-chunk and aggregate evaluation results.

    Example:
        evaluator = StreamingEvaluator(config)
        evaluator.add_eval("toxicity", toxicity_scorer, threshold=0.7, pass_above=False)
        evaluator.set_policy(EarlyStopPolicy.default())

        # Synchronous iteration
        for token in stream:
            result = evaluator.process_token(token)
            if result and result.should_stop:
                break

        final_result = evaluator.finalize()

        # Or async iteration
        async for token in async_stream:
            result = await evaluator.process_token_async(token)
            if result and result.should_stop:
                break
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        policy: Optional[EarlyStopPolicy] = None,
    ):
        """
        Initialize the streaming evaluator.

        Args:
            config: Streaming configuration (uses defaults if None)
            policy: Early stop policy (uses default if None)
        """
        self.config = config or StreamingConfig()
        self._policy = policy or EarlyStopPolicy.default()
        self._buffer = ChunkBuffer(self.config)
        self._evals: List[EvalSpec] = []
        self._chunk_results: List[ChunkResult] = []
        self._state = StreamingState.IDLE
        self._start_time: float = 0.0
        self._stop_reason = EarlyStopReason.NONE
        self._stopped_at_chunk: Optional[int] = None

    def add_eval(
        self,
        name: str,
        eval_fn: Callable[[str, str], float],
        threshold: float = 0.7,
        weight: float = 1.0,
        pass_above: bool = True,
    ) -> "StreamingEvaluator":
        """
        Add an evaluation function.

        Args:
            name: Name of the evaluation
            eval_fn: Function that takes (chunk_text, cumulative_text) and returns score
            threshold: Passing threshold
            weight: Weight for final score calculation
            pass_above: If True, scores above threshold pass; if False, below

        Returns:
            Self for chaining
        """
        self._evals.append(
            EvalSpec(
                name=name,
                eval_fn=eval_fn,
                threshold=threshold,
                weight=weight,
                pass_above=pass_above,
            )
        )
        return self

    def set_policy(self, policy: EarlyStopPolicy) -> "StreamingEvaluator":
        """
        Set the early stop policy.

        Args:
            policy: Policy to use for early stopping

        Returns:
            Self for chaining
        """
        self._policy = policy
        return self

    def reset(self) -> None:
        """Reset the evaluator for a new stream."""
        self._buffer.reset()
        self._policy.reset()
        self._chunk_results = []
        self._state = StreamingState.IDLE
        self._start_time = 0.0
        self._stop_reason = EarlyStopReason.NONE
        self._stopped_at_chunk = None

    def process_token(self, token: str) -> Optional[ChunkResult]:
        """
        Process a single token from the stream.

        Args:
            token: The token text

        Returns:
            ChunkResult if evaluation was triggered, None otherwise
        """
        if self._state == StreamingState.IDLE:
            self._state = StreamingState.STREAMING
            self._start_time = time.perf_counter()

        if self._state in (StreamingState.STOPPED, StreamingState.COMPLETED, StreamingState.ERROR):
            return None

        # Add token to buffer
        self._buffer.add(token)

        # Check for limits
        should_stop_limits, limit_reason = self._buffer.should_stop_for_limits()
        if should_stop_limits:
            return self._handle_limit_stop(limit_reason)

        # Check if we should evaluate
        if not self._buffer.should_evaluate():
            return None

        # Run evaluation
        return self._evaluate_chunk()

    async def process_token_async(self, token: str) -> Optional[ChunkResult]:
        """
        Process a single token asynchronously.

        Args:
            token: The token text

        Returns:
            ChunkResult if evaluation was triggered, None otherwise
        """
        # For now, wrap sync processing - can be optimized later
        return await asyncio.to_thread(self.process_token, token)

    def process_chunk(self, chunk: str) -> Optional[ChunkResult]:
        """
        Process a larger chunk of text (multiple tokens).

        Args:
            chunk: The chunk text

        Returns:
            ChunkResult if evaluation was triggered, None otherwise
        """
        if self._state == StreamingState.IDLE:
            self._state = StreamingState.STREAMING
            self._start_time = time.perf_counter()

        if self._state in (StreamingState.STOPPED, StreamingState.COMPLETED, StreamingState.ERROR):
            return None

        # Add chunk to buffer
        self._buffer.add_chunk(chunk)

        # Check for limits
        should_stop_limits, limit_reason = self._buffer.should_stop_for_limits()
        if should_stop_limits:
            return self._handle_limit_stop(limit_reason)

        # Check if we should evaluate
        if not self._buffer.should_evaluate():
            return None

        # Run evaluation
        return self._evaluate_chunk()

    def _evaluate_chunk(self) -> ChunkResult:
        """Run evaluations on the current chunk."""
        chunk_start = time.perf_counter()

        chunk_text = self._buffer.get_chunk()
        cumulative_text = self._buffer.get_cumulative()
        chunk_index = self._buffer.get_chunk_index()

        scores: Dict[str, float] = {}
        flags: Dict[str, bool] = {}

        # Run all evaluations
        for eval_spec in self._evals:
            try:
                score = eval_spec.eval_fn(chunk_text, cumulative_text)
                scores[eval_spec.name] = score

                # Determine if passed
                if eval_spec.pass_above:
                    flags[eval_spec.name] = score >= eval_spec.threshold
                else:
                    flags[eval_spec.name] = score <= eval_spec.threshold
            except Exception as e:
                # Handle eval errors gracefully
                scores[eval_spec.name] = 0.0
                flags[eval_spec.name] = False

        latency_ms = (time.perf_counter() - chunk_start) * 1000

        # Create chunk result
        chunk_result = ChunkResult(
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            cumulative_text=cumulative_text,
            scores=scores,
            flags=flags,
            latency_ms=latency_ms,
        )

        # Check policy for early stop
        if self.config.enable_early_stop:
            should_stop, stop_reason = self._policy.check(chunk_result)

            if should_stop:
                chunk_result.should_stop = True
                chunk_result.stop_reason = stop_reason
                self._state = StreamingState.STOPPED
                self._stop_reason = stop_reason
                self._stopped_at_chunk = chunk_index

                # Trigger callback if configured
                if self.config.on_stop_callback:
                    self.config.on_stop_callback(stop_reason, cumulative_text)

            # Check stop on first failure
            if self.config.stop_on_first_failure and not chunk_result.all_passed:
                chunk_result.should_stop = True
                chunk_result.stop_reason = EarlyStopReason.THRESHOLD
                self._state = StreamingState.STOPPED
                self._stop_reason = EarlyStopReason.THRESHOLD
                self._stopped_at_chunk = chunk_index

                # Trigger callback if configured
                if self.config.on_stop_callback:
                    self.config.on_stop_callback(EarlyStopReason.THRESHOLD, cumulative_text)

        # Store result
        self._chunk_results.append(chunk_result)

        # Mark as evaluated
        self._buffer.mark_evaluated()

        # Trigger chunk callback if configured
        if self.config.on_chunk_callback:
            self.config.on_chunk_callback(chunk_result)

        return chunk_result

    def _handle_limit_stop(self, reason: str) -> ChunkResult:
        """Handle stopping due to limits."""
        # Evaluate any remaining content first
        if self._buffer.has_pending:
            chunk_result = self._evaluate_chunk()
        else:
            # Create a minimal result for the stop
            chunk_result = ChunkResult(
                chunk_index=self._buffer.get_chunk_index(),
                chunk_text="",
                cumulative_text=self._buffer.get_cumulative(),
                scores={},
                flags={},
            )

        # Map limit reason to stop reason
        if reason == "max_tokens":
            stop_reason = EarlyStopReason.MAX_TOKENS
        elif reason == "timeout":
            stop_reason = EarlyStopReason.TIMEOUT
        else:
            stop_reason = EarlyStopReason.MAX_TOKENS

        chunk_result.should_stop = True
        chunk_result.stop_reason = stop_reason
        self._state = StreamingState.STOPPED
        self._stop_reason = stop_reason
        self._stopped_at_chunk = chunk_result.chunk_index

        return chunk_result

    def finalize(self) -> StreamingEvalResult:
        """
        Finalize evaluation and return results.

        Should be called after stream completes or after early stop.

        Returns:
            StreamingEvalResult with all evaluation data
        """
        # Evaluate any remaining pending content
        if self._buffer.has_pending and self._state == StreamingState.STREAMING:
            self._evaluate_chunk()

        # Mark as completed if not already stopped
        if self._state == StreamingState.STREAMING:
            self._state = StreamingState.COMPLETED

        # Calculate final scores (weighted average across chunks)
        final_scores = self._calculate_final_scores()

        # Determine overall pass/fail
        passed = self._determine_passed(final_scores)

        # Calculate total latency
        total_latency_ms = (time.perf_counter() - self._start_time) * 1000 if self._start_time else 0.0

        return StreamingEvalResult(
            passed=passed,
            final_text=self._buffer.get_cumulative(),
            total_chunks=len(self._chunk_results),
            chunk_results=self._chunk_results,
            final_scores=final_scores,
            early_stopped=self._state == StreamingState.STOPPED,
            stop_reason=self._stop_reason,
            stopped_at_chunk=self._stopped_at_chunk,
            total_latency_ms=total_latency_ms,
            state=self._state,
            metadata={
                "buffer_stats": self._buffer.get_stats(),
                "policy_stats": self._policy.get_stats(),
            },
        )

    def _calculate_final_scores(self) -> Dict[str, float]:
        """Calculate final weighted average scores."""
        if not self._chunk_results:
            return {}

        final_scores: Dict[str, float] = {}
        score_sums: Dict[str, float] = {}
        score_counts: Dict[str, int] = {}

        for chunk_result in self._chunk_results:
            for name, score in chunk_result.scores.items():
                if name not in score_sums:
                    score_sums[name] = 0.0
                    score_counts[name] = 0
                score_sums[name] += score
                score_counts[name] += 1

        for name in score_sums:
            final_scores[name] = score_sums[name] / score_counts[name]

        return final_scores

    def _determine_passed(self, final_scores: Dict[str, float]) -> bool:
        """Determine if evaluation passed overall."""
        if self._state == StreamingState.STOPPED:
            # If stopped early due to safety/toxicity, fail
            if self._stop_reason in (
                EarlyStopReason.TOXICITY,
                EarlyStopReason.SAFETY,
                EarlyStopReason.PII,
                EarlyStopReason.JAILBREAK,
            ):
                return False

        # Check final scores against thresholds
        for eval_spec in self._evals:
            if eval_spec.name in final_scores:
                score = final_scores[eval_spec.name]
                if eval_spec.pass_above:
                    if score < eval_spec.threshold:
                        return False
                else:
                    if score > eval_spec.threshold:
                        return False

        return True

    @property
    def state(self) -> StreamingState:
        """Get current evaluation state."""
        return self._state

    @property
    def is_stopped(self) -> bool:
        """Check if evaluation has stopped."""
        return self._state in (StreamingState.STOPPED, StreamingState.COMPLETED, StreamingState.ERROR)

    @property
    def chunk_count(self) -> int:
        """Get number of chunks evaluated."""
        return len(self._chunk_results)

    def evaluate_stream(
        self,
        stream: Iterator[str],
    ) -> StreamingEvalResult:
        """
        Evaluate an entire stream synchronously.

        Args:
            stream: Iterator yielding tokens/chunks

        Returns:
            StreamingEvalResult after processing complete stream
        """
        self.reset()

        for token in stream:
            result = self.process_token(token)
            if result and result.should_stop:
                break

        return self.finalize()

    async def evaluate_stream_async(
        self,
        stream: AsyncIterator[str],
    ) -> StreamingEvalResult:
        """
        Evaluate an entire stream asynchronously.

        Args:
            stream: Async iterator yielding tokens/chunks

        Returns:
            StreamingEvalResult after processing complete stream
        """
        self.reset()

        async for token in stream:
            result = await self.process_token_async(token)
            if result and result.should_stop:
                break

        return self.finalize()

    @classmethod
    def with_defaults(cls) -> "StreamingEvaluator":
        """
        Create an evaluator with default configuration.

        Returns:
            StreamingEvaluator with default settings
        """
        return cls(
            config=StreamingConfig(),
            policy=EarlyStopPolicy.default(),
        )

    @classmethod
    def for_safety(
        cls,
        toxicity_threshold: float = 0.5,
        safety_threshold: float = 0.5,
    ) -> "StreamingEvaluator":
        """
        Create an evaluator optimized for safety monitoring.

        Args:
            toxicity_threshold: Threshold for toxicity (stop if above)
            safety_threshold: Threshold for safety (stop if below)

        Returns:
            StreamingEvaluator configured for safety
        """
        config = StreamingConfig(
            enable_early_stop=True,
            stop_on_first_failure=True,
            toxicity_threshold=toxicity_threshold,
            safety_threshold=safety_threshold,
        )
        policy = EarlyStopPolicy.strict()
        return cls(config=config, policy=policy)

    @classmethod
    def for_quality(
        cls,
        min_chunk_size: int = 50,
        eval_interval_ms: int = 500,
    ) -> "StreamingEvaluator":
        """
        Create an evaluator optimized for quality assessment.

        Args:
            min_chunk_size: Minimum characters before evaluation
            eval_interval_ms: Milliseconds between evaluations

        Returns:
            StreamingEvaluator configured for quality
        """
        config = StreamingConfig(
            min_chunk_size=min_chunk_size,
            max_chunk_size=200,
            eval_interval_ms=eval_interval_ms,
            enable_early_stop=False,  # Don't stop early for quality
            eval_on_sentence_end=True,
        )
        policy = EarlyStopPolicy.permissive()
        return cls(config=config, policy=policy)
