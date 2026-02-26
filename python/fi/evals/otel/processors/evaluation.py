"""
Evaluation Span Processor.

Enriches LLM spans with evaluation scores by running
configured metrics against the prompt/completion pairs.
"""

from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future
import logging
import time
import random
import hashlib

from .base import BaseSpanProcessor, OTEL_AVAILABLE
from ..conventions import (
    GenAIAttributes,
    EvaluationAttributes,
    create_evaluation_attributes,
)
from ..types import EvaluationResult

if OTEL_AVAILABLE:
    from opentelemetry.sdk.trace import ReadableSpan
else:
    ReadableSpan = Any

logger = logging.getLogger(__name__)


# Type alias for evaluator function
EvaluatorFn = Callable[[str, str, Optional[str]], List[EvaluationResult]]


class EvaluationSpanProcessor(BaseSpanProcessor):
    """
    Processor that evaluates LLM spans and attaches scores.

    This processor runs configured evaluation metrics against
    the prompt/completion pairs in LLM spans and records the
    evaluation results as span attributes or events.

    Supports:
    - Multiple evaluation metrics
    - Sampling (evaluate only a fraction of spans)
    - Async evaluation (non-blocking)
    - Result caching
    - Timeout handling

    Example:
        from fi.evals import evaluate

        processor = EvaluationSpanProcessor(
            metrics=["relevance", "coherence", "faithfulness"],
            sample_rate=0.1,  # Evaluate 10% of spans
            async_evaluation=True,
        )

        # Or with custom evaluator
        processor = EvaluationSpanProcessor(
            evaluator=my_custom_evaluator,
            metrics=["custom_metric"],
        )
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        evaluator: Optional[EvaluatorFn] = None,
        sample_rate: float = 1.0,
        async_evaluation: bool = True,
        timeout_ms: int = 5000,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 3600,
        max_workers: int = 4,
        evaluator_model: Optional[str] = None,
        on_evaluation_complete: Optional[Callable[[str, List[EvaluationResult]], None]] = None,
        enabled: bool = True,
    ):
        """
        Initialize the evaluation processor.

        Args:
            metrics: List of metrics to evaluate (e.g., ["relevance", "coherence"])
            evaluator: Custom evaluator function, or None to use default
            sample_rate: Fraction of spans to evaluate (0.0 to 1.0)
            async_evaluation: Whether to evaluate asynchronously
            timeout_ms: Timeout for evaluation in milliseconds
            cache_enabled: Whether to cache evaluation results
            cache_ttl_seconds: Cache TTL in seconds
            max_workers: Max threads for async evaluation
            evaluator_model: Model to use for LLM-based evaluation
            on_evaluation_complete: Callback when evaluation completes
            enabled: Whether processor is active
        """
        super().__init__(enabled=enabled)
        self._metrics = metrics or ["relevance", "coherence"]
        self._evaluator = evaluator
        self._sample_rate = max(0.0, min(1.0, sample_rate))
        self._async_evaluation = async_evaluation
        self._timeout_ms = timeout_ms
        self._cache_enabled = cache_enabled
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_workers = max_workers
        self._evaluator_model = evaluator_model
        self._on_evaluation_complete = on_evaluation_complete

        # Internal state
        self._executor: Optional[ThreadPoolExecutor] = None
        self._cache: Dict[str, tuple[float, List[EvaluationResult]]] = {}
        self._pending_evaluations: Dict[str, Future] = {}

    def should_process(self, span: ReadableSpan) -> bool:
        """Determine if span should be evaluated."""
        if not self.enabled:
            return False

        # Check if it's an LLM span
        attrs = self._get_attributes(span)
        if not self._is_llm_span(attrs):
            return False

        # Check sampling
        if self._sample_rate < 1.0:
            if random.random() > self._sample_rate:
                return False

        return True

    def _is_llm_span(self, attrs: Dict[str, Any]) -> bool:
        """Check if this is an LLM span worth evaluating."""
        # Has GenAI provider attribute
        if GenAIAttributes.PROVIDER_NAME in attrs:
            return True

        # Has model attribute
        if GenAIAttributes.REQUEST_MODEL in attrs:
            return True

        # Has output messages
        if GenAIAttributes.OUTPUT_MESSAGES in attrs:
            return True

        # Check for common patterns
        for key in attrs:
            if any(pattern in str(key).lower() for pattern in ["llm", "gen_ai", "completion"]):
                return True

        return False

    def _get_attributes(self, span: ReadableSpan) -> Dict[str, Any]:
        """Get span attributes safely."""
        try:
            return dict(span.attributes or {})
        except Exception:
            return {}

    def _get_span_id(self, span: ReadableSpan) -> str:
        """Get unique span identifier."""
        try:
            ctx = span.get_span_context()
            return f"{ctx.trace_id:032x}:{ctx.span_id:016x}"
        except Exception:
            return str(id(span))

    def on_end(self, span: ReadableSpan) -> None:
        """Evaluate the span and record results."""
        if not self.should_process(span):
            return

        try:
            attrs = self._get_attributes(span)
            span_id = self._get_span_id(span)

            # Extract content for evaluation
            prompt = self._extract_prompt(attrs)
            completion = self._extract_completion(attrs)
            model = attrs.get(GenAIAttributes.REQUEST_MODEL)

            if not prompt or not completion:
                logger.debug(f"Skipping evaluation for span {span_id}: no content")
                return

            # Check cache
            cache_key = self._compute_cache_key(prompt, completion, self._metrics)
            cached_results = self._get_cached_results(cache_key)
            if cached_results is not None:
                logger.debug(f"Using cached evaluation for span {span_id}")
                self._handle_results(span_id, cached_results)
                return

            # Run evaluation
            if self._async_evaluation:
                self._evaluate_async(span_id, prompt, completion, model, cache_key)
            else:
                results = self._evaluate_sync(prompt, completion, model)
                self._cache_results(cache_key, results)
                self._handle_results(span_id, results)

        except Exception as e:
            logger.warning(f"Evaluation processing error: {e}")

    def _extract_prompt(self, attrs: Dict[str, Any]) -> Optional[str]:
        """Extract prompt from span attributes."""
        for key in [
            GenAIAttributes.INPUT_MESSAGES,
            "llm.prompt",
            "prompt",
            "input",
        ]:
            if key in attrs:
                return str(attrs[key])
        return None

    def _extract_completion(self, attrs: Dict[str, Any]) -> Optional[str]:
        """Extract completion from span attributes."""
        for key in [
            GenAIAttributes.OUTPUT_MESSAGES,
            "llm.completion",
            "completion",
            "output",
            "response",
        ]:
            if key in attrs:
                return str(attrs[key])
        return None

    def _evaluate_sync(
        self,
        prompt: str,
        completion: str,
        model: Optional[str] = None,
    ) -> List[EvaluationResult]:
        """Run evaluation synchronously."""
        start_time = time.time()
        results: List[EvaluationResult] = []

        try:
            if self._evaluator:
                # Use custom evaluator
                results = self._evaluator(prompt, completion, model)
            else:
                # Use default evaluator (fi.evals)
                results = self._run_default_evaluator(prompt, completion, model)

            # Add latency to results
            latency_ms = (time.time() - start_time) * 1000
            for result in results:
                if result.latency_ms is None:
                    result.latency_ms = latency_ms / len(results)

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            # Return error result
            results = [
                EvaluationResult(
                    metric="error",
                    score=0.0,
                    reason=str(e),
                    latency_ms=(time.time() - start_time) * 1000,
                )
            ]

        return results

    def _run_default_evaluator(
        self,
        prompt: str,
        completion: str,
        model: Optional[str] = None,
    ) -> List[EvaluationResult]:
        """Run the default fi.evals evaluator."""
        results: List[EvaluationResult] = []

        try:
            # Import here to avoid circular deps
            from fi.evals import Evaluator

            evaluator = Evaluator()

            for metric in self._metrics:
                start = time.time()
                try:
                    # Try to evaluate with the metric
                    eval_result = evaluator.evaluate(
                        metric=metric,
                        input=prompt,
                        output=completion,
                        model=self._evaluator_model,
                    )

                    results.append(EvaluationResult(
                        metric=metric,
                        score=float(eval_result.get("score", 0.0)),
                        reason=eval_result.get("reason"),
                        latency_ms=(time.time() - start) * 1000,
                    ))

                except Exception as e:
                    logger.warning(f"Metric '{metric}' evaluation failed: {e}")
                    results.append(EvaluationResult(
                        metric=metric,
                        score=0.0,
                        reason=f"Evaluation error: {e}",
                        latency_ms=(time.time() - start) * 1000,
                    ))

        except ImportError:
            logger.warning("fi.evals not available, skipping default evaluation")

        return results

    def _evaluate_async(
        self,
        span_id: str,
        prompt: str,
        completion: str,
        model: Optional[str],
        cache_key: str,
    ) -> None:
        """Run evaluation asynchronously."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        def evaluate_task():
            results = self._evaluate_sync(prompt, completion, model)
            self._cache_results(cache_key, results)
            self._handle_results(span_id, results)
            return results

        future = self._executor.submit(evaluate_task)
        self._pending_evaluations[span_id] = future

    def _handle_results(
        self,
        span_id: str,
        results: List[EvaluationResult],
    ) -> None:
        """Handle evaluation results."""
        # Log results
        for result in results:
            logger.info(
                f"Evaluation [{span_id}] {result.metric}: "
                f"score={result.score:.3f}"
                + (f" reason={result.reason[:50]}..." if result.reason else "")
            )

        # Call callback if configured
        if self._on_evaluation_complete:
            try:
                self._on_evaluation_complete(span_id, results)
            except Exception as e:
                logger.warning(f"Evaluation callback error: {e}")

        # Clean up pending
        self._pending_evaluations.pop(span_id, None)

    def _compute_cache_key(
        self,
        prompt: str,
        completion: str,
        metrics: List[str],
    ) -> str:
        """Compute cache key for evaluation results."""
        content = f"{prompt}|{completion}|{','.join(sorted(metrics))}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _get_cached_results(self, cache_key: str) -> Optional[List[EvaluationResult]]:
        """Get cached results if valid."""
        if not self._cache_enabled:
            return None

        if cache_key not in self._cache:
            return None

        timestamp, results = self._cache[cache_key]
        if time.time() - timestamp > self._cache_ttl_seconds:
            # Expired
            del self._cache[cache_key]
            return None

        return results

    def _cache_results(self, cache_key: str, results: List[EvaluationResult]) -> None:
        """Cache evaluation results."""
        if not self._cache_enabled:
            return

        self._cache[cache_key] = (time.time(), results)

        # Cleanup old entries if cache is too large
        if len(self._cache) > 10000:
            self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired = [
            key for key, (ts, _) in self._cache.items()
            if current_time - ts > self._cache_ttl_seconds
        ]
        for key in expired:
            del self._cache[key]

    def get_evaluation_attributes(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Any]:
        """
        Convert evaluation results to span attributes.

        Uses gen_ai.evaluation.* namespace.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary of OTEL-compliant attributes
        """
        attrs: Dict[str, Any] = {}

        for result in results:
            # Dual-write via helper
            result_attrs = create_evaluation_attributes(
                metric=result.metric,
                score=result.score,
                reason=result.reason,
                latency_ms=result.latency_ms,
            )
            attrs.update(result_attrs)

        return attrs

    def shutdown(self) -> None:
        """Shutdown the processor."""
        super().shutdown()

        # Wait for pending evaluations
        for span_id, future in list(self._pending_evaluations.items()):
            try:
                future.result(timeout=self._timeout_ms / 1000)
            except Exception as e:
                logger.warning(f"Pending evaluation {span_id} failed: {e}")

        self._pending_evaluations.clear()

        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        # Clear cache
        self._cache.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Wait for pending evaluations."""
        success = True
        per_eval_timeout = timeout_millis / 1000 / max(len(self._pending_evaluations), 1)

        for span_id, future in list(self._pending_evaluations.items()):
            try:
                future.result(timeout=per_eval_timeout)
            except Exception:
                success = False

        return success


class BatchEvaluationProcessor(EvaluationSpanProcessor):
    """
    Evaluation processor that batches evaluations for efficiency.

    Collects spans and evaluates them in batches rather than
    one at a time, which can be more efficient for some evaluators.

    Example:
        processor = BatchEvaluationProcessor(
            batch_size=10,
            batch_timeout_ms=1000,
        )
    """

    def __init__(
        self,
        batch_size: int = 10,
        batch_timeout_ms: int = 1000,
        **kwargs,
    ):
        """
        Initialize batch evaluation processor.

        Args:
            batch_size: Maximum batch size
            batch_timeout_ms: Max time to wait for batch
            **kwargs: Additional args for EvaluationSpanProcessor
        """
        super().__init__(**kwargs)
        self._batch_size = batch_size
        self._batch_timeout_ms = batch_timeout_ms
        self._batch: List[tuple[str, str, str, Optional[str]]] = []
        self._last_batch_time = time.time()

    def on_end(self, span: ReadableSpan) -> None:
        """Add span to batch for evaluation."""
        if not self.should_process(span):
            return

        try:
            attrs = self._get_attributes(span)
            span_id = self._get_span_id(span)
            prompt = self._extract_prompt(attrs)
            completion = self._extract_completion(attrs)
            model = attrs.get(GenAIAttributes.REQUEST_MODEL)

            if not prompt or not completion:
                return

            # Add to batch
            self._batch.append((span_id, prompt, completion, model))

            # Check if we should process batch
            if len(self._batch) >= self._batch_size:
                self._process_batch()
            elif (time.time() - self._last_batch_time) * 1000 >= self._batch_timeout_ms:
                self._process_batch()

        except Exception as e:
            logger.warning(f"Batch evaluation error: {e}")

    def _process_batch(self) -> None:
        """Process the accumulated batch."""
        if not self._batch:
            return

        batch = self._batch
        self._batch = []
        self._last_batch_time = time.time()

        # Evaluate each item in the batch
        for span_id, prompt, completion, model in batch:
            cache_key = self._compute_cache_key(prompt, completion, self._metrics)
            cached = self._get_cached_results(cache_key)

            if cached is not None:
                self._handle_results(span_id, cached)
            else:
                results = self._evaluate_sync(prompt, completion, model)
                self._cache_results(cache_key, results)
                self._handle_results(span_id, results)

    def shutdown(self) -> None:
        """Process remaining batch and shutdown."""
        self._process_batch()
        super().shutdown()


__all__ = [
    "EvaluationSpanProcessor",
    "BatchEvaluationProcessor",
]
