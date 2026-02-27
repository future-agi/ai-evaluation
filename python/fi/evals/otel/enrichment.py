"""
Automatic Span Enrichment.

Automatically adds evaluation results to active OTEL spans
when evaluations are run through fi.evals.

This enables the "evals data automatically goes into spans" workflow.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import time

from .processors import OTEL_AVAILABLE
from .conventions import EvaluationAttributes, GenAIAttributes

if OTEL_AVAILABLE:
    from opentelemetry import trace
    from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger(__name__)

# Global flag to enable/disable automatic enrichment
_auto_enrichment_enabled = True


def enable_auto_enrichment():
    """Enable automatic span enrichment for evaluations."""
    global _auto_enrichment_enabled
    _auto_enrichment_enabled = True
    logger.info("Automatic span enrichment enabled")


def disable_auto_enrichment():
    """Disable automatic span enrichment for evaluations."""
    global _auto_enrichment_enabled
    _auto_enrichment_enabled = False
    logger.info("Automatic span enrichment disabled")


def is_auto_enrichment_enabled() -> bool:
    """Check if automatic span enrichment is enabled."""
    return _auto_enrichment_enabled


def get_current_span() -> Optional[Any]:
    """
    Get the current active span.

    Returns:
        Current span or None if no active span or OTEL not available
    """
    if not OTEL_AVAILABLE:
        return None

    try:
        span = trace.get_current_span()
        # Check if it's a valid, recording span
        if span and span.is_recording():
            return span
        return None
    except Exception as e:
        logger.debug(f"Failed to get current span: {e}")
        return None


def enrich_span_with_evaluation(
    metric_name: str,
    score: Union[float, int, bool],
    reason: Optional[str] = None,
    latency_ms: Optional[float] = None,
    span: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Enrich a span with evaluation results.

    Args:
        metric_name: Name of the evaluation metric
        score: Evaluation score (float 0-1, int, or bool)
        reason: Explanation for the score
        latency_ms: Time taken to evaluate
        span: Specific span to enrich, or None for current span
        metadata: Additional metadata to add

    Returns:
        True if enrichment succeeded, False otherwise
    """
    if not OTEL_AVAILABLE:
        return False

    if not _auto_enrichment_enabled:
        return False

    # Get span to enrich
    target_span = span or get_current_span()
    if target_span is None:
        logger.debug(f"No active span to enrich with {metric_name} evaluation")
        return False

    try:
        # Normalize score to float
        if isinstance(score, bool):
            normalized_score = 1.0 if score else 0.0
        elif isinstance(score, (int, float)):
            normalized_score = float(score)
        elif isinstance(score, str):
            try:
                normalized_score = float(score)
            except ValueError:
                low = score.strip().lower()
                if low in ("true", "yes", "pass", "passed"):
                    normalized_score = 1.0
                elif low in ("false", "no", "fail", "failed"):
                    normalized_score = 0.0
                else:
                    logger.debug(f"Cannot normalize score string '{score}' to float")
                    return False
        else:
            normalized_score = float(score)

        target_span.set_attribute(EvaluationAttributes.NAME, metric_name)
        target_span.set_attribute(EvaluationAttributes.SCORE_VALUE, normalized_score)

        if reason:
            reason_text = reason[:1000] if len(reason) > 1000 else reason
            target_span.set_attribute(EvaluationAttributes.EXPLANATION, reason_text)

        if latency_ms is not None:
            target_span.set_attribute(
                EvaluationAttributes.latency(metric_name),
                latency_ms
            )

        logger.debug(f"Enriched span with {metric_name}={normalized_score}")
        return True

    except Exception as e:
        logger.warning(f"Failed to enrich span with evaluation: {e}")
        return False


def enrich_span_with_eval_result(
    eval_result: Any,
    span: Optional[Any] = None,
) -> bool:
    """
    Enrich a span with an EvalResult object.

    Args:
        eval_result: EvalResult from fi.evals
        span: Specific span to enrich, or None for current span

    Returns:
        True if enrichment succeeded
    """
    if eval_result is None:
        return False

    try:
        metric_name = getattr(eval_result, 'name', 'unknown')
        score = getattr(eval_result, 'output', None)
        reason = getattr(eval_result, 'reason', None)
        runtime = getattr(eval_result, 'runtime', None)

        if score is None:
            return False

        return enrich_span_with_evaluation(
            metric_name=metric_name,
            score=score,
            reason=reason,
            latency_ms=float(runtime) if runtime else None,
            span=span,
        )
    except Exception as e:
        logger.warning(f"Failed to enrich span with EvalResult: {e}")
        return False


def enrich_span_with_batch_result(
    batch_result: Any,
    span: Optional[Any] = None,
) -> int:
    """
    Enrich a span with a BatchRunResult.

    Args:
        batch_result: BatchRunResult from fi.evals
        span: Specific span to enrich, or None for current span

    Returns:
        Number of evaluations successfully added
    """
    if batch_result is None:
        return 0

    count = 0
    try:
        eval_results = getattr(batch_result, 'eval_results', [])
        for result in eval_results:
            if result is not None and enrich_span_with_eval_result(result, span):
                count += 1
    except Exception as e:
        logger.warning(f"Failed to enrich span with BatchRunResult: {e}")

    return count


def create_evaluation_span(
    metric_name: str,
    parent_span: Optional[Any] = None,
) -> Any:
    """
    Create a child span for an evaluation operation.

    Args:
        metric_name: Name of the metric being evaluated
        parent_span: Parent span, or None for current span

    Returns:
        Context manager for the evaluation span
    """
    if not OTEL_AVAILABLE:
        from .tracer import _NoOpContextManager
        return _NoOpContextManager()

    try:
        tracer = trace.get_tracer("fi.evals.evaluation")
        return tracer.start_as_current_span(
            f"eval.{metric_name}",
            attributes={
                GenAIAttributes.SPAN_KIND: "EVALUATOR",
                EvaluationAttributes.NAME: metric_name,
                EvaluationAttributes.EVALUATED_AT: time.time(),
            }
        )
    except Exception as e:
        logger.debug(f"Failed to create evaluation span: {e}")
        from .tracer import _NoOpContextManager
        return _NoOpContextManager()


class EvaluationSpanContext:
    """
    Context manager for evaluation operations with automatic span enrichment.

    Example:
        with EvaluationSpanContext("relevance") as ctx:
            result = run_evaluation()
            ctx.record_result(score=0.85, reason="Good match")
    """

    def __init__(self, metric_name: str, create_child_span: bool = True):
        """
        Initialize evaluation context.

        Args:
            metric_name: Name of the metric
            create_child_span: Whether to create a child span
        """
        self.metric_name = metric_name
        self.create_child_span = create_child_span
        self._span = None
        self._parent_span = None
        self._start_time = None
        self._result_recorded = False

    def __enter__(self):
        self._start_time = time.time()

        if OTEL_AVAILABLE and self.create_child_span:
            try:
                self._parent_span = get_current_span()
                tracer = trace.get_tracer("fi.evals.evaluation")
                self._span = tracer.start_span(
                    f"eval.{self.metric_name}",
                    attributes={
                        GenAIAttributes.SPAN_KIND: "EVALUATOR",
                        EvaluationAttributes.NAME: self.metric_name,
                    }
                )
                # Make it the current span
                self._token = trace.use_span(self._span, end_on_exit=False)
                self._token.__enter__()
            except Exception as e:
                logger.debug(f"Failed to create evaluation span: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self._start_time) * 1000

        if self._span:
            try:
                if exc_type:
                    self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                    self._span.record_exception(exc_val)
                else:
                    self._span.set_status(Status(StatusCode.OK))

                self._span.set_attribute(
                    EvaluationAttributes.latency(self.metric_name),
                    latency_ms
                )
                self._token.__exit__(exc_type, exc_val, exc_tb)
                self._span.end()
            except Exception as e:
                logger.debug(f"Error closing evaluation span: {e}")

        return False

    def record_result(
        self,
        score: Union[float, int, bool],
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Record the evaluation result.

        Args:
            score: Evaluation score
            reason: Explanation
            metadata: Additional metadata

        Returns:
            True if recorded successfully
        """
        if self._result_recorded:
            return False

        latency_ms = (time.time() - self._start_time) * 1000 if self._start_time else None

        # Enrich the child span if created
        if self._span:
            enrich_span_with_evaluation(
                metric_name=self.metric_name,
                score=score,
                reason=reason,
                latency_ms=latency_ms,
                span=self._span,
                metadata=metadata,
            )

        # Also enrich the parent span
        if self._parent_span:
            enrich_span_with_evaluation(
                metric_name=self.metric_name,
                score=score,
                reason=reason,
                latency_ms=latency_ms,
                span=self._parent_span,
                metadata=metadata,
            )

        self._result_recorded = True
        return True


__all__ = [
    "enable_auto_enrichment",
    "disable_auto_enrichment",
    "is_auto_enrichment_enabled",
    "get_current_span",
    "enrich_span_with_evaluation",
    "enrich_span_with_eval_result",
    "enrich_span_with_batch_result",
    "create_evaluation_span",
    "EvaluationSpanContext",
]
