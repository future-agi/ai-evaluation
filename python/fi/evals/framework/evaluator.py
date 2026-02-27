"""
Unified Evaluator API.

Provides a single interface for running evaluations in any mode:
- Blocking: Synchronous execution, waits for results
- Non-blocking: Background execution, zero latency
- Distributed: Scalable execution via pluggable backends

Example:
    from fi.evals.framework import Evaluator, ExecutionMode

    # Create evaluator
    evaluator = Evaluator(
        evaluations=[ToxicityEval(), BiasEval()],
        mode=ExecutionMode.NON_BLOCKING,
    )

    # Run evaluations (returns immediately in non-blocking mode)
    result = evaluator.run({"response": "..."})

    # Get results when needed
    if result.is_future:
        batch = result.wait()
    else:
        batch = result.batch
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .types import ExecutionMode, FrameworkEvalResult, BatchEvalResult, EvalStatus

# Internal alias for brevity — this is the framework-level EvalResult
EvalResult = FrameworkEvalResult
from .context import EvalContext
from .protocols import BaseEvaluation, EvalRegistry
from .evaluators.blocking import BlockingEvaluator
from .evaluators.non_blocking import (
    NonBlockingEvaluator,
    BatchEvalFuture,
    EvalFuture,
)
from .backends import Backend, ThreadPoolBackend, ThreadPoolConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorResult:
    """
    Result from an evaluator run.

    Wraps either immediate results (blocking) or futures (non-blocking).

    Attributes:
        batch: Immediate BatchEvalResult (blocking mode)
        future: BatchEvalFuture for async results (non-blocking mode)
        mode: The execution mode used
        submitted_at: When the evaluation was submitted
    """

    batch: Optional[BatchEvalResult] = None
    future: Optional[BatchEvalFuture] = None
    mode: ExecutionMode = ExecutionMode.BLOCKING
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_future(self) -> bool:
        """Whether this result is a future (non-blocking)."""
        return self.future is not None

    @property
    def is_ready(self) -> bool:
        """Whether results are ready."""
        if self.batch is not None:
            return True
        if self.future is not None:
            return self.future.done()
        return False

    def wait(self, timeout: Optional[float] = None) -> BatchEvalResult:
        """
        Wait for and return results.

        Args:
            timeout: Maximum seconds to wait (non-blocking only)

        Returns:
            BatchEvalResult with all evaluation results
        """
        if self.batch is not None:
            return self.batch
        if self.future is not None:
            return self.future.results(timeout=timeout)
        raise ValueError("No results available")

    @property
    def results(self) -> List[EvalResult]:
        """Get individual results (waits if necessary)."""
        return self.wait().results

    @property
    def success_rate(self) -> float:
        """Get success rate (waits if necessary)."""
        return self.wait().success_rate


class FrameworkEvaluator:
    """
    Unified evaluator for all execution modes.

    Provides a consistent interface regardless of how evaluations are executed.
    Supports blocking, non-blocking, and distributed modes with automatic
    span enrichment.

    Example:
        # Simple blocking usage
        evaluator = FrameworkEvaluator([ToxicityEval()])
        result = evaluator.run({"response": "..."})
        print(f"Score: {result.results[0].value}")

        # Non-blocking for production
        evaluator = FrameworkEvaluator(
            [ToxicityEval(), BiasEval()],
            mode=ExecutionMode.NON_BLOCKING,
        )
        result = evaluator.run({"response": "..."})  # Returns immediately
        # ... do other work ...
        batch = result.wait()  # Get results when needed

        # With custom backend
        evaluator = FrameworkEvaluator(
            [ToxicityEval()],
            mode=ExecutionMode.NON_BLOCKING,
            backend=MyTemporalBackend(),
        )

    Thread Safety:
        This class is thread-safe. Multiple threads can call run() concurrently.
    """

    def __init__(
        self,
        evaluations: Optional[List[BaseEvaluation]] = None,
        mode: ExecutionMode = ExecutionMode.BLOCKING,
        auto_enrich_span: bool = True,
        fail_fast: bool = False,
        validate_inputs: bool = True,
        max_workers: int = 4,
        backend: Optional[Backend] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            evaluations: List of evaluations to run
            mode: Execution mode (BLOCKING, NON_BLOCKING, DISTRIBUTED)
            auto_enrich_span: Whether to automatically enrich OTEL spans
            fail_fast: Stop on first failure
            validate_inputs: Whether to validate inputs before evaluation
            max_workers: Max concurrent workers (non-blocking/distributed)
            backend: Custom backend for execution (uses ThreadPool if None)
        """
        self.evaluations = list(evaluations) if evaluations else []
        self.mode = mode
        self.auto_enrich_span = auto_enrich_span
        self.fail_fast = fail_fast
        self.validate_inputs = validate_inputs
        self.max_workers = max_workers
        self._backend = backend

        # Internal evaluators (created lazily)
        self._blocking: Optional[BlockingEvaluator] = None
        self._non_blocking: Optional[NonBlockingEvaluator] = None

    def add(self, evaluation: BaseEvaluation) -> "FrameworkEvaluator":
        """
        Add an evaluation to run.

        Args:
            evaluation: The evaluation to add

        Returns:
            Self for chaining
        """
        self.evaluations.append(evaluation)
        return self

    def add_by_name(self, name: str, version: str = "latest") -> "FrameworkEvaluator":
        """
        Add an evaluation by name from the registry.

        Args:
            name: Evaluation name
            version: Version (default: latest)

        Returns:
            Self for chaining

        Raises:
            KeyError: If evaluation not found
        """
        eval_class = EvalRegistry.get(name, version)
        if eval_class is None:
            raise KeyError(f"Evaluation not found: {name}@{version}")
        self.evaluations.append(eval_class())
        return self

    def run(
        self,
        inputs: Dict[str, Any],
        context: Optional[EvalContext] = None,
        callback: Optional[Callable[[EvalResult], None]] = None,
    ) -> EvaluatorResult:
        """
        Run all evaluations on the given inputs.

        Args:
            inputs: Input data for evaluations
            context: Optional trace context for span enrichment
            callback: Optional callback for each result (non-blocking only)

        Returns:
            EvaluatorResult wrapping results or future

        Raises:
            ValueError: If no evaluations configured
        """
        if not self.evaluations:
            raise ValueError("No evaluations configured")

        if self.mode == ExecutionMode.BLOCKING:
            return self._run_blocking(inputs, context)
        elif self.mode == ExecutionMode.NON_BLOCKING:
            return self._run_non_blocking(inputs, context, callback)
        elif self.mode == ExecutionMode.DISTRIBUTED:
            return self._run_distributed(inputs, context, callback)
        else:
            raise ValueError(f"Unknown execution mode: {self.mode}")

    def run_single(
        self,
        evaluation: BaseEvaluation,
        inputs: Dict[str, Any],
        context: Optional[EvalContext] = None,
    ) -> EvalResult:
        """
        Run a single evaluation.

        Always runs in blocking mode for simplicity.

        Args:
            evaluation: The evaluation to run
            inputs: Input data
            context: Optional trace context

        Returns:
            Single EvalResult
        """
        evaluator = self._get_blocking_evaluator()
        results = evaluator.evaluate(inputs, evaluations=[evaluation], context=context)
        return results[0]

    def _run_blocking(
        self,
        inputs: Dict[str, Any],
        context: Optional[EvalContext],
    ) -> EvaluatorResult:
        """Run evaluations in blocking mode."""
        evaluator = self._get_blocking_evaluator()
        results = evaluator.evaluate(inputs, context=context)
        batch = BatchEvalResult.from_results(results)

        return EvaluatorResult(
            batch=batch,
            mode=ExecutionMode.BLOCKING,
        )

    def _run_non_blocking(
        self,
        inputs: Dict[str, Any],
        context: Optional[EvalContext],
        callback: Optional[Callable[[EvalResult], None]],
    ) -> EvaluatorResult:
        """Run evaluations in non-blocking mode."""
        evaluator = self._get_non_blocking_evaluator()
        future = evaluator.evaluate(inputs, context=context, callback=callback)

        return EvaluatorResult(
            future=future,
            mode=ExecutionMode.NON_BLOCKING,
        )

    def _run_distributed(
        self,
        inputs: Dict[str, Any],
        context: Optional[EvalContext],
        callback: Optional[Callable[[EvalResult], None]],
    ) -> EvaluatorResult:
        """
        Run evaluations in distributed mode.

        Uses the configured backend for execution. Submits each evaluation
        as a task to the backend, collects results, and returns a BatchEvalResult.
        Falls back to non-blocking if no backend is configured.
        """
        if self._backend is None:
            return self._run_non_blocking(inputs, context, callback)

        context_dict = None
        if context and hasattr(context, "to_dict"):
            context_dict = context.to_dict()

        # Submit each evaluation as a task to the backend, collecting handles
        # or immediate failure results
        handles = []  # List of (index, handle) for successful submissions
        results = [None] * len(self.evaluations)  # Pre-allocate result slots

        for i, evaluation in enumerate(self.evaluations):
            eval_name = getattr(evaluation, "name", evaluation.__class__.__name__)
            eval_version = getattr(evaluation, "version", "1.0.0")
            try:
                handle = self._backend.submit(
                    _execute_single_evaluation,
                    args=(evaluation, inputs, self.validate_inputs),
                    context=context_dict,
                )
                handles.append((i, handle))
            except Exception as e:
                error_result = EvalResult(
                    value=None,
                    eval_name=eval_name,
                    eval_version=eval_version,
                    latency_ms=0.0,
                    status=EvalStatus.FAILED,
                    error=str(e),
                )
                results[i] = error_result
                if callback:
                    callback(error_result)

        # Collect results from successful submissions
        timeout = self._backend_timeout
        for i, handle in handles:
            evaluation = self.evaluations[i]
            eval_name = getattr(evaluation, "name", evaluation.__class__.__name__)
            eval_version = getattr(evaluation, "version", "1.0.0")
            try:
                result = self._backend.get_result(handle, timeout=timeout)
                results[i] = result
                if callback:
                    callback(result)
            except Exception as e:
                error_result = EvalResult(
                    value=None,
                    eval_name=eval_name,
                    eval_version=eval_version,
                    latency_ms=0.0,
                    status=EvalStatus.FAILED,
                    error=str(e),
                )
                results[i] = error_result
                if callback:
                    callback(error_result)

        batch = BatchEvalResult.from_results(results)
        return EvaluatorResult(
            batch=batch,
            mode=ExecutionMode.DISTRIBUTED,
        )

    @property
    def _backend_timeout(self) -> float:
        """Get timeout from backend config, defaulting to 300s."""
        if self._backend and hasattr(self._backend, "config"):
            config = self._backend.config
            if hasattr(config, "timeout_seconds"):
                return config.timeout_seconds
        return 300.0

    def _get_blocking_evaluator(self) -> BlockingEvaluator:
        """Get or create the blocking evaluator."""
        if self._blocking is None:
            self._blocking = BlockingEvaluator(
                evaluations=self.evaluations,
                auto_enrich_span=self.auto_enrich_span,
                fail_fast=self.fail_fast,
                validate_inputs=self.validate_inputs,
            )
        return self._blocking

    def _get_non_blocking_evaluator(self) -> NonBlockingEvaluator:
        """Get or create the non-blocking evaluator."""
        if self._non_blocking is None:
            self._non_blocking = NonBlockingEvaluator(
                evaluations=self.evaluations,
                max_workers=self.max_workers,
                auto_enrich_span=self.auto_enrich_span,
                fail_fast=self.fail_fast,
                validate_inputs=self.validate_inputs,
            )
        return self._non_blocking

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the evaluator and release resources.

        Args:
            wait: Whether to wait for pending evaluations
        """
        if self._non_blocking:
            self._non_blocking.shutdown(wait=wait)
            self._non_blocking = None
        if self._backend:
            self._backend.shutdown(wait=wait)

    def __enter__(self) -> "FrameworkEvaluator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown(wait=True)


# Factory functions for common configurations


def blocking_evaluator(
    *evaluations: BaseEvaluation,
    auto_enrich_span: bool = True,
    fail_fast: bool = False,
) -> FrameworkEvaluator:
    """
    Create a blocking evaluator.

    Args:
        *evaluations: Evaluations to run
        auto_enrich_span: Whether to enrich OTEL spans
        fail_fast: Stop on first failure

    Returns:
        Configured Evaluator in blocking mode

    Example:
        evaluator = blocking_evaluator(ToxicityEval(), BiasEval())
        result = evaluator.run({"response": "..."})
    """
    return FrameworkEvaluator(
        evaluations=list(evaluations),
        mode=ExecutionMode.BLOCKING,
        auto_enrich_span=auto_enrich_span,
        fail_fast=fail_fast,
    )


def async_evaluator(
    *evaluations: BaseEvaluation,
    max_workers: int = 4,
    auto_enrich_span: bool = True,
    backend: Optional[Backend] = None,
) -> FrameworkEvaluator:
    """
    Create a non-blocking (async) evaluator.

    Args:
        *evaluations: Evaluations to run
        max_workers: Maximum concurrent evaluations
        auto_enrich_span: Whether to enrich OTEL spans
        backend: Custom execution backend

    Returns:
        Configured Evaluator in non-blocking mode

    Example:
        evaluator = async_evaluator(ToxicityEval(), BiasEval())
        result = evaluator.run({"response": "..."})  # Returns immediately
        batch = result.wait()  # Get results when needed
    """
    return FrameworkEvaluator(
        evaluations=list(evaluations),
        mode=ExecutionMode.NON_BLOCKING,
        max_workers=max_workers,
        auto_enrich_span=auto_enrich_span,
        backend=backend,
    )


def distributed_evaluator(
    *evaluations: BaseEvaluation,
    backend: Backend,
    auto_enrich_span: bool = True,
) -> FrameworkEvaluator:
    """
    Create a distributed evaluator with custom backend.

    Args:
        *evaluations: Evaluations to run
        backend: Execution backend (Temporal, Celery, Ray, etc.)
        auto_enrich_span: Whether to enrich OTEL spans

    Returns:
        Configured Evaluator in distributed mode

    Example:
        backend = TemporalBackend(config)
        evaluator = distributed_evaluator(
            ToxicityEval(),
            backend=backend,
        )
        result = evaluator.run({"response": "..."})
    """
    return FrameworkEvaluator(
        evaluations=list(evaluations),
        mode=ExecutionMode.DISTRIBUTED,
        backend=backend,
        auto_enrich_span=auto_enrich_span,
    )


def _execute_single_evaluation(
    evaluation: BaseEvaluation,
    inputs: Dict[str, Any],
    validate: bool = True,
) -> EvalResult:
    """
    Execute a single evaluation, suitable for submission to any backend.

    Handles validation, timing, and error capture.

    Args:
        evaluation: The evaluation to run
        inputs: Input data
        validate: Whether to validate inputs

    Returns:
        EvalResult with value or error
    """
    eval_name = getattr(evaluation, "name", evaluation.__class__.__name__)
    eval_version = getattr(evaluation, "version", "1.0.0")

    # Validate inputs if enabled
    if validate and hasattr(evaluation, "validate_inputs"):
        try:
            errors = evaluation.validate_inputs(inputs)
            if errors:
                error_msg = "; ".join(str(e) for e in errors) if isinstance(errors, list) else str(errors)
                return EvalResult(
                    value=None,
                    eval_name=eval_name,
                    eval_version=eval_version,
                    latency_ms=0.0,
                    status=EvalStatus.FAILED,
                    error=f"Validation error: {error_msg}",
                )
        except Exception as e:
            return EvalResult(
                value=None,
                eval_name=eval_name,
                eval_version=eval_version,
                latency_ms=0.0,
                status=EvalStatus.FAILED,
                error=f"Validation exception: {e}",
            )

    start = time.perf_counter()
    try:
        value = evaluation.evaluate(inputs)
        latency_ms = (time.perf_counter() - start) * 1000
        return EvalResult(
            value=value,
            eval_name=eval_name,
            eval_version=eval_version,
            latency_ms=latency_ms,
            status=EvalStatus.COMPLETED,
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return EvalResult(
            value=None,
            eval_name=eval_name,
            eval_version=eval_version,
            latency_ms=latency_ms,
            status=EvalStatus.FAILED,
            error=str(e),
        )


def resilient_evaluator(
    *evaluations: BaseEvaluation,
    backend: Optional[Backend] = None,
    resilience: Optional[Any] = None,
    fallback_backend: Optional[Backend] = None,
    event_callback: Optional[Callable] = None,
    auto_enrich_span: bool = True,
) -> FrameworkEvaluator:
    """
    Create a distributed evaluator with resilience protections.

    Wraps a backend with circuit breaker, rate limiting, retry, and fallback
    capabilities, then creates an Evaluator in DISTRIBUTED mode.

    Args:
        *evaluations: Evaluations to run
        backend: Execution backend (uses ThreadPoolBackend if None)
        resilience: ResilienceConfig for resilience settings (optional)
        fallback_backend: Optional fallback backend for degradation
        event_callback: Callback for resilience events
        auto_enrich_span: Whether to enrich OTEL spans

    Returns:
        Configured Evaluator in distributed mode with resilience

    Example:
        from fi.evals.framework import resilient_evaluator, ResilienceConfig
        from fi.evals.framework import CircuitBreakerConfig, RateLimitConfig

        evaluator = resilient_evaluator(
            ToxicityEval(),
            BiasEval(),
            resilience=ResilienceConfig(
                circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
                rate_limit=RateLimitConfig(requests_per_second=10),
            ),
        )
        result = evaluator.run({"response": "..."})
    """
    from .resilience import ResilientBackend, ResilienceConfig

    # Create underlying backend if not provided
    underlying = backend or ThreadPoolBackend()

    # Wrap with resilience
    resilience_config = resilience or ResilienceConfig()
    resilient = ResilientBackend(
        underlying=underlying,
        config=resilience_config,
        fallback_backend=fallback_backend,
        event_callback=event_callback,
    )

    return FrameworkEvaluator(
        evaluations=list(evaluations),
        mode=ExecutionMode.DISTRIBUTED,
        backend=resilient,
        auto_enrich_span=auto_enrich_span,
    )
