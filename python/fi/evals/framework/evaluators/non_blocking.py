"""
Non-blocking evaluation executor.

Provides async/background evaluation with zero latency impact on the main thread.
Uses thread pools for local execution with context propagation.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import uuid
import time

from ..types import EvalResult, EvalStatus, BatchEvalResult
from ..context import EvalContext
from ..protocols import BaseEvaluation
from ..registry import register_span, get_span, register_current_span
from ..propagation import ContextCarrier, enrich_span_by_context


@dataclass
class EvalFuture:
    """
    Future representing a pending evaluation.

    Wraps the underlying executor future and provides convenience methods.

    Example:
        future = evaluator.evaluate(inputs)
        # Do other work...
        result = future.result()  # Block until complete
    """

    future: Future
    eval_name: str
    eval_version: str
    context: Optional[EvalContext] = None
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def done(self) -> bool:
        """Check if evaluation is complete."""
        return self.future.done()

    def result(self, timeout: Optional[float] = None) -> EvalResult:
        """
        Get the evaluation result, blocking if necessary.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            EvalResult from the evaluation

        Raises:
            TimeoutError: If timeout exceeded
            Exception: If evaluation raised an exception
        """
        return self.future.result(timeout=timeout)

    def cancel(self) -> bool:
        """
        Attempt to cancel the evaluation.

        Returns:
            True if cancelled, False if already running/complete
        """
        return self.future.cancel()

    def cancelled(self) -> bool:
        """Check if evaluation was cancelled."""
        return self.future.cancelled()

    def add_done_callback(self, fn: Callable[["EvalFuture"], None]) -> None:
        """
        Add callback to run when evaluation completes.

        Args:
            fn: Callback function taking this EvalFuture
        """

        def wrapper(f):
            fn(self)

        self.future.add_done_callback(wrapper)


@dataclass
class BatchEvalFuture:
    """Future representing a batch of evaluations."""

    futures: List[EvalFuture]
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def done(self) -> bool:
        """Check if all evaluations are complete."""
        return all(f.done() for f in self.futures)

    def results(self, timeout: Optional[float] = None) -> BatchEvalResult:
        """
        Get all results, blocking if necessary.

        Args:
            timeout: Maximum seconds to wait for ALL results

        Returns:
            BatchEvalResult with all evaluation results
        """
        results = []
        for f in self.futures:
            try:
                results.append(f.result(timeout=timeout))
            except Exception as e:
                # Create failure result for exceptions
                results.append(
                    EvalResult.failure(
                        eval_name=f.eval_name,
                        eval_version=f.eval_version,
                        error=str(e),
                    )
                )
        return BatchEvalResult.from_results(results)

    def cancel_all(self) -> int:
        """
        Cancel all pending evaluations.

        Returns:
            Number of successfully cancelled evaluations
        """
        return sum(1 for f in self.futures if f.cancel())


class NonBlockingEvaluator:
    """
    Non-blocking evaluation executor using thread pool.

    Runs evaluations in background threads with zero latency impact.
    Automatically enriches spans with results when complete.

    Example:
        evaluator = NonBlockingEvaluator(
            evaluations=[ToxicityEval(), BiasEval()],
            max_workers=4,
        )

        # Returns immediately
        future = evaluator.evaluate({"response": "..."})

        # Later, check results
        if future.done():
            result = future.result()

        # Or wait for result
        result = future.result(timeout=5.0)

    Thread Safety:
        This class is thread-safe. Multiple threads can call evaluate()
        concurrently.
    """

    def __init__(
        self,
        evaluations: Optional[List[BaseEvaluation]] = None,
        max_workers: int = 4,
        auto_enrich_span: bool = True,
        fail_fast: bool = False,
        validate_inputs: bool = True,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Initialize the non-blocking evaluator.

        Args:
            evaluations: List of evaluations to run
            max_workers: Maximum concurrent evaluations
            auto_enrich_span: Whether to auto-enrich spans with results
            fail_fast: Stop on first failure (for batch operations)
            validate_inputs: Whether to validate inputs before evaluation
            executor: Custom executor (uses internal pool if None)
        """
        self.evaluations = list(evaluations) if evaluations else []
        self.max_workers = max_workers
        self.auto_enrich_span = auto_enrich_span
        self.fail_fast = fail_fast
        self.validate_inputs = validate_inputs

        # Use provided executor or create our own
        self._executor = executor
        self._owns_executor = executor is None
        self._lock = threading.Lock()

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Get or create the thread pool executor."""
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=self.max_workers,
                        thread_name_prefix="eval_worker_",
                    )
        return self._executor

    def add_evaluation(self, evaluation: BaseEvaluation) -> "NonBlockingEvaluator":
        """
        Add an evaluation to run.

        Args:
            evaluation: The evaluation to add

        Returns:
            Self for chaining
        """
        self.evaluations.append(evaluation)
        return self

    def evaluate(
        self,
        inputs: Dict[str, Any],
        evaluations: Optional[List[BaseEvaluation]] = None,
        context: Optional[EvalContext] = None,
        callback: Optional[Callable[[EvalResult], None]] = None,
    ) -> BatchEvalFuture:
        """
        Run evaluations in background, returning immediately.

        Args:
            inputs: Input data for evaluations
            evaluations: Override evaluations (uses instance evals if None)
            context: Trace context for span enrichment
            callback: Optional callback for each result

        Returns:
            BatchEvalFuture to track/retrieve results

        Raises:
            ValueError: If no evaluations configured
        """
        evals_to_run = evaluations if evaluations is not None else self.evaluations
        if not evals_to_run:
            raise ValueError("No evaluations to run")

        # Capture context for span enrichment
        carrier = ContextCarrier.capture() if context is None else ContextCarrier(context)

        # Register current span if enrichment enabled
        if self.auto_enrich_span:
            register_current_span()

        # Submit all evaluations
        futures = []
        for evaluation in evals_to_run:
            future = self._submit_evaluation(
                evaluation=evaluation,
                inputs=inputs,
                carrier=carrier,
                callback=callback,
            )
            futures.append(future)

        return BatchEvalFuture(futures=futures)

    def evaluate_single(
        self,
        evaluation: BaseEvaluation,
        inputs: Dict[str, Any],
        context: Optional[EvalContext] = None,
        callback: Optional[Callable[[EvalResult], None]] = None,
    ) -> EvalFuture:
        """
        Run a single evaluation in background.

        Args:
            evaluation: The evaluation to run
            inputs: Input data
            context: Trace context
            callback: Optional result callback

        Returns:
            EvalFuture to track/retrieve result
        """
        carrier = ContextCarrier.capture() if context is None else ContextCarrier(context)

        if self.auto_enrich_span:
            register_current_span()

        return self._submit_evaluation(
            evaluation=evaluation,
            inputs=inputs,
            carrier=carrier,
            callback=callback,
        )

    def _submit_evaluation(
        self,
        evaluation: BaseEvaluation,
        inputs: Dict[str, Any],
        carrier: ContextCarrier,
        callback: Optional[Callable[[EvalResult], None]] = None,
    ) -> EvalFuture:
        """Submit a single evaluation to the thread pool."""
        eval_name = getattr(evaluation, "name", evaluation.__class__.__name__)
        eval_version = getattr(evaluation, "version", "1.0.0")

        future = self.executor.submit(
            self._run_evaluation,
            evaluation=evaluation,
            inputs=inputs,
            carrier=carrier,
            callback=callback,
        )

        return EvalFuture(
            future=future,
            eval_name=eval_name,
            eval_version=eval_version,
            context=carrier.context,
        )

    def _run_evaluation(
        self,
        evaluation: BaseEvaluation,
        inputs: Dict[str, Any],
        carrier: ContextCarrier,
        callback: Optional[Callable[[EvalResult], None]] = None,
    ) -> EvalResult:
        """
        Run evaluation in background thread.

        Handles:
        - Input validation
        - Timing
        - Error handling
        - Span enrichment
        - Callback invocation
        """
        eval_name = getattr(evaluation, "name", evaluation.__class__.__name__)
        eval_version = getattr(evaluation, "version", "1.0.0")

        start_time = time.perf_counter()

        try:
            # Validate inputs if enabled
            if self.validate_inputs:
                validate = getattr(evaluation, "validate_inputs", None)
                if validate:
                    errors = validate(inputs)
                    if errors:
                        return EvalResult.failure(
                            eval_name=eval_name,
                            eval_version=eval_version,
                            error=f"Validation errors: {errors}",
                        )

            # Run the evaluation
            value = evaluation.evaluate(inputs)

            latency_ms = (time.perf_counter() - start_time) * 1000

            result = EvalResult(
                value=value,
                eval_name=eval_name,
                eval_version=eval_version,
                latency_ms=latency_ms,
                status=EvalStatus.COMPLETED,
            )

            # Enrich span with results
            if self.auto_enrich_span:
                self._enrich_span(evaluation, result, carrier)

            # Invoke callback
            if callback:
                try:
                    callback(result)
                except Exception:
                    pass  # Don't fail evaluation for callback errors

            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            result = EvalResult.failure(
                eval_name=eval_name,
                eval_version=eval_version,
                error=str(e),
            )
            result.latency_ms = latency_ms

            # Enrich span with failure
            if self.auto_enrich_span:
                self._enrich_span_failure(eval_name, result, carrier)

            # Invoke callback even on failure
            if callback:
                try:
                    callback(result)
                except Exception:
                    pass

            return result

    def _enrich_span(
        self,
        evaluation: BaseEvaluation,
        result: EvalResult,
        carrier: ContextCarrier,
    ) -> bool:
        """Enrich the original span with evaluation results."""
        eval_name = result.eval_name.replace("-", "_").replace(" ", "_")
        prefix = f"eval.{eval_name}"

        # Get span attributes from evaluation
        get_attrs = getattr(evaluation, "get_span_attributes", None)
        if get_attrs:
            try:
                eval_attrs = get_attrs(result.value)
            except Exception:
                eval_attrs = {}
        else:
            eval_attrs = {}

        # Build attributes
        attributes = {
            f"{prefix}.status": result.status.value,
            f"{prefix}.latency_ms": result.latency_ms,
            f"{prefix}.version": result.eval_version,
        }

        # Add eval-specific attributes
        for key, value in eval_attrs.items():
            if isinstance(value, (str, int, float, bool)):
                attributes[f"{prefix}.{key}"] = value

        return carrier.enrich_span(attributes)

    def _enrich_span_failure(
        self,
        eval_name: str,
        result: EvalResult,
        carrier: ContextCarrier,
    ) -> bool:
        """Enrich span with failure information."""
        safe_name = eval_name.replace("-", "_").replace(" ", "_")
        prefix = f"eval.{safe_name}"

        attributes = {
            f"{prefix}.status": result.status.value,
            f"{prefix}.latency_ms": result.latency_ms,
            f"{prefix}.error": result.error or "Unknown error",
        }

        return carrier.enrich_span(attributes)

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        if self._executor and self._owns_executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def __enter__(self) -> "NonBlockingEvaluator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown(wait=True)


def non_blocking_evaluate(
    inputs: Dict[str, Any],
    *evaluations: BaseEvaluation,
    max_workers: int = 4,
    auto_enrich_span: bool = True,
    callback: Optional[Callable[[EvalResult], None]] = None,
) -> BatchEvalFuture:
    """
    Convenience function for non-blocking evaluation.

    Runs evaluations in background and returns immediately.

    Args:
        inputs: Input data for evaluations
        *evaluations: Evaluations to run
        max_workers: Maximum concurrent evaluations
        auto_enrich_span: Whether to enrich spans with results
        callback: Optional callback for each result

    Returns:
        BatchEvalFuture to track/retrieve results

    Example:
        future = non_blocking_evaluate(
            {"response": "..."},
            ToxicityEval(),
            BiasEval(),
        )
        # Do other work...
        results = future.results()
    """
    evaluator = NonBlockingEvaluator(
        evaluations=list(evaluations),
        max_workers=max_workers,
        auto_enrich_span=auto_enrich_span,
    )
    return evaluator.evaluate(inputs, callback=callback)


class EvalResultAggregator:
    """
    Aggregates results from multiple async evaluations.

    Useful for collecting results from different evaluation runs.

    Example:
        aggregator = EvalResultAggregator()

        # Add results as they come in
        aggregator.add(future1.result())
        aggregator.add(future2.result())

        # Get aggregated results
        batch = aggregator.to_batch()
        print(f"Success rate: {batch.success_rate}")
    """

    def __init__(self):
        self._results: List[EvalResult] = []
        self._lock = threading.Lock()

    def add(self, result: EvalResult) -> None:
        """Add a result to the aggregator."""
        with self._lock:
            self._results.append(result)

    def add_all(self, results: List[EvalResult]) -> None:
        """Add multiple results."""
        with self._lock:
            self._results.extend(results)

    def to_batch(self) -> BatchEvalResult:
        """Get aggregated results as a batch."""
        with self._lock:
            return BatchEvalResult.from_results(list(self._results))

    def clear(self) -> int:
        """Clear all results and return count cleared."""
        with self._lock:
            count = len(self._results)
            self._results.clear()
            return count

    @property
    def count(self) -> int:
        """Get current result count."""
        with self._lock:
            return len(self._results)
