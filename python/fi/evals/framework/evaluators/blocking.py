"""
Blocking (synchronous) evaluator.

This module provides BlockingEvaluator, which runs evaluations synchronously
and optionally enriches the current OTEL span with results.
"""

import time
from typing import List, Any, Dict, Optional, Union
from ..types import EvalResult, EvalStatus, BatchEvalResult
from ..protocols import BaseEvaluation
from ..context import EvalContext
from ..enrichment import enrich_current_span, add_eval_event


class BlockingEvaluator:
    """
    Synchronous evaluation execution.

    Runs evaluations one-by-one (or in parallel within the same thread using
    concurrent execution if configured) and waits for all results before returning.

    Use when:
    - Results are needed immediately
    - Latency is acceptable
    - Development and testing
    - Simple local evaluations

    Example:
        # Basic usage
        evaluator = BlockingEvaluator([FaithfulnessEval(), RelevanceEval()])
        results = evaluator.evaluate({
            "query": "What is Python?",
            "response": "Python is a programming language.",
            "context": ["Python is a high-level programming language."],
        })

        for result in results:
            print(f"{result.eval_name}: {result.value}")

        # With OTEL span enrichment
        with trace_llm_call("chat", model="gpt-4") as span:
            response = llm.complete(prompt)
            results = evaluator.evaluate({"response": response})
            # Span now has eval.faithfulness.score, eval.relevance.score, etc.
    """

    def __init__(
        self,
        evaluations: Optional[List[BaseEvaluation]] = None,
        auto_enrich_span: bool = True,
        fail_fast: bool = False,
        validate_inputs: bool = True,
    ):
        """
        Initialize blocking evaluator.

        Args:
            evaluations: List of evaluations to run. Can also be added later
                        with add_evaluation() or passed to evaluate().
            auto_enrich_span: Automatically add results to current OTEL span.
                             Set to False if you want to handle span enrichment yourself.
            fail_fast: Stop on first evaluation failure. If False, continues
                      running remaining evaluations even if one fails.
            validate_inputs: Run input validation before each evaluation.
        """
        self.evaluations = list(evaluations) if evaluations else []
        self.auto_enrich_span = auto_enrich_span
        self.fail_fast = fail_fast
        self.validate_inputs = validate_inputs

    def add_evaluation(self, evaluation: BaseEvaluation) -> "BlockingEvaluator":
        """
        Add an evaluation to run.

        Args:
            evaluation: Evaluation to add

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
    ) -> List[EvalResult]:
        """
        Run all evaluations synchronously.

        Args:
            inputs: Evaluation inputs (e.g., query, response, context)
            evaluations: Optional list of evaluations to run instead of
                        the ones configured in __init__
            context: Optional EvalContext. If not provided, captures from
                    current OTEL span.

        Returns:
            List of EvalResult objects, one per evaluation

        Raises:
            ValueError: If no evaluations are configured and none provided
        """
        evals_to_run = evaluations if evaluations is not None else self.evaluations

        if not evals_to_run:
            raise ValueError(
                "No evaluations to run. Either pass evaluations to __init__ "
                "or to evaluate()."
            )

        if context is None:
            context = EvalContext.from_current_span()

        results = []

        for evaluation in evals_to_run:
            result = self._run_single(evaluation, inputs)
            results.append(result)

            # Enrich current span with result
            if self.auto_enrich_span:
                self._enrich_span(evaluation, result)

            # Check fail fast
            if self.fail_fast and result.status == EvalStatus.FAILED:
                # Mark remaining evaluations as cancelled
                for remaining in evals_to_run[len(results):]:
                    results.append(EvalResult(
                        value=None,
                        eval_name=remaining.name,
                        eval_version=remaining.version,
                        latency_ms=0,
                        status=EvalStatus.CANCELLED,
                        error="Cancelled due to previous failure (fail_fast=True)",
                    ))
                break

        return results

    def evaluate_single(
        self,
        evaluation: BaseEvaluation,
        inputs: Dict[str, Any],
    ) -> EvalResult:
        """
        Run a single evaluation.

        Convenience method for running just one evaluation.

        Args:
            evaluation: The evaluation to run
            inputs: Evaluation inputs

        Returns:
            EvalResult for this evaluation
        """
        result = self._run_single(evaluation, inputs)

        if self.auto_enrich_span:
            self._enrich_span(evaluation, result)

        return result

    def evaluate_batch(
        self,
        inputs_batch: List[Dict[str, Any]],
        evaluations: Optional[List[BaseEvaluation]] = None,
    ) -> BatchEvalResult:
        """
        Run evaluations on a batch of inputs.

        Each input in the batch is evaluated against all evaluations.

        Args:
            inputs_batch: List of input dicts
            evaluations: Optional evaluations to run

        Returns:
            BatchEvalResult with aggregated results
        """
        all_results = []

        for inputs in inputs_batch:
            results = self.evaluate(inputs, evaluations=evaluations)
            all_results.extend(results)

        return BatchEvalResult.from_results(all_results)

    def _run_single(
        self,
        evaluation: BaseEvaluation,
        inputs: Dict[str, Any],
    ) -> EvalResult:
        """
        Run a single evaluation with timing and error handling.

        Args:
            evaluation: The evaluation to run
            inputs: Evaluation inputs

        Returns:
            EvalResult with value or error
        """
        eval_name = getattr(evaluation, 'name', evaluation.__class__.__name__)
        eval_version = getattr(evaluation, 'version', '1.0.0')

        # Validate inputs if enabled
        if self.validate_inputs:
            validation_error = self._validate_inputs(evaluation, inputs)
            if validation_error:
                return EvalResult(
                    value=None,
                    eval_name=eval_name,
                    eval_version=eval_version,
                    latency_ms=0,
                    status=EvalStatus.FAILED,
                    error=f"Validation error: {validation_error}",
                )

        # Run evaluation with timing
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

    def _validate_inputs(
        self,
        evaluation: BaseEvaluation,
        inputs: Dict[str, Any],
    ) -> Optional[str]:
        """
        Validate inputs for an evaluation.

        Args:
            evaluation: The evaluation
            inputs: Inputs to validate

        Returns:
            Error message if invalid, None if valid
        """
        # Check if evaluation has validate_inputs method
        if hasattr(evaluation, 'validate_inputs'):
            try:
                return evaluation.validate_inputs(inputs)
            except Exception as e:
                return f"Validation method raised exception: {e}"

        return None

    def _enrich_span(
        self,
        evaluation: BaseEvaluation,
        result: EvalResult,
    ) -> None:
        """
        Add evaluation results to current OTEL span.

        Args:
            evaluation: The evaluation that produced the result
            result: The evaluation result
        """
        eval_name = result.eval_name

        # Build attributes dict
        attributes: Dict[str, Any] = {
            "latency_ms": result.latency_ms,
            "status": result.status.value,
            "version": result.eval_version,
        }

        if result.error:
            attributes["error"] = result.error

        # Add evaluation-specific attributes
        if result.value is not None and hasattr(evaluation, 'get_span_attributes'):
            try:
                eval_attrs = evaluation.get_span_attributes(result.value)
                attributes.update(eval_attrs)
            except Exception:
                # Don't fail enrichment if get_span_attributes fails
                pass

        enrich_current_span(eval_name, attributes)

    def __len__(self) -> int:
        """Return number of configured evaluations."""
        return len(self.evaluations)

    def __iter__(self):
        """Iterate over configured evaluations."""
        return iter(self.evaluations)


def blocking_evaluate(
    inputs: Dict[str, Any],
    *evaluations: BaseEvaluation,
    auto_enrich_span: bool = True,
) -> List[EvalResult]:
    """
    Convenience function for one-shot blocking evaluation.

    Args:
        inputs: Evaluation inputs
        *evaluations: Evaluations to run
        auto_enrich_span: Whether to enrich current span

    Returns:
        List of EvalResult objects

    Example:
        results = blocking_evaluate(
            {"response": "Hello world"},
            ToxicityEval(),
            PIIDetectionEval(),
        )
    """
    evaluator = BlockingEvaluator(
        evaluations=list(evaluations),
        auto_enrich_span=auto_enrich_span,
    )
    return evaluator.evaluate(inputs)
