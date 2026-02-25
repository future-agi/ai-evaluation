"""
Engine implementations for evaluate().

    LocalEngine  — wraps BaseMetric subclasses (no API key)
    TuringEngine — wraps the cloud Evaluator HTTP client (template-based evals)
    LLMEngine    — wraps CustomLLMJudge + LiteLLMProvider
"""

import inspect
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .result import EvalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_score(output: Any) -> Optional[float]:
    """Convert various metric output types to a 0-1 float."""
    if output is None:
        return None
    if isinstance(output, bool):
        return 1.0 if output else 0.0
    if isinstance(output, (int, float)):
        return float(output)
    if isinstance(output, str):
        try:
            return float(output)
        except ValueError:
            low = output.strip().lower()
            if low in ("true", "yes", "pass", "passed"):
                return 1.0
            if low in ("false", "no", "fail", "failed"):
                return 0.0
    return None


def _extract_result(eval_name: str, batch: Any, latency_ms: float) -> EvalResult:
    """Pull the first result out of a BatchRunResult-like object."""
    r = batch.eval_results[0] if batch.eval_results else None
    if r is None:
        return EvalResult(
            eval_name=eval_name,
            status="failed",
            error="Evaluator returned no results",
            latency_ms=latency_ms,
        )
    return EvalResult(
        eval_name=eval_name,
        score=_normalise_score(r.output),
        reason=r.reason or "",
        latency_ms=latency_ms,
        metadata={
            k: getattr(r, k, None)
            for k in ("eval_id", "output_type")
            if getattr(r, k, None) is not None
        },
    )


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Engine(ABC):
    """Base class for evaluation engines."""

    @abstractmethod
    def run(
        self,
        eval_name: str,
        inputs: Dict[str, Any],
        *,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        ...


# ---------------------------------------------------------------------------
# LocalEngine
# ---------------------------------------------------------------------------

class LocalEngine(Engine):
    """Runs evaluations locally using BaseMetric subclasses."""

    @staticmethod
    def _config_keys() -> set:
        from ..types import ConfigPossibleValues
        return set(ConfigPossibleValues.model_fields.keys())

    def run(
        self,
        eval_name: str,
        inputs: Dict[str, Any],
        *,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        _ = model, prompt  # unused — local metrics don't need these
        from ..local.registry import get_registry

        config_keys = self._config_keys()
        merged_config = dict(config or {})
        metric_inputs: Dict[str, Any] = {}
        for k, v in inputs.items():
            (merged_config if k in config_keys else metric_inputs)[k] = v

        metric = get_registry().create(eval_name, merged_config)
        if metric is None:
            return EvalResult(
                eval_name=eval_name,
                status="failed",
                error=f"Local metric '{eval_name}' not found in registry",
            )

        metric_input = self._build_metric_input(metric, metric_inputs, inputs)

        start = time.perf_counter()
        try:
            batch = metric.evaluate([metric_input])
            return _extract_result(eval_name, batch, (time.perf_counter() - start) * 1000)
        except Exception as exc:
            return EvalResult(
                eval_name=eval_name,
                status="failed",
                error=str(exc),
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    @staticmethod
    def _build_metric_input(
        metric: Any,
        inputs: Dict[str, Any],
        all_user_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map user-facing kwargs to the Pydantic input fields the metric expects.

        Common mappings:
            output -> response
            keyword/context/expected_output -> expected_response
        """
        model_fields = set(metric.input_model.model_fields.keys())
        mapped = {k: v for k, v in inputs.items() if k in model_fields}

        if "response" not in mapped and "output" in inputs:
            mapped["response"] = inputs["output"]

        if "expected_response" not in mapped and "expected_response" in model_fields:
            for src in ("expected_response", "keyword", "context", "expected_output"):
                if src in all_user_kwargs:
                    mapped["expected_response"] = all_user_kwargs[src]
                    break

        return mapped


# ---------------------------------------------------------------------------
# TuringEngine
# ---------------------------------------------------------------------------

class TuringEngine(Engine):
    """Runs template-based evaluations on the Turing (FutureAGI) cloud platform.

    Custom prompts are NOT supported — use ``engine="llm"`` instead.
    """

    def __init__(
        self,
        fi_api_key: Optional[str] = None,
        fi_secret_key: Optional[str] = None,
        fi_base_url: Optional[str] = None,
    ):
        self._api_key = fi_api_key
        self._secret_key = fi_secret_key
        self._base_url = fi_base_url
        self._evaluator: Optional[Any] = None

    def _get_evaluator(self):
        if self._evaluator is None:
            from ..evaluator import Evaluator
            self._evaluator = Evaluator(
                fi_api_key=self._api_key,
                fi_secret_key=self._secret_key,
                fi_base_url=self._base_url,
            )
        return self._evaluator

    def run(
        self,
        eval_name: str,
        inputs: Dict[str, Any],
        *,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        _ = config  # unused — template config comes from the template class
        if prompt:
            return EvalResult(
                eval_name=eval_name or "custom_prompt",
                status="failed",
                error=(
                    "Custom prompts are not supported on the Turing engine. "
                    "Use engine='llm' with a model like 'claude-4.5-sonnet' "
                    "or 'gpt-5' instead."
                ),
            )

        template_cls = self._resolve_template_class(eval_name)
        if template_cls is None:
            return EvalResult(
                eval_name=eval_name,
                status="failed",
                error=f"Cloud template '{eval_name}' not found",
            )

        # Filter inputs to only keys the template's backend accepts,
        # and remap output↔input when the template expects one but not the other.
        accepted = set(template_cls.Input.model_fields.keys()) if hasattr(template_cls, 'Input') else None
        if accepted is not None:
            mapped = {}
            for k, v in inputs.items():
                if k in accepted:
                    mapped[k] = v
                elif k == "output" and "input" in accepted and "output" not in accepted:
                    mapped["input"] = v
                elif k == "input" and "output" in accepted and "input" not in accepted:
                    mapped["output"] = v
            inputs = mapped

        start = time.perf_counter()
        try:
            batch = self._get_evaluator().evaluate(
                eval_templates=template_cls(),
                inputs=inputs,
                model_name=model,
            )
            return _extract_result(eval_name, batch, (time.perf_counter() - start) * 1000)
        except Exception as exc:
            return EvalResult(
                eval_name=eval_name,
                status="failed",
                error=str(exc),
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    @staticmethod
    def _resolve_template_class(eval_name: str) -> Optional[type]:
        """Find the EvalTemplate subclass for a given eval_name."""
        from ..templates import EvalTemplate
        from .. import templates as tmpl_mod

        return next(
            (
                obj
                for _, obj in inspect.getmembers(tmpl_mod, inspect.isclass)
                if issubclass(obj, EvalTemplate)
                and obj is not EvalTemplate
                and getattr(obj, "eval_name", None) == eval_name
            ),
            None,
        )


# ---------------------------------------------------------------------------
# LLMEngine
# ---------------------------------------------------------------------------

class LLMEngine(Engine):
    """Runs evaluations using an LLM as a judge (via LiteLLM)."""

    def run(
        self,
        eval_name: str,
        inputs: Dict[str, Any],
        *,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        if not prompt:
            return EvalResult(
                eval_name=eval_name,
                status="failed",
                error="LLMEngine requires a 'prompt' parameter",
            )

        from ..llm.providers.litellm import LiteLLMProvider
        from ..metrics.llm_as_judges.custom_judge.metric import CustomLLMJudge

        judge = CustomLLMJudge(
            provider=LiteLLMProvider(),
            config={
                "grading_criteria": prompt,
                "model": model or "gpt-4o",
                **(config or {}),
            },
        )

        name = eval_name or "custom_llm_judge"
        start = time.perf_counter()
        try:
            batch = judge.evaluate([inputs])
            return _extract_result(name, batch, (time.perf_counter() - start) * 1000)
        except Exception as exc:
            return EvalResult(
                eval_name=name,
                status="failed",
                error=str(exc),
                latency_ms=(time.perf_counter() - start) * 1000,
            )
