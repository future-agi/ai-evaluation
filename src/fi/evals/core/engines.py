"""
Engine implementations for evaluate().

    LocalEngine  — wraps BaseMetric subclasses (no API key)
    TuringEngine — wraps the cloud Evaluator HTTP client (template-based evals)
    LLMEngine    — wraps CustomLLMJudge + LiteLLMProvider
"""

import inspect
import json as _json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .result import EvalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_parse_json(val: Any) -> Any:
    """Try to parse a string as JSON, returning the parsed value or original."""
    if isinstance(val, str):
        try:
            return _json.loads(val)
        except (_json.JSONDecodeError, TypeError):
            return val
    return val


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
            context (str/list) -> contexts (list)
            input/question -> query
            expected_output/expected_response/ground_truth -> reference
        """
        model_fields = set(metric.input_model.model_fields.keys())
        # Merge both dicts so all user kwargs are searchable
        combined = {**all_user_kwargs, **inputs}
        mapped = {k: v for k, v in combined.items() if k in model_fields}

        # output -> response
        if "response" not in mapped and "output" in combined:
            if "response" in model_fields:
                mapped["response"] = combined["output"]

        # expected_response fallback chain (for string metrics)
        if "expected_response" not in mapped and "expected_response" in model_fields:
            for src in ("expected_response", "keyword", "context", "expected_output"):
                if src in combined:
                    mapped["expected_response"] = combined[src]
                    break

        # context (str or list) -> contexts (list) — for RAG metrics
        if "contexts" not in mapped and "contexts" in model_fields:
            for src in ("context", "contexts"):
                if src in combined:
                    val = combined[src]
                    mapped["contexts"] = val if isinstance(val, list) else [val]
                    break

        # input/question -> query — for RAG metrics
        if "query" not in mapped and "query" in model_fields:
            for src in ("query", "input", "question"):
                if src in combined:
                    mapped["query"] = combined[src]
                    break

        # expected_output/ground_truth -> reference — for RAG metrics
        if "reference" not in mapped and "reference" in model_fields:
            for src in ("reference", "expected_response", "expected_output", "ground_truth"):
                if src in combined:
                    mapped["reference"] = combined[src]
                    break

        # expected -> expected (for structured metrics, also try expected_output)
        if "expected" not in mapped and "expected" in model_fields:
            for src in ("expected", "expected_output"):
                if src in combined:
                    mapped["expected"] = _try_parse_json(combined[src])
                    break

        # schema -> schema (try expected_response as schema for structured metrics)
        if "schema" not in mapped and "schema" in model_fields:
            if "expected_response" in combined:
                mapped["schema"] = _try_parse_json(combined["expected_response"])

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

        # Dynamic registry drives input filtering — backend is source of truth
        # for required_keys. Falls back to pass-through when eval isn't known
        # (registry miss, offline, etc.) so the backend can return its own
        # validation error.
        from .cloud_registry import map_inputs_to_backend

        evaluator = self._get_evaluator()
        mapped_inputs = map_inputs_to_backend(
            eval_name,
            inputs,
            base_url=evaluator._base_url,
            api_key=getattr(evaluator, "_fi_api_key", None) or self._api_key,
            secret_key=getattr(evaluator, "_fi_secret_key", None) or self._secret_key,
        )

        # Template class is optional — if the SDK hasn't shipped a class
        # for a new backend eval, just send the name as a string.
        template_cls = self._resolve_template_class(eval_name)
        eval_arg: Any = template_cls() if template_cls is not None else eval_name

        start = time.perf_counter()
        try:
            batch = evaluator.evaluate(
                eval_templates=eval_arg,
                inputs=mapped_inputs,
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
