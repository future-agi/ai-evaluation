"""
Engine implementations for evaluate().

Each engine wraps an existing layer of the SDK and normalises its output
into the unified EvalResult.

    LocalEngine  — wraps BaseMetric subclasses (no API key)
    TuringEngine — wraps the cloud Evaluator HTTP client + custom-prompt support
    LLMEngine    — wraps CustomLLMJudge + LiteLLMProvider
"""

import inspect
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .result import EvalResult

logger = logging.getLogger(__name__)


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

    # Keys that are config params, not metric inputs
    _CONFIG_KEYS = {
        "keyword", "keywords", "pattern", "case_sensitive",
        "comparator", "min_length", "max_length", "substring",
        "schema", "failure_threshold", "code", "url", "headers",
        "payload", "validations", "choices",
    }

    def run(
        self,
        eval_name: str,
        inputs: Dict[str, Any],
        *,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        from ..local.registry import get_registry

        # Split user kwargs into config params and metric inputs
        merged_config = dict(config or {})
        metric_inputs = {}
        for k, v in inputs.items():
            if k in self._CONFIG_KEYS:
                merged_config[k] = v
            else:
                metric_inputs[k] = v

        registry = get_registry()
        metric = registry.create(eval_name, merged_config)
        if metric is None:
            return EvalResult(
                eval_name=eval_name,
                status="failed",
                error=f"Local metric '{eval_name}' not found in registry",
            )

        # Build the input dict expected by the metric's Pydantic model.
        # Local metrics expect a *list* of inputs; we send one.
        metric_input = self._build_metric_input(metric, metric_inputs, inputs)

        start = time.perf_counter()
        try:
            batch = metric.evaluate([metric_input])
            latency_ms = (time.perf_counter() - start) * 1000

            if batch.eval_results:
                r = batch.eval_results[0]
                score = self._normalise_score(r.output)
                return EvalResult(
                    eval_name=eval_name,
                    score=score,
                    reason=r.reason or "",
                    latency_ms=latency_ms,
                )
            return EvalResult(
                eval_name=eval_name,
                status="failed",
                error="Metric returned no results",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return EvalResult(
                eval_name=eval_name,
                status="failed",
                error=str(exc),
                latency_ms=latency_ms,
            )

    # ------------------------------------------------------------------

    @staticmethod
    def _build_metric_input(metric, inputs: Dict[str, Any], all_user_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Map user-facing keyword args to the Pydantic input fields expected
        by the metric.

        Common mappings:
            output -> response
            keyword -> expected_response (for comparison metrics)
            context -> expected_response (for similarity metrics)

        Args:
            metric: The metric instance (has input_model with model_fields)
            inputs: The filtered metric inputs (config keys removed)
            all_user_kwargs: The original unfiltered user kwargs (for keyword/context)
        """
        mapped: Dict[str, Any] = {}
        model_fields = set(metric.input_model.model_fields.keys())

        # Direct pass-through of any keys that match the model
        for k, v in inputs.items():
            if k in model_fields:
                mapped[k] = v

        # Convenience aliases — also check all_user_kwargs for keys that
        # were split into config
        if "response" not in mapped and "output" in inputs:
            mapped["response"] = inputs["output"]
        if "expected_response" not in mapped and "expected_response" in model_fields:
            # Try multiple sources for expected_response
            for src_key in ("expected_response", "keyword", "context", "expected_output"):
                if src_key in all_user_kwargs:
                    mapped["expected_response"] = all_user_kwargs[src_key]
                    break

        return mapped

    @staticmethod
    def _normalise_score(output) -> Optional[float]:
        """Convert various metric output types to a 0-1 float."""
        if output is None:
            return None
        if isinstance(output, (int, float)):
            return float(output)
        if isinstance(output, bool):
            return 1.0 if output else 0.0
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


# ---------------------------------------------------------------------------
# TuringEngine
# ---------------------------------------------------------------------------

class TuringEngine(Engine):
    """Runs evaluations on the Turing (FutureAGI) cloud platform.

    For known templates: delegates to the existing Evaluator HTTP client.
    For custom prompts: sends the prompt to the Turing eval API.
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
        self._evaluator = None

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
        if prompt:
            return self._run_custom_prompt(prompt, inputs, model=model)
        return self._run_template(eval_name, inputs, model=model)

    # ------------------------------------------------------------------
    # Template-based evaluation
    # ------------------------------------------------------------------

    def _run_template(
        self, eval_name: str, inputs: Dict[str, Any], *, model: Optional[str] = None
    ) -> EvalResult:
        template_cls = self._resolve_template_class(eval_name)
        if template_cls is None:
            return EvalResult(
                eval_name=eval_name,
                status="failed",
                error=f"Cloud template '{eval_name}' not found",
            )

        evaluator = self._get_evaluator()

        start = time.perf_counter()
        try:
            batch = evaluator.evaluate(
                eval_templates=template_cls(),
                inputs=inputs,
                model_name=model,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            if batch.eval_results:
                r = batch.eval_results[0]
                score = LocalEngine._normalise_score(r.output)
                return EvalResult(
                    eval_name=eval_name,
                    score=score,
                    reason=r.reason or "",
                    latency_ms=latency_ms,
                    metadata={"eval_id": r.eval_id, "output_type": r.output_type},
                )
            return EvalResult(
                eval_name=eval_name,
                status="failed",
                error="Cloud evaluator returned no results",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return EvalResult(
                eval_name=eval_name,
                status="failed",
                error=str(exc),
                latency_ms=latency_ms,
            )

    # ------------------------------------------------------------------
    # Custom prompt evaluation (NEW capability)
    # ------------------------------------------------------------------

    def _run_custom_prompt(
        self,
        prompt: str,
        inputs: Dict[str, Any],
        *,
        model: Optional[str] = None,
    ) -> EvalResult:
        """Send a user-defined prompt to the Turing eval API.

        The prompt may contain {output}, {context}, {input} placeholders
        which are filled from *inputs*.
        """
        rendered = prompt.format_map(_SafeFormatDict(inputs))

        evaluator = self._get_evaluator()
        payload = {
            "eval_name": "custom_prompt",
            "inputs": {
                "prompt": rendered,
                **inputs,
            },
            "model": model,
        }

        start = time.perf_counter()
        try:
            from fi.api.types import HttpMethod, RequestConfig
            from fi.utils.routes import Routes

            response = evaluator.request(
                config=RequestConfig(
                    method=HttpMethod.POST,
                    url=f"{evaluator._base_url}/{Routes.evaluatev2.value}",
                    json=payload,
                    timeout=evaluator._default_timeout,
                ),
            )
            latency_ms = (time.perf_counter() - start) * 1000
            data = response.json()

            # Parse API response
            results = data.get("result", [])
            if results:
                evals = results[0].get("evaluations", [])
                if evals:
                    e = evals[0]
                    score = LocalEngine._normalise_score(e.get("output"))
                    return EvalResult(
                        eval_name="custom_prompt",
                        score=score,
                        reason=e.get("reason", ""),
                        latency_ms=latency_ms,
                    )

            return EvalResult(
                eval_name="custom_prompt",
                status="failed",
                error="Turing API returned no evaluation results",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return EvalResult(
                eval_name="custom_prompt",
                status="failed",
                error=str(exc),
                latency_ms=latency_ms,
            )

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_template_class(eval_name: str):
        """Find the EvalTemplate subclass for a given eval_name."""
        from ..templates import EvalTemplate
        from .. import templates as tmpl_mod

        for _attr, obj in inspect.getmembers(tmpl_mod, inspect.isclass):
            if (
                issubclass(obj, EvalTemplate)
                and obj is not EvalTemplate
                and getattr(obj, "eval_name", None) == eval_name
            ):
                return obj
        return None


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

        provider = LiteLLMProvider()
        judge_config = {
            "grading_criteria": prompt,
            "model": model or "gpt-4o",
            **(config or {}),
        }

        judge = CustomLLMJudge(provider=provider, config=judge_config)

        # Build a single input from the user's kwargs
        start = time.perf_counter()
        try:
            batch = judge.evaluate([inputs])
            latency_ms = (time.perf_counter() - start) * 1000

            if batch.eval_results:
                r = batch.eval_results[0]
                score = LocalEngine._normalise_score(r.output)
                return EvalResult(
                    eval_name=eval_name or "custom_llm_judge",
                    score=score,
                    reason=r.reason or "",
                    latency_ms=latency_ms,
                )
            return EvalResult(
                eval_name=eval_name or "custom_llm_judge",
                status="failed",
                error="LLM judge returned no results",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return EvalResult(
                eval_name=eval_name or "custom_llm_judge",
                status="failed",
                error=str(exc),
                latency_ms=latency_ms,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SafeFormatDict(dict):
    """dict subclass that returns '{key}' for missing keys in str.format_map."""

    def __missing__(self, key):
        return "{" + key + "}"
