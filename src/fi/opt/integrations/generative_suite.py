"""Generative (LLM prompt-rewriting) optimizer bridge for eval suites.

Runs the real optimizers in :mod:`fi.opt.optimizers` — GEPA, ProTeGi,
MetaPrompt, PromptWizard, RandomSearch, BayesianSearch — against a
promptfoo-style eval suite. Candidate prompts are produced by a task LLM and
scored against the suite's own assertions (including ``fi_eval`` platform
templates), so the optimizer maximizes the suite's pass-rate.

Unlike the deterministic agent/target backends wired through
``_optimizer_cls`` (which mutate a search space and never rewrite prompt
text), these optimizers use an LLM to *generate* new prompts. They are opt-in
via ``optimization.optimizer.algorithm`` set to a generative token.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..types import EvaluationResult

GENERATIVE_TOKENS = {
    "gepa",
    "protegi",
    "metaprompt",
    "promptwizard",
    "random_search",
    "bayesian_search",
}

_VAR_PLACEHOLDER = re.compile(r"{{\s*([A-Za-z_][\w]*)\s*}}")


def _manifest_error(message: str) -> Exception:
    from fi.simulate.manifest import ManifestError

    return ManifestError(message)


def _suite_attr(name: str) -> Any:
    """Resolve a public helper from the installed simulate-sdk."""
    try:
        from fi import simulate as simulate_sdk
    except Exception as exc:  # pragma: no cover - optional dependency clarity
        raise _manifest_error(
            "agent-simulate is required for generative eval-suite optimization."
        ) from exc
    attr = getattr(simulate_sdk, name, None)
    if attr is None:
        from fi.simulate import suite as suite_mod

        attr = getattr(suite_mod, name, None)
    if attr is None:  # pragma: no cover - version clarity
        raise _manifest_error(
            f"agent-simulate with `{name}` is required; upgrade simulate-sdk."
        )
    return attr


def _to_format_template(template: str) -> str:
    """Convert ``{{ var }}`` placeholders to ``{var}`` for str.format generators."""
    return _VAR_PLACEHOLDER.sub(r"{\1}", template or "")


def _dataset_fields(dataset: Sequence[Mapping[str, Any]]) -> List[str]:
    fields: List[str] = []
    for example in dataset:
        for key in example:
            if not key.startswith("__") and key not in fields:
                fields.append(key)
    return fields


class _SuiteAssertionMapper:
    """Maps ``(generated_output, case)`` into the assertion evaluator's input."""

    def map(
        self, generated_output: str, ground_truth_example: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return {
            "response": generated_output,
            "__assertions__": list(ground_truth_example.get("__assertions__", [])),
            "__vars__": {
                key: value
                for key, value in ground_truth_example.items()
                if not key.startswith("__")
            },
        }


class _SuiteAssertionEvaluator:
    """Scores generated outputs against the suite's own assertions."""

    def __init__(self, evaluate_assertions: Any) -> None:
        self._evaluate_assertions = evaluate_assertions

    def evaluate(
        self, inputs: Sequence[Mapping[str, Any]]
    ) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []
        for item in inputs:
            summary = self._evaluate_assertions(
                str(item.get("response", "")),
                item.get("__assertions__", []),
                variables=item.get("__vars__", {}),
            )
            score = float(summary.get("score", 0.0) or 0.0)
            failed = [
                res for res in summary.get("results", []) if not res.get("passed")
            ]
            if not failed:
                reason = "all assertions passed"
            else:
                reason = "; ".join(
                    str(res.get("reason") or res.get("type")) for res in failed[:3]
                )
            results.append(EvaluationResult(score=score, reason=reason))
        return results


def _resolve_model(
    suite: Mapping[str, Any], optimizer_config: Mapping[str, Any]
) -> str:
    for key in ("model", "generator_model", "reflection_model", "teacher_model"):
        value = optimizer_config.get(key)
        if value:
            return str(value)
    for provider in suite.get("providers") or []:
        if not isinstance(provider, Mapping):
            continue
        model = provider.get("model")
        if not model:
            continue
        model = str(model)
        provider_type = str(provider.get("type") or "").strip().lower()
        if provider_type in {"vertex", "vertex_ai", "gemini"} and "/" not in model:
            return f"vertex_ai/{model}"
        return model
    raise _manifest_error(
        "generative optimization requires a task model: set "
        "optimization.optimizer.model or add a provider with a `model`."
    )


def _build_dataset(suite: Mapping[str, Any]) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    for test in suite.get("tests") or []:
        if not isinstance(test, Mapping):
            continue
        example = dict(test.get("vars") or test.get("variables") or {})
        example["__assertions__"] = list(
            test.get("assertions") or test.get("assert") or []
        )
        dataset.append(example)
    return dataset


def _build_seed(suite: Mapping[str, Any]) -> str:
    for prompt in suite.get("prompts") or []:
        if isinstance(prompt, Mapping) and prompt.get("template"):
            return _to_format_template(str(prompt["template"]))
        if isinstance(prompt, str) and prompt.strip():
            return _to_format_template(prompt)
    raise _manifest_error(
        "generative optimization requires at least one prompt template."
    )


def _validate_seed(seed: str, example: Mapping[str, Any]) -> None:
    variables = {k: v for k, v in example.items() if not k.startswith("__")}
    try:
        seed.format(**variables)
    except (KeyError, IndexError, ValueError) as exc:
        raise _manifest_error(
            "generative seed prompt has placeholders the test vars don't cover "
            f"or literal braces str.format can't handle ({exc}). Use `{{var}}` "
            "placeholders that match test `vars` keys."
        ) from exc


def _budget(optimizer_config: Mapping[str, Any], default: int) -> int:
    for key in ("eval_budget", "max_metric_calls", "max_candidates"):
        value = optimizer_config.get(key)
        if value:
            return int(value)
    return default


def _run_optimizer(
    token: str,
    *,
    model: str,
    seed: str,
    dataset: List[Dict[str, Any]],
    evaluator: _SuiteAssertionEvaluator,
    data_mapper: _SuiteAssertionMapper,
    optimizer_config: Mapping[str, Any],
    task_description: str,
) -> Any:
    from ..generators.litellm import LiteLLMGenerator

    common = dict(evaluator=evaluator, data_mapper=data_mapper, dataset=dataset)
    subset = len(dataset) or 1

    try:
        if token == "gepa":
            from ..optimizers.gepa import GEPAOptimizer

            optimizer = GEPAOptimizer(reflection_model=model, generator_model=model)
            return optimizer.optimize(
                **common,
                initial_prompts=[seed],
                max_metric_calls=_budget(optimizer_config, 25),
            )
        if token == "protegi":
            from ..optimizers.protegi import ProTeGi

            optimizer = ProTeGi(
                teacher_generator=LiteLLMGenerator(model, "{prompt}"),
                task_model=model,
                num_gradients=int(optimizer_config.get("num_gradients", 2)),
                beam_size=int(optimizer_config.get("beam_size", 2)),
            )
            return optimizer.optimize(
                **common,
                initial_prompts=[seed],
                num_rounds=int(optimizer_config.get("num_rounds", 2)),
                eval_subset_size=subset,
            )
        if token == "metaprompt":
            from ..optimizers.metaprompt import MetaPromptOptimizer

            optimizer = MetaPromptOptimizer(
                teacher_generator=LiteLLMGenerator(model, "{prompt}"),
                task_model=model,
            )
            return optimizer.optimize(
                **common,
                initial_prompts=[seed],
                task_description=task_description,
                num_rounds=int(optimizer_config.get("num_rounds", 3)),
                eval_subset_size=subset,
            )
        if token == "promptwizard":
            from ..optimizers.promptwizard import PromptWizardOptimizer

            optimizer = PromptWizardOptimizer(
                teacher_generator=LiteLLMGenerator(model, "{prompt}"),
                task_model=model,
                mutate_rounds=int(optimizer_config.get("mutate_rounds", 2)),
                refine_iterations=int(optimizer_config.get("refine_iterations", 1)),
            )
            return optimizer.optimize(
                **common,
                initial_prompts=[seed],
                task_description=task_description,
            )
        if token == "random_search":
            from ..optimizers.random_search import RandomSearchOptimizer

            optimizer = RandomSearchOptimizer(
                generator=LiteLLMGenerator(model, seed),
                teacher_model=model,
                num_variations=int(optimizer_config.get("num_variations", 4)),
            )
            return optimizer.optimize(**common)
        if token == "bayesian_search":
            from ..optimizers.bayesian_search import BayesianSearchOptimizer

            optimizer = BayesianSearchOptimizer(
                inference_model_name=model,
                n_trials=int(optimizer_config.get("n_trials", 6)),
                example_template_fields=_dataset_fields(dataset) or None,
            )
            return optimizer.optimize(**common, initial_prompts=[seed])
    except ImportError as exc:
        raise _manifest_error(
            f"the `{token}` optimizer needs an optional dependency: {exc}"
        ) from exc

    raise _manifest_error(
        f"unknown generative optimizer token: {token!r}; expected one of "
        f"{sorted(GENERATIVE_TOKENS)}"
    )


def _build_payload(
    result: Any,
    *,
    token: str,
    model: str,
    seed: str,
    suite: Mapping[str, Any],
    suite_path: str | Path,
    threshold: float,
    started: float,
) -> Dict[str, Any]:
    from fi.simulate.suite import (
        CLI_SCHEMA_VERSION,
        EVAL_SUITE_OPTIMIZATION_SCHEMA_VERSION,
    )

    try:
        best_prompt = result.best_generator.get_prompt_template()
    except Exception:  # pragma: no cover - defensive
        best_prompt = seed
    final_score = float(getattr(result, "final_score", 0.0) or 0.0)
    history = [
        {
            "prompt": history_item.prompt,
            "average_score": round(float(history_item.average_score), 4),
        }
        for history_item in getattr(result, "history", []) or []
    ]
    status = "passed" if final_score >= threshold else "failed"
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "kind": EVAL_SUITE_OPTIMIZATION_SCHEMA_VERSION,
        "name": str(suite.get("name") or Path(suite_path).stem),
        "status": status,
        "exit_code": 0 if status == "passed" else 1,
        "summary": {
            "optimizer_algorithm": token,
            "optimizer_family": "generative",
            "final_score": round(final_score, 4),
            "threshold": threshold,
            "iterations": len(history),
        },
        "optimization": {
            "source": "eval_suite",
            "family": "generative",
            "optimizer": token,
            "model": model,
            "threshold": threshold,
            "final_score": round(final_score, 4),
            "seed_prompt": seed,
            "best_prompt": best_prompt,
            "history": history,
            "early_stopped": bool(getattr(result, "early_stopped", False)),
            "stop_reason": getattr(result, "stop_reason", None),
            "total_evaluations": getattr(result, "total_evaluations", None),
        },
        "duration_seconds": round(time.time() - started, 4),
    }


def optimize_eval_suite_generative(
    suite: Mapping[str, Any],
    *,
    token: str,
    suite_path: str | Path = ".",
    name: Optional[str] = None,
    optimizer_config: Optional[Mapping[str, Any]] = None,
    threshold: float = 0.5,
    started: Optional[float] = None,
) -> Dict[str, Any]:
    """Optimize an eval suite with a generative (LLM prompt-rewriting) optimizer.

    Builds a task-LLM generator + an assertion-scoring evaluator from the
    suite, runs the requested optimizer, and returns an eval-suite optimization
    payload with the seed and best prompts, score, and iteration history.
    """
    if token not in GENERATIVE_TOKENS:
        raise _manifest_error(
            f"unknown generative optimizer token: {token!r}; expected one of "
            f"{sorted(GENERATIVE_TOKENS)}"
        )
    started = time.time() if started is None else started
    optimizer_config = dict(optimizer_config or {})

    evaluate_assertions = _suite_attr("evaluate_assertions")
    model = _resolve_model(suite, optimizer_config)
    dataset = _build_dataset(suite)
    if not dataset:
        raise _manifest_error(
            "generative optimization requires at least one test case with vars."
        )
    seed = _build_seed(suite)
    _validate_seed(seed, dataset[0])

    evaluator = _SuiteAssertionEvaluator(evaluate_assertions)
    data_mapper = _SuiteAssertionMapper()
    suite_label = name or suite.get("name") or "the eval suite"
    task_description = str(
        suite.get("description")
        or f"Rewrite the assistant prompt so its responses satisfy the eval "
        f"assertions for {suite_label}."
    )

    result = _run_optimizer(
        token,
        model=model,
        seed=seed,
        dataset=dataset,
        evaluator=evaluator,
        data_mapper=data_mapper,
        optimizer_config=optimizer_config,
        task_description=task_description,
    )
    return _build_payload(
        result,
        token=token,
        model=model,
        seed=seed,
        suite=suite,
        suite_path=suite_path,
        threshold=threshold,
        started=started,
    )
