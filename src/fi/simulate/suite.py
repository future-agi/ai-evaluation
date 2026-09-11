from __future__ import annotations

import asyncio
import copy
import importlib
import importlib.util
import inspect
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .manifest import CLI_SCHEMA_VERSION, ManifestError


EVAL_SUITE_SCHEMA_VERSION = "agent-simulate.eval.v1"
AGENT_LEARNING_EVAL_SUITE_SCHEMA_VERSION = "agent-learning.eval.v1"
EVAL_SUITE_OPTIMIZATION_SCHEMA_VERSION = "agent-learning.eval-optimization.v1"

_JSON_PATH_EQUALS_ASSERTIONS = {
    "json_path_equals",
    "json_path_equal",
    "json_equals",
    "path_equals",
    "path_equal",
}
_JSON_PATH_EXISTS_ASSERTIONS = {
    "json_path_exists",
    "json_exists",
    "path_exists",
}
_JSON_PATH_GTE_ASSERTIONS = {
    "json_path_gte",
    "json_path_ge",
    "json_gte",
    "path_gte",
    "greater_than_or_equal",
}
_JSON_PATH_LTE_ASSERTIONS = {
    "json_path_lte",
    "json_path_le",
    "json_lte",
    "path_lte",
    "less_than_or_equal",
}
_JSON_PATH_CONTAINS_ASSERTIONS = {
    "json_path_contains",
    "json_contains",
    "path_contains",
}
_JSON_PATH_NOT_CONTAINS_ASSERTIONS = {
    "json_path_not_contains",
    "json_not_contains",
    "path_not_contains",
}
_JSON_PATH_ASSERTIONS = (
    _JSON_PATH_EQUALS_ASSERTIONS
    | _JSON_PATH_EXISTS_ASSERTIONS
    | _JSON_PATH_GTE_ASSERTIONS
    | _JSON_PATH_LTE_ASSERTIONS
    | _JSON_PATH_CONTAINS_ASSERTIONS
    | _JSON_PATH_NOT_CONTAINS_ASSERTIONS
)

# Assertion tokens that score the case output with a hosted FutureAGI eval
# template via ``fi.evals.evaluate`` (platform/turing engine by default,
# using FI_API_KEY / FI_SECRET_KEY / FI_BASE_URL). Pass = score >= threshold.
_FI_EVAL_ASSERTIONS = {
    "fi_eval",
    "fi_evals",
    "platform_eval",
    "platform",
    "turing",
}

# Extra eval-input keys an ``fi_eval`` assertion may template from the case
# vars before forwarding to the platform (``output`` is always the case output).
_FI_EVAL_INPUT_KEYS = (
    "input",
    "context",
    "expected",
    "query",
    "reference",
    "instructions",
    "prompt",
    "criteria",
)

# Generative (LLM prompt-rewriting) optimizer tokens. These run the real
# optimizers in ``fi.opt.optimizers`` through the generative eval-suite bridge
# rather than the deterministic agent/target search backends.
_GENERATIVE_OPTIMIZER_TOKENS = {
    "gepa": "gepa",
    "protegi": "protegi",
    "pro_te_gi": "protegi",
    "metaprompt": "metaprompt",
    "meta_prompt": "metaprompt",
    "promptwizard": "promptwizard",
    "prompt_wizard": "promptwizard",
    "random_search": "random_search",
    "random": "random_search",
    "bayesian_search": "bayesian_search",
    "bayesian": "bayesian_search",
    "bayes": "bayesian_search",
}


def _generative_optimizer_token(optimizer_cfg: Any) -> Optional[str]:
    """Return the canonical generative optimizer token, or None for Family A."""
    if not isinstance(optimizer_cfg, Mapping):
        return None
    raw = (
        optimizer_cfg.get("algorithm")
        or optimizer_cfg.get("type")
        or optimizer_cfg.get("name")
        or optimizer_cfg.get("strategy")
    )
    if not raw:
        return None
    norm = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    return _GENERATIVE_OPTIMIZER_TOKENS.get(norm)


@dataclass(frozen=True)
class EvalSuiteOptions:
    name: Optional[str] = None
    threshold: Optional[float] = None
    dry_run: bool = False


@dataclass(frozen=True)
class EvalSuiteOptimizationOptions:
    name: Optional[str] = None
    threshold: Optional[float] = None
    max_candidates: Optional[int] = None
    dry_run: bool = False


def build_eval_suite_manifest(
    *,
    name: str,
    providers: Optional[Sequence[Mapping[str, Any]]] = None,
    prompts: Optional[Sequence[Mapping[str, Any]]] = None,
    tests: Optional[Sequence[Mapping[str, Any]]] = None,
    threshold: float = 1.0,
    outputs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    version: str = AGENT_LEARNING_EVAL_SUITE_SCHEMA_VERSION,
) -> Dict[str, Any]:
    """Build a promptfoo-style eval suite manifest from SDK data."""

    if not name:
        raise ValueError("name is required")
    provider_values = _copy_mapping_sequence(
        providers
        if providers is not None
        else (
            {
                "id": "echo",
                "type": "echo",
            },
        ),
        field="providers",
    )
    prompt_values = _copy_mapping_sequence(
        prompts
        if prompts is not None
        else (
            {
                "id": "support-policy-question",
                "template": "{{question}}",
            },
        ),
        field="prompts",
    )
    test_values = _copy_mapping_sequence(
        tests
        if tests is not None
        else (
            {
                "id": "policy-grounding",
                "vars": {"question": "Where is the refund policy?"},
                "assert": [{"type": "contains", "value": "policy"}],
            },
        ),
        field="tests",
    )
    manifest: Dict[str, Any] = {
        "version": str(version),
        "name": str(name),
        "threshold": float(threshold),
        "providers": provider_values,
        "prompts": prompt_values,
        "tests": test_values,
    }
    if outputs:
        manifest["outputs"] = copy.deepcopy(dict(outputs))
    if metadata:
        manifest["metadata"] = copy.deepcopy(dict(metadata))
    return manifest


def write_eval_suite_file(suite: Mapping[str, Any], path: str | Path) -> Path:
    """Write an eval suite manifest as formatted JSON and return the path."""

    suite_path = Path(path).expanduser().resolve()
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(
        json.dumps(dict(suite), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return suite_path


def load_eval_suite_file(path: str | Path) -> Dict[str, Any]:
    suite_path = Path(path).expanduser().resolve()
    suite = _load_json_or_yaml(suite_path)
    if not isinstance(suite, Mapping):
        raise ManifestError("eval suite root must be an object")
    return _prepare_eval_suite(dict(suite), base_dir=suite_path.parent)


def run_eval_suite_file(
    path: str | Path,
    *,
    options: Optional[EvalSuiteOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    suite_path = Path(path).expanduser().resolve()
    suite = load_eval_suite_file(suite_path)
    return run_eval_suite(
        suite,
        suite_path=suite_path,
        options=_merge_eval_suite_options(
            options,
            name=name,
            threshold=threshold,
            dry_run=dry_run,
        ),
    )


def optimize_eval_suite_file(
    path: str | Path,
    *,
    options: Optional[EvalSuiteOptimizationOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load and optimize a promptfoo-style eval suite with Agent Learning."""

    suite_path = Path(path).expanduser().resolve()
    suite = load_eval_suite_file(suite_path)
    return optimize_eval_suite(
        suite,
        suite_path=suite_path,
        options=_merge_eval_suite_optimization_options(
            options,
            name=name,
            threshold=threshold,
            max_candidates=max_candidates,
            dry_run=dry_run,
        ),
    )


def optimize_eval_suite(
    suite: Mapping[str, Any],
    *,
    suite_path: str | Path = ".",
    options: Optional[EvalSuiteOptimizationOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Optimize an in-memory eval suite and return a unified artifact payload."""

    started = time.time()
    opts = _merge_eval_suite_optimization_options(
        options,
        name=name,
        threshold=threshold,
        max_candidates=max_candidates,
        dry_run=dry_run,
    )
    suite_path = _suite_file_like_path(suite_path)
    runtime_suite = copy.deepcopy(dict(suite))
    if opts.name:
        runtime_suite["name"] = opts.name
    if opts.threshold is not None:
        runtime_suite.setdefault("optimization", {})["threshold"] = opts.threshold
    if opts.max_candidates is not None:
        runtime_suite.setdefault("optimization", {}).setdefault(
            "optimizer", {}
        )["max_candidates"] = opts.max_candidates

    prepared = _prepare_eval_suite(runtime_suite, base_dir=suite_path.parent)
    cli = _cli()
    optimization = cli._optimization_config(prepared)
    optimizer_config = cli._optimizer_config(optimization)

    # Generative (LLM prompt-rewriting) optimizers run through their own bridge
    # and don't need a deterministic search-space target, so route them before
    # `_target_config` (which requires one).
    generative_token = _generative_optimizer_token(optimization.get("optimizer"))
    if generative_token and not opts.dry_run:
        try:
            from fi.opt.integrations.generative_suite import (
                optimize_eval_suite_generative,
            )
        except Exception as exc:  # pragma: no cover - optional dependency clarity
            raise ManifestError(
                "Agent Learning Kit generative optimizer engine is required for "
                f"the `{generative_token}` optimizer."
            ) from exc
        payload = optimize_eval_suite_generative(
            prepared,
            suite_path=suite_path,
            name=str(prepared.get("name") or suite_path.stem),
            token=generative_token,
            optimizer_config=dict(optimizer_config or {}),
            threshold=float(optimization.get("threshold", 0.5)),
            started=started,
        )
        payload["eval_suite"] = _eval_suite_descriptor(prepared)
        payload.setdefault("summary", {})
        payload["summary"]["provider_count"] = len(_as_list(prepared.get("providers")))
        payload["summary"]["prompt_count"] = len(_as_list(prepared.get("prompts")))
        payload["summary"]["test_count"] = len(_as_list(prepared.get("tests")))
        return payload

    target_config = cli._target_config(optimization)
    if opts.dry_run:
        return {
            "schema_version": CLI_SCHEMA_VERSION,
            "kind": EVAL_SUITE_OPTIMIZATION_SCHEMA_VERSION,
            "name": str(prepared.get("name") or suite_path.stem),
            "status": "passed",
            "exit_code": 0,
            "dry_run": True,
            "summary": {
                "provider_count": len(_as_list(prepared.get("providers"))),
                "prompt_count": len(_as_list(prepared.get("prompts"))),
                "test_count": len(_as_list(prepared.get("tests"))),
                "search_path_count": len(target_config.get("search_space", {})),
                "max_candidates": optimizer_config.get("max_candidates"),
            },
            "eval_suite": _eval_suite_descriptor(prepared),
            "duration_seconds": round(time.time() - started, 4),
        }

    try:
        from fi.opt import problem_from_eval_suite
    except Exception as exc:  # pragma: no cover - optional dependency clarity
        raise ManifestError(
            "Agent Learning Kit optimizer engine is required for eval-suite optimization."
        ) from exc

    problem = problem_from_eval_suite(
        prepared,
        suite_path=suite_path,
        name=str(prepared.get("name") or suite_path.stem),
    )
    optimization_result = problem.optimize()
    payload = cli._optimization_result(
        manifest=prepared,
        manifest_path=suite_path,
        optimization_result=optimization_result,
        threshold=float(optimization.get("threshold", 1.0)),
        duration_seconds=round(time.time() - started, 4),
    )
    payload["kind"] = EVAL_SUITE_OPTIMIZATION_SCHEMA_VERSION
    payload["eval_suite"] = _eval_suite_descriptor(prepared)
    payload["summary"]["provider_count"] = len(_as_list(prepared.get("providers")))
    payload["summary"]["prompt_count"] = len(_as_list(prepared.get("prompts")))
    payload["summary"]["test_count"] = len(_as_list(prepared.get("tests")))
    payload["optimization"]["source"] = "eval_suite"
    if "manifest_optimization" in payload["optimization"]:
        artifact = copy.deepcopy(payload["optimization"]["manifest_optimization"])
        artifact["kind"] = "eval_suite_optimization"
        artifact["source"] = "eval_suite"
        payload["optimization"]["eval_suite_optimization"] = artifact
    return payload


def run_eval_suite(
    suite: Mapping[str, Any],
    *,
    suite_path: str | Path = ".",
    options: Optional[EvalSuiteOptions] = None,
) -> Dict[str, Any]:
    started = time.time()
    opts = options or EvalSuiteOptions()
    base_dir = Path(suite_path).expanduser().resolve().parent
    prepared = _prepare_eval_suite(dict(suite), base_dir=base_dir)
    name = str(opts.name or prepared.get("name") or "agent-learning-eval")
    threshold = float(opts.threshold if opts.threshold is not None else prepared.get("threshold", 1.0))
    if opts.dry_run:
        return _suite_result(
            name=name,
            suite=prepared,
            cases=[],
            threshold=threshold,
            duration_seconds=round(time.time() - started, 4),
            dry_run=True,
        )

    cases: List[Dict[str, Any]] = []
    providers = [_as_dict(provider) for provider in _as_list(prepared.get("providers"))]
    prompts = [_as_dict(prompt) for prompt in _as_list(prepared.get("prompts"))]
    tests = [_as_dict(test) for test in _as_list(prepared.get("tests"))]
    for provider in providers:
        for prompt in prompts:
            for test_index, test in enumerate(tests, start=1):
                cases.append(
                    _run_eval_case(
                        provider=provider,
                        prompt=prompt,
                        test=test,
                        test_index=test_index,
                        base_dir=base_dir,
                    )
                )
    return _suite_result(
        name=name,
        suite=prepared,
        cases=cases,
        threshold=threshold,
        duration_seconds=round(time.time() - started, 4),
        dry_run=False,
    )


def _prepare_eval_suite(suite: Dict[str, Any], *, base_dir: Path) -> Dict[str, Any]:
    providers = [_as_dict(item) for item in _as_list(suite.get("providers") or suite.get("provider"))]
    prompts = [_as_dict(item) for item in _as_list(suite.get("prompts") or suite.get("prompt"))]
    tests = _suite_tests(suite, base_dir=base_dir)
    if not providers:
        raise ManifestError("eval suite requires at least one provider")
    if not prompts:
        raise ManifestError("eval suite requires at least one prompt")
    if not tests:
        raise ManifestError("eval suite requires at least one test")
    suite["providers"] = [_normalize_provider(item, index) for index, item in enumerate(providers, start=1)]
    suite["prompts"] = [_normalize_prompt(item, index) for index, item in enumerate(prompts, start=1)]
    suite["tests"] = [_normalize_test(item, index) for index, item in enumerate(tests, start=1)]
    suite.pop("tests_file", None)
    suite.pop("data_file", None)
    suite.pop("data", None)
    suite.setdefault("version", EVAL_SUITE_SCHEMA_VERSION)
    return suite


def _suite_tests(suite: Mapping[str, Any], *, base_dir: Path) -> List[Dict[str, Any]]:
    tests_value = suite.get("tests")
    tests_file = suite.get("tests_file") or suite.get("data") or suite.get("data_file")
    records: List[Dict[str, Any]] = []
    if isinstance(tests_value, str):
        records.extend(_load_test_records(base_dir / tests_value))
    else:
        records.extend(_as_dict(item) for item in _as_list(tests_value))
    for path in _as_list(tests_file):
        records.extend(_load_test_records(base_dir / str(path)))
    return records


def _load_test_records(path: Path) -> List[Dict[str, Any]]:
    source = path.expanduser().resolve()
    if not source.exists():
        raise ManifestError(f"eval suite tests file not found: {source}")
    if source.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"invalid JSONL in {source}:{line_number}") from exc
            records.append(_as_dict(item))
        return records
    data = _load_json_or_yaml(source)
    if isinstance(data, Mapping) and "tests" in data:
        return [_as_dict(item) for item in _as_list(data.get("tests"))]
    return [_as_dict(item) for item in _as_list(data)]


def _normalize_provider(provider: Mapping[str, Any], index: int) -> Dict[str, Any]:
    item = dict(provider)
    item["id"] = str(item.get("id") or item.get("name") or f"provider_{index}")
    item["type"] = str(item.get("type") or item.get("kind") or "echo")
    return item


def _normalize_prompt(prompt: Mapping[str, Any], index: int) -> Dict[str, Any]:
    item = dict(prompt)
    item["id"] = str(item.get("id") or item.get("name") or f"prompt_{index}")
    item["template"] = str(item.get("template") or item.get("content") or item.get("prompt") or "")
    if not item["template"]:
        raise ManifestError(f"prompt `{item['id']}` requires a template")
    return item


def _normalize_test(test: Mapping[str, Any], index: int) -> Dict[str, Any]:
    item = dict(test)
    item["id"] = str(item.get("id") or item.get("name") or f"test_{index}")
    item["vars"] = _as_dict(item.get("vars") or item.get("variables"))
    assertions = _as_list(item.get("assert") or item.get("assertions") or item.get("checks"))
    item["assertions"] = [_normalize_assertion(assertion, item["id"], offset) for offset, assertion in enumerate(assertions, start=1)]
    if not item["assertions"]:
        raise ManifestError(f"test `{item['id']}` requires at least one assertion")
    return item


def _normalize_assertion(assertion: Any, test_id: str, index: int) -> Dict[str, Any]:
    if isinstance(assertion, str):
        item = {"type": "contains", "value": assertion}
    else:
        item = _as_dict(assertion)
    item["type"] = str(item.get("type") or item.get("kind") or "contains").lower().replace("-", "_")
    if "path" not in item:
        for alias in ("json_path", "field"):
            if alias in item:
                item["path"] = item.get(alias)
                break
    if "value" not in item and "expected" in item:
        item["value"] = item.get("expected")
    assertion_type = str(item["type"])
    if assertion_type in _JSON_PATH_ASSERTIONS and not item.get("path"):
        raise ManifestError(f"assertion {index} in test `{test_id}` requires a path")
    if assertion_type in _FI_EVAL_ASSERTIONS:
        if not (item.get("eval") or item.get("metric") or item.get("name") or item.get("value")):
            raise ManifestError(
                f"assertion {index} in test `{test_id}` requires an `eval` "
                "(hosted template name)"
            )
        return item
    requires_value = assertion_type not in _JSON_PATH_EXISTS_ASSERTIONS
    if requires_value and "value" not in item:
        raise ManifestError(f"assertion {index} in test `{test_id}` requires a value")
    return item


def _run_eval_case(
    *,
    provider: Mapping[str, Any],
    prompt: Mapping[str, Any],
    test: Mapping[str, Any],
    test_index: int,
    base_dir: Path,
) -> Dict[str, Any]:
    variables = _as_dict(test.get("vars"))
    rendered_prompt = _render_template(str(prompt.get("template") or ""), variables)
    output = _provider_output(
        provider=provider,
        prompt=rendered_prompt,
        variables=variables,
        test=test,
        base_dir=base_dir,
    )
    assertion_results = [
        _evaluate_assertion(assertion, output, variables)
        for assertion in _as_list(test.get("assertions"))
    ]
    failures = [item for item in assertion_results if not item.get("passed")]
    case_id = f"{provider.get('id')}::{prompt.get('id')}::{test.get('id')}"
    score = 1.0 if not assertion_results else (len(assertion_results) - len(failures)) / len(assertion_results)
    findings = [
        {
            "type": "eval_assertion_failed",
            "severity": "high",
            "case_id": case_id,
            "provider_id": provider.get("id"),
            "prompt_id": prompt.get("id"),
            "test_id": test.get("id"),
            "assertion_type": failure.get("type"),
            "expected": failure.get("expected"),
            "actual": failure.get("actual", output),
            "path": failure.get("path"),
            "error": failure.get("error"),
        }
        for failure in failures
    ]
    return {
        "index": test_index,
        "id": case_id,
        "name": case_id,
        "provider_id": provider.get("id"),
        "provider_type": provider.get("type"),
        "prompt_id": prompt.get("id"),
        "test_id": test.get("id"),
        "input": rendered_prompt,
        "output": output,
        "score": round(score, 4),
        "passed": not failures,
        "assertions": assertion_results,
        "findings": findings,
        "metrics": [
            {
                "name": "eval_assertions",
                "score": round(score, 4),
                "details": {"assertions": assertion_results, "findings": findings},
            }
        ],
    }


def _provider_output(
    *,
    provider: Mapping[str, Any],
    prompt: str,
    variables: Mapping[str, Any],
    test: Mapping[str, Any],
    base_dir: Path,
) -> str:
    provider_type = str(provider.get("type") or "echo").lower().replace("-", "_")
    if provider_type == "echo":
        return prompt
    if provider_type == "scripted":
        template = str(provider.get("response") or provider.get("output") or provider.get("template") or "")
        if not template:
            responses = _as_list(provider.get("responses"))
            template = str(responses[0]) if responses else prompt
        return _render_template(template, {**variables, "prompt": prompt, "input": prompt})
    if provider_type in {"artifact", "artifact_json", "artifact_file"}:
        return _artifact_provider_output(
            provider=provider,
            prompt=prompt,
            variables=variables,
            base_dir=base_dir,
        )
    if provider_type in {"python", "python_callable", "callable"}:
        target = str(provider.get("target") or provider.get("callable") or "")
        if not target:
            raise ManifestError(f"provider `{provider.get('id')}` requires target")
        callback = _load_callable(target, base_dir)
        value = callback(prompt=prompt, vars=dict(variables), test=dict(test), provider=dict(provider))
        if inspect.isawaitable(value):
            value = asyncio.run(value)
        return str(value)
    if provider_type in {"litellm", "llm", "vertex", "vertex_ai", "gemini"}:
        return _litellm_provider_output(
            provider=provider,
            prompt=prompt,
            variables=variables,
            provider_type=provider_type,
        )
    raise ManifestError(f"unsupported eval suite provider type: {provider_type}")


def _litellm_provider_output(
    *,
    provider: Mapping[str, Any],
    prompt: str,
    variables: Mapping[str, Any],
    provider_type: str,
) -> str:
    """Call a live LLM through litellm.

    Routes any litellm-supported model. Use ``type: vertex`` (or ``gemini``)
    with a bare ``model`` name to reach Vertex AI — authentication comes from
    ``GOOGLE_APPLICATION_CREDENTIALS`` and routing from the ``vertex_project`` /
    ``vertex_location`` provider fields (or the matching ``VERTEXAI_*`` env
    vars). Use ``type: litellm`` with a fully-qualified model string
    (``vertex_ai/gemini-2.5-flash``, ``gpt-4o-mini``, ``claude-3-5-sonnet``)
    for any other provider.
    """
    try:
        import litellm
    except Exception as exc:  # pragma: no cover - import guard
        raise ManifestError(
            f"provider type `{provider_type}` requires litellm; reinstall "
            "agent-learning-kit"
        ) from exc

    model = str(provider.get("model") or "").strip()
    if not model:
        raise ManifestError(f"provider `{provider.get('id')}` requires a model")
    if provider_type in {"vertex", "vertex_ai", "gemini"} and "/" not in model:
        model = f"vertex_ai/{model}"

    render_ctx = {**variables, "prompt": prompt, "input": prompt}
    messages: List[Dict[str, Any]] = []
    system_prompt = provider.get("system") or provider.get("system_prompt")
    if system_prompt:
        messages.append(
            {"role": "system", "content": _render_template(str(system_prompt), render_ctx)}
        )
    messages.append({"role": "user", "content": prompt})

    kwargs: Dict[str, Any] = dict(_as_dict(provider.get("params")))
    for key in (
        "vertex_project",
        "vertex_location",
        "vertex_credentials",
        "temperature",
        "max_tokens",
        "top_p",
        "api_base",
        "api_key",
    ):
        value = provider.get(key)
        if value is not None and key not in kwargs:
            kwargs[key] = value

    litellm.drop_params = True
    response = litellm.completion(model=model, messages=messages, **kwargs)
    content = response.choices[0].message.content
    return str(content or "")


def _artifact_provider_output(
    *,
    provider: Mapping[str, Any],
    prompt: str,
    variables: Mapping[str, Any],
    base_dir: Path,
) -> str:
    raw_path = (
        provider.get("path")
        or provider.get("source")
        or provider.get("artifact")
        or variables.get("artifact_path")
        or variables.get("artifact")
    )
    if not raw_path:
        raise ManifestError(f"provider `{provider.get('id')}` requires artifact path")
    rendered_path = _render_template(
        str(raw_path),
        {**variables, "prompt": prompt, "input": prompt},
    )
    artifact_path = Path(rendered_path).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = base_dir / artifact_path
    artifact_path = artifact_path.resolve()
    artifact = _load_json_or_yaml(artifact_path)
    fields = _artifact_fields(provider)
    if not fields:
        return json.dumps(artifact, indent=2, sort_keys=True, default=str)
    extracted = {
        label: _extract_artifact_path(artifact, path)
        for label, path in fields
    }
    return json.dumps(
        {
            "artifact_path": str(artifact_path),
            "fields": extracted,
        },
        indent=2,
        sort_keys=True,
        default=str,
    )


def _artifact_fields(provider: Mapping[str, Any]) -> List[tuple[str, str]]:
    raw_fields = (
        provider.get("fields")
        or provider.get("extract")
        or provider.get("paths")
        or provider.get("json_paths")
    )
    fields: List[tuple[str, str]] = []
    for index, raw_field in enumerate(_as_list(raw_fields), start=1):
        if isinstance(raw_field, str):
            fields.append((raw_field, raw_field))
            continue
        item = _as_dict(raw_field)
        path = str(item.get("path") or item.get("json_path") or item.get("field") or "")
        if not path:
            raise ManifestError(f"artifact field {index} requires path")
        label = str(item.get("id") or item.get("name") or item.get("as") or path)
        fields.append((label, path))
    return fields


def _extract_artifact_path(value: Any, path: str) -> Any:
    current = value
    for token in _artifact_path_tokens(path):
        if isinstance(current, Mapping):
            if token not in current:
                raise ManifestError(f"artifact path `{path}` missing key `{token}`")
            current = current[token]
        elif isinstance(current, list):
            try:
                index = int(token)
            except ValueError as exc:
                raise ManifestError(
                    f"artifact path `{path}` expected list index, got `{token}`"
                ) from exc
            try:
                current = current[index]
            except IndexError as exc:
                raise ManifestError(
                    f"artifact path `{path}` index out of range: {index}"
                ) from exc
        else:
            raise ManifestError(f"artifact path `{path}` cannot traverse `{token}`")
    return current


def _artifact_path_tokens(path: str) -> List[str]:
    normalized = path.strip()
    if not normalized:
        raise ManifestError("artifact path cannot be empty")
    if normalized.startswith("$."):
        normalized = normalized[2:]
    elif normalized == "$":
        return []
    tokens: List[str] = []
    for segment in normalized.split("."):
        if not segment:
            continue
        while "[" in segment:
            before, _, rest = segment.partition("[")
            if before:
                tokens.append(before)
            index, marker, tail = rest.partition("]")
            if not marker:
                raise ManifestError(f"invalid artifact path segment `{segment}`")
            tokens.append(index)
            segment = tail
        if segment:
            tokens.append(segment)
    return tokens


def _evaluate_assertion(
    assertion: Mapping[str, Any],
    output: str,
    variables: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    assertion_type = str(assertion.get("type") or "contains").lower().replace("-", "_")
    if assertion_type in _JSON_PATH_ASSERTIONS:
        return _evaluate_json_path_assertion(assertion, output, assertion_type)
    if assertion_type in _FI_EVAL_ASSERTIONS:
        return _evaluate_fi_eval_assertion(assertion, output, variables or {})
    expected = assertion.get("value")
    text = str(output)
    expected_text = str(expected)
    if assertion_type == "contains":
        passed = expected_text in text
    elif assertion_type == "not_contains":
        passed = expected_text not in text
    elif assertion_type in {"equals", "equal", "is"}:
        passed = text.strip() == expected_text.strip()
    elif assertion_type in {"regex", "matches"}:
        passed = re.search(expected_text, text, flags=re.MULTILINE) is not None
    else:
        raise ManifestError(f"unsupported assertion type: {assertion_type}")
    return {
        "type": assertion_type,
        "expected": expected,
        "actual": output,
        "passed": bool(passed),
    }


def _evaluate_fi_eval_assertion(
    assertion: Mapping[str, Any],
    output: str,
    variables: Mapping[str, Any],
) -> Dict[str, Any]:
    """Score the case output with a hosted FutureAGI eval template.

    Dispatches to ``fi.evals.evaluate`` on the platform (turing) engine by
    default; credentials come from ``FI_API_KEY`` / ``FI_SECRET_KEY`` /
    ``FI_BASE_URL`` (or per-assertion overrides). Pass when the returned score
    is >= ``threshold`` (default 0.5).
    """
    eval_name = (
        assertion.get("eval")
        or assertion.get("metric")
        or assertion.get("name")
        or assertion.get("value")
    )
    if not eval_name:
        raise ManifestError(
            "fi_eval assertion requires an `eval` (hosted template name)."
        )
    threshold = float(assertion.get("threshold", 0.5))
    engine = str(assertion.get("engine") or "turing").strip().lower()
    model = assertion.get("model")

    inputs: Dict[str, Any] = {"output": output}
    for key in _FI_EVAL_INPUT_KEYS:
        if key in assertion:
            inputs[key] = _render_template(str(assertion[key]), variables)
    extra_inputs = assertion.get("inputs")
    if isinstance(extra_inputs, Mapping):
        for key, val in extra_inputs.items():
            inputs[str(key)] = (
                _render_template(val, variables) if isinstance(val, str) else val
            )

    try:
        from fi.evals import evaluate as _fi_evaluate
    except Exception as exc:  # pragma: no cover - optional dependency clarity
        raise ManifestError(
            "fi_eval assertion requires the FutureAGI evals engine (fi.evals). "
            "Install the evals extra to score against platform templates."
        ) from exc

    call_kwargs: Dict[str, Any] = dict(inputs)
    if engine and engine != "auto":
        call_kwargs["engine"] = engine
    if model:
        call_kwargs["model"] = model
    for cred_key, cfg_key in (
        ("fi_api_key", "api_key"),
        ("fi_secret_key", "secret_key"),
        ("fi_base_url", "base_url"),
    ):
        if assertion.get(cfg_key):
            call_kwargs[cred_key] = assertion[cfg_key]

    try:
        result = _fi_evaluate(str(eval_name), **call_kwargs)
    except Exception as exc:
        return {
            "type": "fi_eval",
            "eval": eval_name,
            "engine": engine,
            "threshold": threshold,
            "expected": f">= {threshold}",
            "actual": None,
            "passed": False,
            "error": str(exc),
        }

    score = getattr(result, "score", None)
    reason = getattr(result, "reason", "") or ""
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0
    return {
        "type": "fi_eval",
        "eval": eval_name,
        "engine": engine,
        "threshold": threshold,
        "score": round(score_value, 4),
        "reason": reason,
        "expected": f">= {threshold}",
        "actual": round(score_value, 4),
        "passed": bool(score_value >= threshold),
    }


def evaluate_assertions(
    output: str,
    assertions: Sequence[Mapping[str, Any]],
    *,
    variables: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Score an output against a list of eval-suite assertions.

    Public helper used by the generative optimizer bridge so candidate prompts
    are scored against the suite's own assertions (including ``fi_eval``
    platform templates). Returns pass-rate plus per-assertion detail.
    """
    variables = dict(variables or {})
    results = [
        _evaluate_assertion(assertion, output, variables)
        for assertion in (assertions or [])
    ]
    if not results:
        return {"score": 1.0, "passed": True, "results": []}
    passed = sum(1 for item in results if item.get("passed"))
    return {
        "score": passed / len(results),
        "passed": passed == len(results),
        "results": results,
    }


def _evaluate_json_path_assertion(
    assertion: Mapping[str, Any],
    output: str,
    assertion_type: str,
) -> Dict[str, Any]:
    path = str(
        assertion.get("path")
        or assertion.get("json_path")
        or assertion.get("field")
        or ""
    )
    expected = assertion.get("value")
    result: Dict[str, Any] = {
        "type": assertion_type,
        "path": path,
        "expected": True if assertion_type in _JSON_PATH_EXISTS_ASSERTIONS else expected,
        "actual": None,
        "passed": False,
    }
    if not path:
        result["error"] = "json path assertion requires a path"
        return result
    try:
        document = json.loads(output)
    except json.JSONDecodeError as exc:
        result["error"] = f"output is not valid JSON: {exc.msg}"
        return result
    try:
        actual = _extract_artifact_path(document, path)
    except ManifestError as exc:
        result["error"] = str(exc)
        return result
    result["actual"] = actual

    if assertion_type in _JSON_PATH_EXISTS_ASSERTIONS:
        result["passed"] = True
    elif assertion_type in _JSON_PATH_EQUALS_ASSERTIONS:
        result["passed"] = actual == expected
    elif assertion_type in _JSON_PATH_GTE_ASSERTIONS:
        passed, error = _json_path_numeric_compare(actual, expected, "gte")
        result["passed"] = passed
        if error:
            result["error"] = error
    elif assertion_type in _JSON_PATH_LTE_ASSERTIONS:
        passed, error = _json_path_numeric_compare(actual, expected, "lte")
        result["passed"] = passed
        if error:
            result["error"] = error
    elif assertion_type in _JSON_PATH_CONTAINS_ASSERTIONS:
        result["passed"] = _json_path_contains(actual, expected)
    elif assertion_type in _JSON_PATH_NOT_CONTAINS_ASSERTIONS:
        result["passed"] = not _json_path_contains(actual, expected)
    return result


def _json_path_numeric_compare(
    actual: Any,
    expected: Any,
    operator: str,
) -> tuple[bool, str | None]:
    try:
        actual_number = float(actual)
        expected_number = float(expected)
    except (TypeError, ValueError):
        return (
            False,
            f"expected numeric JSON path values, got actual={actual!r} expected={expected!r}",
        )
    if operator == "gte":
        return actual_number >= expected_number, None
    return actual_number <= expected_number, None


def _json_path_contains(actual: Any, expected: Any) -> bool:
    if isinstance(actual, Mapping):
        return expected in actual or str(expected) in actual
    if isinstance(actual, (list, tuple, set)):
        return expected in actual
    return str(expected) in str(actual)


def _suite_result(
    *,
    name: str,
    suite: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    threshold: float,
    duration_seconds: float,
    dry_run: bool,
) -> Dict[str, Any]:
    case_count = len(cases)
    passed_count = sum(1 for case in cases if case.get("passed"))
    assertion_count = sum(len(_as_list(case.get("assertions"))) for case in cases)
    failed_assertion_count = sum(
        1
        for case in cases
        for assertion in _as_list(case.get("assertions"))
        if not _as_dict(assertion).get("passed")
    )
    score = 1.0 if not assertion_count else (assertion_count - failed_assertion_count) / assertion_count
    passed = (score >= threshold) and (passed_count == case_count)
    if dry_run:
        passed = True
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "kind": EVAL_SUITE_SCHEMA_VERSION,
        "name": name,
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 1,
        "summary": {
            "score": round(score, 4),
            "threshold": threshold,
            "provider_count": len(_as_list(suite.get("providers"))),
            "prompt_count": len(_as_list(suite.get("prompts"))),
            "test_count": len(_as_list(suite.get("tests"))),
            "case_count": case_count,
            "passed_case_count": passed_count,
            "failed_case_count": case_count - passed_count,
            "assertion_count": assertion_count,
            "passed_assertion_count": assertion_count - failed_assertion_count,
            "failed_assertion_count": failed_assertion_count,
            "dry_run": dry_run,
        },
        "eval_suite": {
            "version": suite.get("version") or EVAL_SUITE_SCHEMA_VERSION,
            "providers": [
                {"id": provider.get("id"), "type": provider.get("type")}
                for provider in _as_list(suite.get("providers"))
                if isinstance(provider, Mapping)
            ],
            "prompts": [
                {"id": prompt.get("id")}
                for prompt in _as_list(suite.get("prompts"))
                if isinstance(prompt, Mapping)
            ],
            "tests": [
                {"id": test.get("id")}
                for test in _as_list(suite.get("tests"))
                if isinstance(test, Mapping)
            ],
            "cases": list(cases),
        },
        "evaluation": {
            "passed": passed,
            "score": round(score, 4),
            "threshold": threshold,
            "cases": list(cases),
            "findings": [
                finding
                for case in cases
                for finding in _as_list(case.get("findings"))
                if isinstance(finding, Mapping)
            ],
        },
        "duration_seconds": duration_seconds,
    }


def _render_template(template: str, variables: Mapping[str, Any]) -> str:
    result = template
    for key, value in variables.items():
        result = result.replace("{{" + str(key) + "}}", str(value))
        result = result.replace("{{ " + str(key) + " }}", str(value))
    return result


def _load_json_or_yaml(path: Path) -> Any:
    if not path.exists():
        raise ManifestError(f"eval suite file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency clarity
            raise ManifestError("YAML eval suites require PyYAML; use JSON or install PyYAML.") from exc
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_callable(target: str, base_dir: Path) -> Callable[..., Any]:
    module_name, _, function_name = target.partition(":")
    if not module_name or not function_name:
        raise ManifestError("python callable must use 'module:function' or 'path.py:function'")
    if module_name.endswith(".py") or "/" in module_name:
        module_path = Path(module_name)
        if not module_path.is_absolute():
            module_path = base_dir / module_path
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise ManifestError(f"cannot load python module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    callback = getattr(module, function_name, None)
    if not callable(callback):
        raise ManifestError(f"python callable not found: {target}")
    return callback


def _merge_eval_suite_options(
    options: Optional[EvalSuiteOptions],
    *,
    name: Optional[str],
    threshold: Optional[float],
    dry_run: Optional[bool],
) -> EvalSuiteOptions:
    opts = options or EvalSuiteOptions()
    return EvalSuiteOptions(
        name=opts.name if name is None else name,
        threshold=opts.threshold if threshold is None else threshold,
        dry_run=opts.dry_run if dry_run is None else dry_run,
    )


def _merge_eval_suite_optimization_options(
    options: Optional[EvalSuiteOptimizationOptions],
    *,
    name: Optional[str],
    threshold: Optional[float],
    max_candidates: Optional[int],
    dry_run: Optional[bool],
) -> EvalSuiteOptimizationOptions:
    opts = options or EvalSuiteOptimizationOptions()
    return EvalSuiteOptimizationOptions(
        name=opts.name if name is None else name,
        threshold=opts.threshold if threshold is None else threshold,
        max_candidates=opts.max_candidates if max_candidates is None else max_candidates,
        dry_run=opts.dry_run if dry_run is None else dry_run,
    )


def _suite_file_like_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        return resolved / "eval_suite.json"
    return resolved


def _eval_suite_descriptor(suite: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "version": suite.get("version") or EVAL_SUITE_SCHEMA_VERSION,
        "providers": [
            {"id": provider.get("id"), "type": provider.get("type")}
            for provider in _as_list(suite.get("providers"))
            if isinstance(provider, Mapping)
        ],
        "prompts": [
            {"id": prompt.get("id")}
            for prompt in _as_list(suite.get("prompts"))
            if isinstance(prompt, Mapping)
        ],
        "tests": [
            {"id": test.get("id")}
            for test in _as_list(suite.get("tests"))
            if isinstance(test, Mapping)
        ],
    }


def _cli() -> Any:
    return importlib.import_module("fi.simulate.cli")


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _copy_mapping_sequence(
    values: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> List[Dict[str, Any]]:
    if isinstance(values, (str, bytes)) or isinstance(values, Mapping):
        raise ValueError(f"{field} must be a sequence of mappings")
    copied = [copy.deepcopy(dict(value)) for value in values]
    if not copied:
        raise ValueError(f"{field} must contain at least one item")
    return copied


__all__ = [
    "AGENT_LEARNING_EVAL_SUITE_SCHEMA_VERSION",
    "EVAL_SUITE_SCHEMA_VERSION",
    "EVAL_SUITE_OPTIMIZATION_SCHEMA_VERSION",
    "EvalSuiteOptimizationOptions",
    "EvalSuiteOptions",
    "build_eval_suite_manifest",
    "load_eval_suite_file",
    "optimize_eval_suite",
    "optimize_eval_suite_file",
    "run_eval_suite",
    "run_eval_suite_file",
    "write_eval_suite_file",
]
