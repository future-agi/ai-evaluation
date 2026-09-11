from __future__ import annotations

import copy
import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence


CLI_SCHEMA_VERSION = "agent-simulate.cli.v1"
MANIFEST_SCHEMA_VERSION = CLI_SCHEMA_VERSION


class ManifestError(ValueError):
    """Raised when a simulation manifest cannot be executed safely."""


@dataclass(frozen=True)
class ManifestRunOptions:
    name: Optional[str] = None
    threshold: Optional[float] = None
    no_eval: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class ManifestOptimizationOptions:
    name: Optional[str] = None
    threshold: Optional[float] = None
    max_candidates: Optional[int] = None
    dry_run: bool = False


def load_manifest_file(path: str | Path) -> Dict[str, Any]:
    """Load a JSON/YAML manifest using the same validation as the CLI."""

    return _cli().load_manifest(Path(path).expanduser().resolve())


load_manifest = load_manifest_file


def write_manifest_file(manifest: Mapping[str, Any], path: str | Path) -> Path:
    """Write a portable simulation manifest as formatted JSON."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return destination


def public_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe result payload without local output bookkeeping."""

    return _cli()._public_result(result)


def detect_manifest_command(
    manifest: Mapping[str, Any],
) -> Literal["run", "redteam", "optimize"]:
    """Return the default command for a manifest shape."""

    if manifest.get("optimization") is not None:
        return "optimize"
    if _has_redteam_block(manifest):
        return "redteam"
    return "run"


def required_manifest_env(manifest: Mapping[str, Any]) -> list[str]:
    """Return all environment keys required before a manifest can execute."""

    return sorted(_cli()._required_env(manifest))


def missing_manifest_env(manifest: Mapping[str, Any]) -> list[str]:
    """Return required environment keys that are not present in os.environ."""

    return [key for key in required_manifest_env(manifest) if not os.environ.get(key)]


def validate_manifest_env(manifest: Mapping[str, Any]) -> None:
    """Raise ManifestError if required environment variables are missing."""

    missing_env = missing_manifest_env(manifest)
    if missing_env:
        raise ManifestError(
            "missing required environment variable(s): "
            f"{', '.join(sorted(missing_env))}"
        )


def apply_manifest_env(manifest: Mapping[str, Any]) -> None:
    """Apply manifest env.set values to process env, matching CLI behavior."""

    _cli()._apply_manifest_env(manifest)


def build_manifest_agent_callback(
    agent: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> Any:
    """Build the runtime callback for a manifest ``agent`` block.

    Supports scripted, echo, python callable, and framework adapter agents using
    the same loader/wrapper path as the CLI.
    """

    return _cli()._build_agent_callback(
        copy.deepcopy(dict(agent)),
        Path(base_dir).expanduser().resolve(),
    )


def supported_manifest_environment_types() -> list[str]:
    """Return the manifest environment types supported by the CLI/SDK registry."""

    return sorted(_cli().MANIFEST_ENVIRONMENT_TYPES)


def build_manifest_environments(
    environments: Any,
    *,
    base_dir: str | Path = ".",
) -> list[Any]:
    """Build environment adapters from manifest ``simulation.environments`` data."""

    if environments in (None, ""):
        specs: list[Any] = []
    elif isinstance(environments, Mapping):
        specs = [copy.deepcopy(dict(environments))]
    else:
        specs = copy.deepcopy(list(environments))
    return _cli()._build_environments(specs, Path(base_dir).expanduser().resolve())


def prepare_redteam_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a red-team manifest with CLI defaults and generated environments applied."""

    runtime_manifest = copy.deepcopy(dict(manifest))
    _cli()._prepare_redteam_manifest(runtime_manifest)
    return runtime_manifest


def _has_redteam_block(manifest: Mapping[str, Any]) -> bool:
    return manifest.get("redteam") not in (None, "", [], {}) or manifest.get(
        "red_team"
    ) not in (None, "", [], {})


def _prepare_redteam_if_present(
    manifest: Dict[str, Any],
    cli: Any,
) -> Optional[Dict[str, Any]]:
    if not _has_redteam_block(manifest):
        return None
    return cli._prepare_redteam_manifest(manifest)


def _redteam_optimization_summary(
    manifest: Mapping[str, Any],
    cli: Any,
) -> Optional[Dict[str, Any]]:
    if not _has_redteam_block(manifest):
        return None
    runtime_manifest = copy.deepcopy(dict(manifest))
    try:
        return _prepare_redteam_if_present(runtime_manifest, cli)
    except ManifestError as exc:
        if "requires at least one adversarial_attack_pack" in str(exc):
            return None
        raise


async def run_local_text_manifest(
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
) -> Any:
    """Run a local_text/local manifest and return the raw TestReport."""

    return await _cli()._run_local_text_manifest(
        manifest,
        Path(manifest_path).expanduser().resolve(),
    )


def evaluate_manifest_report(manifest: Mapping[str, Any], report: Any) -> Any:
    """Score a report with the manifest's evaluation.agent_report block."""

    return _cli()._evaluate_manifest_report(manifest, report)


def render_junit(result: Mapping[str, Any]) -> str:
    """Render a CLI-compatible JUnit XML string for any manifest result."""

    return _cli()._junit_xml(result)


def render_sarif(
    result: Mapping[str, Any],
    *,
    manifest_path: str | Path = ".",
) -> str:
    """Render a SARIF 2.1.0 JSON string for result findings."""

    return _cli()._sarif_json(result, Path(manifest_path).expanduser().resolve())


def render_markdown(
    result: Mapping[str, Any],
    *,
    source_path: str | Path = ".",
) -> str:
    """Render the same Markdown report text produced by the CLI."""

    return _cli()._markdown_text(result, Path(source_path).expanduser().resolve())


def create_baseline_file(
    path: str | Path,
    *,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a result artifact and return a compact compare-safe baseline."""

    source_path = Path(path).expanduser().resolve()
    return create_baseline(
        load_manifest_file(source_path),
        source_path=source_path,
        name=name,
    )


def create_baseline(
    source: Mapping[str, Any],
    *,
    source_path: str | Path = ".",
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a compact compare-safe baseline for a run/redteam/optimize result."""

    started = time.time()
    return _cli()._baseline_result(
        source=copy.deepcopy(dict(source)),
        source_path=Path(source_path).expanduser().resolve(),
        name=name,
        duration_seconds=round(time.time() - started, 4),
    )


def compare_result_files(
    baseline_path: str | Path,
    current_path: str | Path,
    *,
    min_score_delta: float = 0.0,
    max_new_findings: int = 0,
    max_new_error_findings: int = 0,
    min_metric_delta: Optional[float] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Load two result artifacts and return the same compare payload as the CLI."""

    baseline_path = Path(baseline_path).expanduser().resolve()
    current_path = Path(current_path).expanduser().resolve()
    return compare_results(
        load_manifest_file(baseline_path),
        load_manifest_file(current_path),
        baseline_path=baseline_path,
        current_path=current_path,
        min_score_delta=min_score_delta,
        max_new_findings=max_new_findings,
        max_new_error_findings=max_new_error_findings,
        min_metric_delta=min_metric_delta,
        name=name,
    )


def compare_results(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    baseline_path: str | Path = "baseline.json",
    current_path: str | Path = "current.json",
    min_score_delta: float = 0.0,
    max_new_findings: int = 0,
    max_new_error_findings: int = 0,
    min_metric_delta: Optional[float] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare baseline/current result payloads with stable gate semantics."""

    started = time.time()
    return _cli()._compare_results(
        baseline=copy.deepcopy(dict(baseline)),
        current=copy.deepcopy(dict(current)),
        baseline_path=Path(baseline_path).expanduser().resolve(),
        current_path=Path(current_path).expanduser().resolve(),
        min_score_delta=float(min_score_delta),
        max_new_findings=int(max_new_findings),
        max_new_error_findings=int(max_new_error_findings),
        min_metric_delta=min_metric_delta,
        name=name,
        duration_seconds=round(time.time() - started, 4),
    )


def render_report_file(
    path: str | Path,
    *,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a result artifact and return a Markdown report payload."""

    source_path = Path(path).expanduser().resolve()
    return render_report(
        load_manifest_file(source_path),
        source_path=source_path,
        name=name,
    )


def render_report(
    source: Mapping[str, Any],
    *,
    source_path: str | Path = ".",
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a Markdown report payload for any manifest result."""

    started = time.time()
    return _cli()._report_result(
        source=copy.deepcopy(dict(source)),
        source_path=Path(source_path).expanduser().resolve(),
        name=name,
        duration_seconds=round(time.time() - started, 4),
    )


def promote_to_regression_file(
    path: str | Path,
    *,
    name: Optional[str] = None,
    min_level: Literal["note", "warning", "error"] = "warning",
    max_findings: int = 25,
    required_env: Sequence[str] = (),
) -> Dict[str, Any]:
    """Load a result artifact and return a runnable red-team regression manifest."""

    source_path = Path(path).expanduser().resolve()
    return promote_to_regression(
        load_manifest_file(source_path),
        source_path=source_path,
        name=name,
        min_level=min_level,
        max_findings=max_findings,
        required_env=required_env,
    )


def promote_to_regression(
    source: Mapping[str, Any],
    *,
    source_path: str | Path = ".",
    name: Optional[str] = None,
    min_level: Literal["note", "warning", "error"] = "warning",
    max_findings: int = 25,
    required_env: Sequence[str] = (),
) -> Dict[str, Any]:
    """Return a regression-promotion payload with an embedded runnable manifest."""

    started = time.time()
    return _cli()._regression_promotion_result(
        source=copy.deepcopy(dict(source)),
        source_path=Path(source_path).expanduser().resolve(),
        name=name,
        min_level=str(min_level),
        max_findings=int(max_findings),
        required_env=list(required_env),
        duration_seconds=round(time.time() - started, 4),
    )


def shrink_attack_evolution_file(
    path: str | Path,
    *,
    name: Optional[str] = None,
    manifest_name: Optional[str] = None,
    required_env: Sequence[str] = (),
) -> Dict[str, Any]:
    """Load an attack-evolution artifact and return a minimized replay manifest."""

    source_path = Path(path).expanduser().resolve()
    return shrink_attack_evolution(
        load_manifest_file(source_path),
        source_path=source_path,
        name=name,
        manifest_name=manifest_name,
        required_env=required_env,
    )


def shrink_attack_evolution(
    source: Mapping[str, Any],
    *,
    source_path: str | Path = ".",
    name: Optional[str] = None,
    manifest_name: Optional[str] = None,
    required_env: Sequence[str] = (),
) -> Dict[str, Any]:
    """Return a minimized attack-evolution regression payload."""

    started = time.time()
    return _cli()._attack_evolution_shrink_result(
        source=copy.deepcopy(dict(source)),
        source_path=Path(source_path).expanduser().resolve(),
        name=name,
        manifest_name=manifest_name,
        required_env=list(required_env),
        duration_seconds=round(time.time() - started, 4),
    )


def replay_manifests(
    manifests: Sequence[str | Path],
    *,
    name: Optional[str] = None,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """Run files/directories/globs as one manifest replay suite."""

    cli = _cli()
    started = time.time()
    requested = [str(item) for item in manifests]
    paths = cli._replay_manifest_paths(requested)
    children = []
    for path in paths:
        child = cli._execute_replay_manifest(path, dry_run=bool(dry_run))
        children.append(child)
        if child.get("exit_code") != 0 and fail_fast:
            break
    return cli._replay_result(
        children=children,
        requested=requested,
        name=name,
        duration_seconds=round(time.time() - started, 4),
        dry_run=bool(dry_run),
        fail_fast=bool(fail_fast),
    )


async def run_manifest_file(
    path: str | Path,
    *,
    options: Optional[ManifestRunOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    no_eval: Optional[bool] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load and execute a run manifest, returning the CLI-compatible payload."""

    manifest_path = Path(path).expanduser().resolve()
    return await run_manifest(
        load_manifest_file(manifest_path),
        manifest_path=manifest_path,
        options=_run_options(
            options,
            name=name,
            threshold=threshold,
            no_eval=no_eval,
            dry_run=dry_run,
        ),
    )


async def run_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path = ".",
    options: Optional[ManifestRunOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    no_eval: Optional[bool] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute a run manifest without writing CLI artifacts."""

    cli = _cli()
    opts = _run_options(
        options,
        name=name,
        threshold=threshold,
        no_eval=no_eval,
        dry_run=dry_run,
    )
    runtime_manifest = copy.deepcopy(dict(manifest))
    manifest_path = Path(manifest_path).expanduser().resolve()
    if opts.name:
        runtime_manifest["name"] = opts.name
    if opts.threshold is not None:
        runtime_manifest.setdefault("evaluation", {}).setdefault("agent_report", {})[
            "threshold"
        ] = opts.threshold
    if opts.no_eval:
        runtime_manifest.setdefault("evaluation", {})["enabled"] = False

    started = time.time()
    validate_manifest_env(runtime_manifest)
    apply_manifest_env(runtime_manifest)
    if opts.dry_run:
        return {
            "schema_version": CLI_SCHEMA_VERSION,
            "name": str(runtime_manifest.get("name") or manifest_path.stem),
            "status": "passed",
            "exit_code": 0,
            "dry_run": True,
            "summary": {
                "required_env": required_manifest_env(runtime_manifest),
                "scenario_cases": len(
                    cli._scenario_dataset(runtime_manifest, manifest_path.parent)
                ),
                "environment_count": len(cli._environment_specs(runtime_manifest)),
            },
            "duration_seconds": round(time.time() - started, 4),
        }

    report = await cli._run_manifest(runtime_manifest, manifest_path)
    evaluation = evaluate_manifest_report(runtime_manifest, report)
    result = cli._run_result(
        manifest=runtime_manifest,
        report=report,
        evaluation=evaluation,
        duration_seconds=round(time.time() - started, 4),
    )
    orchestration_strategy = cli._orchestration_strategy_card(
        result,
        source_path=manifest_path,
        source_manifest_path=manifest_path,
    )
    if orchestration_strategy is not None:
        result["orchestration_strategy"] = orchestration_strategy
    framework_readiness = cli._framework_readiness_card(
        result,
        source_path=manifest_path,
        source_manifest_path=manifest_path,
    )
    if framework_readiness is not None:
        result["framework_readiness"] = framework_readiness
    agent_integration_readiness = cli._agent_integration_readiness_card(
        result,
        source_path=manifest_path,
        source_manifest_path=manifest_path,
    )
    if agent_integration_readiness is not None:
        result["agent_integration_readiness"] = agent_integration_readiness
    return result


async def redteam_manifest_file(
    path: str | Path,
    *,
    options: Optional[ManifestRunOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load and execute a red-team manifest, returning the CLI-compatible payload."""

    manifest_path = Path(path).expanduser().resolve()
    return await redteam_manifest(
        load_manifest_file(manifest_path),
        manifest_path=manifest_path,
        options=_run_options(
            options,
            name=name,
            threshold=threshold,
            dry_run=dry_run,
        ),
    )


run_redteam_manifest_file = redteam_manifest_file


async def redteam_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path = ".",
    options: Optional[ManifestRunOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute a red-team manifest without writing CLI artifacts."""

    cli = _cli()
    opts = _run_options(
        options,
        name=name,
        threshold=threshold,
        dry_run=dry_run,
    )
    runtime_manifest = copy.deepcopy(dict(manifest))
    manifest_path = Path(manifest_path).expanduser().resolve()
    if opts.name:
        runtime_manifest["name"] = opts.name
    if opts.threshold is not None:
        runtime_manifest.setdefault("evaluation", {}).setdefault("agent_report", {})[
            "threshold"
        ] = opts.threshold

    started = time.time()
    redteam_summary = cli._prepare_redteam_manifest(runtime_manifest)
    validate_manifest_env(runtime_manifest)
    apply_manifest_env(runtime_manifest)
    if opts.dry_run:
        result = {
            "schema_version": CLI_SCHEMA_VERSION,
            "name": str(runtime_manifest.get("name") or manifest_path.stem),
            "status": "passed",
            "exit_code": 0,
            "dry_run": True,
            "summary": {
                "required_env": required_manifest_env(runtime_manifest),
                "scenario_cases": len(
                    cli._scenario_dataset(runtime_manifest, manifest_path.parent)
                ),
                "environment_count": len(cli._environment_specs(runtime_manifest)),
                "redteam": redteam_summary,
            },
            "redteam": redteam_summary,
            "duration_seconds": round(time.time() - started, 4),
        }
        redteam_strategy = cli._redteam_strategy_card(
            result,
            source_path=manifest_path,
            source_manifest_path=manifest_path,
        )
        if redteam_strategy is not None:
            result["redteam_strategy"] = redteam_strategy
        return result

    report = await run_local_text_manifest(runtime_manifest, manifest_path)
    evaluation = evaluate_manifest_report(runtime_manifest, report)
    result = cli._run_result(
        manifest=runtime_manifest,
        report=report,
        evaluation=evaluation,
        duration_seconds=round(time.time() - started, 4),
    )
    redteam_result = cli._redteam_result_summary(
        runtime_manifest,
        result.get("evaluation"),
    )
    result["redteam"] = redteam_result
    result["summary"]["redteam"] = redteam_result
    redteam_strategy = cli._redteam_strategy_card(
        result,
        source_path=manifest_path,
        source_manifest_path=manifest_path,
    )
    if redteam_strategy is not None:
        result["redteam_strategy"] = redteam_strategy
    return result


run_redteam_manifest = redteam_manifest


def optimize_manifest_file(
    path: str | Path,
    *,
    options: Optional[ManifestOptimizationOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load and optimize a manifest, returning the CLI-compatible payload."""

    manifest_path = Path(path).expanduser().resolve()
    return optimize_manifest(
        load_manifest_file(manifest_path),
        manifest_path=manifest_path,
        options=_optimization_options(
            options,
            name=name,
            threshold=threshold,
            max_candidates=max_candidates,
            dry_run=dry_run,
        ),
    )


def optimize_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path = ".",
    options: Optional[ManifestOptimizationOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Optimize a manifest without writing CLI artifacts."""

    cli = _cli()
    opts = _optimization_options(
        options,
        name=name,
        threshold=threshold,
        max_candidates=max_candidates,
        dry_run=dry_run,
    )
    runtime_manifest = copy.deepcopy(dict(manifest))
    manifest_path = Path(manifest_path).expanduser().resolve()
    if opts.name:
        runtime_manifest["name"] = opts.name
    if opts.threshold is not None:
        runtime_manifest.setdefault("optimization", {})["threshold"] = opts.threshold
    if opts.max_candidates is not None:
        runtime_manifest.setdefault("optimization", {}).setdefault("optimizer", {})[
            "max_candidates"
        ] = opts.max_candidates

    started = time.time()
    validate_manifest_env(runtime_manifest)
    apply_manifest_env(runtime_manifest)
    optimization = cli._optimization_config(runtime_manifest)
    redteam_summary = _redteam_optimization_summary(runtime_manifest, cli)
    if opts.dry_run:
        summary = {
            "required_env": required_manifest_env(runtime_manifest),
            "search_path_count": len(
                cli._target_config(optimization).get("search_space", {})
            ),
            "max_candidates": cli._optimizer_config(optimization).get("max_candidates"),
        }
        if redteam_summary is not None:
            summary["redteam"] = redteam_summary
        result = {
            "schema_version": CLI_SCHEMA_VERSION,
            "name": str(runtime_manifest.get("name") or manifest_path.stem),
            "status": "passed",
            "exit_code": 0,
            "dry_run": True,
            "summary": summary,
            "duration_seconds": round(time.time() - started, 4),
        }
        if redteam_summary is not None:
            result["redteam"] = redteam_summary
        return result

    problem = build_manifest_optimization_problem(
        runtime_manifest,
        manifest_path=manifest_path,
        name=str(runtime_manifest.get("name") or manifest_path.stem),
    )
    result = problem.optimize()
    payload = cli._optimization_result(
        manifest=runtime_manifest,
        manifest_path=manifest_path,
        optimization_result=result,
        threshold=float(optimization.get("threshold", 0.7)),
        duration_seconds=round(time.time() - started, 4),
    )
    if redteam_summary is not None:
        payload["redteam"] = redteam_summary
        payload.setdefault("summary", {})["redteam"] = redteam_summary
    orchestration_strategy = cli._orchestration_strategy_card(
        payload,
        source_path=manifest_path,
        source_manifest_path=manifest_path,
    )
    if orchestration_strategy is not None:
        payload["orchestration_strategy"] = orchestration_strategy
    framework_readiness = cli._framework_readiness_card(
        payload,
        source_path=manifest_path,
        source_manifest_path=manifest_path,
    )
    if framework_readiness is not None:
        payload["framework_readiness"] = framework_readiness
    agent_integration_readiness = cli._agent_integration_readiness_card(
        payload,
        source_path=manifest_path,
        source_manifest_path=manifest_path,
    )
    if agent_integration_readiness is not None:
        payload["agent_integration_readiness"] = agent_integration_readiness
    return payload


def build_manifest_optimization_problem(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path = ".",
    name: Optional[str] = None,
) -> Any:
    """Build an Agent Learning ManifestOptimizationProblem for this manifest."""

    cli = _cli()
    manifest_path = Path(manifest_path).expanduser().resolve()
    runtime_manifest = copy.deepcopy(dict(manifest))
    optimization = cli._optimization_config(runtime_manifest)
    manifest_base = copy.deepcopy(dict(runtime_manifest))
    manifest_base.pop("optimization", None)

    try:
        from fi.opt import ManifestOptimizationProblem
    except Exception as exc:  # pragma: no cover - optional dependency clarity
        raise ManifestError(
            "Agent Learning Kit optimizer engine is required for manifest optimization."
        ) from exc

    def evaluate_manifest(candidate_manifest: Mapping[str, Any], candidate: Any) -> Any:
        if isinstance(candidate_manifest, dict):
            runtime_candidate = candidate_manifest
        else:
            runtime_candidate = copy.deepcopy(dict(candidate_manifest))
        _prepare_redteam_if_present(runtime_candidate, cli)
        return run_local_text_manifest(runtime_candidate, manifest_path)

    def score_manifest(
        candidate_manifest: Mapping[str, Any],
        report: Any,
        candidate: Any,
    ) -> Dict[str, Any]:
        evaluation = evaluate_manifest_report(candidate_manifest, report)
        evidence_config = _simulation_evidence_scoring_config(optimization)
        evidence_evaluation = None
        if evidence_config is not None:
            from fi.opt import score_simulation_evidence

            evidence_evaluation = score_simulation_evidence(
                report,
                manifest=candidate_manifest,
                candidate=candidate,
                config=evidence_config,
            )
            score = float(evidence_evaluation.score)
        else:
            score = float(
                getattr(evaluation, "score", 1.0 if evaluation is None else 0.0)
            )
            # bug #2: when the manifest DECLARES an anchor objective, score the
            # candidate on it (real dynamic range) instead of the all-metrics-mean
            # evaluation score. Scoped: no declared-anchor objective -> unchanged.
            from fi.opt.integrations.simulate import _score_from_value

            eval_plain = cli._to_plain(evaluation) if evaluation is not None else {}
            objective = (
                candidate_manifest.get("objective")
                or (
                    (candidate_manifest.get("simulation") or {}).get("inline") or {}
                ).get("objective")
                or (candidate_manifest.get("evaluation") or {}).get("objective")
            )
            anchored = _score_from_value(
                {
                    "objective": objective,
                    "summary": (
                        eval_plain.get("summary")
                        if isinstance(eval_plain, Mapping)
                        else {}
                    )
                    or {},
                }
            )
            if anchored is not None:
                score = float(anchored)
        metadata = {
            "agent_report_evaluation": (
                cli._to_plain(evaluation) if evaluation is not None else None
            ),
            "report_summary": cli._report_summary(report),
        }
        if evidence_evaluation is not None:
            metadata["simulation_evidence_score"] = evidence_evaluation.metadata.get(
                "simulation_evidence_score"
            )
        return {
            "score": score,
            "reason": getattr(evidence_evaluation, "reason", "")
            if evidence_evaluation is not None
            else "",
            "metadata": metadata,
        }

    return ManifestOptimizationProblem.from_manifest(
        {**manifest_base, "optimization": optimization},
        evaluate_manifest=evaluate_manifest,
        score_manifest=score_manifest,
        name=name or str(manifest.get("name") or manifest_path.stem),
    )


def _simulation_evidence_scoring_config(
    optimization: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    raw = (
        optimization.get("simulation_evidence")
        or optimization.get("evidence_scorer")
        or optimization.get("scoring")
    )
    if raw is True:
        return {"enabled": True, "method": "simulation_evidence"}
    if isinstance(raw, str):
        normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"simulation_evidence", "evidence", "environment_evidence"}:
            return {"enabled": True, "method": "simulation_evidence"}
        return None
    if not isinstance(raw, Mapping):
        return None
    method = (
        str(
            raw.get("method")
            or raw.get("type")
            or raw.get("name")
            or raw.get("strategy")
            or "simulation_evidence"
        )
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if not bool(raw.get("enabled", True)):
        return None
    if method not in {
        "simulation_evidence",
        "evidence",
        "environment_evidence",
        "trace_evidence",
    }:
        return None
    config = copy.deepcopy(dict(raw))
    config["method"] = "simulation_evidence"
    return config


def _cli() -> Any:
    return importlib.import_module("fi.simulate.cli")


def _run_options(
    options: Optional[ManifestRunOptions],
    *,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    no_eval: Optional[bool] = None,
    dry_run: Optional[bool] = None,
) -> ManifestRunOptions:
    opts = options or ManifestRunOptions()
    return ManifestRunOptions(
        name=opts.name if name is None else name,
        threshold=opts.threshold if threshold is None else threshold,
        no_eval=opts.no_eval if no_eval is None else no_eval,
        dry_run=opts.dry_run if dry_run is None else dry_run,
    )


def _optimization_options(
    options: Optional[ManifestOptimizationOptions],
    *,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
) -> ManifestOptimizationOptions:
    opts = options or ManifestOptimizationOptions()
    return ManifestOptimizationOptions(
        name=opts.name if name is None else name,
        threshold=opts.threshold if threshold is None else threshold,
        max_candidates=opts.max_candidates
        if max_candidates is None
        else max_candidates,
        dry_run=opts.dry_run if dry_run is None else dry_run,
    )


__all__ = [
    "CLI_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestError",
    "ManifestOptimizationOptions",
    "ManifestRunOptions",
    "apply_manifest_env",
    "build_manifest_agent_callback",
    "build_manifest_environments",
    "build_manifest_optimization_problem",
    "compare_result_files",
    "compare_results",
    "create_baseline",
    "create_baseline_file",
    "detect_manifest_command",
    "evaluate_manifest_report",
    "load_manifest",
    "load_manifest_file",
    "missing_manifest_env",
    "optimize_manifest",
    "optimize_manifest_file",
    "prepare_redteam_manifest",
    "promote_to_regression",
    "promote_to_regression_file",
    "public_result",
    "redteam_manifest",
    "redteam_manifest_file",
    "required_manifest_env",
    "render_junit",
    "render_markdown",
    "render_report",
    "render_report_file",
    "render_sarif",
    "replay_manifests",
    "run_local_text_manifest",
    "run_manifest",
    "run_manifest_file",
    "run_redteam_manifest",
    "run_redteam_manifest_file",
    "shrink_attack_evolution",
    "shrink_attack_evolution_file",
    "supported_manifest_environment_types",
    "validate_manifest_env",
]
