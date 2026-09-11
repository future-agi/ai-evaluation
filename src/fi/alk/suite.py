from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from xml.sax.saxutils import escape

from ._schema import AGENT_LEARNING_CLI_SCHEMA_VERSION, public_payload


AGENT_LEARNING_SUITE_KIND = "agent-learning.suite.v1"
AGENT_LEARNING_SUITE_OPTIMIZATION_KIND = "agent-learning.suite-optimization.v1"
AGENT_LEARNING_OPTIMIZATION_LIFECYCLE_KIND = (
    "agent-learning.optimization-lifecycle.v1"
)
AGENT_LEARNING_SUITE_TRUST_CERTIFICATE_KIND = (
    "agent-learning.suite.trust-certificate.v1"
)
AGENT_LEARNING_SUITE_TRUST_VERIFICATION_KIND = (
    "agent-learning.suite.trust-verification.v1"
)

_CHILD_COMMANDS = {
    "action_run",
    "baseline",
    "compare",
    "promote_to_regression",
    "replay",
    "report",
    "run",
    "shrink",
    "suite",
    "eval",
    "eval_artifact",
    "eval_task",
    "redteam",
    "optimize",
    "optimize_eval",
    "optimize_suite",
}

_ADMITTED_EVIDENCE_ROLES = {
    "admitted",
    "claim",
    "primary",
    "paper_facing",
    "paper_facing_evidence",
}

_NON_ADMITTED_EVIDENCE_ROLES = {
    "calibration",
    "diagnostic",
    "fixture",
    "preflight",
    "smoke",
    "support",
}


class SuiteError(ValueError):
    """Raised when an Agent Learning suite manifest cannot run."""


@dataclass(frozen=True)
class SuiteRunOptions:
    name: Optional[str] = None
    threshold: Optional[float] = None
    max_candidates: Optional[int] = None
    dry_run: bool = False
    fail_fast: bool = False
    require_optimizer_governance: bool = False


@dataclass(frozen=True)
class SuiteOptimizationOptions:
    name: Optional[str] = None
    threshold: Optional[float] = None
    max_candidates: Optional[int] = None
    dry_run: bool = False


def load_suite_file(path: str | Path) -> dict[str, Any]:
    suite_path = Path(path).expanduser().resolve()
    if not suite_path.exists():
        raise SuiteError(f"suite manifest not found: {suite_path}")
    suite = _load_json_or_yaml(suite_path)
    if not isinstance(suite, Mapping):
        raise SuiteError("suite manifest root must be an object")
    return _prepare_suite(dict(suite), base_dir=suite_path.parent)


def load_suite_artifact_file(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        raise SuiteError(f"suite artifact not found: {artifact_path}")
    artifact = _load_json_or_yaml(artifact_path)
    if not isinstance(artifact, Mapping):
        raise SuiteError("suite artifact root must be an object")
    return dict(artifact)


def verify_trust_certificate_file(
    path: str | Path,
    *,
    required_verdict: str = "approved",
    require_promotion_ready: bool = True,
) -> dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    artifact = load_suite_artifact_file(artifact_path)
    return verify_trust_certificate(
        artifact,
        required_verdict=required_verdict,
        require_promotion_ready=require_promotion_ready,
        source_path=artifact_path,
    )


def verify_trust_certificate(
    artifact: Mapping[str, Any],
    *,
    required_verdict: str = "approved",
    require_promotion_ready: bool = True,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a saved suite trust certificate without re-running the suite."""
    required = _suite_key(required_verdict)
    if required not in _TRUST_VERDICT_RANK:
        allowed = ", ".join(sorted(_TRUST_VERDICT_RANK))
        raise SuiteError(f"required_verdict must be one of: {allowed}")

    source = Path(source_path).expanduser().resolve() if source_path else None
    result_kind = str(artifact.get("kind") or artifact.get("version") or "")
    summary = _as_mapping(artifact.get("summary"))
    certificate = _as_mapping(artifact.get("trust_certificate"))
    if not certificate and result_kind == AGENT_LEARNING_SUITE_TRUST_CERTIFICATE_KIND:
        certificate = dict(artifact)

    findings: list[dict[str, Any]] = []
    certificate_kind = str(certificate.get("kind") or "") if certificate else ""
    if not certificate:
        findings.append({
            "type": "suite_trust_certificate_missing",
            "level": "error",
            "reason": "Suite artifact does not contain a trust_certificate block.",
        })
    elif certificate_kind != AGENT_LEARNING_SUITE_TRUST_CERTIFICATE_KIND:
        findings.append({
            "type": "suite_trust_certificate_kind_mismatch",
            "level": "error",
            "reason": (
                "Suite trust certificate kind must be "
                f"{AGENT_LEARNING_SUITE_TRUST_CERTIFICATE_KIND}."
            ),
            "observed_kind": certificate_kind,
        })

    observed = _suite_key(
        certificate.get("verdict") if certificate else None
    ) or _suite_key(summary.get("trust_certificate_verdict"))
    verdict_rank_passed = False
    if certificate:
        if observed not in _TRUST_VERDICT_RANK:
            findings.append({
                "type": "suite_trust_certificate_verdict_unknown",
                "level": "error",
                "reason": "Suite trust certificate verdict is missing or unknown.",
                "observed_verdict": observed or None,
            })
        else:
            verdict_rank_passed = (
                _TRUST_VERDICT_RANK[observed] >= _TRUST_VERDICT_RANK[required]
            )
            if not verdict_rank_passed:
                findings.append({
                    "type": "suite_trust_certificate_verdict_too_low",
                    "level": "error",
                    "reason": (
                        f"Suite trust certificate verdict {observed} is below "
                        f"required verdict {required}."
                    ),
                    "required_verdict": required,
                    "observed_verdict": observed,
                })

    promotion_ready = _optional_bool(
        certificate.get("promotion_ready") if certificate else None,
        summary.get("trust_certificate_promotion_ready"),
    )
    promotion_gate_passed = not require_promotion_ready or promotion_ready is True
    if certificate and not promotion_gate_passed:
        findings.append({
            "type": "suite_trust_certificate_not_promotion_ready",
            "level": "error",
            "reason": "Suite trust certificate is not marked promotion_ready.",
            "promotion_ready": promotion_ready,
        })

    passed = not findings
    return {
        "kind": AGENT_LEARNING_SUITE_TRUST_VERIFICATION_KIND,
        "version": AGENT_LEARNING_SUITE_TRUST_VERIFICATION_KIND,
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 1,
        "source_path": str(source) if source else None,
        "result_kind": result_kind or None,
        "required_verdict": required,
        "require_promotion_ready": bool(require_promotion_ready),
        "observed_verdict": observed or None,
        "promotion_ready": promotion_ready,
        "certificate_kind": certificate_kind or None,
        "assurance_level": (
            certificate.get("assurance_level") if certificate else None
        ),
        "summary": {
            "certificate_present": bool(certificate),
            "certificate_kind_passed": (
                certificate_kind == AGENT_LEARNING_SUITE_TRUST_CERTIFICATE_KIND
            ),
            "verdict_rank_passed": verdict_rank_passed,
            "promotion_gate_passed": promotion_gate_passed,
            "finding_count": len(findings),
        },
        "trust_certificate": copy.deepcopy(certificate),
        "findings": findings,
    }


load_suite = load_suite_file


def required_suite_env(
    suite: Mapping[str, Any],
    *,
    suite_path: str | Path = ".",
) -> list[str]:
    base_dir = _suite_base_dir(suite_path)
    required = set(_as_string_list(suite.get("required_env")))
    for job in _suite_jobs(suite):
        try:
            child = _load_child_source(job, base_dir=base_dir)
        except Exception:
            continue
        if _normalize_command(job.get("command") or job.get("type")) == "suite":
            required.update(
                required_suite_env(
                    child,
                    suite_path=_job_path(job, base_dir=base_dir),
                )
            )
            continue
        required.update(_as_string_list(child.get("required_env")))
    return sorted(required)


def missing_suite_env(
    suite: Mapping[str, Any],
    *,
    suite_path: str | Path = ".",
) -> list[str]:
    return [
        key
        for key in required_suite_env(suite, suite_path=suite_path)
        if not os.environ.get(key)
    ]


def validate_suite_env(
    suite: Mapping[str, Any],
    *,
    suite_path: str | Path = ".",
) -> None:
    missing = missing_suite_env(suite, suite_path=suite_path)
    if missing:
        raise SuiteError(
            "missing required environment variable(s): "
            f"{', '.join(sorted(missing))}"
        )


def build_suite_manifest(
    *,
    name: str,
    jobs: Sequence[Mapping[str, Any]],
    required_env: Sequence[str] = (),
    required_capabilities: Optional[Mapping[str, Sequence[str]]] = None,
    outputs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    optimizer_governance_policy: Optional[Mapping[str, Any]] = None,
    threshold: Optional[float] = None,
    fail_fast: Optional[bool] = None,
) -> dict[str, Any]:
    """Build an Agent Learning suite manifest from SDK data.

    This is the SDK counterpart to writing ``agent-learning.suite.v1`` JSON by
    hand: users can compose run/eval/red-team/optimization jobs in Python and
    execute them through ``run_suite`` or ``run_suite_file``.
    """

    if not name:
        raise ValueError("name is required")
    if not jobs:
        raise ValueError("jobs must contain at least one suite job")
    manifest: dict[str, Any] = {
        "version": AGENT_LEARNING_SUITE_KIND,
        "name": str(name),
        "required_env": _unique_strings(required_env),
        "jobs": [
            _normalize_suite_job(job, index)
            for index, job in enumerate(jobs, start=1)
        ],
    }
    if required_capabilities:
        manifest["required_capabilities"] = {
            str(key): _unique_strings(value)
            for key, value in dict(required_capabilities).items()
            if _unique_strings(value)
        }
    if outputs:
        manifest["outputs"] = copy.deepcopy(dict(outputs))
    if metadata:
        manifest["metadata"] = copy.deepcopy(dict(metadata))
    if optimizer_governance_policy:
        manifest["optimizer_governance_policy"] = copy.deepcopy(
            dict(optimizer_governance_policy)
        )
    if threshold is not None:
        manifest["threshold"] = float(threshold)
    if fail_fast is not None:
        manifest["fail_fast"] = bool(fail_fast)
    return manifest


def build_trinity_suite_manifest(
    *,
    name: str,
    run_path: str | Path,
    eval_path: str | Path,
    artifact_eval_path: str | Path,
    artifact_report_path: str | Path,
    redteam_path: str | Path,
    eval_optimization_path: str | Path,
    optimization_path: str | Path,
    world_model_optimization_path: str | Path | None = None,
    artifact_action_id: str | None = "report_orchestration_strategy",
    artifact_action_cwd: str | Path | None = "artifacts/action-loop/workspace",
    artifact_optimization_path: str | Path | None = None,
    artifact_eval_config_path: str | Path | None = None,
    required_env: Sequence[str] = (),
    max_candidates: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a run/eval/artifact/red-team/optimization suite.

    The manifest mirrors the promptfoo-style trinity workflow: simulation,
    text eval, saved-artifact eval, direct artifact-report eval, optional
    artifact-evidence optimization, red-team, eval-suite optimization, and
    full manifest optimization in one capability-gated suite.
    """

    suite_name = str(name)
    jobs: list[dict[str, Any]] = [
        {
            "id": "local-simulation",
            "command": "run",
            "path": _suite_path_text(run_path),
            "name": f"{suite_name}-run",
        },
        {
            "id": "promptfoo-style-eval",
            "command": "eval",
            "path": _suite_path_text(eval_path),
            "name": f"{suite_name}-eval",
        },
        {
            "id": "artifact-task-eval",
            "command": "eval",
            "path": _suite_path_text(artifact_eval_path),
            "name": f"{suite_name}-artifact-eval",
        },
        {
            "id": "direct-artifact-report-eval",
            "command": "eval-artifact",
            "path": _suite_path_text(artifact_report_path),
            "name": f"{suite_name}-direct-artifact",
        },
    ]
    if artifact_action_id:
        action_job = {
            "id": "artifact-action-report",
            "command": "action-run",
            "path": _suite_path_text(artifact_report_path),
            "action_id": str(artifact_action_id),
            "name": f"{suite_name}-artifact-action-report",
            "output": "../../artifacts/action-loop/action-run.json",
            "outputs": {
                "junit": "../../artifacts/action-loop/action-run.junit.xml",
                "sarif": "../../artifacts/action-loop/action-run.sarif.json",
                "markdown": "../../artifacts/action-loop/action-run.md",
            },
        }
        if artifact_action_cwd is not None:
            action_job["cwd"] = _suite_path_text(artifact_action_cwd)
        jobs.append(action_job)
    if artifact_optimization_path is not None:
        jobs.append(
            {
                "id": "artifact-evidence-optimizer",
                "command": "optimize-eval",
                "path": _suite_path_text(artifact_optimization_path),
                "name": f"{suite_name}-artifact-optimizer",
            }
        )
    jobs.extend(
        [
            {
                "id": "agent-red-team",
                "command": "redteam",
                "path": _suite_path_text(redteam_path),
                "name": f"{suite_name}-redteam",
            },
            {
                "id": "eval-suite-optimizer",
                "command": "optimize-eval",
                "path": _suite_path_text(eval_optimization_path),
                "name": f"{suite_name}-eval-optimizer",
            },
            {
                "id": "agent-optimizer",
                "command": "optimize",
                "path": _suite_path_text(optimization_path),
                "name": f"{suite_name}-optimizer",
            },
        ]
    )
    required_metrics = ["eval_assertions"]
    if world_model_optimization_path is not None:
        jobs.append(
            {
                "id": "world-model-optimizer",
                "command": "optimize",
                "path": _suite_path_text(world_model_optimization_path),
                "name": f"{suite_name}-world-model-optimizer",
            }
        )
        required_metrics.extend(
            [
                "world_contract_quality",
                "world_contract_coverage",
                "tool_selection_accuracy",
            ]
        )
    if artifact_eval_config_path is not None:
        jobs[3]["config"] = _suite_path_text(artifact_eval_config_path)
    if max_candidates is not None:
        for job in jobs:
            if job["command"] in {"optimize", "optimize-eval"}:
                job["max_candidates"] = int(max_candidates)
    return build_suite_manifest(
        name=suite_name,
        required_env=required_env,
        jobs=jobs,
        required_capabilities={
            "commands": [
                "run",
                "eval",
                "eval_artifact",
                "action_run",
                "redteam",
                "optimize_eval",
                "optimize",
            ],
            "result_kinds": [
                "agent-learning.run.v1",
                "agent-learning.eval.v1",
                "agent-learning.artifact-evaluation.v1",
                "agent-learning.action-run.v1",
                "agent-learning.redteam.v1",
                "agent-learning.eval-optimization.v1",
                "agent-learning.optimization.v1",
            ],
            "metrics": required_metrics,
        },
        metadata={
            "source": "fi.alk.suite.build_trinity_suite_manifest",
            **copy.deepcopy(dict(metadata or {})),
        },
        optimizer_governance_policy={
            "require_optimizer_governance": True,
            "min_governed": 1,
        },
    )


def build_framework_adapter_trinity_suite_manifest(
    *,
    name: str,
    run_path: str | Path,
    redteam_path: str | Path,
    required_env: Sequence[str] = (),
    required_frameworks: Sequence[str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
    outputs: Optional[Mapping[str, Any]] = None,
    threshold: Optional[float] = None,
    fail_fast: bool = True,
) -> dict[str, Any]:
    """Build a focused suite for framework simulation, eval, and red-team gates."""

    if not name:
        raise ValueError("name is required")
    suite_name = str(name)
    frameworks = _unique_strings(required_frameworks)
    required_capabilities: dict[str, list[str]] = {
        "commands": ["run", "redteam"],
        "result_kinds": [
            "agent-learning.run.v1",
            "agent-learning.redteam.v1",
        ],
        "metrics": [
            "framework_runtime_contract",
            "framework_adapter_contract_quality",
            "adversarial_resilience",
            "red_team_campaign_quality",
        ],
    }
    if frameworks:
        required_capabilities["frameworks"] = frameworks
    return build_suite_manifest(
        name=suite_name,
        required_env=required_env,
        jobs=[
            {
                "id": "optimized-framework-run",
                "command": "run",
                "path": _suite_path_text(run_path),
                "name": f"{suite_name}-run",
            },
            {
                "id": "framework-red-team",
                "command": "redteam",
                "path": _suite_path_text(redteam_path),
                "name": f"{suite_name}-redteam",
            },
        ],
        required_capabilities=required_capabilities,
        outputs=outputs,
        threshold=threshold,
        fail_fast=fail_fast,
        metadata={
            "source": "fi.alk.suite.build_framework_adapter_trinity_suite_manifest",
            "task_kind": "framework_adapter_trinity_suite",
            **copy.deepcopy(dict(metadata or {})),
        },
    )


def write_framework_adapter_trinity_suite_workspace(
    *,
    name: str,
    framework: str,
    target: str,
    directory: str | Path,
    adapter_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
    agent: Any = None,
    agent_factory: Any = None,
    cases: Sequence[Mapping[str, Any]] = (),
    target_base_dir: str | Path = ".",
    target_factory: Optional[bool] = None,
    method_candidates: Optional[Sequence[str | None]] = None,
    input_mode_candidates: Optional[Sequence[str]] = None,
    required_env: Sequence[str] = (),
    scenario: Optional[Mapping[str, Any]] = None,
    framework_trace: Optional[Mapping[str, Any]] = None,
    evaluation_config: Optional[Mapping[str, Any]] = None,
    auto_evaluation_config: bool = True,
    threshold: float = 0.9,
    trace_runtime: bool = True,
    allow_external_target: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
    discovery_max_candidates: Optional[int] = 8,
    max_candidates: Optional[int] = None,
    include_seed: bool = True,
    factory: Optional[bool] = None,
    min_turns: int = 1,
    max_turns: int = 1,
    redteam_attacks: Sequence[str] = ("prompt_injection", "credential_exfiltration"),
    redteam_surfaces: Sequence[str] = ("instruction", "tool"),
    redteam_taxonomies: Sequence[str] = ("owasp_llm_top_10", "owasp_agentic_ai"),
    redteam_channels: Sequence[str] = ("chat",),
    redteam_providers: Sequence[str] = ("local_cli",),
    redteam_agent: Optional[Mapping[str, Any]] = None,
    redteam_config: Optional[Mapping[str, Any]] = None,
    redteam_overrides: Optional[Mapping[str, Any]] = None,
    canaries: Sequence[Any] = (),
    blocked_tools: Sequence[str] = (),
    redteam_min_turns: int = 3,
    redteam_max_turns: int = 3,
) -> dict[str, Any]:
    """Write a runnable framework adapter run+red-team suite workspace."""

    if not name:
        raise ValueError("name is required")
    if not framework:
        raise ValueError("framework is required")
    if not target:
        raise ValueError("target is required")

    workspace = Path(directory).expanduser().resolve()
    manifests_dir = workspace / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    selected_target = _suite_local_target_text(target, base_dir=target_base_dir)
    suite_metadata = copy.deepcopy(dict(metadata or {}))

    from fi.alk import optimize, redteam

    run_manifest = optimize.build_framework_run_manifest_from_local_adapter(
        name=f"{name}-run",
        framework=framework,
        target=selected_target,
        adapter_candidates=adapter_candidates,
        agent=agent,
        agent_factory=agent_factory,
        cases=cases,
        target_base_dir=target_base_dir,
        target_factory=target_factory,
        method_candidates=method_candidates,
        input_mode_candidates=input_mode_candidates,
        required_env=required_env,
        scenario=scenario,
        framework_trace=framework_trace,
        evaluation_config=evaluation_config,
        auto_evaluation_config=auto_evaluation_config,
        threshold=threshold,
        trace_runtime=trace_runtime,
        allow_external_target=allow_external_target,
        metadata={
            "suite": name,
            "suite_role": "optimized_framework_run",
            **suite_metadata,
        },
        discovery_max_candidates=discovery_max_candidates,
        max_candidates=max_candidates,
        include_seed=include_seed,
        factory=factory,
        min_turns=min_turns,
        max_turns=max_turns,
    )
    run_path = _write_suite_json(
        run_manifest,
        manifests_dir / "optimized-framework-run.json",
    )

    agent_config = copy.deepcopy(dict(run_manifest.get("agent") or {}))
    agent_metadata = copy.deepcopy(dict(agent_config.get("metadata") or {}))
    redteam_manifest = redteam.build_redteam_manifest(
        name=f"{name}-redteam",
        attacks=redteam_attacks,
        surfaces=redteam_surfaces,
        taxonomies=redteam_taxonomies,
        channels=redteam_channels,
        providers=redteam_providers,
        frameworks=[framework],
        required_env=required_env,
        target={
            "agent": run_manifest.get("name") or f"{name}-run",
            "framework": framework,
            "adapter_target": selected_target,
            "framework_adapter_contract": copy.deepcopy(
                agent_metadata.get("framework_adapter_probe_contract")
                or agent_metadata.get("framework_adapter_contract")
            ),
            "framework_adapter_probe_proof_status": (
                copy.deepcopy(
                    dict(agent_metadata.get("framework_adapter_probe_proof") or {})
                ).get("status")
            ),
            "framework_adapter_discovery_used": bool(
                agent_metadata.get("framework_adapter_discovery_used")
            ),
            "suite": name,
        },
        agent=redteam_agent,
        redteam=redteam_overrides,
        evaluation_config=redteam_config,
        threshold=threshold,
        canaries=canaries,
        blocked_tools=blocked_tools,
        min_turns=redteam_min_turns,
        max_turns=redteam_max_turns,
    )
    redteam_path = _write_suite_json(
        redteam_manifest,
        manifests_dir / "framework-redteam.json",
    )

    suite_manifest = build_framework_adapter_trinity_suite_manifest(
        name=name,
        run_path=Path("manifests") / run_path.name,
        redteam_path=Path("manifests") / redteam_path.name,
        required_env=required_env,
        required_frameworks=[framework],
        threshold=threshold,
        metadata={
            "source": "fi.alk.suite.write_framework_adapter_trinity_suite_workspace",
            "framework": framework,
            "target": selected_target,
            **suite_metadata,
        },
    )
    suite_path = write_suite_file(suite_manifest, workspace / "suite.json")
    return {
        "kind": "agent-learning.framework-adapter-trinity-workspace.v1",
        "status": "passed",
        "name": str(name),
        "summary": {
            "framework": framework,
            "target": selected_target,
            "suite_job_count": len(suite_manifest["jobs"]),
            "run_manifest": str(run_path),
            "redteam_manifest": str(redteam_path),
            "suite_manifest": str(suite_path),
        },
        "paths": {
            "workspace": str(workspace),
            "suite": str(suite_path),
            "run": str(run_path),
            "redteam": str(redteam_path),
        },
        "suite": suite_manifest,
        "run_manifest": run_manifest,
        "redteam_manifest": redteam_manifest,
    }


def build_framework_adapter_trinity_suite_optimization_manifest(
    *,
    name: str,
    run_path: str | Path,
    trinity_suite_path: str | Path,
    framework: str,
    required_env: Sequence[str] = (),
    required_frameworks: Sequence[str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
    threshold: float = 1.0,
    optimizer: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a suite optimization that selects full framework trinity coverage."""

    if not name:
        raise ValueError("name is required")
    if not framework:
        raise ValueError("framework is required")
    suite_name = str(name)
    frameworks = _unique_strings(required_frameworks) or [str(framework)]
    seed_job = {
        "id": "optimized-framework-run",
        "command": "run",
        "path": _suite_path_text(run_path),
        "name": f"{suite_name}-run-only-seed",
    }
    trinity_job = {
        "id": "framework-adapter-trinity",
        "command": "suite",
        "path": _suite_path_text(trinity_suite_path),
        "name": f"{suite_name}-full-trinity",
    }
    manifest = build_suite_manifest(
        name=suite_name,
        required_env=required_env,
        jobs=[seed_job],
        required_capabilities={
            "commands": ["run", "redteam", "suite"],
            "result_kinds": [
                "agent-learning.run.v1",
                "agent-learning.redteam.v1",
                "agent-learning.suite.v1",
            ],
            "frameworks": frameworks,
            "metrics": [
                "framework_runtime_contract",
                "framework_adapter_contract_quality",
                "adversarial_resilience",
                "red_team_campaign_quality",
            ],
        },
        metadata={
            "source": (
                "fi.alk.suite."
                "build_framework_adapter_trinity_suite_optimization_manifest"
            ),
            "task_kind": "framework_adapter_trinity_suite_optimization",
            "framework": framework,
            **copy.deepcopy(dict(metadata or {})),
        },
    )
    manifest["optimization"] = {
        "threshold": float(threshold),
        "target": {
            "name": suite_name,
            "layers": ["harness", "framework", "security", "evaluator"],
            "base_config": {"jobs": [copy.deepcopy(seed_job)]},
            "search_space": {
                "jobs.0": [
                    copy.deepcopy(seed_job),
                    copy.deepcopy(trinity_job),
                ]
            },
            "metadata": {
                "source": (
                    "fi.alk.suite."
                    "build_framework_adapter_trinity_suite_optimization_manifest"
                ),
                "task_kind": "framework_adapter_trinity_suite_optimization",
                "framework": framework,
                **copy.deepcopy(dict(metadata or {})),
            },
        },
        "optimizer": copy.deepcopy(
            dict(
                optimizer
                or {
                    "algorithm": "agent",
                    "max_candidates": 3,
                    "include_seed": True,
                    "auto_diagnose": False,
                }
            )
        ),
    }
    return manifest


def write_framework_adapter_trinity_suite_optimization_workspace(
    *,
    name: str,
    framework: str,
    target: str,
    directory: str | Path,
    suite_optimization_threshold: float = 1.0,
    suite_optimizer: Optional[Mapping[str, Any]] = None,
    **workspace_kwargs: Any,
) -> dict[str, Any]:
    """Write a framework trinity workspace plus an optimizable outer suite."""

    workspace = write_framework_adapter_trinity_suite_workspace(
        name=name,
        framework=framework,
        target=target,
        directory=directory,
        **workspace_kwargs,
    )
    workspace_root = Path(workspace["paths"]["workspace"]).expanduser().resolve()
    metadata = copy.deepcopy(dict(workspace_kwargs.get("metadata") or {}))
    optimization_manifest = build_framework_adapter_trinity_suite_optimization_manifest(
        name=f"{name}-optimization",
        run_path=Path("manifests") / Path(workspace["paths"]["run"]).name,
        trinity_suite_path=Path("suite.json"),
        framework=framework,
        required_env=workspace_kwargs.get("required_env", ()),
        required_frameworks=[framework],
        metadata={
            "source": (
                "fi.alk.suite."
                "write_framework_adapter_trinity_suite_optimization_workspace"
            ),
            "framework": framework,
            "target": workspace["summary"]["target"],
            **metadata,
        },
        threshold=suite_optimization_threshold,
        optimizer=suite_optimizer,
    )
    optimization_path = write_suite_file(
        optimization_manifest,
        workspace_root / "suite-optimization.json",
    )
    return {
        "kind": "agent-learning.framework-adapter-trinity-optimization-workspace.v1",
        "status": "passed",
        "name": str(name),
        "summary": {
            **copy.deepcopy(dict(workspace.get("summary") or {})),
            "suite_optimization_manifest": str(optimization_path),
            "suite_optimization_search_paths": ["jobs.0"],
        },
        "paths": {
            **copy.deepcopy(dict(workspace.get("paths") or {})),
            "suite_optimization": str(optimization_path),
        },
        "suite_optimization": optimization_manifest,
        "trinity_workspace": workspace,
    }


def build_regression_artifact_suite_manifest(
    *,
    name: str,
    baseline_path: str | Path,
    current_path: str | Path,
    finding_path: str | Path,
    replay_manifest_paths: Sequence[str | Path],
    required_env: Sequence[str] = (),
    min_score_delta: float = 0.0,
    max_new_findings: int = 0,
    max_new_error_findings: int = 0,
    min_level: str = "warning",
    max_findings: int = 1,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the artifact-regression lifecycle suite from SDK paths.

    This composes the lifecycle users usually script around CI artifacts:
    create a compact baseline, compare current vs baseline, render a report,
    promote a red-team finding into a regression manifest, and replay one or
    more regression manifests.
    """

    replay_paths = [_suite_path_text(path) for path in replay_manifest_paths]
    if not replay_paths:
        raise ValueError("replay_manifest_paths must contain at least one manifest")

    suite_name = str(name)
    jobs = [
        {
            "id": "baseline-current-run",
            "command": "baseline",
            "path": _suite_path_text(current_path),
            "name": f"{suite_name}-baseline",
        },
        {
            "id": "compare-baseline-to-current",
            "command": "compare",
            "path": _suite_path_text(current_path),
            "baseline": _suite_path_text(baseline_path),
            "current": _suite_path_text(current_path),
            "name": f"{suite_name}-compare",
            "min_score_delta": float(min_score_delta),
            "max_new_findings": int(max_new_findings),
            "max_new_error_findings": int(max_new_error_findings),
        },
        {
            "id": "report-current-run",
            "command": "report",
            "path": _suite_path_text(current_path),
            "name": f"{suite_name}-report",
        },
        {
            "id": "promote-redteam-finding",
            "command": "promote_to_regression",
            "path": _suite_path_text(finding_path),
            "name": f"{suite_name}-promoted-regression",
            "min_level": str(min_level),
            "max_findings": int(max_findings),
        },
        {
            "id": "replay-regression-manifest",
            "command": "replay",
            "path": replay_paths[0],
            "manifests": replay_paths,
            "name": f"{suite_name}-replay",
        },
    ]
    return build_suite_manifest(
        name=suite_name,
        required_env=required_env,
        jobs=jobs,
        required_capabilities={
            "commands": [
                "baseline",
                "compare",
                "report",
                "promote_to_regression",
                "replay",
            ],
            "result_kinds": [
                "agent_learning.baseline.v1",
                "agent_learning.compare.v1",
                "agent_learning.report.v1",
                "agent_learning.regression_promotion.v1",
                "agent_learning.replay.v1",
            ],
            "metrics": [
                "compare_score_delta",
                "replay_pass_rate",
            ],
        },
        metadata={
            "source": "fi.alk.suite.build_regression_artifact_suite_manifest",
            "task_kind": "regression_artifact_lifecycle",
            **copy.deepcopy(dict(metadata or {})),
        },
    )


def build_optimization_lifecycle_plan(
    *,
    optimize_manifest_path: str | Path,
    workspace_dir: str | Path | None = None,
    name: str = "optimization-lifecycle",
    required_env: Sequence[str] = (),
    frozen_profile_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build an executable optimize -> promote -> replay lifecycle plan.

    When ``frozen_profile_path`` names a frozen capability-profile contract
    (kind ``agent-learning.frozen-capability-profile.v1``, ARCH §2a), the plan
    gains a ``replay_frozen_profile`` step between the promotion and the
    regression replay: every frozen row is re-closed against the optimization
    artifact and an improving-but-row-breaking candidate is vetoed
    (hetvabhasa class ``badhita``) before any replay runs.
    """

    paths = _optimization_lifecycle_paths(
        optimize_manifest_path=optimize_manifest_path,
        workspace_dir=workspace_dir,
    )
    if frozen_profile_path is not None:
        frozen_path = Path(frozen_profile_path).expanduser().resolve()
        paths["frozen_profile"] = frozen_path
        paths["frozen_profile_replay"] = (
            paths["optimization"].parent / "frozen-profile-replay.json"
        )
    required_env_args = _required_env_cli_args(required_env)
    steps = [
        _lifecycle_step(
            "dry_run_optimization",
            "Dry Run Optimization",
            ["agent-learn", "optimize", paths["optimize_manifest"], "--dry-run"],
        ),
        _lifecycle_step(
            "optimize",
            "Run Optimization",
            [
                "agent-learn",
                "optimize",
                paths["optimize_manifest"],
                "--output",
                paths["optimization"],
                "--junit",
                paths["optimization_junit"],
                "--sarif",
                paths["optimization_sarif"],
                "--markdown",
                paths["optimization_markdown"],
            ],
            outputs={
                "json": paths["optimization"],
                "junit": paths["optimization_junit"],
                "sarif": paths["optimization_sarif"],
                "markdown": paths["optimization_markdown"],
            },
        ),
        _lifecycle_step(
            "report_optimization",
            "Report Optimization",
            [
                "agent-learn",
                "report",
                paths["optimization"],
                "--output",
                paths["optimization_report"],
                "--markdown",
                paths["optimization_report_markdown"],
            ],
            outputs={
                "json": paths["optimization_report"],
                "markdown": paths["optimization_report_markdown"],
            },
        ),
        _lifecycle_step(
            "promote_to_regression",
            "Promote To Regression",
            [
                "agent-learn",
                "promote-to-regression",
                paths["optimization"],
                "--output",
                paths["promotion"],
                "--manifest",
                paths["regression_manifest"],
                "--min-level",
                "note",
                "--max-findings",
                "1",
                *required_env_args,
            ],
            outputs={
                "json": paths["promotion"],
                "manifest": paths["regression_manifest"],
            },
        ),
        _lifecycle_step(
            "report_promotion",
            "Report Promotion",
            [
                "agent-learn",
                "report",
                paths["promotion"],
                "--output",
                paths["promotion_report"],
                "--markdown",
                paths["promotion_report_markdown"],
            ],
            outputs={
                "json": paths["promotion_report"],
                "markdown": paths["promotion_report_markdown"],
            },
        ),
        *(
            [
                _lifecycle_step(
                    "replay_frozen_profile",
                    "Replay Frozen Capability Profile",
                    [
                        sys.executable,
                        "-c",
                        (
                            "import json, pathlib; "
                            "from fi.alk import optimize; "
                            "result = json.loads(pathlib.Path("
                            f"{str(paths['optimization'])!r}"
                            ").read_text(encoding='utf-8')); "
                            "frozen = json.loads(pathlib.Path("
                            f"{str(paths['frozen_profile'])!r}"
                            ").read_text(encoding='utf-8')); "
                            "verdict = optimize.replay_frozen_profile(result, frozen); "
                            "pathlib.Path("
                            f"{str(paths['frozen_profile_replay'])!r}"
                            ").write_text(json.dumps(verdict, indent=2, "
                            "sort_keys=True, default=str), encoding='utf-8'); "
                            "raise SystemExit(1 if verdict.get('veto') else 0)"
                        ),
                    ],
                    outputs={"json": paths["frozen_profile_replay"]},
                )
            ]
            if frozen_profile_path is not None
            else []
        ),
        _lifecycle_step(
            "replay_regression",
            "Replay Regression",
            [
                "agent-learn",
                "replay",
                paths["regression_manifest"],
                "--output",
                paths["replay"],
                "--junit",
                paths["replay_junit"],
                "--sarif",
                paths["replay_sarif"],
                "--markdown",
                paths["replay_markdown"],
            ],
            outputs={
                "json": paths["replay"],
                "junit": paths["replay_junit"],
                "sarif": paths["replay_sarif"],
                "markdown": paths["replay_markdown"],
            },
        ),
        _lifecycle_step(
            "report_replay",
            "Report Replay",
            [
                "agent-learn",
                "report",
                paths["replay"],
                "--output",
                paths["replay_report"],
                "--markdown",
                paths["replay_report_markdown"],
            ],
            outputs={
                "json": paths["replay_report"],
                "markdown": paths["replay_report_markdown"],
            },
        ),
    ]
    return {
        "kind": AGENT_LEARNING_OPTIMIZATION_LIFECYCLE_KIND,
        "name": str(name),
        "required_env": _unique_strings(required_env),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "steps": steps,
        "metadata": {
            "source": "fi.alk.suite.build_optimization_lifecycle_plan",
            "research_synthesis": (
                "Deterministic optimization transactions: diagnose/search, "
                "export, promote, replay, and expose action cards over one "
                "shared evidence trail."
            ),
        },
    }


def run_optimization_lifecycle_file(
    optimize_manifest_path: str | Path,
    *,
    workspace_dir: str | Path | None = None,
    name: str = "optimization-lifecycle",
    required_env: Sequence[str] = (),
) -> dict[str, Any]:
    """Run optimize, report, promote, replay, and report replay via SDK."""

    from fi.alk import optimize, simulate

    plan = build_optimization_lifecycle_plan(
        optimize_manifest_path=optimize_manifest_path,
        workspace_dir=workspace_dir,
        name=name,
        required_env=required_env,
    )
    paths = {key: Path(value) for key, value in plan["artifacts"].items()}
    outputs_written: list[str] = []

    optimization = optimize.optimize_manifest_file(paths["optimize_manifest"])
    outputs_written.extend(
        _write_lifecycle_result_bundle(
            optimization,
            json_path=paths["optimization"],
            junit_path=paths["optimization_junit"],
            sarif_path=paths["optimization_sarif"],
            markdown_path=paths["optimization_markdown"],
            source_path=paths["optimize_manifest"],
        )
    )

    optimization_report = simulate.render_report(
        optimization,
        source_path=paths["optimization"],
    )
    outputs_written.extend(
        _write_lifecycle_report_bundle(
            optimization_report,
            json_path=paths["optimization_report"],
            markdown_path=paths["optimization_report_markdown"],
            source_path=paths["optimization"],
        )
    )

    promotion = simulate.promote_to_regression(
        optimization,
        source_path=paths["optimization"],
        min_level="note",
        max_findings=1,
        required_env=required_env,
    )
    outputs_written.append(_write_json(paths["promotion"], promotion))
    manifest = promotion.get("manifest")
    if isinstance(manifest, Mapping):
        outputs_written.append(_write_json(paths["regression_manifest"], manifest))

    promotion_report = simulate.render_report(
        promotion,
        source_path=paths["promotion"],
    )
    outputs_written.extend(
        _write_lifecycle_report_bundle(
            promotion_report,
            json_path=paths["promotion_report"],
            markdown_path=paths["promotion_report_markdown"],
            source_path=paths["promotion"],
        )
    )

    replay = simulate.replay_manifests([paths["regression_manifest"]])
    outputs_written.extend(
        _write_lifecycle_result_bundle(
            replay,
            json_path=paths["replay"],
            junit_path=paths["replay_junit"],
            sarif_path=paths["replay_sarif"],
            markdown_path=paths["replay_markdown"],
            source_path=paths["regression_manifest"],
        )
    )

    replay_report = simulate.render_report(replay, source_path=paths["replay"])
    outputs_written.extend(
        _write_lifecycle_report_bundle(
            replay_report,
            json_path=paths["replay_report"],
            markdown_path=paths["replay_report_markdown"],
            source_path=paths["replay"],
        )
    )

    passed = all(
        payload.get("status") == "passed"
        for payload in (optimization, promotion, replay)
    )
    return {
        "kind": AGENT_LEARNING_OPTIMIZATION_LIFECYCLE_KIND,
        "name": str(name),
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 1,
        "summary": {
            "optimization_score": dict(optimization.get("summary") or {}).get(
                "optimization_score"
            ),
            "promotion_kind": dict(promotion.get("summary") or {}).get(
                "promotion_kind"
            ),
            "promoted_manifest_count": dict(promotion.get("summary") or {}).get(
                "promoted_manifest_count"
            ),
            "replay_pass_rate": dict(replay.get("summary") or {}).get(
                "replay_pass_rate"
            ),
            "step_count": len(plan["steps"]),
            "outputs_written_count": len(outputs_written),
        },
        "plan": plan,
        "artifacts": {
            "optimization": optimization,
            "optimization_report": optimization_report,
            "promotion": promotion,
            "promotion_report": promotion_report,
            "replay": replay,
            "replay_report": replay_report,
        },
        "outputs_written": outputs_written,
    }


def write_suite_file(manifest: Mapping[str, Any], path: str | Path) -> Path:
    """Write a suite manifest as formatted JSON and return the resolved path."""

    suite_path = Path(path).expanduser().resolve()
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return suite_path


def _write_suite_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return output_path


def run_suite_file(
    path: str | Path,
    *,
    options: Optional[SuiteRunOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
    fail_fast: Optional[bool] = None,
    require_optimizer_governance: Optional[bool] = None,
) -> dict[str, Any]:
    suite_path = Path(path).expanduser().resolve()
    suite = load_suite_file(suite_path)
    return run_suite(
        suite,
        suite_path=suite_path,
        options=_merge_options(
            options,
            name=name,
            threshold=threshold,
            max_candidates=max_candidates,
            dry_run=dry_run,
            fail_fast=fail_fast,
            require_optimizer_governance=require_optimizer_governance,
        ),
    )


def run_suite(
    suite: Mapping[str, Any],
    *,
    suite_path: str | Path = ".",
    options: Optional[SuiteRunOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
    fail_fast: Optional[bool] = None,
    require_optimizer_governance: Optional[bool] = None,
) -> dict[str, Any]:
    started = time.time()
    opts = _merge_options(
        options,
        name=name,
        threshold=threshold,
        max_candidates=max_candidates,
        dry_run=dry_run,
        fail_fast=fail_fast,
        require_optimizer_governance=require_optimizer_governance,
    )
    suite_path = Path(suite_path).expanduser().resolve()
    base_dir = _suite_base_dir(suite_path)
    runtime_suite = _prepare_suite(copy.deepcopy(dict(suite)), base_dir=base_dir)
    if opts.require_optimizer_governance:
        optimizer_policy = _suite_optimizer_governance_policy(runtime_suite)
        optimizer_policy["require_optimizer_governance"] = True
        optimizer_policy["require_passed"] = True
        optimizer_policy["min_governed"] = max(
            int(optimizer_policy.get("min_governed") or 0),
            1,
        )
        runtime_suite["optimizer_governance_policy"] = {
            **optimizer_policy,
        }
    validate_suite_env(runtime_suite, suite_path=suite_path)

    children: list[dict[str, Any]] = []
    for index, job in enumerate(_suite_jobs(runtime_suite), start=1):
        child = _execute_job(
            job,
            index=index,
            base_dir=base_dir,
            suite_options=opts,
        )
        children.append(child)
        if int(child.get("exit_code", 1)) != 0 and opts.fail_fast:
            break

    payload = _suite_result(
        suite=runtime_suite,
        suite_path=suite_path,
        children=children,
        name=opts.name,
        dry_run=opts.dry_run,
        fail_fast=opts.fail_fast,
        duration_seconds=round(time.time() - started, 4),
    )
    return public_payload(payload, kind=AGENT_LEARNING_SUITE_KIND)


def optimize_suite_file(
    path: str | Path,
    *,
    options: Optional[SuiteOptimizationOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
) -> dict[str, Any]:
    """Load and optimize a full Agent Learning suite."""

    suite_path = Path(path).expanduser().resolve()
    suite = load_suite_file(suite_path)
    return optimize_suite(
        suite,
        suite_path=suite_path,
        options=_merge_optimization_options(
            options,
            name=name,
            threshold=threshold,
            max_candidates=max_candidates,
            dry_run=dry_run,
        ),
    )


def optimize_suite(
    suite: Mapping[str, Any],
    *,
    suite_path: str | Path = ".",
    options: Optional[SuiteOptimizationOptions] = None,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
) -> dict[str, Any]:
    """Optimize a mixed Agent Learning suite and return a unified artifact."""

    started = time.time()
    opts = _merge_optimization_options(
        options,
        name=name,
        threshold=threshold,
        max_candidates=max_candidates,
        dry_run=dry_run,
    )
    suite_path = Path(suite_path).expanduser().resolve()
    base_dir = _suite_base_dir(suite_path)
    runtime_suite = copy.deepcopy(dict(suite))
    if opts.name:
        runtime_suite["name"] = opts.name
    if opts.threshold is not None:
        runtime_suite.setdefault("optimization", {})["threshold"] = opts.threshold
    if opts.max_candidates is not None:
        runtime_suite.setdefault("optimization", {}).setdefault(
            "optimizer", {}
        )["max_candidates"] = opts.max_candidates

    prepared = _prepare_suite(runtime_suite, base_dir=base_dir)
    validate_suite_env(prepared, suite_path=suite_path)
    cli = _optimization_cli()
    optimization = cli._optimization_config(prepared)
    target_config = cli._target_config(optimization)
    optimizer_config = cli._optimizer_config(optimization)
    if opts.dry_run:
        return public_payload({
            "schema_version": AGENT_LEARNING_CLI_SCHEMA_VERSION,
            "kind": AGENT_LEARNING_SUITE_OPTIMIZATION_KIND,
            "name": str(prepared.get("name") or suite_path.stem),
            "status": "passed",
            "exit_code": 0,
            "dry_run": True,
            "summary": {
                "job_count": len(_suite_jobs(prepared)),
                "required_env": required_suite_env(prepared, suite_path=suite_path),
                "search_path_count": len(target_config.get("search_space", {})),
                "max_candidates": optimizer_config.get("max_candidates"),
            },
            "duration_seconds": round(time.time() - started, 4),
        }, kind=AGENT_LEARNING_SUITE_OPTIMIZATION_KIND)

    try:
        from fi.alk import optimize as agent_optimize
    except Exception as exc:  # pragma: no cover - optional dependency clarity
        raise SuiteError(
            "Agent Learning Kit optimizer engine is required for suite optimization."
        ) from exc

    problem = agent_optimize.problem_from_agent_learning_suite(
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
    payload["kind"] = AGENT_LEARNING_SUITE_OPTIMIZATION_KIND
    payload["suite"] = _suite_descriptor(prepared)
    payload["optimization"]["source"] = "agent_learning_suite"
    if "manifest_optimization" in payload["optimization"]:
        artifact = copy.deepcopy(payload["optimization"]["manifest_optimization"])
        artifact["kind"] = "agent_learning_suite_optimization"
        artifact["source"] = "agent_learning_suite"
        payload["optimization"]["suite_optimization"] = artifact
    payload["summary"]["job_count"] = len(_suite_jobs(prepared))
    payload["summary"]["child_command_count"] = _suite_job_command_counts(prepared)
    action_plan = _artifact_action_plan_card(payload)
    if action_plan is not None:
        payload["artifact_action_plan"] = action_plan
        payload["optimization"]["artifact_action_plan"] = copy.deepcopy(action_plan)
        payload["summary"]["artifact_action_best_action_id"] = action_plan.get(
            "selected_action_id"
        )
    return public_payload(payload, kind=AGENT_LEARNING_SUITE_OPTIMIZATION_KIND)


def render_junit(result: Mapping[str, Any]) -> str:
    name = escape(str(result.get("name") or "agent-learning-suite"))
    children = list(result.get("children") or result.get("jobs") or [])
    finding_failures = [
        finding
        for finding in list(result.get("findings") or [])
        if str(_as_mapping(finding).get("type"))
        in {
            "suite_required_capability_missing",
            "suite_evidence_admission_missing",
            "suite_evidence_freeze_missing",
            "suite_framework_adapter_conformance_failed",
            "suite_framework_coverage_missing",
            "suite_optimizer_governance_failed",
            "suite_optimizer_governance_missing",
            "suite_optimizer_governance_warning",
        }
    ]
    failures = (
        sum(1 for child in children if int(child.get("exit_code", 1)) != 0)
        + len(finding_failures)
    )
    lines = [
        (
            f'<testsuite name="{name}" tests="{len(children) + len(finding_failures)}" '
            f'failures="{failures}" errors="0">'
        )
    ]
    for child in children:
        child_name = escape(str(child.get("id") or child.get("name") or "job"))
        class_name = escape(str(child.get("command") or "suite"))
        duration = float(child.get("duration_seconds") or 0.0)
        lines.append(
            f'  <testcase classname="{class_name}" name="{child_name}" '
            f'time="{duration:.4f}">'
        )
        if int(child.get("exit_code", 1)) != 0:
            message = escape(str(child.get("error") or child.get("status") or "failed"))
            lines.append(f'    <failure message="{message}">{message}</failure>')
        lines.append("  </testcase>")
    for index, finding in enumerate(finding_failures, start=1):
        item = _as_mapping(finding)
        finding_name = escape(str(item.get("type") or f"suite_finding_{index}"))
        message = escape(str(item.get("reason") or finding_name))
        lines.append(f'  <testcase classname="suite" name="{finding_name}" time="0.0000">')
        lines.append(f'    <failure message="{message}">{message}</failure>')
        lines.append("  </testcase>")
    lines.append("</testsuite>")
    return "\n".join(lines)


def render_sarif(
    result: Mapping[str, Any],
    *,
    manifest_path: str | Path = ".",
) -> str:
    suite_path = Path(manifest_path).expanduser().resolve()
    findings = _suite_sarif_findings(result)
    sarif_results = []
    for finding in findings:
        rule_id = str(finding.get("type") or finding.get("rule_id") or "suite_finding")
        level = str(finding.get("level") or finding.get("severity") or "error").lower()
        if level not in {"none", "note", "warning", "error"}:
            level = "warning"
        location_path = str(finding.get("path") or suite_path)
        sarif_results.append(
            {
                "ruleId": rule_id,
                "level": level,
                "message": {"text": str(finding.get("reason") or rule_id)},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": location_path},
                        }
                    }
                ],
            }
        )
    payload = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "agent-learning-suite",
                        "informationUri": "https://futureagi.com",
                        "rules": [],
                    }
                },
                "results": sarif_results,
            }
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def render_markdown(
    result: Mapping[str, Any],
    *,
    source_path: str | Path = ".",
) -> str:
    summary = dict(result.get("summary") or {})
    certificate = _as_mapping(result.get("trust_certificate"))
    lines = [
        f"# {result.get('name') or 'agent-learning-suite'}",
        "",
        f"- Source: `{Path(source_path)}`",
        f"- Status: `{result.get('status')}`",
        f"- Jobs: {summary.get('passed_count', 0)}/{summary.get('job_count', 0)} passed",
        f"- Score: {summary.get('score', 0.0)}",
        (
            "- Trust Certificate: "
            f"{certificate.get('verdict') or summary.get('trust_certificate_verdict')}"
            f" ({certificate.get('assurance_level') or summary.get('trust_certificate_assurance_level')})"
        ),
        (
            "- Evidence: "
            f"{summary.get('admitted_evidence_count', 0)} admitted, "
            f"{summary.get('non_admitted_evidence_count', 0)} non-admitted, "
            f"{summary.get('rejected_evidence_count', 0)} rejected, "
            f"{summary.get('frozen_evidence_count', 0)} frozen"
        ),
        (
            "- Frameworks: "
            f"{summary.get('observed_framework_count', 0)} observed, "
            f"{summary.get('missing_framework_count', 0)} missing, "
            f"{summary.get('adapter_conformance_failed_count', 0)} adapter-failed"
        ),
        "",
        "## Trust Certificate",
        "",
        f"- Verdict: `{certificate.get('verdict')}`",
        f"- Assurance Level: `{certificate.get('assurance_level')}`",
        f"- Promotion Ready: `{certificate.get('promotion_ready')}`",
        f"- Reason: {certificate.get('reason') or ''}",
        "",
        "| Gate | Status | Required |",
        "| --- | --- | --- |",
    ]
    for gate in _as_list(certificate.get("gates")):
        gate_item = _as_mapping(gate)
        if not gate_item:
            continue
        lines.append(
            "| "
            f"{_md_cell(gate_item.get('id') or '')} | "
            f"{_md_cell(gate_item.get('status') or '')} | "
            f"{_md_cell(str(bool(gate_item.get('required'))))} |"
        )
    lines.extend([
        "",
        "| Job | Command | Status | Evidence | Exit |",
        "| --- | --- | --- | --- | --- |",
    ])
    for child in list(result.get("children") or result.get("jobs") or []):
        evidence = _as_mapping(child.get("evidence"))
        evidence_cell = evidence.get("status") or ""
        if evidence.get("role") and evidence.get("role") != evidence_cell:
            evidence_cell = f"{evidence_cell} ({evidence.get('role')})"
        lines.append(
            "| "
            f"{_md_cell(child.get('id') or child.get('name') or '')} | "
            f"{_md_cell(child.get('command') or '')} | "
            f"{_md_cell(child.get('status') or '')} | "
            f"{_md_cell(evidence_cell)} | "
            f"{int(child.get('exit_code', 1))} |"
        )
    return "\n".join(lines) + "\n"


def _prepare_suite(suite: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    jobs = _as_list(suite.get("jobs") or suite.get("runs") or suite.get("steps"))
    if not jobs:
        raise SuiteError("suite manifest requires at least one job")
    prepared_jobs = []
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, Mapping):
            raise SuiteError(f"suite job[{index}] must be an object")
        prepared = dict(job)
        prepared["command"] = _normalize_command(
            prepared.get("command") or prepared.get("type") or prepared.get("kind")
        )
        prepared.setdefault("id", f"{prepared['command']}-{index}")
        _job_path(prepared, base_dir=base_dir)
        prepared_jobs.append(prepared)
    suite["jobs"] = prepared_jobs
    suite.setdefault("version", AGENT_LEARNING_SUITE_KIND)
    suite.setdefault("name", "agent-learning-suite")
    return suite


def _execute_job(
    job: Mapping[str, Any],
    *,
    index: int,
    base_dir: Path,
    suite_options: SuiteRunOptions,
) -> dict[str, Any]:
    started = time.time()
    command = _normalize_command(job.get("command") or job.get("type"))
    path = _job_path(job, base_dir=base_dir)
    job_id = str(job.get("id") or f"{command}-{index}")
    try:
        payload = _execute_child_payload(
            command,
            path=path,
            base_dir=base_dir,
            job=job,
            suite_options=suite_options,
        )
        payload = copy.deepcopy(dict(payload))
        outputs_written = _write_child_outputs(
            payload,
            command=command,
            job=job,
            path=path,
        )
        payload["outputs_written"] = outputs_written
        result = {
            "id": job_id,
            "command": command,
            "path": str(path),
            "kind": payload.get("kind"),
            "name": payload.get("name"),
            "status": str(payload.get("status") or "unknown"),
            "exit_code": int(payload.get("exit_code", 1)),
            "summary": copy.deepcopy(dict(payload.get("summary") or {})),
            "findings": copy.deepcopy(list(payload.get("findings") or [])),
            "outputs_written": outputs_written,
            "duration_seconds": round(time.time() - started, 4),
            "result": payload,
        }
        result["evidence"] = _suite_child_evidence(
            job,
            result,
            base_dir=base_dir,
        )
        return result
    except Exception as exc:
        result = {
            "id": job_id,
            "command": command,
            "path": str(path),
            "kind": None,
            "name": job.get("name"),
            "status": "failed",
            "exit_code": 1,
            "summary": {},
            "findings": [
                {
                    "type": "suite_child_failed",
                    "level": "error",
                    "reason": str(exc),
                    "job": job_id,
                    "command": command,
                    "path": str(path),
                }
            ],
            "outputs_written": [],
            "duration_seconds": round(time.time() - started, 4),
            "error": str(exc),
        }
        result["evidence"] = _suite_child_evidence(
            job,
            result,
            base_dir=base_dir,
        )
        return result


def _execute_child_payload(
    command: str,
    *,
    path: Path,
    base_dir: Path,
    job: Mapping[str, Any],
    suite_options: SuiteRunOptions,
) -> dict[str, Any]:
    if command == "run":
        from fi.alk import simulate
        from fi.alk.cli import AGENT_LEARNING_RUN_KIND

        payload = _run_async(
            simulate.run_manifest_file(
                path,
                name=_job_name(job),
                threshold=_job_threshold(job, suite_options),
                no_eval=bool(job.get("no_eval", job.get("no-eval", False))),
                dry_run=_job_dry_run(job, suite_options),
            )
        )
        payload["kind"] = AGENT_LEARNING_RUN_KIND
        return payload
    if command == "suite":
        payload = run_suite_file(
            path,
            options=SuiteRunOptions(
                name=_job_name(job),
                threshold=_job_threshold(job, suite_options),
                max_candidates=_job_max_candidates(job, suite_options),
                dry_run=_job_dry_run(job, suite_options),
                fail_fast=bool(
                    suite_options.fail_fast
                    or job.get("fail_fast")
                    or job.get("fail-fast")
                ),
                require_optimizer_governance=suite_options.require_optimizer_governance,
            ),
        )
        payload["kind"] = AGENT_LEARNING_SUITE_KIND
        return payload
    if command == "action_run":
        from fi.alk import actions

        artifact = actions.load_artifact_file(path)
        return actions.run_action(
            artifact,
            _job_action_id(job),
            source_path=path,
            inputs=_job_action_inputs(job),
            cwd=_job_action_cwd(job, base_dir=base_dir),
            dry_run=_job_dry_run(job, suite_options),
            name=_job_name(job),
            artifact_output_path=_job_action_artifact_output(job),
        )
    if command == "eval":
        from fi.alk import evals
        from fi.alk.cli import AGENT_LEARNING_EVAL_KIND

        payload = evals.run_eval_suite_file(
            path,
            name=_job_name(job),
            threshold=_job_threshold(job, suite_options),
            dry_run=_job_dry_run(job, suite_options),
        )
        payload["kind"] = AGENT_LEARNING_EVAL_KIND
        return payload
    if command == "eval_artifact":
        from fi.alk import evals
        from fi.alk.cli import AGENT_LEARNING_ARTIFACT_EVAL_KIND

        config_path = _job_optional_path(
            job,
            base_dir=base_dir,
            keys=("config", "eval_config", "agent_report_config"),
        )
        config = evals.load_artifact_file(config_path) if config_path else None
        payload = evals.evaluate_artifact_file(
            path,
            config=config,
            name=_job_name(job),
            threshold=float(_job_threshold(job, suite_options) or 0.7),
        )
        payload["kind"] = AGENT_LEARNING_ARTIFACT_EVAL_KIND
        return payload
    if command == "eval_task":
        from fi.alk import evals
        from fi.alk.cli import AGENT_LEARNING_ARTIFACT_EVAL_KIND

        config_path = _job_optional_path(
            job,
            base_dir=base_dir,
            keys=("config", "eval_config", "agent_report_config"),
        )
        config = evals.load_artifact_file(config_path) if config_path else None
        payload = evals.evaluate_task_evidence_file(
            path,
            config=config,
            name=_job_name(job),
            threshold=float(_job_threshold(job, suite_options) or 0.7),
        )
        payload["kind"] = AGENT_LEARNING_ARTIFACT_EVAL_KIND
        return payload
    if command == "redteam":
        from fi.alk import redteam

        payload = _run_async(
            redteam.redteam_manifest_file(
                path,
                name=_job_name(job),
                threshold=_job_threshold(job, suite_options),
                dry_run=_job_dry_run(job, suite_options),
            )
        )
        return payload
    if command == "optimize":
        from fi.alk import optimize
        from fi.alk.cli import AGENT_LEARNING_OPTIMIZATION_KIND

        payload = optimize.optimize_manifest_file(
            path,
            name=_job_name(job),
            threshold=_job_threshold(job, suite_options),
            max_candidates=_job_max_candidates(job, suite_options),
            dry_run=_job_dry_run(job, suite_options),
        )
        payload["kind"] = AGENT_LEARNING_OPTIMIZATION_KIND
        return payload
    if command == "optimize_eval":
        from fi.alk import optimize
        from fi.alk.cli import AGENT_LEARNING_EVAL_OPTIMIZATION_KIND

        payload = optimize.optimize_eval_suite_file(
            path,
            name=_job_name(job),
            threshold=_job_threshold(job, suite_options),
            max_candidates=_job_max_candidates(job, suite_options),
            dry_run=_job_dry_run(job, suite_options),
        )
        payload["kind"] = AGENT_LEARNING_EVAL_OPTIMIZATION_KIND
        return payload
    if command == "optimize_suite":
        from fi.alk import optimize
        from fi.alk.cli import AGENT_LEARNING_SUITE_OPTIMIZATION_KIND

        payload = optimize.optimize_suite_file(
            path,
            name=_job_name(job),
            threshold=_job_threshold(job, suite_options),
            max_candidates=_job_max_candidates(job, suite_options),
            dry_run=_job_dry_run(job, suite_options),
        )
        payload["kind"] = AGENT_LEARNING_SUITE_OPTIMIZATION_KIND
        return payload
    if command == "baseline":
        from fi.alk import simulate

        return simulate.create_baseline_file(
            path,
            name=_job_name(job),
        )
    if command == "compare":
        from fi.alk import simulate

        return simulate.compare_result_files(
            _job_compare_baseline_path(job, base_dir=base_dir),
            path,
            min_score_delta=_job_float(job, "min_score_delta", "min-score-delta", default=0.0),
            max_new_findings=_job_int(job, "max_new_findings", "max-new-findings", default=0),
            max_new_error_findings=_job_int(
                job,
                "max_new_error_findings",
                "max-new-error-findings",
                default=0,
            ),
            min_metric_delta=_job_optional_float(
                job,
                "min_metric_delta",
                "min-metric-delta",
            ),
            name=_job_name(job),
        )
    if command == "report":
        from fi.alk import simulate

        return simulate.render_report_file(
            path,
            name=_job_name(job),
        )
    if command == "promote_to_regression":
        from fi.alk import simulate

        return simulate.promote_to_regression_file(
            path,
            name=_job_name(job),
            min_level=str(job.get("min_level") or job.get("min-level") or "warning"),
            max_findings=_job_int(job, "max_findings", "max-findings", default=25),
            required_env=_as_string_list(job.get("required_env")),
        )
    if command == "shrink":
        from fi.alk import simulate

        return simulate.shrink_attack_evolution_file(
            path,
            name=_job_name(job),
            manifest_name=str(
                job.get("manifest_name")
                or job.get("manifest-name")
                or ""
            )
            or None,
            required_env=_as_string_list(job.get("required_env")),
        )
    if command == "replay":
        from fi.alk import simulate

        return simulate.replay_manifests(
            _job_replay_manifest_paths(job, base_dir=base_dir),
            name=_job_name(job),
            dry_run=_job_dry_run(job, suite_options),
            fail_fast=bool(suite_options.fail_fast or job.get("fail_fast") or job.get("fail-fast")),
        )
    raise SuiteError(f"unsupported suite job command: {command}")


def _write_child_outputs(
    payload: Mapping[str, Any],
    *,
    command: str,
    job: Mapping[str, Any],
    path: Path,
) -> list[str]:
    output_paths = _job_output_paths(job, path.parent)
    if not any(output_paths.values()):
        return []
    render_junit_fn, render_sarif_fn, render_markdown_fn = _child_renderers(command)
    written: list[str] = []
    for output_path in output_paths["json"]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        written.append(str(output_path))
    for output_path in output_paths["junit"]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_junit_fn(payload), encoding="utf-8")
        written.append(str(output_path))
    for output_path in output_paths["sarif"]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            render_sarif_fn(payload, manifest_path=path),
            encoding="utf-8",
        )
        written.append(str(output_path))
    for output_path in output_paths["markdown"]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            render_markdown_fn(payload, source_path=path),
            encoding="utf-8",
        )
        written.append(str(output_path))
    return written


def _child_renderers(command: str) -> tuple[Any, Any, Any]:
    if command == "suite":
        return render_junit, render_sarif, render_markdown
    if command == "action_run":
        from fi.alk import actions, simulate

        def render_action_run_markdown(
            payload: Mapping[str, Any],
            *,
            source_path: Path,
        ) -> str:
            return actions.render_action_run_markdown(payload)

        return simulate.render_junit, simulate.render_sarif, render_action_run_markdown
    if command == "redteam":
        from fi.alk import redteam

        return redteam.render_junit, redteam.render_sarif, redteam.render_markdown
    from fi.alk import simulate

    return simulate.render_junit, simulate.render_sarif, simulate.render_markdown


def _suite_child_evidence(
    job: Mapping[str, Any],
    child: Mapping[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    role = _suite_evidence_role(job, child)
    exit_code = int(child.get("exit_code", 1))
    manifest_path = Path(str(child.get("path") or ""))
    replay_class = str(
        job.get("replay_class")
        or job.get("replay")
        or _as_mapping(job.get("metadata")).get("replay_class")
        or "r0"
    )
    output_digests = _suite_output_digests(
        child.get("outputs_written"),
        base_dir=base_dir,
    )
    freeze = {
        "kind": "agent-learning.suite.evidence-freeze.v1",
        "hash_algorithm": "sha256",
        "replay_class": replay_class,
        "manifest": _suite_file_digest(manifest_path),
        "result_sha256": _suite_json_digest(child.get("result")),
        "outputs": output_digests,
        "outputs_sha256": _suite_json_digest(output_digests),
    }
    freeze["content_addressed"] = bool(
        _as_mapping(freeze.get("manifest")).get("sha256")
        and freeze.get("result_sha256")
    )
    reasons: list[str] = []
    if exit_code != 0:
        status = "rejected"
        admitted = False
        reasons.append("child_failed")
    elif role in _ADMITTED_EVIDENCE_ROLES:
        status = "admitted"
        admitted = True
    else:
        status = role if role in _NON_ADMITTED_EVIDENCE_ROLES else "diagnostic"
        admitted = False
        reasons.append(f"evidence_role_{status}")
    if _suite_path_is_fixture(child.get("path")) and status != "rejected":
        role = "fixture"
        status = "fixture"
        admitted = False
        if "fixture_path" not in reasons:
            reasons.append("fixture_path")
    metadata = _as_mapping(job.get("metadata"))
    claim_scope = (
        job.get("claim_scope")
        or job.get("claim")
        or metadata.get("claim_scope")
        or ("paper_facing" if admitted else "audit")
    )
    return {
        "kind": "agent-learning.suite.evidence-row.v1",
        "row_id": str(child.get("id") or job.get("id") or ""),
        "status": status,
        "role": role,
        "admitted": admitted,
        "reason": reasons,
        "claim_scope": str(claim_scope),
        "workload": str(job.get("workload") or job.get("id") or child.get("id") or ""),
        "driver": str(job.get("driver") or child.get("command") or ""),
        "command": child.get("command"),
        "path": child.get("path"),
        "result_kind": child.get("kind"),
        "exit_code": exit_code,
        "provenance": {
            "job_id": child.get("id") or job.get("id"),
            "job_name": child.get("name") or job.get("name"),
            "manifest_path": child.get("path"),
            "manifest_sha256": _as_mapping(freeze["manifest"]).get("sha256"),
            "result_sha256": freeze.get("result_sha256"),
            "outputs_written": list(child.get("outputs_written") or []),
            "output_digests": output_digests,
            "outputs_sha256": freeze.get("outputs_sha256"),
            "replay_class": replay_class,
            "content_addressed": freeze["content_addressed"],
        },
        "freeze": freeze,
    }


def _suite_evidence_role(
    job: Mapping[str, Any],
    child: Mapping[str, Any],
) -> str:
    metadata = _as_mapping(job.get("metadata"))
    raw = (
        job.get("evidence_role")
        or job.get("evidence_status")
        or job.get("evidence")
        or metadata.get("evidence_role")
        or metadata.get("evidence_status")
    )
    role = _suite_key(raw) if raw is not None else ""
    if role in _ADMITTED_EVIDENCE_ROLES or role in _NON_ADMITTED_EVIDENCE_ROLES:
        return role
    if _suite_path_is_fixture(child.get("path") or job.get("path")):
        return "fixture"
    return "admitted"


def _suite_path_is_fixture(value: Any) -> bool:
    text = str(value or "").replace("\\", "/").lower()
    return "/fixtures/" in text or text.startswith("fixtures/")


def _suite_file_digest(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser()
    exists = file_path.exists()
    if not exists or not file_path.is_file():
        return {
            "path": str(file_path),
            "exists": exists,
            "sha256": None,
            "bytes": 0,
        }
    data = file_path.read_bytes()
    return {
        "path": str(file_path),
        "exists": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _suite_json_digest(value: Any) -> str:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _suite_output_digests(
    values: Any,
    *,
    base_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value in _as_list(values):
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        records.append(_suite_file_digest(path))
    return records


def _suite_evidence_admission(
    children: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [copy.deepcopy(dict(_as_mapping(child.get("evidence")))) for child in children]
    rows = [row for row in rows if row]
    by_status: dict[str, int] = {}
    by_role: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        role = str(row.get("role") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        by_role[role] = by_role.get(role, 0) + 1
    admitted_rows = [row for row in rows if bool(row.get("admitted"))]
    rejected_rows = [row for row in rows if str(row.get("status") or "") == "rejected"]
    non_admitted_rows = [row for row in rows if not bool(row.get("admitted"))]
    frozen_rows = [row for row in rows if _suite_row_content_addressed(row)]
    admitted_unfrozen_rows = [
        row for row in admitted_rows if not _suite_row_content_addressed(row)
    ]
    return {
        "kind": "agent-learning.suite.evidence-admission.v1",
        "admitted_count": len(admitted_rows),
        "non_admitted_count": len(non_admitted_rows),
        "rejected_count": len(rejected_rows),
        "frozen_count": len(frozen_rows),
        "unfrozen_count": len(rows) - len(frozen_rows),
        "admitted_frozen_count": len(admitted_rows) - len(admitted_unfrozen_rows),
        "by_status": dict(sorted(by_status.items())),
        "by_role": dict(sorted(by_role.items())),
        "admitted_row_ids": [str(row.get("row_id")) for row in admitted_rows],
        "non_admitted_row_ids": [str(row.get("row_id")) for row in non_admitted_rows],
        "admitted_unfrozen_row_ids": [
            str(row.get("row_id")) for row in admitted_unfrozen_rows
        ],
        "rows": rows,
    }


def _suite_row_content_addressed(row: Mapping[str, Any]) -> bool:
    freeze = _as_mapping(row.get("freeze"))
    return bool(freeze.get("content_addressed"))


def _suite_evidence_policy(suite: Mapping[str, Any]) -> dict[str, Any]:
    raw = (
        suite.get("evidence_policy")
        or suite.get("evidence_admission_policy")
        or suite.get("admission_policy")
        or {}
    )
    if isinstance(raw, Mapping):
        policy = copy.deepcopy(dict(raw))
    else:
        policy = {}
    min_admitted = policy.get("min_admitted")
    if min_admitted is None and bool(policy.get("require_admitted")):
        min_admitted = 1
    policy["min_admitted"] = int(min_admitted or 0)
    policy["require_freeze"] = bool(
        policy.get("require_freeze") or policy.get("require_content_addressed")
    )
    return policy


def _suite_optimizer_governance_policy(suite: Mapping[str, Any]) -> dict[str, Any]:
    raw = (
        suite.get("optimizer_governance_policy")
        or suite.get("optimization_governance_policy")
        or {}
    )
    policy = copy.deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}
    required = bool(
        policy.get("require_optimizer_governance")
        or policy.get("required")
        or policy.get("require_passed")
    )
    min_governed = policy.get("min_governed")
    if min_governed is None and required:
        min_governed = 1
    commands = _unique_strings(
        policy.get("commands")
        or policy.get("target_commands")
        or ["optimize"]
    )
    policy["require_optimizer_governance"] = required
    policy["require_passed"] = bool(policy.get("require_passed") or required)
    policy["fail_on_warning"] = bool(policy.get("fail_on_warning"))
    policy["min_governed"] = int(min_governed or 0)
    policy["commands"] = commands or ["optimize"]
    return policy


def _suite_optimizer_governance(
    children: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    target_commands = {
        _normalize_command(command)
        for command in _as_list(policy.get("commands"))
        if command
    }
    rows = [
        _suite_optimizer_governance_row(child)
        for child in children
        if _suite_optimizer_governance_targets_child(child, target_commands)
    ]
    governed_rows = [row for row in rows if bool(row.get("governance_present"))]
    failed_rows = [
        row
        for row in governed_rows
        if row.get("governance_status") != "passed" or row.get("passed") is False
    ]
    missing_rows = [row for row in rows if not bool(row.get("governance_present"))]
    warning_rows = [
        row
        for row in governed_rows
        if _as_list(row.get("warning_check_ids"))
    ]
    return {
        "kind": "agent-learning.suite.optimizer-governance.v1",
        "status": "failed" if failed_rows or missing_rows else "passed",
        "policy": copy.deepcopy(dict(policy)),
        "target_count": len(rows),
        "governed_count": len(governed_rows),
        "passed_count": len(governed_rows) - len(failed_rows),
        "failed_count": len(failed_rows),
        "missing_count": len(missing_rows),
        "warning_count": len(warning_rows),
        "target_child_ids": [str(row.get("child_id")) for row in rows],
        "governed_child_ids": [str(row.get("child_id")) for row in governed_rows],
        "failed_child_ids": [str(row.get("child_id")) for row in failed_rows],
        "missing_child_ids": [str(row.get("child_id")) for row in missing_rows],
        "warning_child_ids": [str(row.get("child_id")) for row in warning_rows],
        "rows": rows,
    }


def _suite_optimizer_governance_targets_child(
    child: Mapping[str, Any],
    target_commands: set[str],
) -> bool:
    result = _as_mapping(child.get("result"))
    if _as_mapping(result.get("optimization_governance")):
        return True
    command = _normalize_command(child.get("command") or "")
    if command in target_commands:
        return True
    return False


def _suite_optimizer_governance_row(child: Mapping[str, Any]) -> dict[str, Any]:
    result = _as_mapping(child.get("result"))
    governance = _as_mapping(result.get("optimization_governance"))
    if not governance:
        governance = _as_mapping(_as_mapping(result.get("optimization")).get("governance"))
    evidence = _as_mapping(governance.get("evidence"))
    return {
        "kind": "agent-learning.suite.optimizer-governance-row.v1",
        "child_id": child.get("id"),
        "command": child.get("command"),
        "path": child.get("path"),
        "result_kind": child.get("kind"),
        "child_status": child.get("status"),
        "child_exit_code": int(child.get("exit_code", 1)),
        "governance_present": bool(governance),
        "governance_kind": governance.get("kind"),
        "governance_status": governance.get("status") if governance else "missing",
        "passed": bool(governance.get("passed")) if governance else False,
        "selected_candidate_id": governance.get("selected_candidate_id"),
        "selected_rank": governance.get("selected_rank"),
        "check_count": int(governance.get("check_count") or 0),
        "failed_check_ids": [
            str(item) for item in _as_list(governance.get("failed_check_ids"))
        ],
        "warning_check_ids": [
            str(item) for item in _as_list(governance.get("warning_check_ids"))
        ],
        "candidate_count": int(evidence.get("candidate_count") or 0),
        "content_addressed_count": int(
            evidence.get("content_addressed_count") or 0
        ),
        "metric_count": int(evidence.get("metric_count") or 0),
        "patch_path_count": int(evidence.get("patch_path_count") or 0),
    }


def _suite_optimizer_governance_findings(
    optimizer_governance: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    min_governed = int(policy.get("min_governed") or 0)
    governed_count = int(optimizer_governance.get("governed_count") or 0)
    if min_governed > governed_count:
        findings.append({
            "type": "suite_optimizer_governance_missing",
            "level": "error",
            "reason": (
                f"Suite optimizer governance gate requires at least {min_governed} "
                f"governed optimizer child row(s), but only {governed_count} "
                "were found."
            ),
            "min_governed": min_governed,
            "governed_count": governed_count,
            "missing_child_ids": list(
                optimizer_governance.get("missing_child_ids") or []
            ),
        })
    if bool(policy.get("require_passed")):
        failed_child_ids = list(optimizer_governance.get("failed_child_ids") or [])
        missing_child_ids = list(optimizer_governance.get("missing_child_ids") or [])
        blocked_child_ids = sorted(
            {str(item) for item in [*failed_child_ids, *missing_child_ids]}
        )
        if blocked_child_ids:
            findings.append({
                "type": "suite_optimizer_governance_failed",
                "level": "error",
                "reason": (
                    "Suite optimizer governance gate requires passed governance "
                    f"for optimizer children, but {len(blocked_child_ids)} child "
                    "row(s) are missing or failed."
                ),
                "failed_child_ids": failed_child_ids,
                "missing_child_ids": missing_child_ids,
            })
    if bool(policy.get("fail_on_warning")):
        warning_child_ids = list(optimizer_governance.get("warning_child_ids") or [])
        if warning_child_ids:
            findings.append({
                "type": "suite_optimizer_governance_warning",
                "level": "error",
                "reason": (
                    "Suite optimizer governance gate is configured to fail on "
                    f"warnings, and {len(warning_child_ids)} child row(s) have "
                    "governance warnings."
                ),
                "warning_child_ids": warning_child_ids,
            })
    return findings


def _suite_evidence_findings(
    admission: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    min_admitted = int(policy.get("min_admitted") or 0)
    admitted_count = int(admission.get("admitted_count") or 0)
    findings: list[dict[str, Any]] = []
    if min_admitted > admitted_count:
        findings.append({
            "type": "suite_evidence_admission_missing",
            "level": "error",
            "reason": (
                f"Suite evidence gate requires at least {min_admitted} admitted "
                f"row(s), but only {admitted_count} were admitted."
            ),
            "admitted_count": admitted_count,
            "min_admitted": min_admitted,
        })
    if bool(policy.get("require_freeze")):
        missing = [
            str(row_id)
            for row_id in _as_list(admission.get("admitted_unfrozen_row_ids"))
        ]
        if missing:
            findings.append({
                "type": "suite_evidence_freeze_missing",
                "level": "error",
                "reason": (
                    "Suite evidence gate requires content-addressed admitted "
                    f"rows, but {len(missing)} admitted row(s) are missing "
                    "manifest/result digests."
                ),
                "missing": missing,
            })
    return findings


def _suite_result(
    *,
    suite: Mapping[str, Any],
    suite_path: Path,
    children: Sequence[Mapping[str, Any]],
    name: Optional[str],
    dry_run: bool,
    fail_fast: bool,
    duration_seconds: float,
) -> dict[str, Any]:
    job_count = len(_suite_jobs(suite))
    passed = [child for child in children if int(child.get("exit_code", 1)) == 0]
    failed = [child for child in children if int(child.get("exit_code", 1)) != 0]
    score = round(len(passed) / job_count, 4) if job_count else 0.0
    command_counts: dict[str, int] = {}
    for child in children:
        command = str(child.get("command") or "unknown")
        command_counts[command] = command_counts.get(command, 0) + 1
    capabilities = _suite_capability_summary(children)
    required_capabilities = _suite_required_capabilities(suite)
    missing_capabilities = _missing_required_capabilities(
        required_capabilities,
        capabilities,
    )
    capability_findings = _suite_capability_findings(missing_capabilities)
    framework_coverage = _suite_framework_coverage(
        children,
        required_frameworks=required_capabilities.get("frameworks", []),
    )
    framework_findings = _suite_framework_findings(framework_coverage)
    evidence_admission = _suite_evidence_admission(children)
    evidence_policy = _suite_evidence_policy(suite)
    evidence_findings = _suite_evidence_findings(
        evidence_admission,
        evidence_policy,
    )
    optimizer_governance_policy = _suite_optimizer_governance_policy(suite)
    optimizer_governance = _suite_optimizer_governance(
        children,
        optimizer_governance_policy,
    )
    optimizer_governance_findings = _suite_optimizer_governance_findings(
        optimizer_governance,
        optimizer_governance_policy,
    )
    suite_findings = [
        *capability_findings,
        *framework_findings,
        *evidence_findings,
        *optimizer_governance_findings,
        *_suite_findings(children),
    ]
    suite_passed = (
        len(failed) == 0
        and len(children) == job_count
        and not capability_findings
        and not framework_findings
        and not evidence_findings
        and not optimizer_governance_findings
    )
    trust_certificate = _suite_trust_certificate(
        suite=suite,
        suite_path=suite_path,
        children=children,
        capabilities=capabilities,
        framework_coverage=framework_coverage,
        evidence_admission=evidence_admission,
        optimizer_governance=optimizer_governance,
        missing_capabilities=missing_capabilities,
        suite_passed=suite_passed,
        job_count=job_count,
        executed_count=len(children),
        passed_count=len(passed),
        failed_count=len(failed),
        score=score,
    )
    return {
        "kind": AGENT_LEARNING_SUITE_KIND,
        "version": AGENT_LEARNING_SUITE_KIND,
        "name": str(name or suite.get("name") or suite_path.stem),
        "status": "passed" if suite_passed else "failed",
        "exit_code": 0 if suite_passed else 1,
        "dry_run": dry_run,
        "fail_fast": fail_fast,
        "summary": {
            "job_count": job_count,
            "executed_count": len(children),
            "passed_count": len(passed),
            "failed_count": len(failed),
            "skipped_count": max(job_count - len(children), 0),
            "score": score,
            "trust_certificate_verdict": trust_certificate["verdict"],
            "trust_certificate_assurance_level": trust_certificate[
                "assurance_level"
            ],
            "trust_certificate_promotion_ready": trust_certificate[
                "promotion_ready"
            ],
            "trust_certificate_failed_gate_count": len(
                trust_certificate["failed_gate_ids"]
            ),
            "trust_certificate_conditional_gate_count": len(
                trust_certificate["conditional_gate_ids"]
            ),
            "commands": command_counts,
            "capabilities": capabilities,
            "required_capabilities": required_capabilities,
            "missing_required_capabilities": missing_capabilities,
            "capability_gate_passed": not capability_findings,
            "framework_coverage_passed": not framework_findings,
            "observed_framework_count": framework_coverage["observed_count"],
            "required_framework_count": framework_coverage["required_count"],
            "missing_framework_count": framework_coverage["missing_count"],
            "adapter_conformance_failed_count": framework_coverage[
                "adapter_conformance_failed_count"
            ],
            "framework_coverage": {
                key: value
                for key, value in framework_coverage.items()
                if key != "rows"
            },
            "evidence_gate_passed": not evidence_findings,
            "optimizer_governance_gate_passed": not optimizer_governance_findings,
            "optimizer_governance_policy": optimizer_governance_policy,
            "optimizer_governance_target_count": optimizer_governance[
                "target_count"
            ],
            "optimizer_governance_governed_count": optimizer_governance[
                "governed_count"
            ],
            "optimizer_governance_passed_count": optimizer_governance[
                "passed_count"
            ],
            "optimizer_governance_failed_count": optimizer_governance[
                "failed_count"
            ],
            "optimizer_governance_missing_count": optimizer_governance[
                "missing_count"
            ],
            "optimizer_governance_warning_count": optimizer_governance[
                "warning_count"
            ],
            "admitted_evidence_count": evidence_admission["admitted_count"],
            "non_admitted_evidence_count": evidence_admission[
                "non_admitted_count"
            ],
            "rejected_evidence_count": evidence_admission["rejected_count"],
            "frozen_evidence_count": evidence_admission["frozen_count"],
            "unfrozen_evidence_count": evidence_admission["unfrozen_count"],
            "admitted_frozen_evidence_count": evidence_admission[
                "admitted_frozen_count"
            ],
            "evidence_admission": {
                key: value
                for key, value in evidence_admission.items()
                if key != "rows"
            },
        },
        "framework_coverage": framework_coverage,
        "evidence_admission": evidence_admission,
        "optimizer_governance": optimizer_governance,
        "trust_certificate": trust_certificate,
        "children": list(children),
        "jobs": list(children),
        "findings": suite_findings,
        "duration_seconds": duration_seconds,
    }


def _suite_descriptor(suite: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": suite.get("version") or AGENT_LEARNING_SUITE_KIND,
        "name": suite.get("name"),
        "job_count": len(_suite_jobs(suite)),
        "jobs": [
            {
                "id": job.get("id"),
                "command": job.get("command"),
                "path": job.get("path"),
            }
            for job in _suite_jobs(suite)
        ],
        "required_capabilities": _suite_required_capabilities(suite),
    }


def _suite_trust_certificate(
    *,
    suite: Mapping[str, Any],
    suite_path: Path,
    children: Sequence[Mapping[str, Any]],
    capabilities: Mapping[str, Sequence[str]],
    framework_coverage: Mapping[str, Any],
    evidence_admission: Mapping[str, Any],
    optimizer_governance: Mapping[str, Any],
    missing_capabilities: Mapping[str, Sequence[str]],
    suite_passed: bool,
    job_count: int,
    executed_count: int,
    passed_count: int,
    failed_count: int,
    score: float,
) -> dict[str, Any]:
    coverage = _suite_trinity_coverage(capabilities)
    admitted_count = int(evidence_admission.get("admitted_count") or 0)
    admitted_frozen_count = int(evidence_admission.get("admitted_frozen_count") or 0)
    governed_count = int(optimizer_governance.get("governed_count") or 0)
    optimizer_failed_count = int(optimizer_governance.get("failed_count") or 0)
    optimizer_missing_count = int(optimizer_governance.get("missing_count") or 0)
    gates = [
        _trust_gate(
            "execution",
            passed=failed_count == 0 and executed_count == job_count and suite_passed,
            required=True,
            reason="all declared suite jobs executed and exited successfully",
            evidence={
                "job_count": job_count,
                "executed_count": executed_count,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "score": score,
            },
        ),
        _trust_gate(
            "capability_gate",
            passed=not missing_capabilities,
            required=True,
            reason="declared required capabilities were observed",
            evidence={"missing_required_capabilities": dict(missing_capabilities)},
        ),
        _trust_gate(
            "framework_coverage",
            passed=int(framework_coverage.get("missing_count") or 0) == 0
            and int(framework_coverage.get("adapter_conformance_failed_count") or 0)
            == 0,
            required=True,
            reason="required framework coverage and adapter conformance passed",
            evidence={
                "observed_count": framework_coverage.get("observed_count"),
                "required_count": framework_coverage.get("required_count"),
                "missing_count": framework_coverage.get("missing_count"),
                "adapter_conformance_failed_count": framework_coverage.get(
                    "adapter_conformance_failed_count"
                ),
            },
        ),
        _trust_gate(
            "evidence_admission",
            passed=admitted_count > 0
            and int(evidence_admission.get("rejected_count") or 0) == 0,
            required=False,
            reason="at least one child artifact is admitted evidence",
            evidence={
                "admitted_count": admitted_count,
                "rejected_count": evidence_admission.get("rejected_count"),
                "by_status": evidence_admission.get("by_status"),
            },
        ),
        _trust_gate(
            "evidence_freeze",
            passed=admitted_count > 0 and admitted_frozen_count == admitted_count,
            required=False,
            reason="admitted evidence rows are content-addressed",
            evidence={
                "admitted_count": admitted_count,
                "admitted_frozen_count": admitted_frozen_count,
            },
        ),
        _trust_gate(
            "optimizer_governance",
            passed=governed_count > 0
            and optimizer_failed_count == 0
            and optimizer_missing_count == 0,
            required=False,
            reason="optimizer children expose passed governance verdicts",
            evidence={
                "target_count": optimizer_governance.get("target_count"),
                "governed_count": governed_count,
                "failed_count": optimizer_failed_count,
                "missing_count": optimizer_missing_count,
                "warning_count": optimizer_governance.get("warning_count"),
            },
        ),
        _trust_gate(
            "trinity_coverage",
            passed=all(coverage.values()),
            required=False,
            reason="suite covers simulation, evaluation, red-team, and optimization",
            evidence=coverage,
        ),
    ]
    failed_gate_ids = [
        gate["id"] for gate in gates if gate["required"] and not gate["passed"]
    ]
    conditional_gate_ids = [
        gate["id"] for gate in gates if not gate["required"] and not gate["passed"]
    ]
    if not suite_passed or failed_gate_ids:
        verdict = "rejected"
    elif conditional_gate_ids:
        verdict = "conditional"
    else:
        verdict = "approved"
    return {
        "kind": "agent-learning.suite.trust-certificate.v1",
        "verdict": verdict,
        "promotion_ready": verdict == "approved",
        "assurance_level": _suite_assurance_level(verdict, coverage, governed_count),
        "subject": {
            "suite_name": str(suite.get("name") or suite_path.stem),
            "suite_path": str(suite_path),
            "suite_version": suite.get("version") or AGENT_LEARNING_SUITE_KIND,
            "job_count": job_count,
        },
        "coverage": coverage,
        "evidence": {
            "admitted_count": admitted_count,
            "admitted_frozen_count": admitted_frozen_count,
            "optimizer_governed_count": governed_count,
            "optimizer_failed_count": optimizer_failed_count,
            "optimizer_missing_count": optimizer_missing_count,
            "framework_observed_count": framework_coverage.get("observed_count"),
            "framework_missing_count": framework_coverage.get("missing_count"),
        },
        "failed_gate_ids": failed_gate_ids,
        "conditional_gate_ids": conditional_gate_ids,
        "reason": _suite_trust_reason(verdict, failed_gate_ids, conditional_gate_ids),
        "gates": gates,
        "child_ids": [str(child.get("id") or "") for child in children],
    }


def _suite_trinity_coverage(capabilities: Mapping[str, Sequence[str]]) -> dict[str, bool]:
    commands = {_suite_key(command) for command in _as_list(capabilities.get("commands"))}
    result_kinds = {
        str(item)
        for item in _as_list(capabilities.get("result_kinds"))
        if str(item)
    }
    return {
        "simulation": "run" in commands or "agent-learning.run.v1" in result_kinds,
        "evaluation": bool(
            commands & {"eval", "eval_artifact", "eval_task", "optimize_eval"}
        )
        or "agent-learning.eval.v1" in result_kinds,
        "redteam": "redteam" in commands or "agent-learning.redteam.v1" in result_kinds,
        "optimization": bool(commands & {"optimize", "optimize_eval", "optimize_suite"})
        or "agent-learning.optimization.v1" in result_kinds
        or "agent-learning.suite-optimization.v1" in result_kinds,
    }


def _suite_assurance_level(
    verdict: str,
    coverage: Mapping[str, bool],
    governed_count: int,
) -> str:
    if verdict == "rejected":
        return "rejected"
    if all(coverage.values()) and governed_count > 0:
        return "l3_trinity_governed"
    if coverage.get("simulation") and coverage.get("evaluation"):
        return "l2_evaluated_simulation"
    return "l1_partial_evidence"


def _suite_trust_reason(
    verdict: str,
    failed_gate_ids: Sequence[str],
    conditional_gate_ids: Sequence[str],
) -> str:
    if verdict == "approved":
        return (
            "Approved: execution, evidence, framework coverage, red-team, "
            "simulation, evaluation, optimization, and optimizer governance closed."
        )
    if verdict == "rejected":
        return (
            "Rejected: required suite gates failed"
            + (f" ({', '.join(failed_gate_ids)})." if failed_gate_ids else ".")
        )
    return (
        "Conditional: required gates passed but advisory deployment evidence is "
        f"incomplete ({', '.join(conditional_gate_ids)})."
    )


def _trust_gate(
    gate_id: str,
    *,
    passed: bool,
    required: bool,
    reason: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "status": "passed" if passed else "failed" if required else "conditional",
        "passed": passed,
        "required": required,
        "reason": reason,
        "evidence": copy.deepcopy(dict(evidence)),
    }


def _suite_framework_coverage(
    children: Sequence[Mapping[str, Any]],
    *,
    required_frameworks: Sequence[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for child in children:
        rows.extend(_suite_framework_rows_for_child(_as_mapping(child)))
    observed = sorted(
        {
            _suite_key(row.get("framework"))
            for row in rows
            if _suite_key(row.get("framework"))
        }
    )
    required = sorted(
        {
            _suite_key(item)
            for item in _as_list(required_frameworks)
            if _suite_key(item)
        }
    )
    missing = sorted(set(required) - set(observed))
    adapter_failures = [
        row
        for row in rows
        if row.get("adapter_conformance_passed") is False
    ]
    methods: dict[str, set[str]] = {}
    input_modes: dict[str, set[str]] = {}
    modalities: dict[str, set[str]] = {}
    for row in rows:
        framework = _suite_key(row.get("framework"))
        if not framework:
            continue
        methods.setdefault(framework, set()).update(
            _suite_key(item)
            for item in _as_list(row.get("methods"))
            if _suite_key(item)
        )
        input_modes.setdefault(framework, set()).update(
            _suite_key(item)
            for item in _as_list(row.get("input_modes"))
            if _suite_key(item)
        )
        modality = _suite_key(row.get("modality"))
        if modality:
            modalities.setdefault(framework, set()).add(modality)
    return {
        "kind": "agent-learning.suite.framework-coverage.v1",
        "observed_frameworks": observed,
        "required_frameworks": required,
        "missing_required_frameworks": missing,
        "observed_count": len(observed),
        "required_count": len(required),
        "missing_count": len(missing),
        "adapter_conformance_failed_count": len(adapter_failures),
        "adapter_conformance_failed_child_ids": [
            str(row.get("child_id")) for row in adapter_failures
        ],
        "methods_by_framework": {
            key: sorted(values) for key, values in sorted(methods.items())
        },
        "input_modes_by_framework": {
            key: sorted(values) for key, values in sorted(input_modes.items())
        },
        "modalities_by_framework": {
            key: sorted(values) for key, values in sorted(modalities.items())
        },
        "rows": rows,
    }


def _suite_framework_findings(
    coverage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    missing = [
        _suite_key(item)
        for item in _as_list(coverage.get("missing_required_frameworks"))
        if _suite_key(item)
    ]
    if missing:
        findings.append(
            {
                "type": "suite_framework_coverage_missing",
                "level": "error",
                "reason": (
                    "Suite framework coverage is missing required framework(s): "
                    f"{', '.join(sorted(missing))}."
                ),
                "missing": sorted(missing),
            }
        )
    failed = [
        str(item)
        for item in _as_list(coverage.get("adapter_conformance_failed_child_ids"))
        if str(item)
    ]
    if failed:
        findings.append(
            {
                "type": "suite_framework_adapter_conformance_failed",
                "level": "error",
                "reason": (
                    "Suite framework coverage found adapter conformance failures "
                    f"in {len(failed)} child row(s)."
                ),
                "failed_child_ids": failed,
            }
        )
    return findings


def _suite_framework_rows_for_child(child: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    result = _as_mapping(child.get("result"))
    for nested in _as_list(result.get("children") or result.get("jobs")):
        nested_child = _as_mapping(nested)
        if nested_child:
            rows.extend(_suite_framework_rows_for_child(nested_child))
    for state in _suite_framework_environment_states(result):
        row = _suite_framework_row_from_state(child, state)
        if row:
            rows.append(row)
    return rows


def _suite_framework_environment_states(
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for report in (
        _as_mapping(result.get("report")),
        _as_mapping(_as_mapping(result.get("evaluation")).get("report")),
    ):
        for case in _as_list(report.get("results")):
            metadata = _as_mapping(_as_mapping(case).get("metadata"))
            state = _as_mapping(metadata.get("environment_state"))
            if state:
                states.append(state)
    return states


def _suite_framework_row_from_state(
    child: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    runtime = _as_mapping(state.get("framework_runtime"))
    trace = _as_mapping(state.get("framework_trace"))
    capability = _as_mapping(state.get("framework_capability_matrix"))
    framework = (
        runtime.get("framework")
        or trace.get("framework")
        or capability.get("framework")
    )
    framework_key = _suite_key(framework)
    if not framework_key:
        return None
    runtime_summary = _as_mapping(runtime.get("summary"))
    trace_spans = [
        _as_mapping(span)
        for span in _as_list(trace.get("spans"))
        if _as_mapping(span)
    ]
    trace_signals = sorted(
        {
            _suite_key(signal)
            for span in trace_spans
            for signal in _as_list(span.get("signals"))
            if _suite_key(signal)
        }
    )
    conformance = _as_mapping(trace.get("adapter_conformance"))
    conformance_passed = (
        bool(conformance.get("passed")) if conformance else None
    )
    return {
        "kind": "agent-learning.suite.framework-coverage-row.v1",
        "child_id": child.get("id"),
        "child_name": child.get("name"),
        "command": child.get("command"),
        "result_kind": child.get("kind"),
        "framework": framework_key,
        "modality": _suite_key(runtime.get("modality") or trace.get("modality")),
        "methods": sorted(
            {
                _suite_key(item)
                for item in _as_list(runtime_summary.get("methods"))
                if _suite_key(item)
            }
        ),
        "input_modes": sorted(
            {
                _suite_key(item)
                for item in _as_list(runtime_summary.get("input_modes"))
                if _suite_key(item)
            }
        ),
        "tool_call_count": int(runtime_summary.get("tool_call_count") or 0),
        "trace_span_count": len(trace_spans),
        "trace_signals": trace_signals,
        "adapter_conformance_passed": conformance_passed,
    }


def _suite_job_command_counts(suite: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in _suite_jobs(suite):
        command = str(job.get("command") or "unknown")
        counts[command] = counts.get(command, 0) + 1
    return counts


def _artifact_action_plan_card(result: Mapping[str, Any]) -> dict[str, Any] | None:
    optimization = _as_mapping(result.get("optimization"))
    history = [
        _as_mapping(item)
        for item in _as_list(optimization.get("history"))
        if _as_mapping(item)
    ]
    candidate_records = [
        record
        for item in history
        for record in _artifact_action_candidate_records(item)
    ]
    if not candidate_records:
        return None
    selected_action_id = _artifact_action_selected_id(optimization, candidate_records)
    for record in candidate_records:
        record["selected"] = bool(record.get("action_id") == selected_action_id)
    selected = next(
        (
            record
            for record in candidate_records
            if record.get("action_id") == selected_action_id
        ),
        max(candidate_records, key=lambda record: float(record.get("score") or 0.0)),
    )
    return {
        "kind": "artifact_action_plan",
        "status": "selected" if selected_action_id else "observed",
        "source": "agent_learning_suite_optimization",
        "selected_action_id": selected.get("action_id"),
        "selected_candidate_id": selected.get("candidate_id"),
        "selected_score": selected.get("score"),
        "selection_reason": _artifact_action_selection_reason(selected),
        "candidate_count": len(candidate_records),
        "candidate_score_lineage": candidate_records,
        "search_paths": _as_string_list(result.get("summary", {}).get("search_paths")),
        "source_manifest_path": optimization.get("source_manifest_path"),
    }


def _artifact_action_candidate_records(
    history_item: Mapping[str, Any],
) -> list[dict[str, Any]]:
    report = _as_mapping(history_item.get("report"))
    records: list[dict[str, Any]] = []
    for child in _as_list(report.get("children") or report.get("jobs")):
        child_item = _as_mapping(child)
        if str(child_item.get("command") or "").replace("-", "_") != "action_run":
            continue
        action_result = _as_mapping(child_item.get("result"))
        action_summary = _as_mapping(action_result.get("summary"))
        action_id = str(
            action_summary.get("action_id")
            or _artifact_action_id_from_patch(history_item)
            or child_item.get("id")
            or ""
        )
        output_count = int(action_summary.get("output_count") or 0)
        outputs_written_count = int(action_summary.get("outputs_written_count") or 0)
        completion = _artifact_action_completion_rate(
            action_summary,
            output_count=output_count,
            outputs_written_count=outputs_written_count,
        )
        action_kind = str(action_summary.get("action_kind") or "cli")
        evidence_denominator = 1.0 if action_kind == "download" else 4.0
        evidence_depth = round(
            min(outputs_written_count / evidence_denominator, 1.0),
            4,
        )
        records.append(
            {
                "candidate_id": history_item.get("candidate_id"),
                "action_id": action_id,
                "action_label": action_summary.get("action_label"),
                "action_kind": action_kind,
                "artifact_ref": action_summary.get("artifact_ref"),
                "source_card_path": action_summary.get("source_card_path"),
                "score": history_item.get("score"),
                "action_score": round((0.8 * completion) + (0.2 * evidence_depth), 4),
                "status": action_result.get("status") or child_item.get("status"),
                "exit_code": action_result.get("exit_code", child_item.get("exit_code")),
                "output_count": output_count,
                "outputs_written_count": outputs_written_count,
                "output_completion_rate": completion,
                "evidence_depth": evidence_depth,
                "outputs_written": list(action_result.get("outputs_written") or []),
                "outputs": [
                    {
                        "flag": _as_mapping(output).get("flag"),
                        "path": _as_mapping(output).get("path"),
                        "exists": _as_mapping(output).get("exists"),
                    }
                    for output in _as_list(action_result.get("outputs"))
                    if _as_mapping(output)
                ],
                "command_args": list(action_result.get("command_args") or []),
                "patch": copy.deepcopy(dict(history_item.get("patch") or {})),
            }
        )
    return records


def _artifact_action_completion_rate(
    summary: Mapping[str, Any],
    *,
    output_count: int,
    outputs_written_count: int,
) -> float:
    if summary.get("output_completion_rate") is not None:
        return round(float(summary.get("output_completion_rate") or 0.0), 4)
    if output_count:
        return round(outputs_written_count / output_count, 4)
    return 1.0


def _artifact_action_selected_id(
    optimization: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> str | None:
    best_config = _as_mapping(optimization.get("best_config"))
    for job in _as_list(best_config.get("jobs")):
        action_id = _as_mapping(job).get("action_id")
        if action_id:
            return str(action_id)
    if not candidates:
        return None
    best = max(candidates, key=lambda record: float(record.get("score") or 0.0))
    return str(best.get("action_id")) if best.get("action_id") else None


def _artifact_action_id_from_patch(history_item: Mapping[str, Any]) -> str | None:
    patch = _as_mapping(history_item.get("patch") or history_item.get("candidate_patch"))
    job = _as_mapping(patch.get("jobs.0"))
    action_id = job.get("action_id")
    return str(action_id) if action_id else None


def _artifact_action_selection_reason(selected: Mapping[str, Any]) -> str:
    action_id = selected.get("action_id") or "selected action"
    status = selected.get("status") or "unknown"
    output_count = selected.get("output_count")
    outputs_written = selected.get("outputs_written_count")
    completion = selected.get("output_completion_rate")
    score = selected.get("score")
    return (
        f"Selected {action_id} because it finished with status {status}, "
        f"score {score}, output completion {completion}, and "
        f"{outputs_written}/{output_count} declared outputs written."
    )


def _suite_capability_summary(children: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    caps: dict[str, set[str]] = {
        "channels": set(),
        "child_ids": set(),
        "commands": set(),
        "environment_state_keys": set(),
        "environment_types": set(),
        "evidence_roles": set(),
        "evidence_statuses": set(),
        "frameworks": set(),
        "metrics": set(),
        "modalities": set(),
        "providers": set(),
        "result_kinds": set(),
        "search_paths": set(),
    }
    for child in children:
        _add_capability(caps, "child_ids", child.get("id"))
        _add_capability(caps, "commands", child.get("command"))
        _add_capability(caps, "result_kinds", child.get("kind"))
        evidence = _as_mapping(child.get("evidence"))
        _add_capability(caps, "evidence_roles", evidence.get("role"))
        _add_capability(caps, "evidence_statuses", evidence.get("status"))
        result = _as_mapping(child.get("result"))
        _collect_result_capabilities(result, caps)
    return {key: sorted(values) for key, values in caps.items()}


def _suite_required_capabilities(suite: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = (
        suite.get("required_capabilities")
        or suite.get("capability_requirements")
        or suite.get("capabilities_required")
        or {}
    )
    if not isinstance(raw, Mapping):
        return {}
    requirements: dict[str, list[str]] = {}
    for key, values in raw.items():
        normalized_key = _suite_key(key)
        if not normalized_key:
            continue
        normalized_values = sorted(
            {
                _suite_key(value)
                for value in _as_list(values)
                if _suite_key(value)
            }
        )
        if normalized_values:
            requirements[normalized_key] = normalized_values
    return requirements


def _missing_required_capabilities(
    required: Mapping[str, Sequence[str]],
    observed: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for key, required_values in required.items():
        observed_values = {_suite_key(value) for value in _as_list(observed.get(key))}
        missing_values = sorted(
            {
                _suite_key(value)
                for value in _as_list(required_values)
                if _suite_key(value) and _suite_key(value) not in observed_values
            }
        )
        if missing_values:
            missing[key] = missing_values
    return missing


def _suite_capability_findings(
    missing_capabilities: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for capability, missing_values in sorted(missing_capabilities.items()):
        values = sorted(_suite_key(value) for value in missing_values if _suite_key(value))
        if not values:
            continue
        findings.append(
            {
                "type": "suite_required_capability_missing",
                "level": "error",
                "reason": (
                    f"Missing required suite capability `{capability}`: "
                    f"{', '.join(values)}."
                ),
                "capability": capability,
                "missing": values,
            }
        )
    return findings


def _collect_result_capabilities(payload: Mapping[str, Any], caps: dict[str, set[str]]) -> None:
    for child in _as_list(payload.get("children") or payload.get("jobs")):
        child_item = _as_mapping(child)
        if not child_item:
            continue
        _add_capability(caps, "child_ids", child_item.get("id"))
        _add_capability(caps, "commands", child_item.get("command"))
        _add_capability(caps, "result_kinds", child_item.get("kind"))
        _collect_result_capabilities(_as_mapping(child_item.get("result")), caps)
    _collect_summary_capabilities(_as_mapping(payload.get("summary")), caps)
    optimization = _as_mapping(payload.get("optimization"))
    best_config = _as_mapping(optimization.get("best_config"))
    simulation = _as_mapping(best_config.get("simulation"))
    for environment in _as_list(simulation.get("environments")):
        env = _as_mapping(environment)
        _add_capability(caps, "environment_types", env.get("type"))
    for history in _as_list(optimization.get("history")):
        item = _as_mapping(history)
        _add_capabilities(caps, "metrics", _as_mapping(item.get("metrics")).keys())
        _collect_report_capabilities(_as_mapping(item.get("report")), caps)
    _collect_report_capabilities(_as_mapping(payload.get("report")), caps)
    _collect_report_capabilities(_as_mapping(_as_mapping(payload.get("evaluation")).get("report")), caps)
    _collect_payload_capabilities(payload, caps)


def _collect_report_capabilities(report: Mapping[str, Any], caps: dict[str, set[str]]) -> None:
    for result in _as_list(report.get("results")):
        case = _as_mapping(result)
        metadata = _as_mapping(case.get("metadata"))
        environment_state = _as_mapping(metadata.get("environment_state"))
        _add_capabilities(caps, "environment_state_keys", environment_state.keys())
        for state in environment_state.values():
            _collect_payload_capabilities(state, caps)
        _collect_payload_capabilities(_as_mapping(case.get("evaluation")), caps)


def _collect_payload_capabilities(
    value: Any,
    caps: dict[str, set[str]],
    *,
    depth: int = 0,
) -> None:
    if depth > 12:
        return
    if isinstance(value, Mapping):
        item = _as_mapping(value)
        _collect_summary_capabilities(_as_mapping(item.get("summary")), caps)
        _add_capability(caps, "frameworks", item.get("framework"))
        _add_capability(caps, "providers", item.get("provider"))
        _add_capability(caps, "providers", item.get("provider_id"))
        _add_capability(caps, "providers", item.get("provider_type"))
        _add_capability(caps, "channels", item.get("channel"))
        _add_capability(caps, "channels", item.get("modality"))
        _add_capability(caps, "modalities", item.get("modality"))
        _add_capabilities(caps, "metrics", _as_mapping(item.get("metrics")).keys())
        if _suite_key(item.get("type")) in _KNOWN_ENVIRONMENT_TYPES:
            _add_capability(caps, "environment_types", item.get("type"))
        for metric in _as_list(item.get("metrics")):
            metric_item = _as_mapping(metric)
            _add_capability(caps, "metrics", metric_item.get("name"))
        for child in item.values():
            _collect_payload_capabilities(child, caps, depth=depth + 1)
    elif isinstance(value, list):
        for child in value:
            _collect_payload_capabilities(child, caps, depth=depth + 1)


def _collect_summary_capabilities(summary: Mapping[str, Any], caps: dict[str, set[str]]) -> None:
    if not summary:
        return
    _add_capabilities(caps, "search_paths", summary.get("search_paths"))
    _add_capabilities(caps, "providers", summary.get("observed_providers"))
    _add_capabilities(caps, "providers", summary.get("required_providers"))
    _add_capabilities(caps, "channels", summary.get("observed_channels"))
    _add_capabilities(caps, "channels", summary.get("required_channels"))
    _add_capabilities(caps, "frameworks", summary.get("trace_frameworks"))
    _add_capabilities(caps, "frameworks", summary.get("observed_frameworks"))
    _add_capabilities(caps, "frameworks", summary.get("required_trace_frameworks"))
    _add_capabilities(caps, "frameworks", summary.get("frameworks"))
    _add_capabilities(caps, "environment_state_keys", summary.get("environment_state_keys"))
    evidence_admission = _as_mapping(summary.get("evidence_admission"))
    _add_capabilities(caps, "evidence_statuses", evidence_admission.get("by_status"))
    _add_capabilities(caps, "evidence_roles", evidence_admission.get("by_role"))
    _add_capabilities(caps, "metrics", summary.get("observed_metrics"))
    _add_capabilities(caps, "metrics", summary.get("required_metrics"))
    _add_capabilities(caps, "metrics", summary.get("eval_metrics"))
    _add_capabilities(caps, "metrics", _as_mapping(summary.get("metric_averages")).keys())
    provider_channels = _as_mapping(summary.get("provider_channels"))
    _add_capabilities(caps, "providers", provider_channels.keys())
    for channels in provider_channels.values():
        _add_capabilities(caps, "channels", channels)


def _add_capabilities(
    caps: dict[str, set[str]],
    key: str,
    values: Any,
) -> None:
    if isinstance(values, Mapping):
        values = values.keys()
    elif values is None:
        return
    elif isinstance(values, (str, bytes)):
        values = [values]
    else:
        try:
            values = list(values)
        except TypeError:
            values = [values]
    for value in values:
        _add_capability(caps, key, value)


def _add_capability(caps: dict[str, set[str]], key: str, value: Any) -> None:
    normalized = _suite_key(value)
    if normalized:
        caps[key].add(normalized)


def _suite_findings(children: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for child in children:
        exit_code = int(child.get("exit_code", 1))
        if exit_code != 0:
            findings.append(
                {
                    "type": "suite_child_failed",
                    "level": "error",
                    "reason": (
                        f"{child.get('command')} {child.get('id')} exited "
                        f"{exit_code}."
                    ),
                    "job": child.get("id"),
                    "command": child.get("command"),
                    "path": child.get("path"),
                }
            )
        for finding in list(child.get("findings") or []):
            if isinstance(finding, Mapping):
                copied = copy.deepcopy(dict(finding))
                copied.setdefault("job", child.get("id"))
                copied.setdefault("command", child.get("command"))
                copied.setdefault("path", child.get("path"))
                findings.append(copied)
    return findings


def _suite_sarif_findings(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    findings = []
    for finding in list(result.get("findings") or []):
        if isinstance(finding, Mapping):
            findings.append(copy.deepcopy(dict(finding)))
    return findings


def _load_child_source(job: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    path = _job_path(job, base_dir=base_dir)
    loaded = _load_json_or_yaml(path)
    if not isinstance(loaded, Mapping):
        raise SuiteError(f"suite job source must be an object: {path}")
    return dict(loaded)


def _suite_jobs(suite: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [dict(job) for job in _as_list(suite.get("jobs"))]


def _job_path(job: Mapping[str, Any], *, base_dir: Path) -> Path:
    raw = (
        job.get("path")
        or job.get("manifest")
        or job.get("suite")
        or job.get("file")
        or job.get("current")
        or job.get("result")
    )
    if not raw:
        replay_paths = _as_list(job.get("manifests") or job.get("paths"))
        if replay_paths:
            raw = replay_paths[0]
    if not raw:
        raise SuiteError(f"suite job {job.get('id') or ''} requires path")
    return _resolve_path(str(raw), base_dir)


def _job_compare_baseline_path(job: Mapping[str, Any], *, base_dir: Path) -> Path:
    raw = job.get("baseline") or job.get("baseline_path") or job.get("baseline-path")
    if not raw:
        raise SuiteError(f"suite compare job {job.get('id') or ''} requires baseline")
    return _resolve_path(str(raw), base_dir)


def _job_replay_manifest_paths(job: Mapping[str, Any], *, base_dir: Path) -> list[Path]:
    raw_values = _as_list(
        job.get("manifests")
        or job.get("paths")
        or job.get("path")
        or job.get("manifest")
    )
    paths = [_resolve_path(str(value), base_dir) for value in raw_values if str(value)]
    if not paths:
        raise SuiteError(f"suite replay job {job.get('id') or ''} requires manifests")
    return paths


def _job_optional_path(
    job: Mapping[str, Any],
    *,
    base_dir: Path,
    keys: Sequence[str],
) -> Optional[Path]:
    for key in keys:
        raw = job.get(key)
        if raw not in (None, ""):
            return _resolve_path(str(raw), base_dir)
    return None


def _job_action_id(job: Mapping[str, Any]) -> str:
    raw = (
        job.get("action_id")
        or job.get("action-id")
        or job.get("action")
        or job.get("actionId")
    )
    if raw in (None, ""):
        raise SuiteError(f"suite action-run job {job.get('id') or ''} requires action_id")
    return str(raw)


def _job_action_inputs(job: Mapping[str, Any]) -> dict[str, Any]:
    raw = job.get("inputs") or job.get("action_inputs") or job.get("action-inputs")
    if raw in (None, ""):
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    parsed: dict[str, Any] = {}
    for value in _as_list(raw):
        text = str(value)
        if "=" not in text:
            raise SuiteError(f"suite action-run input must be name=value: {text!r}")
        key, item = text.split("=", 1)
        if not key.strip():
            raise SuiteError(f"suite action-run input has empty name: {text!r}")
        parsed[key.strip()] = item
    return parsed


def _job_action_artifact_output(job: Mapping[str, Any]) -> Optional[str]:
    raw = (
        job.get("artifact_output")
        or job.get("artifact-output")
        or job.get("artifact_output_path")
        or job.get("artifact-output-path")
    )
    if raw in (None, ""):
        return None
    return str(raw)


def _job_action_cwd(job: Mapping[str, Any], *, base_dir: Path) -> Path:
    raw = (
        job.get("cwd")
        or job.get("working_dir")
        or job.get("working-dir")
        or job.get("workdir")
    )
    if raw in (None, ""):
        return base_dir
    return _resolve_path(str(raw), base_dir)


def _job_output_paths(job: Mapping[str, Any], base_dir: Path) -> dict[str, list[Path]]:
    outputs: dict[str, list[Path]] = {
        "json": [],
        "junit": [],
        "sarif": [],
        "markdown": [],
    }
    suite_outputs = dict(job.get("outputs") or {})
    raw_json = [*_as_list(job.get("output")), *_as_list(suite_outputs.get("json"))]
    raw_junit = _as_list(suite_outputs.get("junit"))
    raw_sarif = _as_list(suite_outputs.get("sarif"))
    raw_markdown = [
        *_as_list(suite_outputs.get("markdown")),
        *_as_list(suite_outputs.get("md")),
    ]
    for value in raw_json:
        path = _resolve_path(str(value), base_dir)
        if path.name.endswith((".junit.xml", ".xml")):
            outputs["junit"].append(path)
        elif path.name.endswith((".sarif", ".sarif.json")):
            outputs["sarif"].append(path)
        else:
            outputs["json"].append(path)
    outputs["junit"].extend(_resolve_path(str(value), base_dir) for value in raw_junit)
    outputs["sarif"].extend(_resolve_path(str(value), base_dir) for value in raw_sarif)
    outputs["markdown"].extend(
        _resolve_path(str(value), base_dir) for value in raw_markdown
    )
    return outputs


def _normalize_command(value: Any) -> str:
    command = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "simulation": "run",
        "simulate": "run",
        "evaluation": "eval",
        "evalartifact": "eval_artifact",
        "eval_artifacts": "eval_artifact",
        "eval_report": "eval_artifact",
        "eval_reports": "eval_artifact",
        "artifact_eval": "eval_artifact",
        "artifact_evaluation": "eval_artifact",
        "evaltask": "eval_task",
        "eval_tasks": "eval_task",
        "eval_evidence": "eval_task",
        "action": "action_run",
        "actions": "action_run",
        "actionrun": "action_run",
        "run_action": "action_run",
        "task_eval": "eval_task",
        "task_evaluation": "eval_task",
        "task_evidence_eval": "eval_task",
        "red_team": "redteam",
        "optimization": "optimize",
        "optimizeeval": "optimize_eval",
        "optimizesuite": "optimize_suite",
        "suite_optimization": "optimize_suite",
        "suite_optimizer": "optimize_suite",
        "subsuite": "suite",
        "sub_suite": "suite",
        "promotion": "promote_to_regression",
        "regression_promotion": "promote_to_regression",
        "promote": "promote_to_regression",
        "minimize": "shrink",
        "minimize_counterexample": "shrink",
    }
    command = aliases.get(command, command)
    if command not in _CHILD_COMMANDS:
        allowed = ", ".join(sorted(_CHILD_COMMANDS))
        raise SuiteError(f"unsupported suite job command: {command}; expected {allowed}")
    return command


def _normalize_suite_job(job: Mapping[str, Any], index: int) -> dict[str, Any]:
    item = copy.deepcopy(dict(job))
    command = _normalize_command(item.get("command") or item.get("type"))
    path = item.get("path") or item.get("manifest") or item.get("suite")
    if path in (None, ""):
        raise ValueError(f"suite job {index} requires a path")
    item["command"] = command
    item["path"] = _suite_path_text(path)
    item["id"] = str(item.get("id") or item.get("name") or f"{command}-{index}")
    return item


def _suite_path_text(path: str | Path) -> str:
    return str(path)


def _suite_local_target_text(target: str | Path, *, base_dir: str | Path = ".") -> str:
    target_text = str(target)
    module_name, separator, attribute_path = target_text.partition(":")
    if (
        separator
        and attribute_path
        and (
            module_name.endswith(".py")
            or "/" in module_name
            or "\\" in module_name
        )
    ):
        module_path = Path(module_name).expanduser()
        if not module_path.is_absolute():
            module_path = Path(base_dir).expanduser() / module_path
        return f"{module_path.resolve()}:{attribute_path}"
    return target_text


def _unique_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    items = values if isinstance(values, (list, tuple, set)) else _as_list(values)
    for value in items:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _optimization_lifecycle_paths(
    *,
    optimize_manifest_path: str | Path,
    workspace_dir: str | Path | None,
) -> dict[str, Path]:
    manifest_path = Path(optimize_manifest_path).expanduser().resolve()
    if workspace_dir is None:
        workspace = (
            manifest_path.parent.parent
            if manifest_path.parent.name == "manifests"
            else manifest_path.parent
        )
    else:
        workspace = Path(workspace_dir).expanduser().resolve()
    artifacts = workspace / "artifacts"
    regressions = workspace / "regressions"
    return {
        "optimize_manifest": manifest_path,
        "optimization": artifacts / "optimization.json",
        "optimization_junit": artifacts / "optimization.junit.xml",
        "optimization_sarif": artifacts / "optimization.sarif.json",
        "optimization_markdown": artifacts / "optimization.md",
        "optimization_report": artifacts / "optimization-report.json",
        "optimization_report_markdown": artifacts / "optimization-report.md",
        "promotion": artifacts / "promotion.json",
        "promotion_report": artifacts / "promotion-report.json",
        "promotion_report_markdown": artifacts / "promotion-report.md",
        "regression_manifest": regressions / "optimized-regression.json",
        "replay": artifacts / "replay.json",
        "replay_junit": artifacts / "replay.junit.xml",
        "replay_sarif": artifacts / "replay.sarif.json",
        "replay_markdown": artifacts / "replay.md",
        "replay_report": artifacts / "replay-report.json",
        "replay_report_markdown": artifacts / "replay-report.md",
    }


def _required_env_cli_args(required_env: Sequence[str]) -> list[str]:
    args: list[str] = []
    for key in _unique_strings(required_env):
        args.extend(["--required-env", key])
    return args


def _lifecycle_step(
    step_id: str,
    label: str,
    command_args: Sequence[Any],
    *,
    outputs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    step = {
        "id": step_id,
        "label": label,
        "kind": "cli",
        "command": " ".join(shlex.quote(str(arg)) for arg in command_args),
        "command_args": [str(arg) for arg in command_args],
    }
    if outputs:
        step["outputs"] = {key: str(value) for key, value in outputs.items()}
    return step


def _write_lifecycle_result_bundle(
    result: Mapping[str, Any],
    *,
    json_path: Path,
    junit_path: Path,
    sarif_path: Path,
    markdown_path: Path,
    source_path: Path,
) -> list[str]:
    from fi.alk import simulate

    return [
        _write_json(json_path, result),
        _write_text(junit_path, simulate.render_junit(result)),
        _write_text(sarif_path, simulate.render_sarif(result, manifest_path=source_path)),
        _write_text(
            markdown_path,
            simulate.render_markdown(result, source_path=source_path),
        ),
    ]


def _write_lifecycle_report_bundle(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
    source_path: Path,
) -> list[str]:
    from fi.alk import simulate

    return [
        _write_json(json_path, report),
        _write_text(
            markdown_path,
            simulate.render_markdown(report, source_path=source_path),
        ),
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    return _write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def _write_text(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return str(path)


def _job_name(job: Mapping[str, Any]) -> Optional[str]:
    value = job.get("name")
    if value in (None, ""):
        return None
    return str(value)


def _job_threshold(
    job: Mapping[str, Any],
    suite_options: SuiteRunOptions,
) -> Optional[float]:
    if job.get("threshold") is not None:
        return float(job["threshold"])
    return suite_options.threshold


def _job_max_candidates(
    job: Mapping[str, Any],
    suite_options: SuiteRunOptions,
) -> Optional[int]:
    if job.get("max_candidates") is not None:
        return int(job["max_candidates"])
    if job.get("max-candidates") is not None:
        return int(job["max-candidates"])
    return suite_options.max_candidates


def _job_dry_run(job: Mapping[str, Any], suite_options: SuiteRunOptions) -> bool:
    return bool(suite_options.dry_run or job.get("dry_run") or job.get("dry-run"))


def _job_int(
    job: Mapping[str, Any],
    *keys: str,
    default: int,
) -> int:
    for key in keys:
        if job.get(key) is not None:
            return int(job[key])
    return default


def _job_float(
    job: Mapping[str, Any],
    *keys: str,
    default: float,
) -> float:
    for key in keys:
        if job.get(key) is not None:
            return float(job[key])
    return default


def _job_optional_float(
    job: Mapping[str, Any],
    *keys: str,
) -> Optional[float]:
    for key in keys:
        if job.get(key) is not None:
            return float(job[key])
    return None


def _merge_options(
    options: Optional[SuiteRunOptions],
    *,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
    fail_fast: Optional[bool] = None,
    require_optimizer_governance: Optional[bool] = None,
) -> SuiteRunOptions:
    base = options or SuiteRunOptions()
    return SuiteRunOptions(
        name=name if name is not None else base.name,
        threshold=threshold if threshold is not None else base.threshold,
        max_candidates=(
            max_candidates if max_candidates is not None else base.max_candidates
        ),
        dry_run=dry_run if dry_run is not None else base.dry_run,
        fail_fast=fail_fast if fail_fast is not None else base.fail_fast,
        require_optimizer_governance=(
            require_optimizer_governance
            if require_optimizer_governance is not None
            else base.require_optimizer_governance
        ),
    )


def _merge_optimization_options(
    options: Optional[SuiteOptimizationOptions],
    *,
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_candidates: Optional[int] = None,
    dry_run: Optional[bool] = None,
) -> SuiteOptimizationOptions:
    base = options or SuiteOptimizationOptions()
    return SuiteOptimizationOptions(
        name=name if name is not None else base.name,
        threshold=threshold if threshold is not None else base.threshold,
        max_candidates=(
            max_candidates if max_candidates is not None else base.max_candidates
        ),
        dry_run=dry_run if dry_run is not None else base.dry_run,
    )


def _optimization_cli() -> Any:
    import importlib

    return importlib.import_module("fi.alk.simulate.cli")


def _load_json_or_yaml(path: Path) -> Any:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency clarity
            raise SuiteError("YAML suite manifests require PyYAML.") from exc
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _suite_base_dir(suite_path: str | Path) -> Path:
    path = Path(suite_path).expanduser().resolve()
    if path.suffix:
        return path.parent
    return path


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _run_async(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item)]


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _suite_key(value: Any) -> str:
    if isinstance(value, Mapping):
        return ""
    if isinstance(value, (list, tuple, set)):
        return ""
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


_TRUST_VERDICT_RANK = {
    "rejected": 0,
    "conditional": 1,
    "approved": 2,
}


def _optional_bool(value: Any, fallback: Any = None) -> bool | None:
    candidate = value if value is not None else fallback
    if candidate is None:
        return None
    if isinstance(candidate, bool):
        return candidate
    if isinstance(candidate, str):
        normalized = candidate.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


_KNOWN_ENVIRONMENT_TYPES = {
    "adversarial_attack_pack",
    "agent_control_plane",
    "agent_integration",
    "agent_memory_lineage",
    "agent_trust_boundary",
    "autonomy_loop",
    "browser",
    "domain_package",
    "framework_capability",
    "framework_lifecycle",
    "framework_portability",
    "framework_probe",
    "framework_trace",
    "multimodal_image",
    "multi_agent_room",
    "observability_replay",
    "openenv",
    "optimizer_trace",
    "persistent_state_attack",
    "red_team_campaign",
    "red_team_readiness",
    "retrieval_memory",
    "stateful_tool_world",
    "streaming_trace",
    "voice",
    "workspace_run_manifest",
    "world_attack_replay",
    "world_contract",
    "world_orchestration_replay",
}


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


__all__ = [
    "AGENT_LEARNING_OPTIMIZATION_LIFECYCLE_KIND",
    "AGENT_LEARNING_SUITE_KIND",
    "AGENT_LEARNING_SUITE_OPTIMIZATION_KIND",
    "AGENT_LEARNING_SUITE_TRUST_CERTIFICATE_KIND",
    "AGENT_LEARNING_SUITE_TRUST_VERIFICATION_KIND",
    "SuiteError",
    "SuiteOptimizationOptions",
    "SuiteRunOptions",
    "build_framework_adapter_trinity_suite_optimization_manifest",
    "build_framework_adapter_trinity_suite_manifest",
    "build_optimization_lifecycle_plan",
    "build_regression_artifact_suite_manifest",
    "build_suite_manifest",
    "build_trinity_suite_manifest",
    "load_suite",
    "load_suite_artifact_file",
    "load_suite_file",
    "missing_suite_env",
    "optimize_suite",
    "optimize_suite_file",
    "render_junit",
    "render_markdown",
    "render_sarif",
    "required_suite_env",
    "run_optimization_lifecycle_file",
    "run_suite",
    "run_suite_file",
    "verify_trust_certificate",
    "verify_trust_certificate_file",
    "validate_suite_env",
    "write_framework_adapter_trinity_suite_optimization_workspace",
    "write_framework_adapter_trinity_suite_workspace",
    "write_suite_file",
]
