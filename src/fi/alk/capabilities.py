from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ._schema import AGENT_LEARNING_CLI_SCHEMA_VERSION, normalize_public_payload

AGENT_LEARNING_CAPABILITIES_KIND = "agent-learning.capabilities.v1"

COMMANDS = [
    "actions",
    "action_optimize",
    "action_run",
    "baseline",
    "capabilities",
    "compare",
    "doctor",
    "eval",
    "eval_artifact",
    "eval_cli",
    "eval_task",
    "init",
    "optimize",
    "optimize_eval",
    "optimize_suite",
    "promote_to_regression",
    "redteam",
    "replay",
    "report",
    "run",
    "shrink",
    "suite",
]

RESULT_KINDS = [
    "agent_learning.action_run.v1",
    "agent_learning.actions.v1",
    "agent_learning.artifact_evaluation.v1",
    "agent_learning.capabilities.v1",
    "agent_learning.eval.behavior_entropy.v1",
    "agent_learning.eval.collaborative_competence.v1",
    "agent_learning.eval.redteam_adaptive_loop.v1",
    "agent_learning.eval.redteam_attack_evolution.v1",
    "agent_learning.eval.v1",
    "agent_learning.eval_optimization.v1",
    "agent_learning.optimization.v1",
    "agent_learning.redteam.v1",
    "agent_learning.report.v1",
    "agent_learning.run.v1",
    "agent_learning.attack_evolution_shrink.v1",
    "agent_learning.suite.v1",
    "agent_learning.suite_optimization.v1",
]

DEFAULT_METRICS = [
    "action_safety",
    "agent_control_plane_quality",
    "agent_goal_accuracy",
    "agent_integration_quality",
    "agent_memory_lineage_coverage",
    "agent_memory_lineage_quality",
    "artifact_coverage",
    "artifact_grounding_quality",
    "artifact_semantics_quality",
    "browser_action_outcome",
    "browser_cua_probe_action_quality",
    "browser_cua_probe_local_contract_quality",
    "browser_cua_probe_mutation_grounding_quality",
    "browser_cua_probe_pass_rate",
    "browser_cua_probe_score",
    "browser_cua_probe_state_quality",
    "browser_cua_probe_tool_evidence",
    "browser_cua_probe_trace_quality",
    "behavior_entropy_quality",
    "collaborative_competence_quality",
    "domain_package_quality",
    "environment_injection_resistance",
    "eval_assertions",
    "evaluation_hook_probe_agent_report_quality",
    "evaluation_hook_probe_auth_redaction",
    "evaluation_hook_probe_local_contract_quality",
    "evaluation_hook_probe_metric_response_quality",
    "evaluation_hook_probe_pass_rate",
    "evaluation_hook_probe_score",
    "evaluation_hook_probe_task_evidence",
    "framework_capability_coverage",
    "framework_capability_quality",
    "framework_adapter_call_contract_quality",
    "framework_adapter_observed_io_quality",
    "framework_adapter_probe_finding_quality",
    "framework_adapter_probe_io_contract_quality",
    "framework_adapter_probe_local_contract_quality",
    "framework_adapter_probe_pass_rate",
    "framework_adapter_probe_runtime_trace_coverage",
    "framework_adapter_probe_score",
    "framework_adapter_probe_tool_evidence",
    "framework_import_readiness",
    "framework_lifecycle_quality",
    "framework_portability_quality",
    "framework_probe_quality",
    "framework_runtime_contract",
    "framework_runtime_coverage",
    "framework_trace_coverage",
    "framework_transcript_quality",
    "goal_progress",
    "harness_trajectory_replay_quality",
    "memory_correctness",
    "memory_layer_probe_finding_quality",
    "memory_layer_probe_governance_quality",
    "memory_layer_probe_lineage_quality",
    "memory_layer_probe_local_contract_quality",
    "memory_layer_probe_pass_rate",
    "memory_layer_probe_retrieval_grounding",
    "memory_layer_probe_score",
    "multi_agent_coordination_quality",
    "multi_agent_room_probe_coordination_quality",
    "multi_agent_room_probe_finding_quality",
    "multi_agent_room_probe_handoff_contract",
    "multi_agent_room_probe_local_contract_quality",
    "multi_agent_room_probe_pass_rate",
    "multi_agent_room_probe_role_boundary",
    "multi_agent_room_probe_score",
    "multi_agent_trace_coverage",
    "multimodal_faithfulness",
    "observability_replay_quality",
    "orchestration_stack_probe_framework_quality",
    "orchestration_stack_probe_local_contract_quality",
    "orchestration_stack_probe_memory_quality",
    "orchestration_stack_probe_multi_agent_quality",
    "orchestration_stack_probe_pass_rate",
    "orchestration_stack_probe_retrieval_quality",
    "orchestration_stack_probe_score",
    "orchestration_stack_probe_tool_evidence",
    "orchestration_stack_probe_world_quality",
    "trinity_stack_probe_evaluation_hook_quality",
    "trinity_stack_probe_local_contracts",
    "trinity_stack_probe_orchestration_quality",
    "trinity_stack_probe_promotion_ready",
    "trinity_stack_probe_same_agent",
    "trinity_stack_probe_score",
    "optimizer_trace_coverage",
    "optimizer_trace_quality",
    "persistent_state_attack_coverage",
    "persistent_state_attack_quality",
    "policy_adherence",
    "prompt_injection_resistance",
    "reasoning_quality",
    "realtime_stack_probe_local_contract_quality",
    "realtime_stack_probe_pass_rate",
    "realtime_stack_probe_routing_quality",
    "realtime_stack_probe_score",
    "realtime_stack_probe_streaming_quality",
    "realtime_stack_probe_tool_evidence",
    "realtime_stack_probe_voice_quality",
    "realtime_trace_coverage",
    "realtime_trace_quality",
    "red_team_campaign_quality",
    "red_team_adaptive_loop_quality",
    "red_team_attack_evolution_coverage",
    "red_team_attack_evolution_quality",
    "retrieval_context_quality",
    "step_efficiency",
    "task_completion",
    "tool_fault_tolerance",
    "tool_selection_accuracy",
    "trajectory_score",
    "voice_trace_coverage",
    "workspace_run_quality",
    "world_contract_quality",
]

RESEARCH_SOURCES = [
    {
        "id": "agent_identity_uri_capability_discovery",
        "title": (
            "Agent Identity URI Scheme: Topology-Independent Naming and "
            "Capability-Based Discovery for Multi-Agent Systems"
        ),
        "source": "arxiv:2601.14567",
        "url": "https://arxiv.org/abs/2601.14567",
        "year": 2026,
    },
    {
        "id": "structured_agentic_discovery",
        "title": (
            "Declarative Data Services: Structured Agentic Discovery for "
            "Composing Data Systems"
        ),
        "source": "arxiv:2605.20690",
        "url": "https://arxiv.org/abs/2605.20690",
        "year": 2026,
    },
    {
        "id": "capability_governance",
        "title": (
            "Beyond Static Sandboxing: Learned Capability Governance for "
            "Autonomous AI Agents"
        ),
        "source": "arxiv:2604.11839",
        "url": "https://arxiv.org/abs/2604.11839",
        "year": 2026,
    },
    {
        "id": "recuse_signal_agent_governance",
        "title": (
            "Will the Agent Recuse Itself? Measuring LLM-Agent Compliance "
            "with In-Band Access-Deny Signals"
        ),
        "source": "arxiv:2606.06460",
        "url": "https://arxiv.org/abs/2606.06460",
        "year": 2026,
    },
]


def capability_catalog(
    artifacts: Sequence[Mapping[str, Any]] = (),
    *,
    source_paths: Sequence[str | Path] = (),
    required_capabilities: Optional[Mapping[str, Sequence[Any]]] = None,
    name: Optional[str] = None,
) -> dict[str, Any]:
    """Return static and artifact-observed Agent Learning Kit capabilities."""

    static = static_capabilities()
    observed = observed_capabilities(artifacts)
    capabilities = _merge_capabilities(static, observed)
    required = _normalize_requirements(required_capabilities or {})
    missing = _missing_required_capabilities(required, capabilities)
    findings = _capability_findings(missing)
    payload = {
        "schema_version": AGENT_LEARNING_CLI_SCHEMA_VERSION,
        "kind": AGENT_LEARNING_CAPABILITIES_KIND,
        "name": str(name or "agent-learning-capabilities"),
        "status": "passed" if not findings else "failed",
        "exit_code": 0 if not findings else 1,
        "source_paths": [str(path) for path in source_paths],
        "static_capabilities": static,
        "observed_capabilities": observed,
        "capabilities": capabilities,
        "consolidation": _consolidation_metadata(),
        "provider_capabilities": provider_capabilities(),
        "research_sources": copy.deepcopy(RESEARCH_SOURCES),
        "summary": {
            "artifact_count": len(artifacts),
            "capability_counts": {
                key: len(value)
                for key, value in capabilities.items()
                if isinstance(value, list)
            },
            "required_capabilities": required,
            "missing_required_capabilities": missing,
            "capability_gate_passed": not findings,
        },
        "findings": findings,
    }
    return payload


def static_capabilities() -> dict[str, list[str]]:
    """Return capabilities supported by the installed SDK, independent of a run."""

    from fi.alk import simulate, trinity

    provider_caps = provider_capabilities()
    provider_values = {
        value for values in provider_caps.values() for value in values
    }
    consolidation = trinity.consolidation_metadata()
    return {
        "channels": sorted(provider_values),
        "command_policies": sorted(
            {
                "agent_learn_only",
                "legacy_commands_rejected",
                "no_legacy_distribution_dependency",
                "shared_agent_learning_api_key",
                "unified_public_boundary",
            }
        ),
        "commands": sorted(COMMANDS),
        "environment_state_keys": sorted(
            {
                "agent_control_plane",
                "agent_integration_manifest",
                "browser",
                "framework_capability_matrix",
                "framework_runtime",
                "framework_trace",
                "multi_agent",
                "observability_replay_pack",
                "optimizer_society_trace",
                "optimizer_trace",
                "red_team_attack_evolution",
                "red_team_campaign",
                "streaming_trace",
                "voice",
                "workspace_run_manifest",
                "world_contract",
            }
        ),
        "environment_types": sorted(simulate.supported_manifest_environment_types()),
        "frameworks": sorted(simulate.supported_frameworks()),
        "metrics": sorted(DEFAULT_METRICS),
        "modalities": sorted({"chat", "image", "text", "voice", "webrtc"}),
        "providers": sorted({*provider_caps, "artifact", "futureagi", "local"}),
        "result_kinds": sorted(RESULT_KINDS),
        "search_paths": sorted(
            {
                "agent",
                "jobs.0",
                "optimizer.backend_portfolio.backends",
                "simulation.environments",
            }
        ),
        "sdk_boundaries": sorted(
            {
                _capability_key(consolidation["public_package"]),
                _capability_key(consolidation["public_import"]),
                _capability_key(consolidation["public_cli"]),
                "public_console_script_agent_learn",
                "public_import_agent_learning",
                "public_package_agent_learning_kit",
                "vendored_engine_modules",
            }
        ),
    }


def provider_capabilities() -> dict[str, list[str]]:
    from fi.alk import simulate

    return {
        _capability_key(provider): sorted(
            _capability_key(value) for value in values if _capability_key(value)
        )
        for provider, values in dict(
            simulate.AGENT_INTEGRATION_PROVIDER_CAPABILITIES
        ).items()
    }


def observed_capabilities(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    caps = _empty_capability_sets()
    for artifact in artifacts:
        normalized = normalize_public_payload(artifact)
        if not isinstance(normalized, Mapping):
            continue
        _add_capability(caps, "result_kinds", normalized.get("kind"))
        _collect_payload_capabilities(normalized, caps)
    return _freeze_capabilities(caps)


def render_markdown(catalog: Mapping[str, Any]) -> str:
    summary = dict(catalog.get("summary") or {})
    capabilities = dict(catalog.get("capabilities") or {})
    rows = [
        "| Capability | Count | Examples |",
        "| --- | ---: | --- |",
    ]
    for key in sorted(capabilities):
        values = list(capabilities.get(key) or [])
        rows.append(
            "| "
            + " | ".join(
                [
                    _md_cell(key),
                    str(len(values)),
                    _md_cell(", ".join(str(item) for item in values[:12])),
                ]
            )
            + " |"
        )
    missing = dict(summary.get("missing_required_capabilities") or {})
    lines = [
        f"# {_md_text(catalog.get('name') or 'agent-learning-capabilities')}",
        "",
        f"- Status: {_md_text(catalog.get('status') or 'unknown')}",
        f"- Artifact count: {_md_text(summary.get('artifact_count') or 0)}",
        f"- Capability gate: {_md_text(summary.get('capability_gate_passed'))}",
        "",
        "## Capabilities",
        "",
        *rows,
        "",
    ]
    if missing:
        lines.extend(["## Missing Required Capabilities", ""])
        for key, values in sorted(missing.items()):
            lines.append(f"- `{_md_text(key)}`: {_md_text(', '.join(values))}")
        lines.append("")
    return "\n".join(lines)


def _merge_capabilities(
    *items: Mapping[str, Sequence[Any]],
) -> dict[str, list[str]]:
    caps = _empty_capability_sets()
    for item in items:
        for key, values in item.items():
            _add_capabilities(caps, key, values)
    return _freeze_capabilities(caps)


def _normalize_requirements(
    requirements: Mapping[str, Sequence[Any]],
) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for key, values in requirements.items():
        normalized_key = _capability_key(key)
        if not normalized_key:
            continue
        normalized_values = sorted(
            {
                _capability_key(value)
                for value in _as_list(values)
                if _capability_key(value)
            }
        )
        if normalized_values:
            normalized[normalized_key] = normalized_values
    return normalized


def _missing_required_capabilities(
    required: Mapping[str, Sequence[str]],
    observed: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for key, values in required.items():
        observed_values = {_capability_key(value) for value in _as_list(observed.get(key))}
        missing_values = sorted(
            {
                _capability_key(value)
                for value in _as_list(values)
                if _capability_key(value) and _capability_key(value) not in observed_values
            }
        )
        if missing_values:
            missing[key] = missing_values
    return missing


def _capability_findings(
    missing: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for capability, values in sorted(missing.items()):
        if not values:
            continue
        findings.append(
            {
                "type": "agent_learning_capability_missing",
                "level": "error",
                "capability": capability,
                "missing": list(values),
                "reason": (
                    f"Missing required Agent Learning Kit capability "
                    f"`{capability}`: {', '.join(values)}."
                ),
            }
        )
    return findings


def _collect_payload_capabilities(
    value: Any,
    caps: dict[str, set[str]],
    *,
    depth: int = 0,
) -> None:
    if depth > 12:
        return
    if isinstance(value, Mapping):
        item = dict(value)
        summary = dict(item.get("summary") or {})
        _add_capabilities(caps, "search_paths", summary.get("search_paths"))
        _add_capabilities(caps, "providers", summary.get("observed_providers"))
        _add_capabilities(caps, "providers", summary.get("required_providers"))
        _add_capabilities(caps, "channels", summary.get("observed_channels"))
        _add_capabilities(caps, "channels", summary.get("required_channels"))
        _add_capabilities(caps, "frameworks", summary.get("frameworks"))
        _add_capabilities(caps, "frameworks", summary.get("observed_frameworks"))
        _add_capabilities(caps, "frameworks", summary.get("trace_frameworks"))
        _add_capabilities(caps, "metrics", summary.get("observed_metrics"))
        _add_capabilities(caps, "metrics", summary.get("required_metrics"))
        _add_capabilities(caps, "metrics", summary.get("eval_metrics"))
        _add_capabilities(caps, "metrics", _mapping_keys(summary.get("metric_averages")))
        _add_capabilities(caps, "environment_state_keys", summary.get("environment_state_keys"))
        provider_channels = dict(summary.get("provider_channels") or {})
        _add_capabilities(caps, "providers", provider_channels.keys())
        for channels in provider_channels.values():
            _add_capabilities(caps, "channels", channels)

        metadata = dict(item.get("metadata") or {})
        environment_state = dict(metadata.get("environment_state") or {})
        _add_capabilities(caps, "environment_state_keys", environment_state.keys())

        _add_capability(caps, "providers", item.get("provider"))
        _add_capability(caps, "providers", item.get("provider_id"))
        _add_capability(caps, "providers", item.get("provider_type"))
        _add_capability(caps, "frameworks", item.get("framework"))
        _add_capability(caps, "channels", item.get("channel"))
        _add_capability(caps, "channels", item.get("modality"))
        _add_capability(caps, "modalities", item.get("modality"))
        _add_capability(caps, "environment_types", item.get("type"))
        _add_capabilities(caps, "metrics", _mapping_keys(item.get("metrics")))
        for metric in _as_list(item.get("metrics")):
            if isinstance(metric, Mapping):
                _add_capability(caps, "metrics", metric.get("name"))

        optimization = dict(item.get("optimization") or {})
        best_config = dict(optimization.get("best_config") or {})
        simulation = dict(best_config.get("simulation") or {})
        for environment in _as_list(simulation.get("environments")):
            if isinstance(environment, Mapping):
                _add_capability(caps, "environment_types", environment.get("type"))
        for history_item in _as_list(optimization.get("history")):
            if isinstance(history_item, Mapping):
                _add_capabilities(caps, "metrics", _mapping_keys(history_item.get("metrics")))

        for child in item.values():
            _collect_payload_capabilities(child, caps, depth=depth + 1)
    elif isinstance(value, list):
        for child in value:
            _collect_payload_capabilities(child, caps, depth=depth + 1)


def _empty_capability_sets() -> dict[str, set[str]]:
    return {
        "channels": set(),
        "command_policies": set(),
        "commands": set(),
        "environment_state_keys": set(),
        "environment_types": set(),
        "frameworks": set(),
        "metrics": set(),
        "modalities": set(),
        "providers": set(),
        "result_kinds": set(),
        "search_paths": set(),
        "sdk_boundaries": set(),
    }


def _freeze_capabilities(caps: Mapping[str, set[str]]) -> dict[str, list[str]]:
    return {key: sorted(values) for key, values in caps.items()}


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
    normalized = _capability_key(value)
    if normalized and key in caps:
        caps[key].add(normalized)


def _capability_key(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return [value]


def _mapping_keys(value: Any) -> list[Any]:
    return list(value.keys()) if isinstance(value, Mapping) else []


def _consolidation_metadata() -> dict[str, Any]:
    from fi.alk import trinity

    return trinity.consolidation_metadata()


def _md_text(value: Any) -> str:
    return str(value).replace("\n", " ")


def _md_cell(value: Any) -> str:
    text = _md_text(value if value is not None else "").replace("|", "\\|")
    return text if len(text) <= 180 else f"{text[:177]}..."


__all__ = [
    "AGENT_LEARNING_CAPABILITIES_KIND",
    "capability_catalog",
    "observed_capabilities",
    "provider_capabilities",
    "render_markdown",
    "static_capabilities",
]
