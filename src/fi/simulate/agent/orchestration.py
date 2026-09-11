from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import urlparse

from fi.simulate.environment import (
    AgentMemoryLineageEnvironment,
    FrameworkTraceEnvironment,
    MultiAgentRoomEnvironment,
    RetrievalMemoryEnvironment,
    WorldContractEnvironment,
    WorldOrchestrationReplayEnvironment,
)


DEFAULT_ORCHESTRATION_PROBE_TOOLS = (
    "apply_world_transition",
    "framework_trace_status",
    "retrieve_documents",
    "read_document",
    "cite_sources",
    "agent_memory_lineage_status",
    "retrieval_memory_status",
    "room_status",
    "request_review",
    "reconcile",
)


def orchestration_stack_contract(
    *,
    target: str | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    external_sources: Sequence[str] = (),
    environment_types: Sequence[str] = (),
) -> dict[str, Any]:
    """Return an import-free local contract for a whole orchestration stack."""

    target_scheme = urlparse(str(target or "")).scheme.lower()
    external_source_list = _unique_strings(external_sources)
    requires_external = target_scheme in {"http", "https"} or bool(external_source_list)
    return {
        "kind": "agent-learning.orchestration-stack-contract.v1",
        "runtime": "in_process",
        "target": str(target) if target else "",
        "target_scheme": target_scheme,
        "requires_external_service": requires_external,
        "local_executable_fixture": not requires_external,
        "environment_types": _unique_strings(environment_types),
        "external_sources": external_source_list,
        "evidence_requirements": [
            "world_contract",
            "world_transition",
            "framework_trace",
            "retrieval_memory",
            "current_source_citation",
            "agent_memory_lineage",
            "memory_governance",
            "multi_agent_room",
            "critic_review",
            "reconciliation",
            "tool_execution",
            "trace_artifact",
        ],
        "metadata": _plain_mapping(metadata),
    }


def run_orchestration_stack_probe(
    stack: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility alias for the synchronous orchestration stack probe."""

    return probe_orchestration_stack(stack=stack, **kwargs)


def probe_orchestration_stack(
    *,
    stack: Mapping[str, Any],
    agent: Optional[Mapping[str, Any]] = None,
    target: str | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    allow_external_target: bool = False,
    expected_transition: str = "approve_refund",
    expected_state: Optional[Mapping[str, Any]] = None,
    expected_document_id: str = "doc_refund_2026",
    expected_roles: Sequence[str] = ("planner", "retriever", "critic"),
    expected_review_target: str = "refund",
    expected_reconciliation: str = "approved refund",
    required_tools: Sequence[str] = DEFAULT_ORCHESTRATION_PROBE_TOOLS,
) -> dict[str, Any]:
    """Probe local world/framework/retrieval/memory/multi-agent stack evidence."""

    if target and _is_external_target(target) and not allow_external_target:
        raise ValueError(
            "external targets are disabled for orchestration stack probes; "
            "set allow_external_target=True only when the user explicitly "
            "wants to test that live workload"
        )
    stack_data = _orchestration_stack_data(stack)
    external_sources = _external_sources(stack_data)
    if external_sources and not allow_external_target:
        raise ValueError(
            "external export sources are disabled for orchestration stack probes; "
            "set allow_external_target=True only when the user explicitly "
            "wants to test live exports"
        )
    contract = orchestration_stack_contract(
        target=target,
        metadata=metadata,
        external_sources=external_sources,
        environment_types=[item["type"] for item in stack_data["environments"]],
    )

    environments = _stack_environments(stack_data["environments"])
    for _environment_type, environment, _data in environments:
        environment.reset()

    active_agent = agent or _default_orchestration_probe_agent(
        expected_transition=expected_transition,
        expected_document_id=expected_document_id,
        expected_review_target=expected_review_target,
        expected_reconciliation=expected_reconciliation,
    )
    tool_calls = _agent_tool_calls(active_agent)
    handled_tool_calls = 0
    successful_tool_calls = 0
    failed_tool_calls = 0
    observed_tool_names: list[str] = []
    handled_tool_names: list[str] = []

    for turn_index, tool_call in enumerate(tool_calls, start=1):
        name = str(tool_call.get("name") or "")
        if name:
            observed_tool_names.append(name)
        handled = False
        success = False
        for _environment_type, environment, _data in environments:
            result = environment.handle_tool_call(tool_call, turn_index=turn_index)
            if result is None:
                continue
            handled = True
            success = success or bool(getattr(result, "success", True))
        if handled:
            handled_tool_calls += 1
            handled_tool_names.append(name)
            if success:
                successful_tool_calls += 1
            else:
                failed_tool_calls += 1

    state = _environment_state(environments)
    summary = _orchestration_probe_summary(
        state,
        stack_data,
        contract=contract,
        tool_calls=tool_calls,
        handled_tool_calls=handled_tool_calls,
        successful_tool_calls=successful_tool_calls,
        failed_tool_calls=failed_tool_calls,
        observed_tool_names=observed_tool_names,
        handled_tool_names=handled_tool_names,
        expected_transition=expected_transition,
        expected_state=expected_state or {"refund.status": "approved"},
        expected_document_id=expected_document_id,
        expected_roles=expected_roles,
        expected_review_target=expected_review_target,
        expected_reconciliation=expected_reconciliation,
        required_tools=required_tools,
    )
    findings = _orchestration_probe_findings(summary, contract=contract)
    summary["finding_count"] = len(findings)
    summary["passed_case_count"] = 1 if not findings else 0
    summary["failed_case_count"] = 0 if not findings else 1
    status = "passed" if not findings else "failed"
    return {
        "kind": "agent-learning.orchestration-stack-probe.v1",
        "status": status,
        "passed": status == "passed",
        "requires_external_service": bool(contract["requires_external_service"]),
        "allow_external_target": bool(allow_external_target),
        "contract": contract,
        "summary": summary,
        "stack": stack_data,
        "environments": copy.deepcopy(stack_data["environments"]),
        "state": state,
        "findings": findings,
        "metadata": {
            "source": "fi.simulate.agent.orchestration.probe_orchestration_stack",
            **_plain_mapping(metadata),
        },
    }


_ORCHESTRATION_ENVIRONMENT_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("world_orchestration_replay", "world_replay", "world_orchestration"),
        "world_orchestration_replay",
    ),
    (("world_contract", "world"), "world_contract"),
    (("framework_trace", "framework"), "framework_trace"),
    (("retrieval_memory", "retrieval"), "retrieval_memory"),
    (
        ("agent_memory_lineage", "memory_lineage", "lineage"),
        "agent_memory_lineage",
    ),
    (("multi_agent_room", "room", "multi_agent"), "multi_agent_room"),
)


def _orchestration_stack_data(stack: Mapping[str, Any]) -> dict[str, Any]:
    source = copy.deepcopy(dict(stack or {}))
    explicit_environments = source.pop("environments", None)
    metadata = _plain_mapping(source.pop("metadata", None))
    name = str(source.pop("name", source.pop("id", "")) or "")
    source.pop("description", None)
    source.pop("target", None)
    source.pop("allow_external_target", None)
    if explicit_environments is not None:
        environments = _environment_list(explicit_environments)
    else:
        environments = []
        for aliases, environment_type in _ORCHESTRATION_ENVIRONMENT_ALIASES:
            data = _pop_first(source, aliases)
            if data is not None:
                environments.append(_typed_environment(environment_type, data))
    if source:
        raise ValueError(
            "orchestration stack has unsupported key(s): "
            f"{', '.join(sorted(source))}"
        )
    if not environments:
        raise ValueError("orchestration stack must define at least one environment")
    return {
        "name": name,
        "metadata": metadata,
        "environments": environments,
        **{
            item["type"]: copy.deepcopy(item.get("data", {}))
            for item in environments
            if item.get("type")
        },
    }


def _environment_list(environments: Any) -> list[dict[str, Any]]:
    if isinstance(environments, Mapping):
        environments = [environments]
    if isinstance(environments, (str, bytes)) or environments is None:
        raise ValueError("environments must be a mapping or sequence of mappings")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(environments, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"environment {index} must be a mapping")
        item = copy.deepcopy(dict(raw))
        environment_type = _scope_key(item.get("type"))
        if not environment_type:
            raise ValueError(f"environment {index} requires type")
        if item.get("data") is None:
            data = {
                key: value
                for key, value in item.items()
                if key not in {"type", "kind", "name", "description"}
            }
        else:
            data = item["data"]
        result.append(_typed_environment(environment_type, data))
    return result


def _typed_environment(environment_type: str, data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{environment_type} candidate data must be a mapping")
    return {"type": _scope_key(environment_type), "data": copy.deepcopy(dict(data))}


def _stack_environments(
    environments: Sequence[Mapping[str, Any]],
) -> list[tuple[str, Any, dict[str, Any]]]:
    result: list[tuple[str, Any, dict[str, Any]]] = []
    for item in environments:
        environment_type = _scope_key(item.get("type"))
        data = _plain_mapping(item.get("data"))
        if environment_type == "world_contract":
            result.append((environment_type, WorldContractEnvironment(**data), data))
        elif environment_type == "world_orchestration_replay":
            result.append((environment_type, WorldOrchestrationReplayEnvironment(**data), data))
        elif environment_type == "framework_trace":
            source = dict(data)
            framework = str(source.pop("framework", source.pop("provider", "traceai")))
            result.append(
                (
                    environment_type,
                    FrameworkTraceEnvironment(framework=framework, **source),
                    data,
                )
            )
        elif environment_type == "retrieval_memory":
            source = dict(data)
            documents = source.pop("documents", source.pop("docs", []))
            result.append(
                (
                    environment_type,
                    RetrievalMemoryEnvironment(
                        documents,
                        memory=_plain_mapping(source.pop("memory", None)),
                        top_k=_as_int(source.pop("top_k", 3)) or 3,
                        require_current=bool(source.pop("require_current", True)),
                        metadata=_plain_mapping(source.pop("metadata", None)),
                    ),
                    data,
                )
            )
        elif environment_type == "agent_memory_lineage":
            result.append(
                (
                    environment_type,
                    AgentMemoryLineageEnvironment(data),
                    data,
                )
            )
        elif environment_type == "multi_agent_room":
            source = dict(data)
            participants = (
                source.pop("participants", None)
                or source.pop("agents", None)
                or source.pop("roles", None)
            )
            result.append(
                (
                    environment_type,
                    MultiAgentRoomEnvironment(
                        participants,
                        handoff_contracts=source.pop("handoff_contracts", None),
                        expected_handoffs=source.pop("expected_handoffs", None),
                        expected_reviews=source.pop("expected_reviews", None),
                        expected_reconciliation=source.pop("expected_reconciliation", None),
                        messages=source.pop("messages", None),
                        handoffs=source.pop("handoffs", None),
                        reviews=source.pop("reviews", None),
                        reconciliations=source.pop("reconciliations", None),
                        state=_plain_mapping(source.pop("state", None)),
                        allow_unknown_roles=bool(source.pop("allow_unknown_roles", True)),
                        extra_trace=_plain_mapping(source.pop("extra_trace", None)),
                    ),
                    data,
                )
            )
    return result


def _environment_state(environments: Sequence[tuple[str, Any, dict[str, Any]]]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for environment_type, environment, _data in environments:
        payload_factory = getattr(environment, "_state_payload", None)
        if not callable(payload_factory):
            payload_factory = getattr(environment, "_trace_payload", None)
        if not callable(payload_factory):
            payload_factory = getattr(environment, "_payload", None)
        payload = payload_factory() if callable(payload_factory) else {}
        state[_state_key(environment_type)] = copy.deepcopy(payload)
    return state


def _state_key(environment_type: str) -> str:
    if environment_type == "multi_agent_room":
        return "multi_agent"
    return environment_type


def _orchestration_probe_summary(
    state: Mapping[str, Any],
    stack_data: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    tool_calls: Sequence[Mapping[str, Any]],
    handled_tool_calls: int,
    successful_tool_calls: int,
    failed_tool_calls: int,
    observed_tool_names: Sequence[str],
    handled_tool_names: Sequence[str],
    expected_transition: str,
    expected_state: Mapping[str, Any],
    expected_document_id: str,
    expected_roles: Sequence[str],
    expected_review_target: str,
    expected_reconciliation: str,
    required_tools: Sequence[str],
) -> dict[str, Any]:
    world = _plain_mapping(state.get("world_contract"))
    framework = _plain_mapping(state.get("framework_trace"))
    retrieval = _plain_mapping(state.get("retrieval_memory"))
    lineage = _plain_mapping(state.get("agent_memory_lineage"))
    room = _plain_mapping(state.get("multi_agent"))
    lineage_summary = _plain_mapping(lineage.get("summary"))
    room_state = _plain_mapping(room.get("state"))
    case_state = _plain_mapping(room_state.get("case"))

    transition_log = [_plain_mapping(item) for item in _plain_list(world.get("transition_log"))]
    completed_transitions = [
        item for item in transition_log if _scope_key(item.get("status")) == "success"
    ]
    framework_spans = [_plain_mapping(item) for item in _plain_list(framework.get("spans"))]
    framework_events = [_plain_mapping(item) for item in _plain_list(framework.get("events"))]
    framework_signals = set(_unique_strings(framework.get("signals")))
    framework_required = _unique_strings(
        _plain_mapping(stack_data.get("framework_trace")).get("adapter_required_signals")
    )
    documents = [_plain_mapping(item) for item in _plain_list(retrieval.get("documents"))]
    current_doc_ids = {
        str(item.get("id") or "")
        for item in documents
        if item.get("current") is True and str(item.get("id") or "")
    }
    citations = [_plain_mapping(item) for item in _plain_list(retrieval.get("citations"))]
    cited_doc_ids = {
        str(doc_id)
        for citation in citations
        for doc_id in _plain_list(citation.get("doc_ids"))
        if str(doc_id or "")
    }
    required_operations = {"read", "write", "recall"}
    operation_types = set(_plain_list(lineage_summary.get("operation_types")))
    participants = _unique_strings(room.get("participants"))
    reviews = [_plain_mapping(item) for item in _plain_list(room.get("reviews"))]
    reconciliations = [
        _plain_mapping(item) for item in _plain_list(room.get("reconciliations"))
    ]
    required_tool_names = _unique_strings(required_tools)
    observed_tool_name_set = set(_unique_strings(observed_tool_names))
    handled_tool_name_set = set(_unique_strings(handled_tool_names))
    required_roles = set(_unique_strings(expected_roles))
    participant_set = set(participants)
    terminal_status = _scope_key(case_state.get("status") or room_state.get("status"))
    expected_review_present = any(
        _scope_key(item.get("reviewer")) == _scope_key("critic")
        and _scope_key(expected_review_target) in _scope_key(item.get("target"))
        for item in reviews
    )
    expected_reconciliation_present = any(
        _scope_key(expected_reconciliation) in _scope_key(
            item.get("summary") or item.get("decision")
        )
        for item in reconciliations
    )
    return {
        "case_count": 1,
        "passed_case_count": 0,
        "failed_case_count": 1,
        "finding_count": 0,
        "environment_types": _unique_strings(
            [item["type"] for item in stack_data["environments"]]
        ),
        "requires_external_service": bool(contract.get("requires_external_service")),
        "local_executable_fixture": bool(contract.get("local_executable_fixture")),
        "world_present": bool(world),
        "world_transition_count": len(_plain_list(world.get("transitions"))),
        "world_completed_transition_count": len(completed_transitions),
        "expected_transition": str(expected_transition),
        "expected_transition_completed": any(
            _scope_key(item.get("id")) == _scope_key(expected_transition)
            for item in completed_transitions
        ),
        "world_state_match": _state_matches(world.get("state"), expected_state),
        "world_terminal_success": any(
            item.get("pass") is True
            for item in _plain_list(world.get("success_results"))
        ),
        "framework_present": bool(framework),
        "framework": str(framework.get("framework") or ""),
        "framework_span_count": len(framework_spans),
        "framework_event_count": len(framework_events),
        "framework_signal_count": len(framework_signals),
        "framework_required_signal_count": len(framework_required),
        "framework_required_signal_match_count": len(
            set(framework_required) & framework_signals
        ),
        "framework_tool_signal_present": "tool" in framework_signals
        or any(_plain_list(item.get("tool_calls")) for item in framework_spans),
        "retrieval_present": bool(retrieval),
        "retrieval_document_count": len(documents),
        "retrieval_current_document_count": len(current_doc_ids),
        "retrieval_citation_count": len(citations),
        "retrieval_cited_document_count": len(cited_doc_ids),
        "retrieval_citations_current": bool(cited_doc_ids)
        and cited_doc_ids.issubset(current_doc_ids),
        "retrieval_expected_document_id": str(expected_document_id),
        "retrieval_expected_document_cited": str(expected_document_id) in cited_doc_ids,
        "retrieval_freshness_checked_count": sum(
            1 for citation in citations if citation.get("freshness_checked") is True
        ),
        "memory_present": bool(lineage),
        "memory_store_count": _as_int(lineage_summary.get("store_count")),
        "memory_record_count": _as_int(lineage_summary.get("memory_count")),
        "memory_operation_count": _as_int(lineage_summary.get("operation_count")),
        "memory_audited_operation_count": _as_int(
            lineage_summary.get("audited_operation_count")
        ),
        "memory_required_operations_present": required_operations.issubset(
            operation_types
        ),
        "memory_operation_types": sorted(operation_types),
        "has_source_attribution": bool(lineage_summary.get("has_source_attribution")),
        "has_tenant_isolation": bool(lineage_summary.get("has_tenant_isolation")),
        "has_audit": bool(lineage_summary.get("has_audit")),
        "has_retention_policy": bool(lineage_summary.get("has_retention_policy")),
        "has_deletion_policy": bool(lineage_summary.get("has_deletion_policy")),
        "has_redaction": bool(lineage_summary.get("has_redaction")),
        "has_canaries": bool(lineage_summary.get("has_canaries")),
        "has_observability": bool(lineage_summary.get("has_observability")),
        "has_artifacts": bool(lineage_summary.get("has_artifacts")),
        "policy_violation_count": _as_int(lineage_summary.get("policy_violation_count")),
        "open_poisoning_count": _as_int(lineage_summary.get("open_poisoning_count")),
        "isolation_violation_count": _as_int(
            lineage_summary.get("isolation_violation_count")
        ),
        "retention_violation_count": _as_int(
            lineage_summary.get("retention_violation_count")
        ),
        "blocking_gap_count": _as_int(lineage_summary.get("blocking_gap_count")),
        "room_present": bool(room),
        "participant_count": len(participants),
        "participants": participants,
        "required_roles": sorted(required_roles),
        "role_match": required_roles.issubset(participant_set),
        "allow_unknown_roles": bool(
            _plain_mapping(stack_data.get("multi_agent_room")).get("allow_unknown_roles", True)
        ),
        "review_count": len(reviews),
        "reconciliation_count": len(reconciliations),
        "expected_review_present": expected_review_present,
        "expected_reconciliation_present": expected_reconciliation_present,
        "reconciliation_conflict_count": sum(
            len(_plain_list(item.get("conflicts"))) for item in reconciliations
        ),
        "terminal_room_state": terminal_status in {
            "approved",
            "complete",
            "completed",
            "closed",
            "done",
            "resolved",
        },
        "terminal_status": terminal_status,
        "tool_call_count": len(tool_calls),
        "handled_tool_call_count": int(handled_tool_calls),
        "successful_tool_call_count": int(successful_tool_calls),
        "failed_tool_call_count": int(failed_tool_calls),
        "observed_tool_names": _unique_strings(observed_tool_names),
        "handled_tool_names": _unique_strings(handled_tool_names),
        "required_tool_count": len(required_tool_names),
        "required_tools_present": set(required_tool_names).issubset(
            observed_tool_name_set
        ),
        "required_tools_handled": set(required_tool_names).issubset(
            handled_tool_name_set
        ),
    }


def _orchestration_probe_findings(
    summary: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    _append_finding(
        findings,
        "orchestration_probe_local_contract",
        bool(summary.get("local_executable_fixture"))
        and not bool(summary.get("requires_external_service")),
        "orchestration probe target must be local and no-external-service",
        {"contract": dict(contract)},
    )
    _append_finding(
        findings,
        "orchestration_probe_environment_bundle",
        all(
            bool(summary.get(key))
            for key in (
                "world_present",
                "framework_present",
                "retrieval_present",
                "memory_present",
                "room_present",
            )
        ),
        "stack must include world, framework, retrieval, memory lineage, and room evidence",
        summary,
    )
    _append_finding(
        findings,
        "orchestration_probe_world_transition",
        summary.get("expected_transition_completed") is True
        and summary.get("world_state_match") is True
        and summary.get("world_terminal_success") is True,
        "world contract must complete the expected transition and terminal state",
        summary,
    )
    _append_finding(
        findings,
        "orchestration_probe_framework_trace",
        _as_int(summary.get("framework_span_count")) > 0
        and _as_int(summary.get("framework_required_signal_match_count"))
        >= _as_int(summary.get("framework_required_signal_count"))
        and summary.get("framework_tool_signal_present") is True,
        "framework trace must include spans, required signals, and tool evidence",
        summary,
    )
    _append_finding(
        findings,
        "orchestration_probe_retrieval_grounding",
        _as_int(summary.get("retrieval_current_document_count")) > 0
        and _as_int(summary.get("retrieval_citation_count")) > 0
        and summary.get("retrieval_citations_current") is True
        and summary.get("retrieval_expected_document_cited") is True
        and _as_int(summary.get("retrieval_freshness_checked_count"))
        >= _as_int(summary.get("retrieval_citation_count")),
        "retrieval must cite the expected current document with freshness checks",
        summary,
    )
    _append_finding(
        findings,
        "orchestration_probe_memory_lineage_governance",
        _as_int(summary.get("memory_record_count")) > 0
        and summary.get("memory_required_operations_present") is True
        and _as_int(summary.get("memory_audited_operation_count"))
        >= _as_int(summary.get("memory_operation_count"))
        and summary.get("has_source_attribution") is True
        and all(
            summary.get(key) is True
            for key in (
                "has_tenant_isolation",
                "has_audit",
                "has_retention_policy",
                "has_deletion_policy",
                "has_redaction",
                "has_canaries",
                "has_observability",
                "has_artifacts",
            )
        )
        and _as_int(summary.get("policy_violation_count")) == 0
        and _as_int(summary.get("open_poisoning_count")) == 0
        and _as_int(summary.get("isolation_violation_count")) == 0
        and _as_int(summary.get("retention_violation_count")) == 0
        and _as_int(summary.get("blocking_gap_count")) == 0,
        "memory lineage must close source attribution and governance checks",
        summary,
    )
    _append_finding(
        findings,
        "orchestration_probe_multi_agent_coordination",
        summary.get("role_match") is True
        and summary.get("allow_unknown_roles") is False
        and _as_int(summary.get("review_count")) > 0
        and _as_int(summary.get("reconciliation_count")) > 0
        and summary.get("expected_review_present") is True
        and summary.get("expected_reconciliation_present") is True
        and _as_int(summary.get("reconciliation_conflict_count")) == 0
        and summary.get("terminal_room_state") is True,
        "multi-agent room must close roles, review, reconciliation, and terminal state",
        summary,
    )
    _append_finding(
        findings,
        "orchestration_probe_tool_evidence",
        _as_int(summary.get("tool_call_count")) > 0
        and summary.get("required_tools_present") is True
        and summary.get("required_tools_handled") is True
        and _as_int(summary.get("successful_tool_call_count"))
        >= _as_int(summary.get("tool_call_count"))
        and _as_int(summary.get("failed_tool_call_count")) == 0,
        "agent must execute and successfully handle all required orchestration tools",
        summary,
    )
    return findings


def _default_orchestration_probe_agent(
    *,
    expected_transition: str,
    expected_document_id: str,
    expected_review_target: str,
    expected_reconciliation: str,
) -> dict[str, Any]:
    return {
        "type": "scripted",
        "responses": [
            {
                "content": "Inspecting world and framework orchestration evidence.",
                "tool_calls": [
                    {
                        "id": "world_transition",
                        "name": "apply_world_transition",
                        "arguments": {"id": expected_transition},
                    },
                    {
                        "id": "framework_status",
                        "name": "framework_trace_status",
                        "arguments": {},
                    },
                ],
            },
            {
                "content": "Inspecting retrieval and memory lineage evidence.",
                "tool_calls": [
                    {
                        "id": "retrieve_current_policy",
                        "name": "retrieve_documents",
                        "arguments": {"query": "current refund policy"},
                    },
                    {
                        "id": "read_current_policy",
                        "name": "read_document",
                        "arguments": {"id": expected_document_id},
                    },
                    {
                        "id": "cite_current_policy",
                        "name": "cite_sources",
                        "arguments": {
                            "doc_ids": [expected_document_id],
                            "claim": "Current policy supports the orchestration decision.",
                            "freshness_checked": True,
                        },
                    },
                    {
                        "id": "memory_lineage",
                        "name": "agent_memory_lineage_status",
                        "arguments": {},
                    },
                    {
                        "id": "retrieval_memory",
                        "name": "retrieval_memory_status",
                        "arguments": {},
                    },
                ],
            },
            {
                "content": "Inspecting review and reconciliation evidence.",
                "tool_calls": [
                    {
                        "id": "room_status",
                        "name": "room_status",
                        "arguments": {},
                    },
                    {
                        "id": "critic_review",
                        "name": "request_review",
                        "arguments": {
                            "reviewer": "critic",
                            "target": expected_review_target,
                            "criteria": ["policy", "memory", "world"],
                        },
                    },
                    {
                        "id": "reconcile",
                        "name": "reconcile",
                        "arguments": {
                            "summary": expected_reconciliation,
                            "accepted_source": "critic",
                            "conflicts": [],
                            "participants": ["planner", "retriever", "critic"],
                        },
                    },
                ],
            },
        ],
    }


def _agent_tool_calls(agent: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not agent:
        return []
    calls: list[dict[str, Any]] = []
    for response in _plain_list(_plain_mapping(agent).get("responses")):
        for call in _plain_list(_plain_mapping(response).get("tool_calls")):
            item = _plain_mapping(call)
            if item:
                calls.append(item)
    return calls


def _state_matches(state: Any, expected: Mapping[str, Any]) -> bool:
    actual = _plain_mapping(state)
    if not expected:
        return True
    for key, value in expected.items():
        if "." in str(key):
            observed = _lookup_dotted(actual, str(key))
            if observed != value:
                return False
        elif isinstance(value, Mapping):
            if not _state_matches(_plain_mapping(actual.get(key)), value):
                return False
        elif actual.get(key) != value:
            return False
    return True


def _lookup_dotted(state: Mapping[str, Any], path: str) -> Any:
    current: Any = state
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _external_sources(value: Any) -> list[str]:
    sources: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = _scope_key(key)
            if key_text in {
                "export_source",
                "trace_source",
                "source",
                "source_url",
                "voice_export_source",
            } and _is_external_target(str(item)):
                sources.append(str(item))
            sources.extend(_external_sources(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            sources.extend(_external_sources(item))
    return _unique_strings(sources)


def _append_finding(
    findings: list[dict[str, Any]],
    check: str,
    passed: bool,
    message: str,
    evidence: Mapping[str, Any],
) -> None:
    if passed:
        return
    findings.append(
        {
            "check": check,
            "level": "error",
            "message": message,
            "evidence": dict(evidence),
        }
    )


def _pop_first(source: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in source:
            return source.pop(key)
    return None


def _plain_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _plain_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _scope_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _unique_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in _plain_list(values):
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_external_target(target: str) -> bool:
    return urlparse(str(target)).scheme.lower() in {"http", "https"}


__all__ = [
    "DEFAULT_ORCHESTRATION_PROBE_TOOLS",
    "orchestration_stack_contract",
    "probe_orchestration_stack",
    "run_orchestration_stack_probe",
]
