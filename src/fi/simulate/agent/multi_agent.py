from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import urlparse

from fi.simulate.environment import MultiAgentRoomEnvironment


def multi_agent_room_contract(
    *,
    target: str | None = None,
    participants: Mapping[str, Any] | Sequence[Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return an import-free local contract for a multi-agent room."""

    target_scheme = urlparse(str(target or "")).scheme.lower()
    participant_keys = _participant_keys(participants)
    return {
        "kind": "agent-learning.multi-agent-room-contract.v1",
        "runtime": "in_process",
        "target": str(target) if target else "",
        "target_scheme": target_scheme,
        "requires_external_service": False,
        "local_executable_fixture": target_scheme not in {"http", "https"},
        "participants": participant_keys,
        "min_participant_count": 2,
        "evidence_requirements": [
            "multi_agent_room",
            "role_boundary",
            "handoff_contract",
            "expected_handoff",
            "expected_review",
            "expected_reconciliation",
            "room_state",
            "trace_artifact",
        ],
        "metadata": _plain_mapping(metadata),
    }


def probe_multi_agent_room(
    *,
    participants: Mapping[str, Any] | Sequence[Any],
    room: Mapping[str, Any],
    agent: Optional[Mapping[str, Any]] = None,
    target: str | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    allow_external_target: bool = False,
) -> dict[str, Any]:
    """Probe local multi-agent room coordination evidence."""

    if target and _is_external_target(target) and not allow_external_target:
        raise ValueError(
            "external targets are disabled for multi-agent room probes; "
            "set allow_external_target=True only when the user explicitly "
            "wants to test that live workload"
        )
    room_data = _room_data(participants=participants, room=room)
    contract = multi_agent_room_contract(
        target=target,
        participants=room_data["participants"],
        metadata=metadata,
    )
    environment = MultiAgentRoomEnvironment(
        room_data["participants"],
        handoff_contracts=room_data.get("handoff_contracts"),
        expected_handoffs=room_data.get("expected_handoffs"),
        expected_reviews=room_data.get("expected_reviews"),
        expected_reconciliation=room_data.get("expected_reconciliation"),
        messages=room_data.get("messages"),
        handoffs=room_data.get("handoffs"),
        reviews=room_data.get("reviews"),
        reconciliations=room_data.get("reconciliations"),
        state=room_data.get("state"),
        allow_unknown_roles=bool(room_data.get("allow_unknown_roles", True)),
        extra_trace={
            **_plain_mapping(room_data.get("extra_trace")),
            "multi_agent_room_contract": contract,
        },
    )
    environment.reset()
    for tool_call in _agent_tool_calls(agent):
        environment.handle_tool_call(tool_call)
    room_state = environment._state_payload()
    findings = _multi_agent_probe_findings(room_state, room_data, contract=contract)
    summary = _multi_agent_probe_summary(
        room_state,
        room_data,
        finding_count=len(findings),
        contract=contract,
    )
    status = "passed" if not findings else "failed"
    return {
        "kind": "agent-learning.multi-agent-room-probe.v1",
        "status": status,
        "passed": status == "passed",
        "requires_external_service": bool(contract["requires_external_service"]),
        "allow_external_target": bool(allow_external_target),
        "contract": contract,
        "summary": summary,
        "room": room_data,
        "environment": {"type": "multi_agent_room", "data": room_data},
        "state": {"multi_agent": room_state},
        "findings": findings,
        "metadata": {
            "source": "fi.simulate.agent.multi_agent.probe_multi_agent_room",
            **_plain_mapping(metadata),
        },
    }


def run_multi_agent_room_probe(**kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for the synchronous multi-agent room probe."""

    return probe_multi_agent_room(**kwargs)


def _room_data(
    *,
    participants: Mapping[str, Any] | Sequence[Any],
    room: Mapping[str, Any],
) -> dict[str, Any]:
    room_data = copy.deepcopy(dict(room or {}))
    configured_participants = (
        room_data.pop("participants", None)
        or room_data.pop("agents", None)
        or room_data.pop("roles", None)
        or participants
    )
    room_data["participants"] = _copy_participants(configured_participants)
    return room_data


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


def _multi_agent_probe_summary(
    room_state: Mapping[str, Any],
    room_data: Mapping[str, Any],
    *,
    finding_count: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    participants = _unique_strings(room_state.get("participants"))
    handoffs = [_plain_mapping(item) for item in _plain_list(room_state.get("handoffs"))]
    reviews = [_plain_mapping(item) for item in _plain_list(room_state.get("reviews"))]
    reconciliations = [
        _plain_mapping(item) for item in _plain_list(room_state.get("reconciliations"))
    ]
    checks = [
        _plain_mapping(item)
        for item in _plain_list(room_state.get("coordination_checks"))
    ]
    matched_checks = [item for item in checks if item.get("match") is True]
    contract_statuses = [
        _plain_mapping(item.get("contract_status")) for item in handoffs
    ]
    known_handoffs = sum(1 for item in handoffs if item.get("known_role") is True)
    known_reviews = sum(1 for item in reviews if item.get("known_role") is True)
    case_status = _scope_key(
        _plain_mapping(_plain_mapping(room_state.get("state")).get("case")).get("status")
    )
    return {
        "case_count": 1,
        "passed_case_count": 1 if finding_count == 0 else 0,
        "failed_case_count": 0 if finding_count == 0 else 1,
        "finding_count": int(finding_count),
        "participant_count": len(participants),
        "participants": participants,
        "allow_unknown_roles": bool(room_data.get("allow_unknown_roles", True)),
        "handoff_count": len(handoffs),
        "known_handoff_count": known_handoffs,
        "review_count": len(reviews),
        "known_review_count": known_reviews,
        "reconciliation_count": len(reconciliations),
        "coordination_check_count": len(checks),
        "matched_coordination_check_count": len(matched_checks),
        "unmatched_coordination_check_count": len(checks) - len(matched_checks),
        "expected_handoff_count": len(_plain_list(room_state.get("expected_handoffs"))),
        "expected_review_count": len(_plain_list(room_state.get("expected_reviews"))),
        "expected_reconciliation_present": bool(
            _plain_mapping(room_state.get("expected_reconciliation"))
        ),
        "handoff_contract_count": len(_plain_mapping(room_state.get("handoff_contracts"))),
        "handoff_contract_matched_count": sum(
            1 for item in contract_statuses if item.get("matched") is True
        ),
        "reconciliation_conflict_count": sum(
            len(_plain_list(item.get("conflicts"))) for item in reconciliations
        ),
        "terminal_state": case_status not in {"", "triage", "open", "pending"},
        "case_status": case_status,
        "requires_external_service": bool(contract.get("requires_external_service")),
        "local_executable_fixture": bool(contract.get("local_executable_fixture")),
    }


def _multi_agent_probe_findings(
    room_state: Mapping[str, Any],
    room_data: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    summary = _multi_agent_probe_summary(
        room_state,
        room_data,
        finding_count=0,
        contract=contract,
    )
    _append_finding(
        findings,
        "multi_agent_probe_local_contract",
        bool(summary["local_executable_fixture"])
        and not bool(summary["requires_external_service"]),
        "multi-agent probe target must be local and no-external-service",
        {"contract": dict(contract)},
    )
    _append_finding(
        findings,
        "multi_agent_probe_role_boundary",
        summary["participant_count"] >= 2
        and summary["allow_unknown_roles"] is False
        and summary["known_handoff_count"] >= summary["handoff_count"]
        and summary["known_review_count"] >= summary["review_count"],
        "participants must be explicit and observed handoffs/reviews must target known roles",
        summary,
    )
    _append_finding(
        findings,
        "multi_agent_probe_handoff_contracts",
        summary["handoff_count"] > 0
        and summary["handoff_contract_count"] > 0
        and summary["handoff_contract_matched_count"] >= summary["handoff_count"],
        "handoffs must be present and satisfy configured contracts",
        summary,
    )
    _append_finding(
        findings,
        "multi_agent_probe_expected_coordination",
        summary["expected_handoff_count"] > 0
        and summary["expected_review_count"] > 0
        and summary["expected_reconciliation_present"] is True
        and summary["unmatched_coordination_check_count"] == 0,
        "expected handoff, review, and reconciliation checks must match",
        summary,
    )
    _append_finding(
        findings,
        "multi_agent_probe_review_reconciliation",
        summary["review_count"] > 0
        and summary["reconciliation_count"] > 0
        and summary["reconciliation_conflict_count"] == 0,
        "review and conflict-free reconciliation evidence must be present",
        summary,
    )
    _append_finding(
        findings,
        "multi_agent_probe_terminal_state",
        summary["terminal_state"] is True,
        "shared room state must reach a non-open terminal status",
        summary,
    )
    return findings


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


def _copy_participants(participants: Mapping[str, Any] | Sequence[Any]) -> Mapping[str, Any] | list[Any]:
    if isinstance(participants, Mapping):
        copied = copy.deepcopy(dict(participants))
        if not copied:
            raise ValueError("participants must not be empty")
        return copied
    if isinstance(participants, (str, bytes)):
        raise ValueError("participants must be a mapping or sequence of roles")
    copied_list = [
        copy.deepcopy(dict(item)) if isinstance(item, Mapping) else str(item)
        for item in participants
        if item not in (None, "")
    ]
    if not copied_list:
        raise ValueError("participants must not be empty")
    return copied_list


def _participant_keys(participants: Mapping[str, Any] | Sequence[Any]) -> list[str]:
    if isinstance(participants, Mapping):
        return _unique_strings(participants.keys())
    return _unique_strings(
        _plain_mapping(item).get("name") or _plain_mapping(item).get("role") or item
        for item in participants
    )


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


def _is_external_target(target: str) -> bool:
    return urlparse(str(target)).scheme.lower() in {"http", "https"}


__all__ = [
    "multi_agent_room_contract",
    "probe_multi_agent_room",
    "run_multi_agent_room_probe",
]
