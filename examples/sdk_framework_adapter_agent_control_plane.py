from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


FRAMEWORK = "agent_learning_kit"
TARGET = f"{Path(__file__).resolve()}:LocalAgentControlPlaneRuntime"


def _evidence(evidence_id: str) -> dict[str, str]:
    return {"id": evidence_id, "type": "trace"}


class LocalAgentControlPlaneRuntime:
    """Local framework-like runtime that exposes agency control-plane evidence."""

    def run(self, text: str) -> str:
        assert text
        return "Weak control-plane response without runtime governance evidence."

    async def execute_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["metadata"]["framework"] == FRAMEWORK
        return {
            "content": (
                "Agent control-plane adapter approved refund after trust-boundary "
                "checks, approval gates, rollback controls, and containment."
            ),
            "metadata": {
                "framework": FRAMEWORK,
                "run_id": "control-plane-run-001",
            },
            "tool_calls": [
                {
                    "id": "trust-status",
                    "name": "agent_trust_boundary_status",
                    "arguments": {"case_id": "refund-42"},
                },
                {
                    "id": "control-status",
                    "name": "agent_control_plane_status",
                    "arguments": {"case_id": "refund-42"},
                },
            ],
            "agent_trust_boundary_model": _trust_boundary_payload(),
            "agent_control_plane": _control_plane_payload(),
            "framework_spans": [
                {
                    "id": "span-trust-boundary",
                    "name": "agent trust boundary check",
                    "type": "span",
                    "signals": ["model", "state", "security", "latency"],
                    "latency_ms": 8,
                    "attributes": {"case_id": "refund-42"},
                },
                {
                    "id": "span-control-plane",
                    "name": "agent control plane decision",
                    "type": "span",
                    "signals": ["tool", "state", "control_plane", "cost"],
                    "tool_name": "agent_control_plane_status",
                    "cost": {"tokens": 12},
                    "attributes": {"decision": "approved"},
                },
            ],
        }


def _trust_boundary_payload() -> dict[str, Any]:
    controls = [
        ("identity", "identity"),
        ("permissions", "permissions"),
        ("sandbox", "sandbox"),
        ("audit", "audit"),
        ("canaries", "canaries"),
        ("hitl_approval", "human_approval"),
        ("memory_isolation", "memory_isolation"),
        ("network_egress", "network_egress"),
        ("tool_allowlist", "tool_allowlist"),
        ("data_boundary", "data_boundary"),
        ("secret_handling", "secret_handling"),
    ]
    return {
        "name": "framework-agent-trust-boundary",
        "framework": FRAMEWORK,
        "actors": [
            {
                "id": "support_agent",
                "type": "agent",
                "trust_level": "internal",
                "privileges": ["least_privilege", "tool_runtime"],
                "evidence": [_evidence("principal-map")],
            }
        ],
        "assets": [
            {
                "id": "customer_secret",
                "type": "credential",
                "sensitivity": "secret",
                "owner": "tenant",
                "evidence": [_evidence("secret-inventory")],
            },
            {
                "id": "customer_pii",
                "type": "profile",
                "sensitivity": "high",
                "owner": "tenant",
                "evidence": [_evidence("pii-boundary")],
            },
        ],
        "tools": [
            {
                "id": "wire_transfer",
                "permissions": ["write"],
                "high_risk": True,
                "destructive": True,
                "auth_required": True,
                "controls": ["human_approval", "tool_allowlist", "audit"],
                "evidence": [_evidence("wire-tool-policy")],
            },
            {
                "id": "webhook_post",
                "permissions": ["network", "write"],
                "high_risk": True,
                "external": True,
                "controls": ["network_egress", "secret_handling", "audit"],
                "evidence": [_evidence("egress-policy")],
            },
            {
                "id": "memory_write",
                "permissions": ["write"],
                "high_risk": True,
                "controls": ["memory_isolation", "data_boundary", "audit"],
                "evidence": [_evidence("memory-policy")],
            },
        ],
        "surfaces": [
            {
                "id": "chat_input",
                "type": "chat",
                "trust_level": "untrusted",
                "threats": ["indirect_prompt_injection"],
                "controls": ["data_boundary", "canaries"],
                "evidence": [_evidence("chat-redteam-trace")],
            },
            {
                "id": "retrieval_memory",
                "type": "memory",
                "trust_level": "untrusted",
                "threats": ["indirect_prompt_injection"],
                "controls": ["memory_isolation", "canaries"],
                "evidence": [_evidence("memory-canary-trace")],
            },
            {
                "id": "webhook_output",
                "type": "network",
                "trust_level": "external",
                "threats": ["secret_exfiltration"],
                "controls": ["network_egress", "secret_handling"],
                "evidence": [_evidence("webhook-egress-trace")],
            },
        ],
        "controls": [
            {
                "id": control_id,
                "category": category,
                "status": "present",
                "required": True,
                "evidence": [_evidence(f"{control_id}-evidence")],
            }
            for control_id, category in controls
        ],
        "canaries": [
            {
                "id": "prompt_canary",
                "surface": "chat_input",
                "status": "present",
                "evidence": [_evidence("prompt-canary-run")],
            },
            {
                "id": "memory_canary",
                "surface": "retrieval_memory",
                "status": "present",
                "evidence": [_evidence("memory-canary-run")],
            },
        ],
        "threats": [
            {
                "id": "indirect_prompt_injection",
                "category": "prompt_injection",
                "severity": "critical",
                "status": "mitigated",
                "controls": ["data_boundary", "canaries", "tool_allowlist"],
                "evidence": [_evidence("prompt-injection-replay")],
            },
            {
                "id": "secret_exfiltration",
                "category": "secret_exfiltration",
                "severity": "critical",
                "status": "mitigated",
                "controls": ["secret_handling", "network_egress", "audit"],
                "evidence": [_evidence("secret-exfil-replay")],
            },
            {
                "id": "tool_abuse",
                "category": "tool_abuse",
                "severity": "high",
                "status": "mitigated",
                "controls": ["hitl_approval", "tool_allowlist", "sandbox"],
                "evidence": [_evidence("tool-abuse-replay")],
            },
        ],
    }


def _control_plane_payload() -> dict[str, Any]:
    controls = [
        ("risk_scoring", "risk_scoring"),
        ("action_policy", "action_policy"),
        ("approval_gate", "approval"),
        ("rollback", "rollback"),
        ("kill_switch", "kill_switch"),
        ("circuit_breaker", "circuit_breaker"),
        ("rate_limit", "rate_limit"),
        ("budget", "budget"),
        ("audit", "audit"),
        ("containment", "containment"),
        ("drift_detection", "drift_detection"),
    ]
    return {
        "name": "framework-agent-control-plane",
        "framework": FRAMEWORK,
        "actions": [
            {
                "id": "wire_transfer",
                "category": "tool",
                "tool": "wire_transfer",
                "risk_level": "critical",
                "status": "approved",
                "reversible": True,
                "requires_approval": True,
                "approved_by": "human_reviewer",
                "controls": [
                    "risk_scoring",
                    "action_policy",
                    "approval",
                    "budget",
                    "audit",
                ],
                "evidence": [_evidence("approval-trace")],
            },
            {
                "id": "wire_transfer_rollback",
                "category": "tool",
                "tool": "wire_transfer",
                "risk_level": "critical",
                "status": "rolled_back",
                "reversible": True,
                "requires_approval": True,
                "approved_by": "human_reviewer",
                "controls": ["rollback", "containment", "audit"],
                "evidence": [_evidence("rollback-trace")],
            },
            {
                "id": "network_egress_block",
                "category": "network",
                "risk_level": "high",
                "status": "blocked",
                "reversible": True,
                "controls": [
                    "network_egress",
                    "kill_switch",
                    "circuit_breaker",
                    "audit",
                ],
                "evidence": [_evidence("egress-block-trace")],
            },
        ],
        "controls": [
            {
                "id": control_id,
                "category": category,
                "status": "present",
                "required": True,
                "evidence": [_evidence(f"{control_id}-evidence")],
            }
            for control_id, category in controls
        ],
        "budgets": [
            {
                "id": "tool_spend",
                "category": "budget",
                "status": "within",
                "limit": 100.0,
                "used": 25.0,
                "remaining": 75.0,
                "evidence": [_evidence("tool-spend-budget")],
            },
            {
                "id": "network_calls",
                "category": "rate_limit",
                "status": "within",
                "limit": 50.0,
                "used": 10.0,
                "remaining": 40.0,
                "evidence": [_evidence("network-budget")],
            },
            {
                "id": "autonomy_minutes",
                "category": "budget",
                "status": "within",
                "limit": 30.0,
                "used": 8.0,
                "remaining": 22.0,
                "evidence": [_evidence("time-budget")],
            },
        ],
        "escalations": [
            {
                "id": "wire_transfer_approval",
                "action": "wire_transfer",
                "status": "approved",
                "reviewer": "human_reviewer",
                "evidence": [_evidence("approval-ticket")],
            }
        ],
        "incidents": [
            {
                "id": "secret_tool_escape",
                "action": "webhook_post",
                "severity": "critical",
                "status": "contained",
                "controls": ["kill_switch", "containment", "rollback", "audit"],
                "evidence": [_evidence("incident-containment-trace")],
            }
        ],
    }


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-agent-control-plane-run",
        framework=FRAMEWORK,
        target=TARGET,
        method_candidates=["run", "execute_task"],
        input_mode_candidates=["text", "dict", "agent_input"],
        discovery_max_candidates=4,
        cases=[
            {
                "id": "agent-control-plane-refund",
                "input": "Approve the refund only if runtime control-plane checks pass.",
                "expected_contains": ["approved refund"],
                "required_tools": [
                    "agent_trust_boundary_status",
                    "agent_control_plane_status",
                ],
                "required_events": [
                    "agent_trust_boundary_status",
                    "agent_control_plane_status",
                    "framework_trace_span",
                ],
                "required_state_keys": [
                    "framework_runtime",
                    "framework_trace",
                    "agent_trust_boundary_model",
                    "agent_control_plane",
                ],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-agent-control-plane"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_agent_control_plane_manifest"] = manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    destination = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts") / "sdk-framework-adapter-agent-control-plane.json"
    )
    run(destination)
