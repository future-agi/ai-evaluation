import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalLangGraphOrchestrationAgent"


@dataclass(frozen=True)
class OrchestrationNode:
    id: str
    name: str
    type: str
    signals: list[str]


@dataclass(frozen=True)
class OrchestrationEdge:
    source: str
    target: str
    type: str
    condition: str
    signals: list[str]


@dataclass(frozen=True)
class OrchestrationStep:
    id: str
    name: str
    type: str
    node: str
    status: str
    route_from: str
    route_to: str
    attempt: int
    recoverable: bool
    recovered: bool
    error: dict[str, Any]
    latency_ms: int
    cost: dict[str, Any]
    signals: list[str]
    state: dict[str, Any]
    input: dict[str, Any]
    output: dict[str, Any]
    tool_name: str = ""
    tool_call_id: str = ""


@dataclass(frozen=True)
class OrchestrationTraceExport:
    content: str
    framework: str
    orchestration_nodes: list[Any]
    orchestration_edges: list[Any]
    orchestration_steps: list[Any]
    orchestration_state: dict[str, Any]
    orchestration_metadata: dict[str, Any]


class LocalLangGraphOrchestrationAgent:
    """Local LangGraph-style supervisor export for adapter promotion."""

    def run(self, text: str) -> str:
        assert text
        return "Weak response without multi-agent orchestration evidence."

    async def execute_task(self, payload: dict[str, Any]) -> OrchestrationTraceExport:
        assert payload["metadata"]["framework"] == "langgraph"
        case_id = "refund-42"
        return OrchestrationTraceExport(
            content=(
                "Orchestration trace adapter approved refund after supervisor "
                "delegation, policy retry recovery, critic vote, and final stop."
            ),
            framework="langgraph",
            orchestration_nodes=[
                OrchestrationNode(
                    id="supervisor",
                    name="supervisor",
                    type="supervisor",
                    signals=["agent", "spawn", "delegate"],
                ),
                OrchestrationNode(
                    id="policy_agent",
                    name="policy_agent",
                    type="tool_agent",
                    signals=["agent", "tool"],
                ),
                OrchestrationNode(
                    id="critic",
                    name="critic",
                    type="review_agent",
                    signals=["agent", "communicate"],
                ),
                OrchestrationNode(
                    id="finalizer",
                    name="finalizer",
                    type="aggregation_agent",
                    signals=["agent", "aggregate", "stop"],
                ),
            ],
            orchestration_edges=[
                OrchestrationEdge(
                    source="supervisor",
                    target="policy_agent",
                    type="delegate",
                    condition="policy_review_required",
                    signals=["route", "delegate"],
                ),
                OrchestrationEdge(
                    source="policy_agent",
                    target="critic",
                    type="handoff",
                    condition="eligible_refund",
                    signals=["route", "handoff", "communicate"],
                ),
                OrchestrationEdge(
                    source="critic",
                    target="finalizer",
                    type="route",
                    condition="critic_approved",
                    signals=["route", "aggregate"],
                ),
            ],
            orchestration_steps=[
                OrchestrationStep(
                    id="step-supervisor-delegate",
                    name="supervisor delegate policy_agent",
                    type="delegate",
                    node="supervisor",
                    status="success",
                    route_from="supervisor",
                    route_to="policy_agent",
                    attempt=1,
                    recoverable=False,
                    recovered=False,
                    error={},
                    latency_ms=8,
                    cost={"tokens": 6},
                    signals=["agent", "spawn", "delegate", "route", "latency", "cost"],
                    state={"assigned_agent": "policy_agent"},
                    input={"request": payload["input"]},
                    output={"assigned_agent": "policy_agent"},
                ),
                OrchestrationStep(
                    id="step-policy-tool-error",
                    name="tool call policy_lookup",
                    type="tool",
                    node="policy_agent",
                    status="error",
                    route_from="",
                    route_to="",
                    attempt=1,
                    recoverable=True,
                    recovered=False,
                    error={"code": "cache_miss", "recoverable": True},
                    latency_ms=14,
                    cost={"tokens": 10},
                    signals=["tool", "error", "latency", "cost"],
                    state={"policy_attempt": "cache_miss"},
                    input={"case_id": case_id, "market": "us"},
                    output={},
                    tool_name="policy_lookup",
                    tool_call_id="policy-lookup-1",
                ),
                OrchestrationStep(
                    id="step-policy-tool-retry",
                    name="tool call policy_lookup",
                    type="tool",
                    node="policy_agent",
                    status="success",
                    route_from="",
                    route_to="",
                    attempt=2,
                    recoverable=False,
                    recovered=True,
                    error={},
                    latency_ms=17,
                    cost={"tokens": 12},
                    signals=["tool", "retry", "recovered", "state", "latency", "cost"],
                    state={"policy_result": "eligible"},
                    input={"case_id": case_id, "market": "us", "retry": True},
                    output={"policy": "standard_refund", "eligible": True},
                    tool_name="policy_lookup",
                    tool_call_id="policy-lookup-2",
                ),
                OrchestrationStep(
                    id="step-policy-handoff",
                    name="policy_agent handoff critic",
                    type="handoff",
                    node="policy_agent",
                    status="success",
                    route_from="policy_agent",
                    route_to="critic",
                    attempt=1,
                    recoverable=False,
                    recovered=False,
                    error={},
                    latency_ms=6,
                    cost={"tokens": 5},
                    signals=["agent", "handoff", "communicate", "route"],
                    state={"handoff": "critic"},
                    input={"policy_result": "eligible"},
                    output={"review_request": "verify refund decision"},
                ),
                OrchestrationStep(
                    id="step-critic-vote",
                    name="critic communicate approval vote",
                    type="communicate",
                    node="critic",
                    status="success",
                    route_from="critic",
                    route_to="finalizer",
                    attempt=1,
                    recoverable=False,
                    recovered=False,
                    error={},
                    latency_ms=9,
                    cost={"tokens": 8},
                    signals=["agent", "communicate", "aggregate", "route"],
                    state={"critic_vote": "approve"},
                    input={"policy_result": "eligible"},
                    output={"vote": "approve"},
                ),
                OrchestrationStep(
                    id="step-finalizer-stop",
                    name="finalizer aggregate stop decision",
                    type="stop",
                    node="finalizer",
                    status="success",
                    route_from="",
                    route_to="",
                    attempt=1,
                    recoverable=False,
                    recovered=False,
                    error={},
                    latency_ms=11,
                    cost={"tokens": 9},
                    signals=["aggregate", "stop", "state", "latency", "cost"],
                    state={"decision": "approved refund"},
                    input={"vote": "approve", "policy_result": "eligible"},
                    output={"decision": "approved refund"},
                ),
            ],
            orchestration_state={
                "case_id": case_id,
                "decision": "approved refund",
                "terminal_agent": "finalizer",
            },
            orchestration_metadata={
                "trace_provider": "local-langgraph-supervisor",
                "scenario": "refund_orchestration",
            },
        )


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-orchestration-trace-run",
        framework="langgraph",
        target=TARGET,
        method_candidates=["run", "execute_task"],
        input_mode_candidates=["text", "dict"],
        discovery_max_candidates=6,
        cases=[
            {
                "id": "orchestration-refund",
                "input": "Approve the refund with full supervisor orchestration evidence.",
                "expected_contains": ["approved refund"],
                "required_tools": ["policy_lookup"],
                "required_events": [
                    "orchestration_step",
                    "orchestration_trace",
                ],
                "required_state_keys": [
                    "framework_runtime",
                    "orchestration_trace",
                ],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-orchestration-trace"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_orchestration_trace_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-orchestration-trace.json"
    )
    run(destination)
