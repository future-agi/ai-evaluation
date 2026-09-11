from __future__ import annotations

import asyncio
import importlib
import json
import textwrap
from pathlib import Path

import pytest

from fi.alk import simulate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FI_ROOT = PROJECT_ROOT / "src" / "fi"


FRAMEWORK_AGENT_MODULE = """
class LocalLangGraphAgent:
    async def ainvoke(self, payload):
        assert payload["metadata"]["framework"] == "langgraph"
        assert payload["metadata"]["suite"] == "agent-learning-kit"
        assert payload["scenario_name"] == "vendored-simulate-runtime"
        assert {"apply_world_transition", "framework_trace_status"} <= {
            tool["name"] for tool in payload["tools"]
        }
        return {
            "content": (
                "The refund world contract completed and the framework adapter "
                "conformance passed."
            ),
            "tool_calls": [
                {
                    "id": "approve_refund",
                    "name": "apply_world_transition",
                    "arguments": {"id": "approve_refund"},
                },
                {
                    "id": "framework_status",
                    "name": "framework_trace_status",
                    "arguments": {},
                },
            ],
            "metadata": {"runtime_contract": {"passed": True}},
        }


def build_agent():
    return LocalLangGraphAgent()
"""


def _module_path(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    return Path(module.__file__).resolve()


def _assert_vendored_under_agent_learning_kit(obj: object) -> None:
    module_path = _module_path(obj.__module__)
    assert module_path.is_relative_to(FI_ROOT)


def _manifest(required_env: str) -> dict:
    return {
        "version": "agent-simulate.cli.v1",
        "name": "vendored-simulate-runtime",
        "required_env": [required_env],
        "scenario": {
            "name": "vendored-simulate-runtime",
            "dataset": [
                {
                    "persona": {"name": "Maya", "role": "sdk-owner"},
                    "situation": "Maya needs a local vendored simulation run.",
                    "outcome": (
                        "The refund world contract completed and the framework "
                        "adapter conformance passed."
                    ),
                }
            ],
        },
        "agent": {
            "type": "framework",
            "framework": "langgraph",
            "target": "framework_agent.py:build_agent",
            "factory": True,
            "method": "ainvoke",
            "input_mode": "dict",
            "trace_runtime": True,
            "metadata": {"suite": "agent-learning-kit"},
        },
        "simulation": {
            "engine": "local_text",
            "max_turns": 1,
            "min_turns": 1,
            "environments": [
                {
                    "type": "world_contract",
                    "data": {
                        "name": "refund-world",
                        "actors": ["agent", "customer"],
                        "resources": ["refund"],
                        "initial_state": {
                            "policy": {"can_refund": True},
                            "refund": {"status": "pending"},
                        },
                        "transitions": [
                            {
                                "id": "approve_refund",
                                "actor": "agent",
                                "resource": "refund",
                                "action": "approve_refund",
                                "required": True,
                                "preconditions": {"refund.status": "pending"},
                                "effects": {"refund.status": "approved"},
                                "postconditions": {"refund.status": "approved"},
                                "signals": ["refund_resolution"],
                            }
                        ],
                        "invariants": [
                            {
                                "id": "policy_allows_refunds",
                                "must": {"policy.can_refund": True},
                            }
                        ],
                        "success_conditions": [
                            {
                                "id": "refund_approved",
                                "must": {"refund.status": "approved"},
                            }
                        ],
                    },
                },
                {
                    "type": "framework_trace",
                    "data": {
                        "framework": "langgraph",
                        "spans": [
                            {
                                "id": "agent_node",
                                "name": "agent_node",
                                "input": "refund request",
                                "output": "approved",
                                "tool_calls": [
                                    {"name": "apply_world_transition"}
                                ],
                                "signals": ["model", "tool", "state"],
                            }
                        ],
                        "adapter_required_signals": ["model", "tool", "state"],
                        "adapter_required_mappings": {
                            "tool": ["tool_name"],
                        },
                    },
                },
            ],
        },
        "evaluation": {"enabled": False},
    }


def _eval_suite() -> dict:
    return {
        "version": "agent-simulate.eval.v1",
        "name": "vendored-local-eval-suite",
        "providers": [
            {
                "id": "scripted",
                "type": "scripted",
                "response": "Policy answer: {{question}} is approved locally.",
            }
        ],
        "prompts": [{"id": "support", "template": "{{question}}"}],
        "tests": [
            {
                "id": "policy_lookup",
                "vars": {"question": "refund policy"},
                "assert": [
                    {"type": "contains", "value": "refund policy"},
                    {"type": "not_contains", "value": "network"},
                ],
            }
        ],
    }


def test_agent_learning_simulate_exports_are_vendored_from_src_fi() -> None:
    for module_name in (
        "fi.simulate",
        "fi.simulate.manifest",
        "fi.simulate.suite",
        "fi.simulate.simulation.engines.local_text",
        "fi.alk.simulate.environment",
        "fi.alk.simulate.manifest",
        "fi.alk.simulate.suite",
        "fi.alk.simulate.simulation.engines.local_text",
        "fi.alk.simulate.agent.definition",
    ):
        assert _module_path(module_name).is_relative_to(FI_ROOT)

    from fi.alk.simulate.agent.definition import AgentDefinition
    from fi.alk.simulate.environment import WorldContractEnvironment
    from fi.alk.simulate.manifest import load_manifest_file
    from fi.alk.simulate.simulation.engines.local_text import LocalTextEngine
    from fi.alk.simulate.suite import run_eval_suite

    assert AgentDefinition is simulate.AgentDefinition
    assert WorldContractEnvironment is simulate.WorldContractEnvironment
    assert LocalTextEngine is simulate.LocalTextEngine
    assert load_manifest_file.__module__ == "fi.simulate.manifest"
    assert run_eval_suite.__module__ == "fi.simulate.suite"
    assert callable(simulate.load_manifest_file)
    assert callable(simulate.run_eval_suite)

    for exported in (
        simulate.AgentResponse,
        simulate.WorldContractEnvironment,
        simulate.FrameworkTraceEnvironment,
        simulate.LocalTextEngine,
        simulate.TestRunner,
    ):
        _assert_vendored_under_agent_learning_kit(exported)

    assert "langgraph" in simulate.supported_frameworks()
    assert callable(simulate.discover_framework_adapter)
    assert callable(simulate.probe_framework_adapter)
    assert callable(simulate.run_framework_adapter_probe)
    assert callable(simulate.memory_layer_contract)
    assert callable(simulate.probe_memory_layer)
    assert callable(simulate.run_memory_layer_probe)
    assert callable(simulate.multi_agent_room_contract)
    assert callable(simulate.probe_multi_agent_room)
    assert callable(simulate.run_multi_agent_room_probe)
    assert callable(simulate.orchestration_stack_contract)
    assert callable(simulate.probe_orchestration_stack)
    assert callable(simulate.run_orchestration_stack_probe)
    assert callable(simulate.realtime_stack_contract)
    assert callable(simulate.probe_realtime_stack)
    assert callable(simulate.run_realtime_stack_probe)
    assert callable(simulate.browser_cua_contract)
    assert callable(simulate.probe_browser_cua)
    assert callable(simulate.run_browser_cua_probe)


def test_browser_cua_probe_scores_local_replay_and_rejects_external_target() -> None:
    from fi.alk import optimize

    manifest = optimize.build_browser_cua_optimization_manifest(
        name="browser-cua-probe-test",
    )
    weak, strong = manifest["optimization"]["target"]["search_space"][
        "simulation.environments"
    ]

    result = simulate.run_browser_cua_probe(
        strong,
        expected_url="https://shop.example.test/confirmation",
        expected_state={"checkout_complete": True, "order_id": "ord_123"},
        expected_order_id="ord_123",
    )

    assert result["kind"] == "agent-learning.browser-cua-probe.v1"
    assert result["status"] == "passed"
    assert result["contract"]["kind"] == "agent-learning.browser-cua-contract.v1"
    assert result["contract"]["local_executable_fixture"] is True
    assert result["summary"]["refreshed_snapshot"] is True
    assert result["summary"]["selector_match_count"] == 1
    assert result["summary"]["mutation_count"] == 2
    assert result["summary"]["screenshot_diff_count"] == 1
    assert result["summary"]["prompt_injection_touched_count"] == 0
    assert result["summary"]["observed_tool_names"] == [
        "browser_snapshot",
        "browser_refresh_snapshot",
        "browser_mutations",
        "browser_click",
        "browser_storage",
        "browser_runtime",
        "browser_network",
    ]
    scoring = optimize.score_browser_cua_probe_result(result)
    assert scoring["metrics"]["browser_cua_probe_score"] == pytest.approx(1.0)

    weak_result = simulate.run_browser_cua_probe(
        weak,
        expected_url="https://shop.example.test/confirmation",
        expected_state={"checkout_complete": True, "order_id": "ord_123"},
        expected_order_id="ord_123",
    )
    assert weak_result["status"] == "failed"
    assert {
        "browser_cua_probe_mutation_grounding",
        "browser_cua_probe_expected_state",
    } <= {finding["check"] for finding in weak_result["findings"]}

    with pytest.raises(ValueError, match="external targets are disabled"):
        simulate.run_browser_cua_probe(
            strong,
            target="https://example.com/browser-agent",
        )


def test_realtime_stack_probe_scores_local_voice_streaming_and_rejects_external_target() -> None:
    example_path = PROJECT_ROOT / "examples" / "sdk_realtime_voice_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_realtime_voice_optimization_probe",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = simulate.run_realtime_stack_probe(
        module.strong_candidate(),
        framework="livekit",
        expected_route="support",
        metadata={"suite": "realtime-stack-probe"},
    )

    assert result["kind"] == "agent-learning.realtime-stack-probe.v1"
    assert result["status"] == "passed"
    assert result["contract"]["kind"] == "agent-learning.realtime-stack-contract.v1"
    assert result["contract"]["local_executable_fixture"] is True
    assert result["summary"]["sample_rate_hz"] == 16000
    assert result["summary"]["current_route"] == "support"
    assert result["summary"]["streaming_tool_delta_count"] == 1
    assert result["summary"]["observed_tool_names"] == [
        "voice_status",
        "voice_timing",
        "transcribe_audio",
        "route_call",
        "speak",
        "streaming_trace_status",
        "list_stream_events",
        "inspect_stream_event",
    ]

    weak = simulate.run_realtime_stack_probe(
        module.weak_candidate(),
        framework="livekit",
        expected_route="support",
    )
    assert weak["status"] == "failed"
    assert "realtime_probe_streaming_trace" in {
        finding["check"] for finding in weak["findings"]
    }

    with pytest.raises(ValueError, match="external targets are disabled"):
        simulate.run_realtime_stack_probe(
            module.strong_candidate(),
            framework="livekit",
            expected_route="support",
            target="https://example.com/realtime-agent",
        )


def test_orchestration_stack_probe_scores_local_stack_and_rejects_external_target() -> None:
    example_path = PROJECT_ROOT / "examples" / "sdk_orchestration_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "sdk_orchestration_optimization_probe",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = simulate.run_orchestration_stack_probe(
        module.strong_stack(),
        agent=module.strong_agent(),
        metadata={"suite": "orchestration-stack-probe"},
    )

    assert result["kind"] == "agent-learning.orchestration-stack-probe.v1"
    assert result["status"] == "passed"
    assert result["contract"]["kind"] == (
        "agent-learning.orchestration-stack-contract.v1"
    )
    assert result["contract"]["local_executable_fixture"] is True
    assert result["summary"]["expected_transition_completed"] is True
    assert result["summary"]["world_state_match"] is True
    assert result["summary"]["framework_required_signal_match_count"] == 3
    assert result["summary"]["retrieval_expected_document_cited"] is True
    assert result["summary"]["has_source_attribution"] is True
    assert result["summary"]["role_match"] is True
    assert result["summary"]["expected_review_present"] is True
    assert result["summary"]["expected_reconciliation_present"] is True
    assert result["summary"]["required_tools_handled"] is True
    assert [env["type"] for env in result["environments"]] == [
        "world_contract",
        "framework_trace",
        "retrieval_memory",
        "agent_memory_lineage",
        "multi_agent_room",
    ]

    weak = simulate.run_orchestration_stack_probe(
        module.weak_stack(),
        agent=module.weak_agent(),
    )
    assert weak["status"] == "failed"
    assert {
        "orchestration_probe_world_transition",
        "orchestration_probe_tool_evidence",
    } <= {finding["check"] for finding in weak["findings"]}

    with pytest.raises(ValueError, match="external targets are disabled"):
        simulate.run_orchestration_stack_probe(
            module.strong_stack(),
            agent=module.strong_agent(),
            target="https://example.com/orchestration-agent",
        )


def test_multi_agent_room_probe_scores_local_coordination_and_rejects_external_target() -> None:
    participants = {
        "planner": {"name": "planner", "role": "task planner"},
        "retriever": {"name": "retriever", "role": "policy evidence retriever"},
        "critic": {"name": "critic", "role": "grounding reviewer"},
    }
    agent = {
        "type": "scripted",
        "responses": [
            {
                "content": "Route evidence and request review.",
                "tool_calls": [
                    {
                        "id": "handoff_retriever",
                        "name": "handoff",
                        "arguments": {
                            "to": "retriever",
                            "task": "Collect the current refund policy evidence.",
                            "reason": "source grounding is required",
                            "context": {
                                "doc_id": "doc_refund_2026",
                                "world_state": "refund_case_open",
                            },
                        },
                    },
                    {
                        "id": "review_critic",
                        "name": "request_review",
                        "arguments": {
                            "reviewer": "critic",
                            "target": "refund policy answer",
                            "criteria": ["policy", "source"],
                        },
                    },
                    {
                        "id": "reconcile_answer",
                        "name": "reconcile",
                        "arguments": {
                            "summary": "approved refund answer",
                            "decision": "ship grounded refund decision",
                            "accepted_source": "critic",
                            "conflicts": [],
                        },
                    },
                ],
            }
        ],
    }
    room = {
        "handoff_contracts": {
            "retriever": {
                "require_reason": True,
                "required_context_keys": ["doc_id", "world_state"],
                "required_task_terms": ["refund policy"],
            }
        },
        "expected_handoffs": [
            {
                "to": "retriever",
                "task_contains": "current refund policy",
                "reason_contains": "source grounding",
                "context_keys": ["doc_id", "world_state"],
                "contract_matched": True,
            }
        ],
        "expected_reviews": [
            {
                "reviewer": "critic",
                "target_contains": "refund policy answer",
                "criteria": ["policy", "source"],
            }
        ],
        "expected_reconciliation": {
            "summary_contains": "approved refund answer",
            "accepted_source": "critic",
            "conflicts_empty": True,
        },
        "allow_unknown_roles": False,
        "state": {"case": {"status": "resolved"}},
    }

    result = simulate.probe_multi_agent_room(
        participants=participants,
        room=room,
        agent=agent,
        target="multi_agent_room.py:local_fixture",
        metadata={"suite": "multi-agent-room-probe"},
    )

    assert result["kind"] == "agent-learning.multi-agent-room-probe.v1"
    assert result["status"] == "passed"
    assert result["contract"]["kind"] == "agent-learning.multi-agent-room-contract.v1"
    assert result["contract"]["local_executable_fixture"] is True
    assert result["summary"]["participant_count"] == 3
    assert result["summary"]["handoff_contract_matched_count"] == 1
    assert result["summary"]["matched_coordination_check_count"] == (
        result["summary"]["coordination_check_count"]
    )
    assert result["summary"]["terminal_state"] is True
    assert result["environment"]["type"] == "multi_agent_room"

    weak = simulate.probe_multi_agent_room(
        participants=participants,
        room={"allow_unknown_roles": True, "state": {"case": {"status": "triage"}}},
        agent={"responses": [{"content": "solo answer", "tool_calls": []}]},
    )
    assert weak["status"] == "failed"
    assert "multi_agent_probe_role_boundary" in {
        finding["check"] for finding in weak["findings"]
    }

    with pytest.raises(ValueError, match="external targets are disabled"):
        simulate.probe_multi_agent_room(
            participants=participants,
            room=room,
            agent=agent,
            target="https://example.com/multi-agent-room",
        )


def test_framework_adapter_probe_runs_custom_framework_runtime() -> None:
    class CustomRefundOrchestrator:
        async def execute_task(self, payload):
            assert payload["metadata"]["framework"] == "custom_refund_orchestrator"
            assert payload["scenario_name"] == "adapter-probe"
            return {
                "content": "Adapter probe approved refund with trace evidence.",
                "tool_calls": [
                    {
                        "id": "framework_status",
                        "name": "framework_trace_status",
                        "arguments": {"status": "passed"},
                    }
                ],
                "events": [
                    {
                        "type": "framework_trace",
                        "name": "execute_task",
                        "payload": {"framework": "custom_refund_orchestrator"},
                    }
                ],
                "metadata": {"runtime_contract": {"passed": True}},
            }

    result = asyncio.run(
        simulate.probe_framework_adapter(
            "custom_refund_orchestrator",
            CustomRefundOrchestrator(),
            target="framework_shims.py:build_custom_refund_orchestrator",
            method="execute_task",
            input_mode="dict",
            cases=[
                {
                    "id": "refund",
                    "scenario_name": "adapter-probe",
                    "input": "Approve the refund.",
                    "expected_contains": ["approved refund"],
                    "required_tools": ["framework_trace_status"],
                    "required_events": ["framework_trace"],
                    "required_state_keys": ["framework_runtime"],
                }
            ],
            metadata={"suite": "adapter-probe"},
        )
    )

    assert result["kind"] == "agent-learning.framework-adapter-probe.v1"
    assert result["status"] == "passed"
    assert result["summary"]["case_count"] == 1
    assert result["summary"]["runtime_trace_count"] == 1
    assert result["summary"]["callable_signature_present"] is True
    assert result["summary"]["observed_io_contract_count"] == 1
    assert result["summary"]["call_contract_count"] == 1
    assert result["summary"]["signature_bound_count"] == 1
    assert result["summary"]["tool_call_count"] == 1
    assert result["contract"]["framework"] == "custom_refund_orchestrator"
    assert result["contract"]["method"] == "execute_task"
    assert result["contract"]["input_mode"] == "dict"
    assert result["contract"]["local_executable_fixture"] is True
    assert result["contract"]["callable_signature"]["kind"] == (
        "agent-learning.framework-adapter-callable-signature.v1"
    )
    case = result["cases"][0]
    assert case["status"] == "passed"
    assert case["runtime_trace"]["summary"]["methods"] == ["execute_task"]
    assert case["runtime_trace"]["summary"]["input_modes"] == ["dict"]
    assert case["runtime_trace"]["summary"]["call_contract_count"] == 1
    invocation = case["runtime_trace"]["invocations"][0]
    assert invocation["call_contract"]["kind"] == (
        "agent-learning.framework-adapter-call-contract.v1"
    )
    assert invocation["call_contract"]["signature_bound"] is True
    assert case["observed_io_contract"]["kind"] == (
        "agent-learning.framework-adapter-observed-io-contract.v1"
    )
    assert case["observed_io_contract"]["summary"]["signature_bound"] is True
    assert case["runtime_trace"]["metadata"]["framework_adapter_contract"] == (
        result["contract"]
    )


def test_framework_adapter_probe_proves_keyword_only_signature_io_contract() -> None:
    class KeywordOnlyRefundOrchestrator:
        async def execute_task(self, *, payload: dict):
            assert payload["metadata"]["framework"] == "keyword_only_refund"
            return {"content": "Keyword-only adapter approved refund."}

    result = simulate.run_framework_adapter_probe(
        "keyword_only_refund",
        KeywordOnlyRefundOrchestrator(),
        method="execute_task",
        input_mode="dict",
        cases=[
            {
                "id": "keyword-only-refund",
                "input": "Approve the refund through keyword payload.",
                "expected_contains": ["approved refund"],
            }
        ],
    )

    assert result["status"] == "passed"
    assert result["summary"]["input_keys"] == ["payload"]
    assert result["summary"]["call_styles"] == ["keyword"]
    assert result["summary"]["input_types"] == ["dict"]
    assert result["summary"]["output_types"] == ["AgentResponse"]
    signature = result["contract"]["callable_signature"]
    assert signature["keyword_only_parameters"] == ["payload"]
    assert signature["preferred_input_key"] == "payload"

    case = result["cases"][0]
    invocation = case["runtime_trace"]["invocations"][0]
    assert invocation["call_style"] == "keyword"
    assert invocation["input_key"] == "payload"
    assert invocation["call_contract"]["input_key"] == "payload"
    assert invocation["call_contract"]["signature"]["selected_input_key"] == "payload"
    assert invocation["call_contract"]["observed_io"]["input"]["type"] == "dict"
    assert invocation["call_contract"]["observed_io"]["output"]["type"] == (
        "AgentResponse"
    )
    assert case["observed_io_contract"]["summary"]["signature_bound"] is True
    check_ids = {check["id"] for check in case["checks"] if check["passed"]}
    assert {
        "framework_adapter_callable_signature_present",
        "framework_adapter_observed_io_contract_present",
        "framework_adapter_observed_io_matches_signature",
    } <= check_ids


def test_framework_adapter_discovery_ranks_custom_methods_and_rejects_external_target() -> None:
    class CustomRefundOrchestrator:
        def run(self, text):
            return f"weak text adapter: {text}"

        async def execute_task(self, payload):
            return {
                "content": "structured refund task",
                "metadata": {"framework": payload["metadata"]["framework"]},
            }

    result = simulate.discover_framework_adapter(
        "custom_refund_orchestrator",
        CustomRefundOrchestrator(),
        target="framework_shims.py:build_custom_refund_orchestrator",
        method_candidates=["run", "execute_task"],
        input_mode_candidates=["text", "dict", "agent_input"],
        metadata={"suite": "adapter-discovery"},
    )

    assert result["kind"] == "agent-learning.framework-adapter-discovery.v1"
    assert result["status"] == "passed"
    assert result["requires_external_service"] is False
    assert result["summary"]["top_method"] == "execute_task"
    assert result["summary"]["top_input_mode"] == "dict"
    assert result["adapter_candidates"][0]["method"] == "execute_task"
    assert result["adapter_candidates"][0]["input_mode"] == "dict"
    assert result["candidates"][0]["contract"]["kind"] == (
        "agent-learning.framework-adapter-contract.v1"
    )
    assert result["candidates"][0]["contract"]["method"] == "execute_task"
    assert "execute_task" in result["agent"]["exposed_methods"]

    with pytest.raises(ValueError, match="external targets are disabled"):
        simulate.discover_framework_adapter(
            "langchain",
            CustomRefundOrchestrator(),
            target="https://example.com/agent",
        )


def test_framework_adapter_probe_runs_sync_callable_and_rejects_external_target() -> None:
    def callable_agent(agent_input):
        return simulate.AgentResponse(
            content="Callable adapter probe passed.",
            tool_calls=[
                {
                    "id": "callable_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed"},
                }
            ],
        )

    result = simulate.run_framework_adapter_probe(
        "callable",
        callable_agent,
        input_mode="agent_input",
        cases=[
            {
                "id": "callable",
                "input": "Run a callable probe.",
                "expected_contains": ["probe passed"],
                "required_tools": ["framework_trace_status"],
            }
        ],
    )

    assert result["status"] == "passed"
    assert result["summary"]["runtime_trace_count"] == 1
    assert result["cases"][0]["runtime_trace"]["framework"] == "callable"

    with pytest.raises(ValueError, match="external targets are disabled"):
        simulate.run_framework_adapter_probe(
            "langchain",
            callable_agent,
            target="https://example.com/agent",
            input_mode="agent_input",
        )


def test_memory_layer_probe_scores_local_retrieval_and_lineage() -> None:
    memory_candidate = {
        "retrieval_memory": {
            "documents": [
                {
                    "id": "doc_refund_2026",
                    "content": "Current refund memory policy.",
                    "current": True,
                }
            ],
            "citations": [
                {
                    "claim": "Refund policy is current.",
                    "doc_ids": ["doc_refund_2026"],
                    "freshness_checked": True,
                }
            ],
        },
        "agent_memory_lineage": {
            "target": {"agent": "refund-agent", "tenant": "tenant_a"},
            "stores": [{"id": "episodic", "tenant": "tenant_a"}],
            "memories": [
                {
                    "id": "refund_decision",
                    "source_ids": ["doc_refund_2026"],
                    "tenant": "tenant_a",
                }
            ],
            "operations": [
                {
                    "id": "read_policy",
                    "operation": "read",
                    "trace_id": "trace_read",
                    "status": "allowed",
                    "policy_decision": "allowed",
                },
                {
                    "id": "write_policy",
                    "operation": "write",
                    "trace_id": "trace_write",
                    "status": "allowed",
                    "policy_decision": "allowed",
                },
                {
                    "id": "recall_policy",
                    "operation": "recall",
                    "trace_id": "trace_recall",
                    "status": "allowed",
                    "policy_decision": "allowed",
                },
            ],
            "lineage": [
                {
                    "from": "doc_refund_2026",
                    "to": "refund_decision",
                    "type": "source_attribution",
                }
            ],
            "policies": {
                "retention": {"status": "enforced"},
                "deletion": {"status": "enforced"},
                "redaction": {"status": "enforced"},
                "tenant_isolation": {"status": "enforced"},
                "audit": {"status": "enforced"},
            },
            "poison_tests": [{"id": "canary", "status": "blocked"}],
            "isolation_tests": [{"id": "tenant", "status": "passed"}],
            "retention_tests": [{"id": "retention", "status": "passed"}],
            "observability": {"traces": ["trace_read"]},
            "artifacts": [{"id": "memory-audit", "type": "json"}],
            "required_evidence": [
                "source_attribution",
                "tenant_isolation",
                "audit",
                "retention_policy",
                "deletion_policy",
                "redaction",
                "canary",
            ],
            "required_signals": ["memory_lineage", "source_attribution", "audit"],
        },
    }

    result = simulate.run_memory_layer_probe(
        memory_candidate,
        cases=[{"id": "refund-memory", "input": "Recall refund policy memory."}],
        target="memory_shims.py:build_memory",
        metadata={"suite": "memory-probe"},
    )

    assert result["kind"] == "agent-learning.memory-layer-probe.v1"
    assert result["status"] == "passed"
    assert result["contract"]["kind"] == "agent-learning.memory-layer-contract.v1"
    assert result["contract"]["local_executable_fixture"] is True
    assert result["summary"]["retrieval_citations_current"] is True
    assert result["summary"]["memory_required_operations_present"] is True
    assert result["summary"]["has_tenant_isolation"] is True
    assert result["summary"]["blocking_gap_count"] == 0
    assert [env["type"] for env in result["environments"]] == [
        "retrieval_memory",
        "agent_memory_lineage",
    ]

    with pytest.raises(ValueError, match="external targets are disabled"):
        simulate.run_memory_layer_probe(
            memory_candidate,
            target="https://example.com/memory",
        )


def test_public_manifest_api_runs_vendored_local_world_and_framework_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required_env = "AGENT_LEARNING_KIT_SIMULATE_TEST_KEY"
    manifest = _manifest(required_env)
    manifest_path = tmp_path / "simulate.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "framework_agent.py").write_text(
        textwrap.dedent(FRAMEWORK_AGENT_MODULE),
        encoding="utf-8",
    )

    monkeypatch.delenv(required_env, raising=False)
    assert simulate.detect_manifest_command(manifest) == "run"
    assert simulate.missing_manifest_env(manifest) == [required_env]
    fi_simulate = importlib.import_module("fi.simulate")
    with pytest.raises(fi_simulate.ManifestError, match=required_env):
        simulate.validate_manifest_env(manifest)

    monkeypatch.setenv(required_env, "local-only-key")
    environments = simulate.build_manifest_environments(
        manifest["simulation"]["environments"],
        base_dir=tmp_path,
    )
    assert [environment.name for environment in environments] == [
        "world_contract",
        "framework_trace",
    ]
    assert {"world_contract", "framework_trace"} <= set(
        simulate.supported_manifest_environment_types()
    )
    assert {
        "multi_agent_room",
        "retrieval_memory",
        "streaming_trace",
        "voice",
        "voice_replay",
        "world_orchestration_replay",
        "structured_artifact",
        "domain_package",
        "world_attack_replay",
        "autonomy_loop",
        "persistent_state_attack",
        "image",
        "vision",
        "framework_lifecycle",
        "framework_capability",
        "framework_probe",
        "framework_portability",
    } <= set(simulate.supported_manifest_environment_types())
    certification_environments = simulate.build_manifest_environments(
        [
            {
                "type": "framework_lifecycle",
                "data": {
                    "framework": "langgraph",
                    "session_id": "thread-123",
                    "phases": [{"id": "init", "stage": "initialize"}],
                },
            },
            {
                "type": "framework_capability",
                "data": {
                    "framework": "langgraph",
                    "capabilities": [{"name": "tool_calling", "category": "tools"}],
                },
            },
            {
                "type": "framework_probe",
                "data": {
                    "framework": "langgraph",
                    "probes": [{"id": "invoke", "operation": "invoke"}],
                },
            },
            {
                "type": "framework_portability",
                "data": {
                    "source_framework": "langgraph",
                    "target_framework": "openai_agents",
                    "mappings": [{"id": "invoke", "source": "invoke", "target": "run"}],
                },
            },
        ],
        base_dir=tmp_path,
    )
    assert [environment.name for environment in certification_environments] == [
        "framework_lifecycle",
        "framework_capability",
        "framework_probe",
        "framework_portability",
    ]
    task_world_environments = simulate.build_manifest_environments(
        [
            {
                "type": "structured_artifact",
                "data": {
                    "domain": "support",
                    "artifacts": {
                        "intake": {
                            "schema": "support_intake",
                            "data": {"ticket_id": "T-1", "priority": "high"},
                        }
                    },
                },
            },
            {
                "type": "domain_package",
                "data": {
                    "domain": "support",
                    "packages": {
                        "case": {
                            "package_type": "support_case",
                            "data": {"status": "ready"},
                        }
                    },
                },
            },
            {
                "type": "world_attack_replay",
                "data": {
                    "world_contract": {
                        "name": "support-world",
                        "transitions": [{"id": "resolve", "required": True}],
                    },
                    "attack_pack": {"attacks": ["prompt_injection"]},
                },
            },
            {
                "type": "autonomy_loop",
                "data": {
                    "goal": "resolve the support case safely",
                    "expected_plan": {"required_steps": ["inspect"]},
                },
            },
            {
                "type": "image",
                "data": {
                    "images": {
                        "receipt": {
                            "uri": "data:image/png;base64,iVBORw0KGgo=",
                            "description": "Refund receipt image fixture.",
                            "labels": ["receipt", "total_42"],
                        }
                    }
                },
            },
        ],
        base_dir=tmp_path,
    )
    assert [environment.name for environment in task_world_environments] == [
        "structured_artifacts",
        "domain_packages",
        "world_attack_replay",
        "autonomy_loop",
        "image",
    ]

    result = asyncio.run(simulate.run_manifest_file(manifest_path, no_eval=True))

    assert result["status"] == "passed"
    assert result["exit_code"] == 0
    assert result["summary"]["case_count"] == 1
    case = result["report"]["results"][0]
    assert "refund world contract completed" in case["transcript"]

    state = case["metadata"]["environment_state"]
    assert state["world_contract"]["state"]["refund"]["status"] == "approved"
    assert state["world_contract"]["summary"]["terminal_status"] == "success"
    assert state["framework_trace"]["adapter_conformance"]["passed"] is True
    assert state["framework_runtime"]["framework"] == "langgraph"
    assert state["framework_runtime"]["summary"]["tool_call_count"] == 2

    event_names = {(event["type"], event.get("name")) for event in case["events"]}
    assert ("world_contract", "world_transition_applied") in event_names
    assert ("framework_trace", "framework_trace_status") in event_names
    assert [
        message["tool_call_id"]
        for message in case["messages"]
        if message["role"] == "tool"
    ] == ["approve_refund", "framework_status"]


def test_task_run_manifest_builder_runs_world_task_with_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required_env = "AGENT_LEARNING_KIT_TASK_RUN_TEST_KEY"
    transition = {
        "id": "approve_refund",
        "actor": "agent",
        "resource": "refund",
        "action": "approve_refund",
        "required": True,
        "preconditions": {"refund.status": "pending"},
        "effects": {"refund.status": "approved"},
        "postconditions": {"refund.status": "approved"},
        "signals": ["refund_resolution"],
    }
    manifest = simulate.build_task_run_manifest(
        name="sdk-task-run-builder",
        required_env=[required_env],
        task_description=(
            "Approve the refund by applying the world transition and produce "
            "a complete final state."
        ),
        expected_result=(
            "The refund world transition is applied and the final state is "
            "approved and complete."
        ),
        agent={
            "type": "scripted",
            "responses": [
                {
                    "content": (
                        "First, because I approve the refund by applying the "
                        "refund world transition, I produce a complete final "
                        "state; the transition is applied, approved, and complete."
                    ),
                    "tool_calls": [
                        {
                            "id": "approve_refund",
                            "name": "apply_world_transition",
                            "arguments": {"id": "approve_refund"},
                        }
                    ],
                },
                {
                    "content": (
                        "Next, since I approve the refund by applying the "
                        "refund world transition, I produce a complete final "
                        "state; the transition is applied, approved, and complete."
                    ),
                },
                {
                    "content": (
                        "Finally, therefore I approve the refund by applying "
                        "the refund world transition and produce a complete "
                        "final state; the transition is applied and approved."
                    ),
                },
            ],
        },
        environments=[
            {
                "type": "world_contract",
                "data": {
                    "name": "sdk-task-run-refund-world",
                    "actors": ["agent", "customer"],
                    "resources": ["refund"],
                    "initial_state": {"refund": {"status": "pending"}},
                    "transitions": [transition],
                    "success_conditions": [
                        {
                            "id": "refund_approved",
                            "must": {"refund.status": "approved"},
                        }
                    ],
                },
            }
        ],
        required_tools=["apply_world_transition"],
        available_tools=["apply_world_transition", "world_contract_status"],
        success_criteria=[
            "refund world transition is applied",
            "final state is approved and complete",
        ],
        evaluation_config={
            "required_world_contract": [
                "world_contract",
                "transition",
                "success_condition",
                "refund",
            ],
            "world_contract_quality": {
                "required_transitions": ["approve_refund"],
                "min_completed_transitions": 1,
                "require_all_required_transitions": True,
                "required_success_conditions": ["refund_approved"],
                "terminal_status": "success",
                "expected_state": {"refund": {"status": "approved"}},
            },
        },
        threshold=0.85,
        min_turns=3,
        max_turns=3,
    )

    assert manifest["version"] == "agent-learning.run.v1"
    assert manifest["required_env"] == [required_env]
    assert manifest["evaluation"]["enabled"] is True
    assert manifest["evaluation"]["agent_report"]["config"]["required_tools"] == [
        "apply_world_transition"
    ]
    assert manifest["simulation"]["environments"][0]["type"] == "world_contract"

    manifest_path = simulate.write_manifest_file(manifest, tmp_path / "task-run.json")
    monkeypatch.setenv(required_env, "real-local-task-run-key")
    result = asyncio.run(simulate.run_manifest_file(manifest_path))

    assert result["status"] == "passed"
    assert result["summary"]["evaluation_score"] == 1.0
    assert result["summary"]["metric_averages"]["task_completion"] == 1.0
    assert result["summary"]["metric_averages"]["tool_selection_accuracy"] == 1.0
    assert result["summary"]["metric_averages"]["world_contract_quality"] == 1.0
    state = result["report"]["results"][0]["metadata"]["environment_state"]
    assert state["world_contract"]["state"]["refund"]["status"] == "approved"
    assert state["world_contract"]["summary"]["terminal_status"] == "success"


def test_public_manifest_command_detection_prioritizes_optimization() -> None:
    assert simulate.detect_manifest_command(
        {
            "redteam": {"auto_generate": True},
            "optimization": {
                "target": {
                    "base_config": {},
                    "search_space": {"redteam.surfaces": [["tool"]]},
                }
            },
        }
    ) == "optimize"


def test_public_eval_suite_api_runs_local_prompt_provider(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(_eval_suite()), encoding="utf-8")

    loaded = simulate.load_eval_suite_file(suite_path)
    result = simulate.run_eval_suite_file(suite_path)

    assert loaded["version"] == "agent-learning.eval.v1"
    assert result["kind"] == "agent-learning.eval.v1"
    assert result["status"] == "passed"
    assert result["exit_code"] == 0
    assert result["summary"]["case_count"] == 1
    assert result["summary"]["assertion_count"] == 2
    assert result["eval_suite"]["cases"][0]["output"] == (
        "Policy answer: refund policy is approved locally."
    )


def test_public_eval_suite_api_evaluates_saved_artifact_provider(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "summary": {"score": 1.0},
                "report": {
                    "results": [
                        {
                            "metadata": {
                                "environment_state": {
                                    "task_evidence": {
                                        "verification_status": "approved",
                                    }
                                }
                            },
                            "evaluation": {
                                "agent_report": {
                                    "summary": {
                                        "metric_averages": {
                                            "task_completion": 1.0,
                                        }
                                    }
                                }
                            },
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    suite_path = tmp_path / "artifact-suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "version": "agent-simulate.eval.v1",
                "name": "artifact-provider-eval",
                "providers": [
                    {
                        "id": "artifact",
                        "type": "artifact",
                        "path": "{{artifact_path}}",
                        "fields": [
                            {"name": "status", "path": "status"},
                            {
                                "name": "task_completion",
                                "path": (
                                    "report.results[0].evaluation.agent_report"
                                    ".summary.metric_averages.task_completion"
                                ),
                            },
                            {
                                "name": "verification_status",
                                "path": (
                                    "report.results[0].metadata.environment_state"
                                    ".task_evidence.verification_status"
                                ),
                            },
                        ],
                    }
                ],
                "prompts": [{"id": "task", "template": "{{artifact_path}}"}],
                "tests": [
                    {
                        "id": "task_artifact",
                        "vars": {"artifact_path": str(artifact_path)},
                        "assert": [
                            {
                                "type": "json_path_equals",
                                "path": "fields.status",
                                "value": "passed",
                            },
                            {
                                "type": "json_path_gte",
                                "path": "fields.task_completion",
                                "value": 1.0,
                            },
                            {
                                "type": "json_path_equals",
                                "path": "fields.verification_status",
                                "value": "approved",
                            },
                            {
                                "type": "json_path_exists",
                                "path": "artifact_path",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = simulate.run_eval_suite_file(suite_path)

    assert result["status"] == "passed"
    assert result["summary"]["assertion_count"] == 4
    case = result["eval_suite"]["cases"][0]
    assert case["provider_type"] == "artifact"
    assert {item["type"] for item in case["assertions"]} == {
        "json_path_equals",
        "json_path_exists",
        "json_path_gte",
    }
    assert '"task_completion": 1.0' in case["output"]
    assert '"verification_status": "approved"' in case["output"]


def test_public_eval_suite_api_reports_json_path_assertion_failures() -> None:
    suite = {
        "version": "agent-simulate.eval.v1",
        "name": "json-path-assertions",
        "providers": [
            {
                "id": "scripted",
                "type": "scripted",
                "response": json.dumps(
                    {
                        "metrics": {"score": 0.4},
                        "items": ["policy"],
                        "status": "warning",
                    }
                ),
            }
        ],
        "prompts": [{"id": "task", "template": "score"}],
        "tests": [
            {
                "id": "structured",
                "assert": [
                    {
                        "type": "json_path_contains",
                        "path": "items",
                        "value": "policy",
                    },
                    {"type": "json_path_lte", "path": "metrics.score", "value": 0.5},
                    {"type": "json_path_gte", "path": "metrics.score", "value": 0.9},
                    {"type": "json_path_exists", "path": "metrics.missing"},
                ],
            }
        ],
    }

    result = simulate.run_eval_suite(suite)

    assert result["status"] == "failed"
    assert result["summary"]["passed_assertion_count"] == 2
    assert result["summary"]["failed_assertion_count"] == 2
    case = result["eval_suite"]["cases"][0]
    failed = [item for item in case["assertions"] if not item["passed"]]
    assert failed[0]["type"] == "json_path_gte"
    assert failed[0]["actual"] == pytest.approx(0.4)
    assert failed[0]["path"] == "metrics.score"
    assert failed[1]["type"] == "json_path_exists"
    assert failed[1]["path"] == "metrics.missing"
    assert "missing key" in failed[1]["error"]
    assert case["findings"][0]["actual"] == pytest.approx(0.4)
    assert case["findings"][0]["path"] == "metrics.score"
