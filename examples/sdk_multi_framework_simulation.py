from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from fi.alk import configure, simulate, suite


REQUIRED_ENV = "AGENT_LEARNING_SDK_MULTI_FRAMEWORK_EXAMPLE_KEY"
EXAMPLES_DIR = Path(__file__).resolve().parent
SHIMS = EXAMPLES_DIR / "framework_shims.py"


FRAMEWORKS = [
    {
        "id": "langchain-runnable",
        "framework": "langchain",
        "factory": "build_langchain_agent",
        "persona": {"name": "Maya", "role": "framework-owner"},
        "situation": (
            "Maya needs a LangChain-style runnable simulated through the "
            "generic framework adapter."
        ),
        "outcome": (
            "The LangChain-style runnable completes with framework runtime "
            "trace evidence."
        ),
        "trace": {
            "span_id": "langchain_runnable",
            "span_name": "RunnableSequence.ainvoke",
            "input": "support workflow",
            "output": "completed",
            "signals": ["model", "tool", "chain"],
        },
    },
    {
        "id": "langgraph-state-graph",
        "framework": "langgraph",
        "factory": "build_langgraph_agent",
        "persona": {"name": "Riya", "role": "orchestration-owner"},
        "situation": (
            "Riya needs a LangGraph-style state graph simulated through the "
            "generic framework adapter."
        ),
        "outcome": (
            "The LangGraph-style graph completes with stateful framework "
            "runtime trace evidence."
        ),
        "trace": {
            "span_id": "langgraph_node",
            "span_name": "refund_graph.ainvoke",
            "input": "refund workflow",
            "output": "completed",
            "signals": ["model", "tool", "state"],
        },
    },
    {
        "id": "llamaindex-chat-engine",
        "framework": "llamaindex",
        "factory": "build_llamaindex_chat_engine",
        "persona": {"name": "Ishan", "role": "retrieval-agent-owner"},
        "situation": (
            "Ishan needs a LlamaIndex-style chat engine simulated through the "
            "generic framework adapter."
        ),
        "outcome": (
            "The LlamaIndex-style chat engine completes with retrieval "
            "framework runtime trace evidence."
        ),
        "trace": {
            "span_id": "llamaindex_chat_engine",
            "span_name": "chat_engine.achat",
            "input": "retrieval workflow",
            "output": "completed",
            "signals": ["retrieval", "index", "tool"],
        },
    },
    {
        "id": "openai-agents-runner",
        "framework": "openai_agents",
        "factory": "build_openai_agents_runner",
        "persona": {"name": "Omar", "role": "agent-sdk-owner"},
        "situation": (
            "Omar needs an OpenAI Agents-style runner simulated through the "
            "generic framework adapter."
        ),
        "outcome": (
            "The OpenAI Agents-style runner completes with handoff framework "
            "runtime trace evidence."
        ),
        "trace": {
            "span_id": "openai_agents_runner",
            "span_name": "Runner.run",
            "input": "handoff workflow",
            "output": "completed",
            "signals": ["agent", "handoff", "tool"],
        },
    },
    {
        "id": "autogen-agent-chat",
        "framework": "autogen",
        "factory": "build_autogen_agent",
        "persona": {"name": "Tara", "role": "multi-agent-owner"},
        "situation": (
            "Tara needs an AutoGen-style agent chat simulated through the "
            "generic framework adapter."
        ),
        "outcome": (
            "The AutoGen-style agent chat completes with group conversation "
            "runtime trace evidence."
        ),
        "trace": {
            "span_id": "autogen_agent_chat",
            "span_name": "AgentChat.run",
            "input": "groupchat workflow",
            "output": "completed",
            "signals": ["agent", "groupchat", "tool"],
        },
    },
    {
        "id": "crewai-crew",
        "framework": "crewai",
        "factory": "build_crewai_crew",
        "persona": {"name": "Leela", "role": "crew-owner"},
        "situation": (
            "Leela needs a CrewAI-style crew simulated through the generic "
            "framework adapter."
        ),
        "outcome": (
            "The CrewAI-style crew completes with manager and worker runtime "
            "trace evidence."
        ),
        "trace": {
            "span_id": "crewai_crew",
            "span_name": "Crew.kickoff",
            "input": "crew workflow",
            "output": "completed",
            "signals": ["crew", "role", "tool"],
        },
    },
    {
        "id": "pydantic-ai-agent",
        "framework": "pydantic_ai",
        "factory": "build_pydantic_ai_agent",
        "persona": {"name": "Sana", "role": "typed-agent-owner"},
        "situation": (
            "Sana needs a PydanticAI-style typed agent simulated through the "
            "generic framework adapter."
        ),
        "outcome": (
            "The PydanticAI-style agent completes with typed output framework "
            "runtime trace evidence."
        ),
        "trace": {
            "span_id": "pydantic_ai_agent",
            "span_name": "Agent.run",
            "input": "typed workflow",
            "output": "completed",
            "signals": ["agent", "schema", "tool"],
        },
    },
    {
        "id": "pipecat-voice-pipeline",
        "framework": "pipecat",
        "factory": "build_pipecat_pipeline",
        "modality": "voice",
        "persona": {"name": "Asha", "role": "voice-agent-owner"},
        "situation": (
            "Asha needs a Pipecat-style voice pipeline simulated through the "
            "generic framework adapter."
        ),
        "outcome": (
            "The Pipecat-style pipeline completes with voice framework runtime "
            "trace evidence."
        ),
        "trace": {
            "span_id": "pipecat_pipeline",
            "span_name": "pipeline.process",
            "input": "voice handoff",
            "output": "completed",
            "signals": ["voice", "frame", "tool"],
        },
    },
    {
        "id": "livekit-realtime-agent",
        "framework": "livekit",
        "factory": "build_livekit_agent",
        "modality": "voice",
        "persona": {"name": "Kabir", "role": "realtime-agent-owner"},
        "situation": (
            "Kabir needs a LiveKit-style realtime agent simulated through the "
            "generic framework adapter."
        ),
        "outcome": (
            "The LiveKit-style agent completes with realtime voice framework "
            "runtime trace evidence."
        ),
        "trace": {
            "span_id": "livekit_room_agent",
            "span_name": "agent.respond",
            "input": "voice room message",
            "output": "completed",
            "signals": ["voice", "room", "tool"],
        },
    },
    {
        "id": "custom-refund-orchestrator",
        "framework": "custom_refund_orchestrator",
        "factory": "build_custom_refund_orchestrator",
        "method": "execute_task",
        "input_mode": "dict",
        "persona": {"name": "Nia", "role": "framework-owner"},
        "situation": (
            "Nia needs a proprietary refund orchestrator simulated through a "
            "custom framework adapter."
        ),
        "outcome": (
            "The custom orchestrator completes with framework runtime trace "
            "evidence."
        ),
        "trace": {
            "span_id": "custom_refund_orchestrator",
            "span_name": "CustomRefundOrchestrator.execute_task",
            "input": "refund workflow",
            "output": "approved",
            "signals": ["planner", "tool", "policy"],
        },
    },
]


def build_framework_manifests() -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for spec in FRAMEWORKS:
        framework = str(spec["framework"])
        trace = spec["trace"]
        manifests[str(spec["id"])] = simulate.build_framework_run_manifest(
            name=f"sdk-{spec['id']}",
            framework=framework,
            target=f"{SHIMS}:{spec['factory']}",
            required_env=[REQUIRED_ENV],
            method=spec.get("method"),
            input_mode=spec.get("input_mode"),
            modality=spec.get("modality"),
            metadata={"cookbook": "multi-framework-simulation"},
            scenario={
                "name": f"sdk-{spec['id']}",
                "dataset": [
                    {
                        "persona": spec["persona"],
                        "situation": spec["situation"],
                        "outcome": spec["outcome"],
                    }
                ],
            },
            framework_trace={
                "framework": framework,
                "spans": [
                    {
                        "id": trace["span_id"],
                        "name": trace["span_name"],
                        "input": trace["input"],
                        "output": trace["output"],
                        "tool_calls": [{"name": "framework_trace_status"}],
                        "signals": trace["signals"],
                    }
                ],
                "adapter_required_signals": trace["signals"],
                "adapter_required_mappings": {"tool": ["tool_name"]},
            },
        )
    return manifests


def write_framework_workspace(directory: str | Path) -> Path:
    root = Path(directory).expanduser().resolve()
    manifests_dir = root / "manifests"
    manifest_paths: list[dict[str, Any]] = []
    for manifest_id, manifest in build_framework_manifests().items():
        path = simulate.write_manifest_file(
            manifest,
            manifests_dir / f"{manifest_id}.json",
        )
        manifest_paths.append(
            {
                "id": manifest_id,
                "framework": manifest["agent"]["framework"],
                "path": path.name,
            }
        )
    suite_manifest = simulate.build_multi_framework_suite_manifest(
        name="sdk-multi-framework-simulation",
        required_env=[REQUIRED_ENV],
        framework_manifests=manifest_paths,
        metadata={"cookbook": "sdk-multi-framework-simulation"},
    )
    return suite.write_suite_file(
        suite_manifest,
        manifests_dir / "multi_framework_suite.json",
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    output = Path(output_path).expanduser() if output_path is not None else None
    workspace = (
        output.parent / "sdk-multi-framework-workspace"
        if output is not None
        else Path(tempfile.gettempdir()) / "agent-learning-sdk-multi-framework-workspace"
    )
    suite_path = write_framework_workspace(workspace)
    result = suite.run_suite_file(suite_path)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
