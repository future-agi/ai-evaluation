from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalLangChainRunnable"


class LocalLangChainRunnable:
    """Local LangChain-style runnable promoted through BYO adapter probing."""

    def run(self, input: str) -> dict[str, Any]:
        assert input
        return {
            "content": "Weak run response without trace or tool evidence.",
            "tool_calls": [],
            "state": {"langchain_status": "weak"},
        }

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(input, dict):
            return {
                "content": "Weak invoke response without dict adapter evidence.",
                "tool_calls": [],
                "state": {"langchain_status": "weak"},
            }
        metadata = input.get("metadata")
        framework = metadata.get("framework") if isinstance(metadata, dict) else None
        if framework != "langchain":
            return {
                "content": "Weak invoke response without LangChain metadata.",
                "tool_calls": [],
                "state": {"langchain_status": "weak"},
            }
        content = str(input.get("content") or input.get("input") or input)
        trace = {
            "framework": "langchain",
            "spans": [
                {
                    "id": "prompt-template",
                    "name": "prompt_template.invoke",
                    "input": content,
                    "output": "refund_policy_prompt",
                    "signals": ["prompt", "chain", "policy"],
                },
                {
                    "id": "tool-policy",
                    "name": "framework_trace_status",
                    "input": {"case_id": "refund-43"},
                    "output": {"status": "passed"},
                    "tool_calls": [{"name": "framework_trace_status"}],
                    "signals": ["tool", "policy"],
                },
            ],
            "summary": {
                "span_count": 2,
                "tool_span_count": 1,
                "status": "passed",
            },
        }
        return {
            "content": (
                "LangChain invoke adapter approved refund with chain trace "
                "and framework_trace_status evidence."
            ),
            "tool_calls": [
                {
                    "id": "framework_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed", "framework": "langchain"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "langchain_invoke",
                    "payload": {"framework": "langchain", "status": "passed"},
                }
            ],
            "state": {
                "framework_trace": trace,
                "langchain_status": "verified",
            },
        }


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-langchain-invoke-promotion-run",
        framework="langchain",
        target=TARGET,
        method_candidates=["run", "invoke"],
        input_mode_candidates=["text", "dict", "agent_input"],
        discovery_max_candidates=4,
        cases=[
            {
                "id": "langchain-refund-status",
                "input": "Approve the refund through the LangChain runnable.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": ["framework_runtime", "framework_trace"],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-langchain-invoke-promotion"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    simulate.write_manifest_file(build_manifest(), manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
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
        else Path("artifacts")
        / "sdk-framework-adapter-langchain-invoke-promotion.json"
    )
    run(destination)
