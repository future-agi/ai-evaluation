import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalTypedOutputAgent"


@dataclass(frozen=True)
class RefundDecision:
    verdict: str
    rationale: str
    policy: str


class TypedAgentResult:
    """PydanticAI/OpenAI-Agents-style typed result shim."""

    def __init__(self, decision: RefundDecision) -> None:
        self.decision = decision

    def model_dump(self) -> dict[str, Any]:
        decision = asdict(self.decision)
        return {
            "content": (
                "Typed output adapter approved refund with structured state "
                f"under {decision['policy']}."
            ),
            "tool_calls": [
                {
                    "id": "typed_framework_status",
                    "name": "framework_trace_status",
                    "arguments": {
                        "status": "passed",
                        "schema": "RefundDecision",
                    },
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "typed_output",
                    "payload": {
                        "framework": "custom_typed_output_agent",
                        "schema": "RefundDecision",
                    },
                }
            ],
            "state": {
                "typed_output": {
                    "schema": "RefundDecision",
                    "decision": decision,
                }
            },
            "metadata": {
                "output_schema": "RefundDecision",
                "typed_output": decision,
            },
        }


class LocalTypedOutputAgent:
    """Local typed-output framework shim for adapter promotion."""

    def run(self, text: str) -> str:
        assert text
        return "Weak typed-output response without structured state."

    async def execute_task(self, payload: dict[str, Any]) -> TypedAgentResult:
        assert payload["metadata"]["framework"] == "custom_typed_output_agent"
        return TypedAgentResult(
            RefundDecision(
                verdict="approved",
                rationale="structured output preserved through the adapter",
                policy="refund_policy_2026",
            )
        )


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-typed-output-run",
        framework="custom_typed_output_agent",
        target=TARGET,
        method_candidates=["run", "execute_task"],
        input_mode_candidates=["text", "dict", "agent_input"],
        discovery_max_candidates=4,
        cases=[
            {
                "id": "typed-refund",
                "input": "Approve the refund and preserve typed output.",
                "expected_contains": ["approved refund"],
                "required_tools": ["framework_trace_status"],
                "required_events": ["framework_trace"],
                "required_state_keys": ["framework_runtime", "typed_output"],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-typed-output"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_typed_output_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-typed-output.json"
    )
    run(destination)
