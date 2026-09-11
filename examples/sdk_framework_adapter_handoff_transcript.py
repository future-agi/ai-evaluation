import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalHandoffTeam"


@dataclass(frozen=True)
class HandoffMessage:
    source: str
    handoff_to: str
    task: str
    content: str
    type: str = "handoff"
    role: str = "assistant"
    reason: str = "specialized agent needed"


@dataclass(frozen=True)
class ReviewMessage:
    source: str
    review_target: str
    review_status: str
    content: str
    type: str = "review"
    role: str = "assistant"


@dataclass(frozen=True)
class ReconciliationMessage:
    source: str
    accepted_source: str
    reconciliation_status: str
    content: str
    type: str = "reconciliation"
    role: str = "assistant"


@dataclass(frozen=True)
class FinalMessage:
    source: str
    content: str
    type: str = "final_answer"
    role: str = "assistant"


@dataclass(frozen=True)
class HandoffTranscript:
    messages: list[Any]
    stop_reason: str


class LocalHandoffTeam:
    """Local multi-agent framework shim with handoff/review/reconciliation transcript."""

    def run(self, text: str) -> str:
        assert text
        return "Weak handoff transcript without coordination evidence."

    async def execute_task(self, payload: dict[str, Any]) -> HandoffTranscript:
        assert payload["metadata"]["framework"] == "openai_agents"
        return HandoffTranscript(
            messages=[
                HandoffMessage(
                    source="triage_agent",
                    handoff_to="retrieval_agent",
                    task="Gather current refund policy evidence.",
                    content="Triage hands refund policy research to retrieval.",
                ),
                HandoffMessage(
                    source="retrieval_agent",
                    handoff_to="critic_agent",
                    task="Review grounded refund recommendation.",
                    content="Retrieval hands cited recommendation to critic.",
                ),
                ReviewMessage(
                    source="critic_agent",
                    review_target="retrieval_agent",
                    review_status="passed",
                    content="Critic review confirms grounded handoff evidence.",
                ),
                ReconciliationMessage(
                    source="critic_agent",
                    accepted_source="retrieval_agent",
                    reconciliation_status="accepted",
                    content="Critic reconciles accepted source and closes handoff chain.",
                ),
                FinalMessage(
                    source="critic_agent",
                    content=(
                        "Handoff transcript adapter approved refund with "
                        "review and reconciliation evidence."
                    ),
                ),
            ],
            stop_reason="completed",
        )


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-handoff-transcript-run",
        framework="openai_agents",
        target=TARGET,
        method_candidates=["run", "execute_task"],
        input_mode_candidates=["text", "dict"],
        discovery_max_candidates=4,
        cases=[
            {
                "id": "handoff-transcript-refund",
                "input": "Approve the refund through a reviewed handoff transcript.",
                "expected_contains": ["approved refund"],
                "required_events": [
                    "framework_handoff",
                    "framework_review",
                    "framework_reconciliation",
                ],
                "required_state_keys": [
                    "framework_runtime",
                    "message_history",
                    "framework_handoffs",
                ],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-handoff-transcript"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_handoff_transcript_manifest"] = manifest

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
        else Path("artifacts") / "sdk-framework-adapter-handoff-transcript.json"
    )
    run(destination)
