"""Spec + Runner quickstart: one SimulationSpec, one SimulationRunner, offline.

Twin for docs/simulate/spec-and-runner.md. Runs fully offline (no API key, no
network): a synthetic user drives a conversation against a plain `.call()`
target through the single `SimulationRunner` spine, and the finished run is
written to the output path as JSON.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import fi.alk.simulate as S
from fi.simulate.agent.wrapper import AgentInput, AgentResponse


class EchoAgent:
    # Any object with an async .call(AgentInput) -> AgentResponse is a valid target.
    async def call(self, agent_input: AgentInput) -> AgentResponse:
        last = agent_input.messages[-1]["content"] if agent_input.messages else "hi"
        return AgentResponse(content=f"You said: {last}. How can I help further?")


def build_spec() -> "S.SimulationSpec":
    return S.SimulationSpec(
        run_id="spec_runner_quickstart",
        environment=S.EnvironmentSpec(
            adapter=S.EnvironmentAdapters.CHAT,
            world_kind=S.WorldKinds.CONVERSATION,
            config={"max_turns": 3, "min_turns": 1},
        ),
        target=S.AgentEndpointSpec(adapter=S.TargetAdapters.CALLABLE),
        simulator=S.SimulatorPolicySpec(adapter=S.SimulatorAdapters.SYNTHETIC_USER),
        scenario=S.Scenario(
            name="late-delivery",
            dataset=[
                S.Persona(
                    persona={"name": "Morgan", "role": "customer"},
                    situation="A delivery is 3 days late; ask for status and ETA.",
                    outcome="Get a clear status and a concrete next step.",
                )
            ],
        ),
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    report = asyncio.run(S.SimulationRunner().run(build_spec(), target=EchoAgent()))
    payload = {
        "run_id": "spec_runner_quickstart",
        "environments": sorted(S.environment_registry.names()),
        "status": str(report.status),
        "transcript": report.test_cases[0].result.transcript,
    }
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return payload


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else None
    payload = run(out)
    if out is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(f"wrote {out}")
