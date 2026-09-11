"""Run one harness voice scenario through ALK's public SimulationRunner.

This is deliberately a thin adapter.  The harness owns the generated world and
exports the scenario/connection through environment variables; the SDK owns the
LiveKit room, simulated caller, transcript, recordings, and terminal status.
"""

from __future__ import annotations

import argparse
import logging
import asyncio
import json
import os
from pathlib import Path

from fi.simulate.runtime import SimulationSpec, new_run_id
from fi.simulate.runtime.runner import SimulationRunner

from fi.alk.harness.simulator_voice import (
    caller_scenario,
    simulation_spec,
    simulator_definition,
)

logger = logging.getLogger(__name__)


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing environment variable: {name}")
    return value


def _env(name: str) -> str:
    return os.environ.get(name, "")


def _json_env(name: str, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    parsed = json.loads(raw)
    return parsed


def build_spec(run_id: str) -> SimulationSpec:
    """The local lane: every value comes from a HARNESS_* environment variable."""
    persona = _json_env("HARNESS_PERSONA", {"name": "customer"})
    simulator = simulator_definition(_env, persona)
    direction = os.environ.get("HARNESS_CONVERSATION_DIRECTION", "agent_first")
    max_seconds = float(os.environ.get("VOICE_MAX_SECONDS", "300"))
    recording_dir = _output_root() / run_id / "1.1.2" / "recordings"
    return simulation_spec(
        run_id=run_id,
        room_name=f"harness-{run_id}",
        agent_name=_required("LIVEKIT_TARGET_AGENT_NAME"),
        system_prompt=_required("LIVEKIT_TARGET_SYSTEM_PROMPT"),
        livekit_url=os.environ.get("ACCEPTANCE_LIVEKIT_URL") or _required("LIVEKIT_URL"),
        recording_dir=recording_dir,
        scenario=caller_scenario(
            name=os.environ.get("HARNESS_SCENARIO", "harness-voice"),
            persona=persona,
            situation=_required("HARNESS_INSTRUCTION"),
            fixture=_json_env("HARNESS_FIXTURE", {}),
            tts_provider=simulator.tts.provider,
            outcome=os.environ.get("HARNESS_OUTCOME", ""),
            initial_message=os.environ.get("HARNESS_INITIAL_MESSAGE", ""),
        ),
        simulator=simulator,
        direction=direction,
        max_seconds=max_seconds,
        min_turn_messages=int(os.environ.get("VOICE_MIN_TURN_MESSAGES", "6")),
        agent_first_silence_seconds=float(
            os.environ.get("VOICE_AGENT_FIRST_SILENCE_SECONDS", "45")
        ),
        run_seconds=max(300.0, max_seconds + 60 + 120 + 30 + 60),
    )


def _output_root() -> Path:
    return Path(
        os.environ.get("HARNESS_VOICE_OUTPUT_ROOT", "artifacts/simulation-acceptance")
    )


async def _run(run_id: str) -> int:
    report = await SimulationRunner().run(build_spec(run_id))
    output = _output_root() / run_id / "1.1.2"
    output.mkdir(parents=True, exist_ok=True)
    # The harness evidence reader still consumes the legacy TestReport envelope.
    # Keep that local compatibility boundary while execution itself uses the new
    # typed runner and report internally.
    legacy = report.to_legacy()
    (output / "report.json").write_text(
        legacy.model_dump_json(indent=2), encoding="utf-8"
    )
    (output / "canonical-report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    case = report.test_cases[0] if report.test_cases else None
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": report.status.value,
                "test_case_status": case.status.value if case else "missing",
                "report": str(output / "canonical-report.json"),
            },
            indent=2,
        )
    )
    return 0 if case is not None and case.status.value == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("1.1.2",))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run_id = new_run_id()
    if args.dry_run:
        print(build_spec(run_id).model_dump_json(indent=2))
        return 0
    return asyncio.run(_run(run_id))


if __name__ == "__main__":
    raise SystemExit(main())
