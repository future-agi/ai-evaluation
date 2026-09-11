from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from pydantic import ValidationError

from fi.simulate.agent.definition import (
    AgentDefinition,
    LiveKitSimulatorRuntime,
    SimulatorAgentDefinition,
)
from fi.simulate.manifest import ManifestError, validate_manifest_env
from fi.simulate.simulation.models import Scenario
from fi.simulate.voice import build_voice_run_manifest, run_voice_simulation


def add_voice_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agent-definition",
        required=True,
        help="Path to an AgentDefinition JSON/YAML file.",
    )
    scenarios = parser.add_mutually_exclusive_group(required=True)
    scenarios.add_argument("--scenario", help="Path to a Scenario JSON/YAML file.")
    scenarios.add_argument("--topic", help="Generate scenarios from this topic.")
    parser.add_argument(
        "--simulator", help="Optional SimulatorAgentDefinition JSON/YAML file."
    )
    parser.add_argument(
        "--livekit-runtime",
        help="FutureAGI LiveKitSimulatorRuntime JSON/YAML file.",
    )
    parser.add_argument("--num-scenarios", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--record-audio", action="store_true")
    parser.add_argument("--recording-root", default="recordings")
    parser.add_argument("--recorder-sample-rate", type=int, default=8000)
    parser.add_argument("--recorder-join-delay", type=float, default=0.2)
    parser.add_argument("--min-turn-messages", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=45.0)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--readiness-timeout", type=float, default=30.0)
    parser.add_argument("--cleanup-timeout", type=float, default=30.0)
    parser.add_argument(
        "--conversation-direction",
        choices=["simulator_first", "agent_first"],
        default="simulator_first",
    )
    parser.add_argument(
        "--agent-first-silence-timeout-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--write-manifest",
        help="Write a portable manifest; requires --scenario.",
    )
    parser.add_argument("-o", "--output", action="append", default=[])
    parser.add_argument("--junit", action="append", default=[])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")


async def run_voice_command(
    args: argparse.Namespace,
    *,
    load_object: Callable[[Path], dict[str, Any]],
    write_manifest: Callable[[Mapping[str, Any], str | Path], Path],
    evaluate_report: Callable[[Mapping[str, Any], Any], Any],
    result_builder: Callable[..., dict[str, Any]],
    write_outputs: Callable[
        [dict[str, Any], Mapping[str, Any], argparse.Namespace, Path], dict[str, Any]
    ],
) -> dict[str, Any]:
    agent_path = Path(args.agent_definition).expanduser().resolve()
    try:
        agent_definition = AgentDefinition(**load_object(agent_path))
        scenario = (
            Scenario(**load_object(Path(args.scenario).expanduser().resolve()))
            if args.scenario
            else None
        )
        simulator = (
            SimulatorAgentDefinition(
                **load_object(Path(args.simulator).expanduser().resolve())
            )
            if args.simulator
            else None
        )
        livekit_runtime = (
            LiveKitSimulatorRuntime(
                **load_object(Path(args.livekit_runtime).expanduser().resolve())
            )
            if args.livekit_runtime
            else None
        )
    except ValidationError as exc:
        raise ManifestError(f"invalid typed voice input: {exc}") from exc
    if args.write_manifest and scenario is None:
        raise ManifestError("--write-manifest requires --scenario")

    manifest = (
        build_voice_run_manifest(
            agent_definition=agent_definition,
            scenario=scenario,
            simulator=simulator,
            livekit_runtime=livekit_runtime,
            name=args.name,
            simulation_run_id=args.run_id,
            record_audio=args.record_audio,
            recording_root=args.recording_root,
            recorder_sample_rate=args.recorder_sample_rate,
            recorder_join_delay=args.recorder_join_delay,
            min_turn_messages=args.min_turn_messages,
            max_seconds=args.max_seconds,
            connect_timeout=args.connect_timeout,
            readiness_timeout=args.readiness_timeout,
            cleanup_timeout=args.cleanup_timeout,
            conversation_direction=args.conversation_direction,
            agent_first_silence_timeout_seconds=(
                args.agent_first_silence_timeout_seconds
            ),
            evaluation_enabled=not args.no_eval,
            threshold=args.threshold if args.threshold is not None else 0.7,
        )
        if scenario is not None
        else {"name": args.name or f"{agent_definition.name}-voice-simulation"}
    )
    if args.write_manifest:
        write_manifest(manifest, args.write_manifest)
    if args.dry_run:
        if scenario is None:
            raise ManifestError("--dry-run requires --scenario")
        validate_manifest_env(manifest)
        result = {
            "schema_version": "agent-simulate.cli.v1",
            "name": manifest["name"],
            "status": "passed",
            "exit_code": 0,
            "dry_run": True,
            "summary": {"scenario_cases": len(scenario.dataset)},
            "duration_seconds": 0.0,
        }
        return write_outputs(result, manifest, args, agent_path)

    started = time.monotonic()
    report = await run_voice_simulation(
        agent_definition=agent_definition,
        scenario=scenario,
        simulator=simulator,
        livekit_runtime=livekit_runtime,
        topic=args.topic,
        num_scenarios=args.num_scenarios,
        simulation_run_id=args.run_id,
        record_audio=args.record_audio,
        recording_root=args.recording_root,
        recorder_sample_rate=args.recorder_sample_rate,
        recorder_join_delay=args.recorder_join_delay,
        min_turn_messages=args.min_turn_messages,
        max_seconds=args.max_seconds,
        connect_timeout=args.connect_timeout,
        readiness_timeout=args.readiness_timeout,
        cleanup_timeout=args.cleanup_timeout,
        conversation_direction=args.conversation_direction,
        agent_first_silence_timeout_seconds=(
            args.agent_first_silence_timeout_seconds
        ),
    )
    evaluation = None if args.no_eval else evaluate_report(manifest, report)
    result = result_builder(
        manifest=manifest,
        report=report,
        evaluation=evaluation,
        duration_seconds=round(time.monotonic() - started, 4),
    )
    return write_outputs(result, manifest, args, agent_path)
