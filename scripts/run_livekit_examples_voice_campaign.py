#!/usr/bin/env python3
"""Run real WebRTC calls against unchanged official LiveKit example agents."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from fi import simulate
from fi.alk.harness.contract import AgentContract, Runtime, ToolSpec
from fi.alk.harness.provision import provision, start_runtime, stop


DRIVE_THRU_CASES = [
    {
        "label": "Large Big Mac meal with explicit drink",
        "instructions": (
            "You are Jordan Lee ordering dinner after a long shift. Start by asking for one "
            "large Big Mac meal with a large Diet Coke. After it is added, ask for one Oreo "
            "McFlurry. Listen to the final order, correct any wrong size or duplicate, then say "
            "that is everything and end politely. Speak naturally and never add anything you "
            "did not request."
        ),
        "agent_expectations": "The final cart contains exactly the requested large meal and one Oreo McFlurry.",
    },
    {
        "label": "Unavailable sweet tea accepts alternative",
        "instructions": (
            "You are Aisha Grant. Ask for a Quarter Pounder with Cheese combo with a medium "
            "sweet tea. Sweet tea is important, but if it is unavailable, ask what unsweetened "
            "tea is available and accept a medium unsweetened iced tea. Finish after the correct "
            "combo is confirmed."
        ),
        "agent_expectations": "The agent does not invent sweet-tea availability and records the accepted substitute.",
    },
    {
        "label": "Two Happy Meals with distinct choices",
        "instructions": (
            "You are Miguel Santos ordering for two children. Order two Happy Meals, one with "
            "hamburger, apple slices, and milk, and the other with four-piece McNuggets, fries, "
            "and chocolate milk. Answer clarification questions one fact at a time. Make sure "
            "both distinct meals are present, then finish."
        ),
        "agent_expectations": "The final cart contains two distinct Happy Meals with the stated choices.",
    },
    {
        "label": "Change drink after ordering",
        "instructions": (
            "You are Priya Shah. Order a medium McCrispy combo with a medium Sprite. Once the "
            "agent confirms it, change only the drink to a medium Coca-Cola. Do not accept both "
            "drinks remaining in the order. Ask for the revised order and finish when correct."
        ),
        "agent_expectations": "The agent removes/replaces the old item rather than leaving duplicate drinks or combos.",
    },
    {
        "label": "Reject nonexistent item and keep valid order",
        "instructions": (
            "You are Elena Rossi. Ask for a Whopper. When told it is not sold here, accept one "
            "regular Big Mac instead and add a small Coca-Cola. Do not accept an invented "
            "Whopper or cheeseburger substitution. Confirm the two valid items and end."
        ),
        "agent_expectations": "The agent refuses the off-menu item and adds only the explicitly accepted valid items.",
    },
]

DIRECT_PROVIDER_CASES: dict[str, list[dict[str, str]]] = {
    "multi_agent": [
        {
            "label": "Nia from Nairobi adventure",
            "instructions": (
                "Your name is Nia and you are from Nairobi. Answer those questions naturally. "
                "Ask for a short adventurous story involving a clever wildlife photographer. "
                "Make one clear choice when the storyteller asks, then after the next story beat "
                "say you are happy with the ending, thank Echo, and say goodbye."
            ),
            "agent_expectations": "The agent gathers Nia and Nairobi, hands off, and personalizes the story.",
        },
        {
            "label": "Mateo from Mexico City mystery",
            "instructions": (
                "Your name is Mateo and you are from Mexico City. Request a compact mystery set "
                "around a missing street-food recipe. When offered a decision, choose to question "
                "the night-market vendor. Let the agent resolve that beat, then decline more story "
                "and end warmly."
            ),
            "agent_expectations": "The handoff preserves Mateo and Mexico City and produces an interactive mystery.",
        },
        {
            "label": "Mei from Singapore science fiction",
            "instructions": (
                "Your name is Mei and you are from Singapore. Ask for a hopeful science-fiction "
                "story about an engineer repairing a floating garden. Choose to repair the water "
                "system when prompted. After a meaningful outcome, say that is enough and goodbye."
            ),
            "agent_expectations": "The story is personalized and the user's choice affects the next story beat.",
        },
        {
            "label": "Arjun from Pune comic folklore",
            "instructions": (
                "Your name is Arjun and you are from Pune. Ask for a funny folklore-inspired story "
                "with a stubborn inventor. Answer one interactive question with 'test the machine "
                "carefully'. After the consequence is narrated, ask to finish and say goodbye."
            ),
            "agent_expectations": "The agent collects both facts, transitions agents, and responds to the choice.",
        },
        {
            "label": "Laila from Cairo gentle bedtime story",
            "instructions": (
                "Your name is Laila and you are from Cairo. Ask for a calm, very short bedtime story "
                "about a lost astronomer finding home. If asked to choose, follow the brightest star. "
                "Once home is found, decline another story, thank the agent, and say goodbye."
            ),
            "agent_expectations": "The story uses Laila and Cairo and reaches a user-confirmed ending.",
        },
    ],
    "weather_agent": [
        {
            "label": "Reykjavik current temperature",
            "instructions": (
                "Ask for the current temperature in Reykjavik, Iceland because you are packing for "
                "an evening walk. Do not provide coordinates; let the agent estimate them. After it "
                "gives a temperature and unit, acknowledge it and say goodbye."
            ),
            "agent_expectations": "The real weather tool is called with estimated Reykjavik coordinates.",
        },
        {
            "label": "Singapore current temperature",
            "instructions": (
                "Ask what the current temperature is in Singapore before leaving the airport. If "
                "the agent asks for latitude or longitude, politely repeat that the city should be "
                "enough. End after receiving a Celsius temperature."
            ),
            "agent_expectations": "The agent estimates coordinates and returns live weather data.",
        },
        {
            "label": "Cape Town current temperature",
            "instructions": (
                "Ask for the current temperature in Cape Town, South Africa. Mention that this is "
                "for deciding whether to take a jacket. Keep the exchange concise and say goodbye "
                "once a temperature with units is supplied."
            ),
            "agent_expectations": "The external weather API is used for Cape Town rather than inventing a value.",
        },
        {
            "label": "Sapporo current temperature",
            "instructions": (
                "Ask for the current temperature in Sapporo, Japan. Do not accept a generic seasonal "
                "description as the answer; briefly ask for the actual current reading if necessary. "
                "Thank the agent and finish after the reading."
            ),
            "agent_expectations": "A numeric current reading is returned from the real tool.",
        },
        {
            "label": "Buenos Aires current temperature",
            "instructions": (
                "Ask for the current temperature in Buenos Aires, Argentina for a bicycle ride. "
                "Answer any normal clarification briefly, then close the conversation naturally "
                "after receiving the current Celsius temperature."
            ),
            "agent_expectations": "The tool resolves Buenos Aires coordinates and returns current data.",
        },
    ],
    "structured_output": [
        {
            "label": "Calm delayed-flight reassurance",
            "instructions": (
                "Say your flight has been delayed twice and ask for one calm sentence to help you "
                "reset before speaking to the airline desk. Respond naturally to the answer, then "
                "thank the assistant and say goodbye."
            ),
            "agent_expectations": "The spoken response is concise and uses a calm emotional directive.",
        },
        {
            "label": "Excited promotion celebration",
            "instructions": (
                "Tell Echo that you just received a long-awaited promotion and ask it to celebrate "
                "with you in an energetic but brief way. Share that you are taking your family to "
                "dinner, then end the conversation."
            ),
            "agent_expectations": "The generated speech adapts to an excited celebratory tone.",
        },
        {
            "label": "Gentle apology rehearsal",
            "instructions": (
                "Ask the assistant to help you rehearse a sincere, gentle apology for missing a "
                "friend's art show. Ask for a slower version if the first answer sounds rushed. "
                "Thank it and finish once you have a usable sentence."
            ),
            "agent_expectations": "Structured voice instructions produce a gentle, measured response.",
        },
        {
            "label": "Suspenseful treasure clue",
            "instructions": (
                "Ask for a single suspenseful clue leading to a fictional treasure hidden in an old "
                "library. Guess that it means looking behind the atlas, react to the reply, and then "
                "say goodbye."
            ),
            "agent_expectations": "The assistant expresses suspense without leaking its structured directive.",
        },
        {
            "label": "Warm first-day encouragement",
            "instructions": (
                "Say tomorrow is your first day managing a new team and ask for warm, confident "
                "encouragement in no more than two sentences. Say which phrase helped most, thank "
                "the assistant, and close the call."
            ),
            "agent_expectations": "The response is brief, warm, and emotionally appropriate.",
        },
    ],
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:72]


def _official_cases(source: Path, limit: int) -> list[dict[str, Any]]:
    document = yaml.safe_load((source / "scenarios.yaml").read_text(encoding="utf-8"))
    return list(document["scenarios"][:limit])


def _cases(name: str, source: Path, limit: int) -> list[dict[str, Any]]:
    if name in DIRECT_PROVIDER_CASES:
        return DIRECT_PROVIDER_CASES[name][:limit]
    if name == "drive_thru":
        return DRIVE_THRU_CASES[:limit]
    return _official_cases(source, limit)


def _source_for(examples_root: Path, name: str) -> Path:
    if name in DIRECT_PROVIDER_CASES:
        return (examples_root / "voice_agents").resolve()
    return (examples_root / name).resolve()


def _simulator() -> simulate.SimulatorAgentDefinition:
    return simulate.SimulatorAgentDefinition(
        llm={
            "provider": os.environ.get("SIMULATOR_LLM_PROVIDER", "google"),
            "model": os.environ.get("SIMULATOR_LLM_MODEL", "gemini-2.5-flash-lite"),
            "temperature": 0.35,
        },
        stt={
            "provider": os.environ.get("SIMULATOR_STT_PROVIDER", "deepgram"),
            "model": os.environ.get("SIMULATOR_STT_MODEL", "nova-2"),
            "language": "en",
        },
        tts={
            "provider": os.environ.get("SIMULATOR_TTS_PROVIDER", "deepgram"),
            "model": os.environ.get("SIMULATOR_TTS_MODEL", "aura-2-thalia-en"),
            "voice": os.environ.get("SIMULATOR_TTS_VOICE", "aura-2-thalia-en"),
        },
        instructions=(
            "Follow the scenario exactly as a natural caller. Keep each turn brief. Reveal facts "
            "only when the agent asks or when the ordered scenario step requires them. Do not "
            "restart the request, repeat an answer more than twice, invent facts, or end before "
            "the requested outcome is resolved. When complete, say one natural closing sentence "
            "and invoke the endCall tool in that same turn. Never wait for another response after "
            "your closing sentence."
        ),
        allow_interruptions=True,
    )


def _dispatch(run_id: str, case: dict[str, Any]) -> dict[str, Any]:
    return {
        "simulation_run_id": run_id,
        "job_id": f"job-{uuid.uuid4().hex[:12]}",
        "scenario": {
            "label": str(case["label"]),
            "instructions": str(case["instructions"]),
            "agent_expectations": str(case.get("agent_expectations") or ""),
            "tags": {
                str(key): str(value) for key, value in (case.get("tags") or {}).items()
            },
            "userdata": json.dumps(case.get("userdata") or {}, sort_keys=True),
        },
    }


def _result_summary(
    report: simulate.TestReport, case: dict[str, Any]
) -> dict[str, Any]:
    result = report.results[0] if report.results else None
    metadata = dict(result.metadata or {}) if result is not None else {}
    return {
        "label": case["label"],
        "status": metadata.get("status", "missing"),
        "stop_reason": metadata.get("stop_reason"),
        "failure": metadata.get("failure"),
        "duration_seconds": metadata.get("duration_seconds"),
        "transcript_characters": len(result.transcript) if result is not None else 0,
        "message_count": len(result.messages) if result is not None else 0,
        "audio_input_path": result.audio_input_path if result is not None else None,
        "audio_output_path": result.audio_output_path if result is not None else None,
        "audio_combined_path": result.audio_combined_path
        if result is not None
        else None,
        "audio_stereo_path": result.audio_stereo_path if result is not None else None,
    }


async def run_campaign(
    *, examples_root: Path, output: Path, agent_names: list[str], count: int
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    overall: dict[str, Any] = {
        "started_at_epoch": time.time(),
        "examples_root": str(examples_root),
        "agents": {},
    }
    failures = 0
    credentials = {
        name: os.environ[name]
        for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    }
    provider_credentials = {
        name: os.environ[name]
        for name in (
            "OPENAI_API_KEY",
            "DEEPGRAM_API_KEY",
            "CARTESIA_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
        )
        if os.environ.get(name)
    }
    for name in agent_names:
        source = _source_for(examples_root, name)
        destination = output / name / "environment"
        worker_name = f"alk-{name.replace('_', '-')}-{uuid.uuid4().hex[:8]}"
        contract = AgentContract(
            agent=f"livekit-{name}",
            modality="voice",
            tools=[ToolSpec(name="upstream_agent_tools")],
            real_use_cases=["run official LiveKit voice scenarios"],
            runtime=Runtime(
                dockerfile=(
                    "ALK.Dockerfile" if name in DIRECT_PROVIDER_CASES else "Dockerfile"
                )
            ),
        )
        agent_record: dict[str, Any] = {"worker_name": worker_name, "calls": []}
        overall["agents"][name] = agent_record
        runtime_container = ""
        print(json.dumps({"event": "agent_provisioning", "agent": name}), flush=True)
        try:
            provision(source, destination, contract)
            runtime_overrides = {
                **credentials,
                **provider_credentials,
                "LIVEKIT_AGENT_NAME": worker_name,
            }
            if name in DIRECT_PROVIDER_CASES:
                runtime_overrides["ALK_AGENT_SCRIPT"] = f"{name}.py"
            if name == "hotel_receptionist":
                runtime_overrides["HOTEL_TODAY"] = "2026-06-08"
            runtime_environment = start_runtime(
                destination, overrides=runtime_overrides
            )
            runtime_container = runtime_environment.runtime_container
            print(json.dumps({"event": "agent_ready", "agent": name}), flush=True)
            for index, case in enumerate(_cases(name, source, count), start=1):
                call_id = f"{index:02d}-{_slug(str(case['label']))}"
                call_dir = output / name / "calls" / call_id
                call_dir.mkdir(parents=True, exist_ok=True)
                run_id = f"{name}-{index}-{uuid.uuid4().hex[:10]}"
                agent_definition = simulate.AgentDefinition(
                    name=name.replace("_", " "),
                    agent_name=worker_name,
                    system_prompt=(
                        "Use the submitted official LiveKit agent implementation and its own "
                        "business rules and tools."
                    ),
                    # Frontdesk and Hotel consume LiveKit SimulationDispatch userdata to seed
                    # their source-owned worlds. Drive-thru has no simulation hook, and generic
                    # templates may interpret arbitrary metadata as an outbound/no-greet job.
                    dispatch_metadata=(
                        _dispatch(run_id, case)
                        if name in {"frontdesk", "hotel_receptionist"}
                        else None
                    ),
                    transport={"kind": "webrtc"},
                )
                scenario = simulate.Scenario(
                    name=call_id,
                    dataset=[
                        simulate.Persona(
                            persona={"name": str(case["label"]), "role": "customer"},
                            situation=str(case["instructions"]),
                            outcome=str(
                                case.get("agent_expectations")
                                or "Complete the requested task accurately."
                            ),
                        )
                    ],
                )
                print(
                    json.dumps(
                        {
                            "event": "call_started",
                            "agent": name,
                            "index": index,
                            "label": case["label"],
                        }
                    ),
                    flush=True,
                )
                try:
                    report = await simulate.run_voice_simulation(
                        agent_definition=agent_definition,
                        livekit_runtime=simulate.LiveKitSimulatorRuntime(
                            url=credentials["LIVEKIT_URL"],
                            room_name=f"alk-{run_id}",
                            room_mode="managed",
                        ),
                        scenario=scenario,
                        simulator=_simulator(),
                        simulation_run_id=run_id,
                        record_audio=True,
                        recording_root=call_dir / "recordings",
                        recording_case_directory=call_dir / "recordings",
                        min_turn_messages=4,
                        max_seconds=240 if name != "hotel_receptionist" else 420,
                        connect_timeout=60,
                        readiness_timeout=120,
                        cleanup_timeout=45,
                        # Drive-thru has no on_enter greeting; Frontdesk and Hotel explicitly do.
                        conversation_direction=(
                            "agent_first"
                            if name
                            in {"frontdesk", "hotel_receptionist", "multi_agent"}
                            else "simulator_first"
                        ),
                        agent_first_silence_timeout_seconds=45,
                    )
                    (call_dir / "report.json").write_text(
                        report.model_dump_json(indent=2), encoding="utf-8"
                    )
                    summary = _result_summary(report, case)
                    if summary.get("status") != "completed":
                        failures += 1
                except Exception as exc:  # keep the campaign and its evidence running
                    failures += 1
                    summary = {
                        "label": case["label"],
                        "status": "infrastructure_exception",
                        "exception_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    (call_dir / "exception.json").write_text(
                        json.dumps(summary, indent=2), encoding="utf-8"
                    )
                agent_record["calls"].append(summary)
                (output / "campaign-summary.json").write_text(
                    json.dumps(overall, indent=2), encoding="utf-8"
                )
                print(
                    json.dumps(
                        {
                            "event": "call_completed",
                            "agent": name,
                            "index": index,
                            "status": summary.get("status"),
                            "transcript_characters": summary.get(
                                "transcript_characters", 0
                            ),
                        }
                    ),
                    flush=True,
                )
        except Exception as exc:
            failures += 1
            agent_record["setup_failure"] = {
                "exception_type": type(exc).__name__,
                "error": str(exc),
            }
            (output / "campaign-summary.json").write_text(
                json.dumps(overall, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "event": "agent_setup_failed",
                        "agent": name,
                        "exception_type": type(exc).__name__,
                        "error": str(exc),
                    }
                ),
                flush=True,
            )
        finally:
            if runtime_container:
                completed = subprocess.run(
                    ["docker", "logs", "--tail", "1000", runtime_container],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                (output / name / "worker.log").write_text(
                    (completed.stdout or "") + (completed.stderr or ""),
                    encoding="utf-8",
                )
            stop(destination)
            print(json.dumps({"event": "agent_stopped", "agent": name}), flush=True)
    overall["finished_at_epoch"] = time.time()
    (output / "campaign-summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=(
            "drive_thru",
            "frontdesk",
            "hotel_receptionist",
            "multi_agent",
            "weather_agent",
            "structured_output",
        ),
        default=["drive_thru", "frontdesk", "hotel_receptionist"],
    )
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    return asyncio.run(
        run_campaign(
            examples_root=args.examples_root.resolve(),
            output=args.output.resolve(),
            agent_names=args.agents,
            count=args.count,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
