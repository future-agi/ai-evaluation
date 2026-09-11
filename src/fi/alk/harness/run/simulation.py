"""The one thing the harness hands a suite to.

The harness does not run scenarios. It builds a world, writes scenarios against it, and calls
`simulate` once. Everything after that belongs to ALK: how many run at a time, whether the person
is typed to or phoned, where the audio goes, what a report looks like.

That split matters more than it looks. While the harness ran scenarios itself, one at a time,
through its own conversation loop, a suite was only as good as the harness's patience: a run took
as many turns of the chat as it had scenarios, and the simulator driving it was not the one the
product ships. Handing over means the suite runs the same way whether a person triggered it from
the UI, a script did, or nobody did.

Chat and voice are one path here. A contract-only chat spec may run as an in-process target; a
repository-backed chat agent runs its submitted service and is reached through its declared HTTP
or WebSocket ingress; a voice agent is reached through its declared realtime transport. All three
receive the same isolated world, setup, checks and report. Only the target adapter differs, and a
repository-backed agent is never reconstructed from its extracted prompt.

A run is a folder. One simulation over a suite is one run, kept whole, so a session accumulates
runs that can be compared rather than one result file that the next run overwrites.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..contract import AgentContract
from ..scenario import Scenario
from ..world.runtime import Call
from .grade import Judgement, Result

if TYPE_CHECKING:
    from .conversation import Exchange

RUNS = "runs"
RUN = "run.json"
RESULT = "result.json"
TRANSCRIPT = "transcript.txt"
CALLS = "calls.json"
logger = logging.getLogger(__name__)

# How many scenarios run at once by default. One, because the shipped default should be the one
# that cannot surprise anybody: a voice suite places real calls that cost real money, and fanning
# out to twenty is a bad thing to learn from a bill.
CONCURRENCY = 1

# What ALK calls the world and the person, per modality. Both are registry names it validates
# against the plugin's own manifest, so a typo is an error here rather than a confusing run.
WORLDS = {"text": ("chat", "chat"), "voice": ("voice", "voice")}
SIMULATORS = {"text": "synthetic_user", "voice": "livekit_simulator"}


def spoken_to(contract: AgentContract) -> bool:
    """Whether this agent is spoken to rather than typed to."""
    return (contract.modality or "text").strip().lower() == "voice"


def new_run_id() -> str:
    return datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")


def run_root(destination: Path, run_id: str) -> Path:
    return Path(destination) / RUNS / run_id


def every_run(destination: Path) -> list[dict[str, Any]]:
    """Every run in this session, newest first, finished or not.

    A run that is still going is reported too, from the results already written. `run.json` is
    written once, at the end, so requiring it meant an hour-long suite showed nothing at all
    while its results sat on disk: the scenario that finished forty minutes ago was as invisible
    as the one that had not started. `finished` says which kind each is.
    """
    root = Path(destination) / RUNS
    if not root.exists():
        return []
    found: list[dict[str, Any]] = []
    for folder in sorted(root.iterdir(), reverse=True):
        if not folder.is_dir():
            continue
        kept = folder / RUN
        if kept.exists():
            try:
                summary = json.loads(kept.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - one unreadable run never hides the rest
                continue
            summary["finished"] = True
            found.append(summary)
            continue
        done = _cases_so_far(folder)
        if done:
            found.append(
                {
                    "run_id": folder.name,
                    "finished": False,
                    "scenarios": len(done),
                    "passed": sum(1 for one in done if one.get("passed")),
                    "seconds": round(sum(one.get("seconds") or 0 for one in done), 1),
                    "results": done,
                }
            )
    return found


def _cases_so_far(folder: Path) -> list[dict[str, Any]]:
    """The scenarios of an unfinished run that have already been written."""
    done: list[dict[str, Any]] = []
    for case in sorted(folder.iterdir()):
        kept = case / RESULT
        if not case.is_dir() or not kept.exists():
            continue
        try:
            one = json.loads(kept.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a result being written this instant is not an error
            continue
        done.append(
            {
                "scenario": one.get("scenario", case.name),
                "passed": bool(one.get("passed")),
                "met": one.get("met"),
                "of": len(one.get("checkpoints") or []),
                "seconds": one.get("seconds"),
                "recording": one.get("recording", ""),
                "problems": one.get("problems") or [],
            }
        )
    return done


def read_run(destination: Path, run_id: str) -> dict[str, Any]:
    """One run in full: its summary, and every scenario's result, transcript and calls.

    Read from the folder rather than held in memory, so the harness can be asked about a run
    that happened before it was started, and about any single call inside one.
    """
    root = run_root(destination, run_id)
    kept = root / RUN
    if not root.exists():
        raise FileNotFoundError(f"no run {run_id} in {destination}")
    # A run still going has no summary yet, but the scenarios it has finished are readable and
    # worth reading. Only a folder that is not there at all is an error.
    summary = (
        json.loads(kept.read_text(encoding="utf-8"))
        if kept.exists()
        else {"run_id": run_id, "finished": False, "passed": 0}
    )
    summary.setdefault("finished", kept.exists())
    scenarios: list[dict[str, Any]] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / RESULT).exists():
            continue
        one = json.loads((folder / RESULT).read_text(encoding="utf-8"))
        one["transcript"] = _text(folder / TRANSCRIPT)
        one["calls_detail"] = _json(folder / CALLS)
        scenarios.append(one)
    summary["scenarios"] = scenarios
    return summary


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


async def simulate(
    scenarios: list[Scenario],
    contract: AgentContract,
    world_root: Path,
    *,
    destination: Path | None = None,
    model: str | None = None,
    concurrency: int = CONCURRENCY,
    run_id: str = "",
    on_case_start: Callable[[Scenario], Any] | None = None,
    on_case_done: Callable[[Result], Any] | None = None,
    on_exchange: Callable[[str, dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Run a whole suite through ALK and write it out as one run.

    Returns the run's summary. Results are in the order they were asked for, not the order they
    finished, so a report reads the same however it was scheduled.
    """
    from .models import for_roles

    destination = Path(destination or world_root)
    if (Path(world_root) / "environment.json").exists() and concurrency != 1:
        # Restored source worlds point at the submitted Compose project's one real datastore.
        # Until a provisioner can clone that entire project per case, parallel cases would reset
        # and mutate the same database underneath each other. Serialize explicitly rather than
        # offering fast but invalid isolation.
        logger.warning(
            "source-provisioned scenarios share one isolated Compose project; forcing "
            "concurrency from %s to 1",
            concurrency,
        )
        concurrency = 1
    run_id = run_id or new_run_id()
    root = run_root(destination, run_id)
    root.mkdir(parents=True, exist_ok=True)
    roles = for_roles(model)

    # Durations must not jump when the host clock is corrected (common on laptops/VMs). Keep the
    # human timestamp separately and measure elapsed time with the monotonic clock.
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    room = asyncio.Semaphore(max(1, concurrency))
    ordered: list[Result | None] = [None] * len(scenarios)

    async def one(index: int, scenario: Scenario) -> None:
        async with room:
            if on_case_start:
                notified = on_case_start(scenario)
                if inspect.isawaitable(notified):
                    await notified
            began = time.monotonic()
            folder = root / scenario.name
            folder.mkdir(parents=True, exist_ok=True)
            try:
                result = await _run_one(
                    scenario,
                    contract,
                    world_root,
                    folder,
                    roles=roles,
                    on_exchange=(
                        (lambda turn: on_exchange(scenario.name, turn))
                        if on_exchange
                        else None
                    ),
                )
            except Exception as failed:  # noqa: BLE001 - one bad scenario never stops the suite
                result = Result(
                    scenario=scenario.name,
                    tests=scenario.tests,
                    problems=[f"{type(failed).__name__}: {failed}"],
                    # This is a terminal outcome for the attempted scenario, but it is not an
                    # agent result.  Keeping an explicit ending prevents downstream artifact
                    # readers from confusing an exception-shaped partial record with a call
                    # that is still in progress.
                    ended="failed",
                )
            result.seconds = round(time.monotonic() - began, 1)
            _write_case(folder, result)
            ordered[index] = result
            if on_case_done:
                notified = on_case_done(result)
                if inspect.isawaitable(notified):
                    await notified

    await asyncio.gather(
        *(one(index, scenario) for index, scenario in enumerate(scenarios))
    )
    results = [one for one in ordered if one is not None]

    summary = {
        "run_id": run_id,
        "agent": contract.agent,
        "modality": contract.modality or "text",
        "started": started_at,
        "seconds": round(time.monotonic() - started, 1),
        "concurrency": concurrency,
        "models": roles,
        "scenarios": len(results),
        "passed": sum(1 for one in results if one.passed),
        # A scenario that could not be executed is an infrastructure/harness outcome, not a
        # weak-agent grade.  The CLI and hosted worker use this count to keep those two result
        # classes distinct all the way to the platform.
        "unrunnable": sum(1 for one in results if one.problems),
        "spent_usd": round(sum(one.spent_usd for one in results), 4),
        # Averaged across the scenarios that reported them, so a suite has one line per metric
        # rather than a number nobody compares. Only over the runs that actually measured it:
        # averaging a missing metric as zero would make a suite look worse the more of it failed
        # to run, which is the opposite of informative.
        "metrics": _averaged([one.measured for one in results]),
        "results": [
            {
                "scenario": one.scenario,
                "passed": one.passed,
                "met": one.met,
                "of": len(one.checkpoints),
                "seconds": one.seconds,
                "recording": one.recording,
                "problems": one.problems,
            }
            for one in results
        ],
    }
    (root / RUN).write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def _write_case(folder: Path, result: Result) -> None:
    """One scenario's result, transcript and calls, each in the form it is read in.

    The transcript is written as text because it is read by people, and the calls as JSON
    because they are read by the UI and by the harness looking into a single call.
    """
    body = asdict(result)
    body["passed"] = result.passed
    body["met"] = result.met
    detail = body.pop("calls_detail", None) or []
    (folder / RESULT).write_text(
        json.dumps(body, indent=2, default=str), encoding="utf-8"
    )
    (folder / TRANSCRIPT).write_text(result.transcript or "", encoding="utf-8")
    (folder / CALLS).write_text(
        json.dumps(detail, indent=2, default=str), encoding="utf-8"
    )


async def _run_one(
    scenario: Scenario,
    contract: AgentContract,
    world_root: Path,
    folder: Path,
    *,
    roles: dict[str, str],
    on_exchange: Callable[[dict[str, Any]], Any] | None = None,
) -> Result:
    """One scenario, in its own world, through ALK's runner.

    The world is prepared here and handed in, rather than named in the spec, because isolation
    is ours to guarantee: every scenario starts from the same frozen base with only its own
    setup applied, and a world shared between cases would let the first one decide what the
    second is graded against.
    """

    from ..folder import apply_setup, check_ready
    from ..world.snapshot import restore

    spoken = spoken_to(contract)
    kind = "voice" if spoken else "text"
    adapter, world_kind = WORLDS[kind]

    world = restore(world_root)
    try:
        world.reset()
        applied = apply_setup(scenario, world)
        if not applied.ok:
            raise RuntimeError(f"the scenario's setup did not run: {applied.said}")
        ready = check_ready(scenario, world)
        if not ready.ok:
            raise RuntimeError(
                f"the world is not ready for this scenario: {ready.said}. Running it would "
                "test us rather than the agent."
            )
        # The setup's own calls are not the agent's.
        world.calls = []

        if not spoken:
            # Typed, and driven by a model rather than by ALK's chat simulator.
            #
            # That simulator is deterministic on purpose: an untyped persona gets three fixed
            # lines ("Can you give me the exact next step…"), and a typed one renders utterances
            # from a compiled behaviour policy. Reproducible, and not a simulation of a person.
            # A suite whose user says the same three things to every agent tests one path and
            # calls it coverage.
            #
            # So the conversation is driven here, by a model reading the simulator prompt the
            # build stage wrote for this agent. Everything around it is unchanged: same world,
            # same setup, same checks, same run folder.
            return await _typed_to(
                scenario, contract, world, world_root, folder, roles=roles
            )

        # Spoken. The agent is not here: it runs in Vapi, with its own prompt, its own model
        # and its own voice, and the only thing that changes is where its tools are answered.
        # ALK places the call and drives a simulated caller that is a real model over STT and
        # TTS, so this half was never deterministic.
        return await _spoken_to(
            scenario,
            contract,
            world,
            world_root,
            folder,
            roles=roles,
            on_exchange=on_exchange,
        )
    finally:
        try:
            world.close()
        except Exception:  # cleanup must never replace a completed scenario result
            logger.exception("world cleanup failed after scenario %s", scenario.name)


def _found_audio(directory: Path) -> Path | None:
    """The recording a run left behind, if it left one.

    Asked of the directory rather than taken on trust from whatever placed the call: a runner
    that exits badly still returns a path, and a path is not a file.
    """
    if not directory.exists():
        return None
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".wav", ".mp3", ".ogg", ".m4a"):
            return path
    return None


async def _typed_to(
    scenario: Scenario,
    contract: AgentContract,
    world: Any,
    world_root: Path,
    folder: Path,
    *,
    roles: dict[str, str],
) -> Result:
    """A typed conversation, with a model on both sides.

    The same grading as every other run: the world it is handed is already set up, and what it
    leaves behind is what the checks read.
    """
    from ..catalogue import load_catalogue
    from . import converse
    from .grade import (
        checkpoints,
        grade_sub_goals,
        judge,
        judge_suite_evals,
        reconcile_task_completion,
        ungraded_sub_goals,
    )
    from .targets import resolve

    repository_backed = bool(
        contract.runtime or contract.tool_entrypoints or contract.implementation
    )
    if repository_backed:
        agent = resolve("repository")(
            contract,
            world,
            world_root=world_root,
            trace_path=folder / "agent-tool-calls.jsonl",
            scenario_name=scenario.name,
        )
    else:
        agent = resolve("local")(contract, world, model=roles["agent"])
    transcript = await converse(
        agent, scenario, contract, world_root=world_root, model=roles["user"]
    )
    catalogue = load_catalogue(world_root)
    settled = grade_sub_goals(world, scenario, catalogue, transcript.calls)
    ending = ", ".join(
        f"{name}: {len(rows)} rows"
        for name, rows in sorted(world.observe().state.items())
    )
    judgements, judged_cost = await judge(
        scenario, transcript, contract, catalogue, model=roles["judge"], ending=ending
    )
    suite_judgements = judge_suite_evals(
        catalogue.suite_evals, scenario, transcript, contract, ending=ending
    )
    judgements += reconcile_task_completion(suite_judgements, settled, judgements)
    result = Result(
        scenario=scenario.name,
        tests=scenario.tests,
        problems=[
            f"{name} is not in this catalogue, so nothing graded it"
            for name in ungraded_sub_goals(scenario, catalogue)
        ],
        state_failures=[f"{one.name}: {one.said}" for one in settled if not one.held],
        conduct=judgements,
        checkpoints=checkpoints(settled, judgements),
        crashes=[f"{call.name}: {call.error}" for call in transcript.crashed()],
        ended=transcript.ended,
        turns=len(transcript.exchanges),
        calls=len(transcript.calls),
        spent_usd=transcript.spent_usd + judged_cost,
        transcript=transcript.spoken(),
        exchanges=[
            {"speaker": turn.speaker, "text": turn.text}
            for turn in transcript.exchanges
        ],
        actions=transcript.actions(),
    )
    result.calls_detail = _calls_of(transcript.calls)
    return result


def _calls_of(calls: Any) -> list[dict[str, Any]]:
    """Every call in full, for the timeline and for anyone asking what one call did."""
    return [
        {
            "name": call.name,
            "arguments": call.arguments,
            "result": str(call.result)[:2000],
            "ok": call.ok,
            "refused": call.refused,
            "error": call.error,
            "at": getattr(call, "at", 0.0),
        }
        for call in calls
    ]


def _said(line: str) -> Exchange:
    """One transcript line as a turn, with its role read off rather than left in the text.

    The line arrives already labelled ("assistant: ..."). Keeping that label in the text made the
    judge read ``agent: assistant: ...``, two speakers deep for every turn.
    """
    from .conversation import Exchange

    role, _, text = line.partition(":")
    named = role.strip().lower()
    if named in ("assistant", "agent"):
        return Exchange("agent", text.strip())
    if named in ("user", "customer"):
        return Exchange("customer", text.strip())
    return Exchange("customer", line.strip())


async def _spoken_to(
    scenario: Scenario,
    contract: AgentContract,
    world: Any,
    world_root: Path,
    folder: Path,
    *,
    roles: dict[str, str],
    on_exchange: Callable[[dict[str, Any]], Any] | None = None,
) -> Result:
    """A real call, with the agent's own tools answered by this world.

    The agent under test is not reconstructed here and is not running in this process. It is the
    hosted assistant, with its own prompt, model and voice; the only thing that changes for the
    duration is where its tool calls are sent. That makes this the more faithful of the two
    paths, and the reason a spoken suite is worth more than a typed one.

    The call itself belongs to ALK, which drives a simulated caller through speech: a real model
    behind STT and TTS, not a script.
    """
    import os
    import time

    from ..catalogue import load_catalogue
    from .call import place_the_call
    from .conversation import Transcript
    from .evidence import measured, newest_report, spoken_times, tracks_in
    from .grade import (
        checkpoints,
        grade_sub_goals,
        judge,
        judge_suite_evals,
        reconcile_task_completion,
        ungraded_sub_goals,
    )
    from .live import wire
    from .tools import configure_source_voice, missing_prerequisites

    configure_source_voice(world_root, contract)
    stopping = missing_prerequisites(world_root, contract)
    if stopping:
        raise RuntimeError("cannot place a call:\n  - " + "\n  - ".join(stopping))

    loop = asyncio.get_running_loop()

    def live_exchange(turn: dict[str, Any]) -> None:
        if on_exchange:
            normalized = _normalize_live_exchange(turn)
            loop.call_soon_threadsafe(on_exchange, normalized)

    def placed_once() -> tuple[int, dict[str, Any], str]:
        """Everything about the call, off the event loop.

        Wiring reads a subprocess's stdout and the call itself blocks for minutes. Run inline
        they freeze whatever loop is hosting this, which for the UI means the stream, the status
        endpoint and the stop button all stop with it.
        """
        _world, instruction, webhook, tunnel, _url, _moved = wire(
            scenario,
            world_root,
            world=world,
            trace_path=folder / "agent-tool-calls.jsonl",
        )
        started = time.time()
        runtime_output = ""
        try:
            sdk_output = folder / "sdk"
            os.environ["HARNESS_VOICE_OUTPUT_ROOT"] = str(sdk_output.resolve())
            os.environ["HARNESS_INSTRUCTION"] = instruction
            os.environ["HARNESS_SCENARIO"] = scenario.name
            # The caller is never handed the grader's pass question: `tests` is written about
            # the agent in the third person, so as an objective it reads as a rubric rather
            # than a motive. What this person wants is already in the instruction.
            os.environ.pop("HARNESS_OUTCOME", None)
            os.environ["HARNESS_PERSONA"] = json.dumps(
                scenario.persona.model_dump(exclude_none=True)
                if scenario.persona is not None
                else {"name": "customer"}
            )
            os.environ["HARNESS_INITIAL_MESSAGE"] = (
                scenario.persona.initial_message if scenario.persona is not None else ""
            )
            os.environ["HARNESS_SCRIPTED_CALLER"] = json.dumps(
                scenario.persona.scripted_caller
                if scenario.persona is not None
                and scenario.persona.scripted_caller is not None
                else {}
            )
            os.environ["HARNESS_FIXTURE"] = json.dumps(
                scenario.fixture, ensure_ascii=False, default=str
            )
            # A scenario that asks to be heard through background noise selects a clip for the
            # caller's environment; the voice engine mixes it under the caller. Cleared otherwise so
            # a previous call's noise never leaks into a quiet one.
            from ..background_noise import scenario_source

            source = scenario_source(
                getattr(scenario, "background_noise", False),
                scenario.fixture,
                seed=scenario.name,
            )
            if source:
                os.environ["HARNESS_BACKGROUND_NOISE"] = source
            else:
                os.environ.pop("HARNESS_BACKGROUND_NOISE", None)
            code = place_the_call(
                os.environ.get("HARNESS_VOICE_CASE", "2.1.2"),
                on_exchange=live_exchange if on_exchange else None,
            )
        finally:
            try:
                webhook.stop()
            except Exception:
                logger.exception(
                    "webhook cleanup failed after scenario %s", scenario.name
                )
            if tunnel is not None:
                try:
                    tunnel.terminate()
                except Exception:
                    logger.exception(
                        "tunnel cleanup failed after scenario %s", scenario.name
                    )
            if (Path(world_root) / "environment.json").exists():
                try:
                    from ..provision import runtime_logs, stop_runtime

                    # Some third-party LiveKit agents execute every tool in-process. They do not
                    # call the harness webhook and may not implement HARNESS_AGENT_TOOL_TRACE,
                    # but the LiveKit worker emits structured execution lifecycle events. Read
                    # those events before removing the per-scenario container. Raw logs are not
                    # retained; only normalized tool evidence is kept below.
                    runtime_output = runtime_logs(world_root)
                    stop_runtime(world_root)
                except Exception:
                    logger.exception(
                        "runtime cleanup failed after scenario %s", scenario.name
                    )
        # Everything the runner recorded about this call, read from the report it wrote.
        return code, newest_report(started, root=sdk_output), runtime_output

    attempts = 1 + max(0, int(os.environ.get("HARNESS_VOICE_INFRA_RETRIES", "1")))
    code, case, runtime_output = 1, {}, ""
    attempts_used = 0
    trace_path = folder / "agent-tool-calls.jsonl"
    for attempt in range(attempts):
        attempts_used = attempt + 1
        # Every attempt owns its trace. A stale line from a failed attempt must not turn a later
        # worker-join failure into something that looks like agent evidence.
        trace_path.unlink(missing_ok=True)
        # The webhook is the transport-level evidence fallback. Clear calls from a failed voice
        # attempt before retrying so only the attempt whose transcript is graded can contribute.
        world.calls = []
        code, case, runtime_output = await asyncio.to_thread(placed_once)
        attempt_calls = _semantic_calls(
            trace_path, contract=contract
        ) or _livekit_log_calls(runtime_output, contract=contract)
        if (
            not _voice_attempt_should_retry(
                code, case, has_agent_calls=bool(attempt_calls or world.calls)
            )
            or attempt + 1 >= attempts
        ):
            break
        logger.warning(
            "retryable voice attempt ended early for %s; retrying attempt %s/%s",
            scenario.name,
            attempt + 2,
            attempts,
        )
    semantic = _semantic_calls(trace_path, contract=contract) or _livekit_log_calls(
        runtime_output, contract=contract
    )
    # A worker trace includes semantic/local actions that never cross HTTP and is preferred when
    # available. In a hosted Docker runner, however, the job artifacts can live in a named volume
    # whose container path cannot be bind-mounted by the host daemon into the submitted runtime.
    # In that case the bound world is still exact evidence: setup calls were cleared in prepare,
    # caller hydration uses record=False, and every remaining call arrived through this call's
    # private webhook. Do not erase that evidence merely because the optional trace is absent.
    if semantic:
        world.calls = semantic
    spoken = str(case.get("transcript") or "")
    # Every track that exists, copied in beside the result so a run is self-contained and the
    # page can fall back when the preferred one is missing.
    kept = _keep_tracks(tracks_in(case), folder)

    if _voice_infrastructure_failure(code, case, has_agent_calls=bool(semantic)):
        status = str((case.get("metadata") or {}).get("status") or "failed")
        result = Result(
            scenario=scenario.name,
            tests=scenario.tests,
            problems=[
                "voice infrastructure failed after retry: the target worker never joined or "
                "produced an assistant turn; this is not graded as an agent failure"
            ],
            ended=status,
            turns=len([line for line in spoken.splitlines() if line.strip()]),
            calls=0,
            transcript=spoken,
            recording=(kept[0]["path"] if kept else ""),
        )
        result.tracks = kept
        result.measured = measured(case)
        result.measured["voice_attempts"] = attempts_used
        return result

    catalogue = load_catalogue(world_root)
    settled = grade_sub_goals(world, scenario, catalogue, world.calls)
    # Judged the same way a typed run is. Without this a spoken scenario reports "1/2" when what
    # happened is that one check passed and the other was never asked, which reads as the agent
    # half-failing rather than as the suite not having looked.
    spoken_transcript = Transcript(
        exchanges=[_said(line) for line in spoken.splitlines() if line.strip()],
        calls=list(world.calls),
        ended=str((case.get("metadata") or {}).get("status") or "finished"),
    )
    judgements, judged_cost = await judge(
        scenario,
        spoken_transcript,
        contract,
        catalogue,
        model=roles["judge"],
        ending=", ".join(
            f"{name}: {len(rows)} rows"
            for name, rows in sorted(world.observe().state.items())
        ),
    )
    _require_action_evidence(judgements, scenario, world.calls)
    suite_judgements = judge_suite_evals(
        catalogue.suite_evals,
        scenario,
        spoken_transcript,
        contract,
        ending=", ".join(
            f"{name}: {len(rows)} rows"
            for name, rows in sorted(world.observe().state.items())
        ),
    )
    judgements += reconcile_task_completion(suite_judgements, settled, judgements)
    result = Result(
        scenario=scenario.name,
        tests=scenario.tests,
        problems=[
            f"{name} is not in this catalogue, so nothing graded it"
            for name in ungraded_sub_goals(scenario, catalogue)
        ],
        state_failures=[f"{one.name}: {one.said}" for one in settled if not one.held],
        conduct=judgements,
        checkpoints=checkpoints(settled, judgements),
        spent_usd=judged_cost,
        ended=str((case.get("metadata") or {}).get("status") or "finished"),
        turns=len([line for line in spoken.splitlines() if line.strip()]),
        calls=len(world.calls),
        transcript=spoken,
        exchanges=_timed_exchanges(spoken_transcript.exchanges, spoken_times(case)),
        recording=(kept[0]["path"] if kept else ""),
    )
    result.tracks = kept
    result.measured = measured(case)
    result.measured["voice_attempts"] = attempts_used
    result.calls_detail = _calls_of(world.calls)
    return result


def _normalize_live_exchange(turn: dict[str, Any]) -> dict[str, Any]:
    """Translate ALK's report roles to the harness UI's conversation roles.

    The callback comes from the simulator's own AgentSession, so its roles are from that
    session's point of view: ``assistant`` is the simulated customer and ``user`` is the tested
    service agent. The completed report later translates them to the test's point of view.
    """
    raw = str(turn.get("speaker") or turn.get("role") or "").strip().lower()
    speaker = (
        "customer"
        if raw in {"assistant", "customer"}
        else "agent"
        if raw in {"user", "agent"}
        else raw or "customer"
    )
    return {
        **turn,
        "speaker": speaker,
        "text": turn.get("text") or turn.get("content") or "",
    }


def _require_action_evidence(
    judgements: list[Judgement], scenario: Scenario, calls: list[Call]
) -> None:
    """Prevent conditional prose checks from passing vacuously when the agent did nothing.

    A judge can reasonably say "surge was disclosed before confirmation" when neither event
    occurred because the implication is technically vacuous. In an executable scenario with a
    reference action sequence, no semantic calls means the scenario conduct was not satisfied.
    """
    if not scenario.solution or calls:
        return
    for judgement in judgements:
        if judgement.holds:
            judgement.holds = False
            judgement.why = (
                "The agent made no semantic tool calls, so this scenario conduct cannot be "
                "credited even if its ordering condition is vacuously true."
            )


def _voice_infrastructure_failure(
    code: int, case: dict[str, Any], *, has_agent_calls: bool = False
) -> bool:
    """A failed room before meaningful target activity is not agent evidence."""
    if code == 0:
        return False
    transcript = str(case.get("transcript") or "")
    lines = [line for line in transcript.splitlines() if line.strip()]
    has_target_turn = any(
        line.strip().lower().startswith(("assistant:", "agent:")) for line in lines
    )
    if not has_target_turn:
        return True
    failure = (case.get("metadata") or {}).get("failure") or case.get("failure") or {}
    retryable = bool(failure.get("retryable")) if isinstance(failure, dict) else False
    failure_code = str(failure.get("code") or "") if isinstance(failure, dict) else ""
    transport_retryable = retryable and failure_code in {
        "target_disconnected",
        "target_not_found",
        "provider_disconnected",
        "room_connection_failed",
        "room_not_ready",
    }
    target_lines = [
        line.split(":", 1)[-1].strip()
        for line in lines
        if line.strip().lower().startswith(("assistant:", "agent:"))
    ]
    truncated_target = any(
        utterance and utterance[-1] not in ".?!" for utterance in target_lines
    )
    # ALK requires six alternating messages by default. A retryable disconnect before that,
    # without one semantic action, is a room/worker lifecycle failure; a longer conversation or
    # any tool trace is enough evidence to grade the agent normally.
    return (
        not has_agent_calls
        and len(lines) < 6
        and (transport_retryable or truncated_target)
    )


def _voice_attempt_should_retry(
    code: int, case: dict[str, Any], *, has_agent_calls: bool = False
) -> bool:
    """Retry one short silence without misclassifying the final result as infrastructure.

    Live speech recognition occasionally drops the caller's first audio turn.  The provider
    labels that timeout retryable, but a greeting proves the worker joined, so if the retry also
    fails it remains an agent-pipeline reliability failure.  A bounded second attempt separates
    a transient dropped turn from a reproducible weak branch while preserving both outcomes via
    ``voice_attempts``.
    """
    if _voice_infrastructure_failure(code, case, has_agent_calls=has_agent_calls):
        return True
    if code == 0 or has_agent_calls:
        return False
    failure = (case.get("metadata") or {}).get("failure") or case.get("failure") or {}
    if not isinstance(failure, dict):
        return False
    lines = [
        line for line in str(case.get("transcript") or "").splitlines() if line.strip()
    ]
    return (
        bool(failure.get("retryable"))
        and str(failure.get("code") or "") == "conversation_silence_timeout"
        and len(lines) < 6
    )


def _semantic_calls(path: Path, *, contract: AgentContract | None = None) -> list[Call]:
    """Read the submitted worker's agent-facing tool trace, tolerating a killed final line."""
    if not path.exists():
        return []
    endpoint_names = {
        entry.endpoint.strip("/"): entry.tool
        for entry in (contract.tool_entrypoints if contract is not None else [])
        if entry.endpoint.strip("/")
    }
    calls: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(record, dict) or not record.get("name"):
            continue
        output = record.get("output")
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                pass
        arguments = record.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        failed = bool(record.get("is_error"))
        recorded_name = str(record["name"]).strip("/")
        call = Call(
            name=endpoint_names.get(recorded_name, recorded_name),
            arguments=arguments,
            result=output,
            ok=not failed,
            refused=failed,
            error=str(output) if failed else "",
            at=float(record.get("at") or 0.0),
        )
        # A harness-aware worker may mirror a local state-machine action to the world for
        # observability and then emit the authoritative function-completion event. They are one
        # logical action. Prefer the completion result, but never collapse ordinary identical
        # retries (payment-status polling is a legitimate example).
        if calls and _telemetry_mirror(calls[-1], call):
            calls[-1] = call
        else:
            calls.append(call)
    return calls


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _livekit_log_calls(
    output: str, *, contract: AgentContract | None = None
) -> list[Call]:
    """Normalize completed LiveKit Python tool executions from bounded runtime logs.

    This fallback is intentionally narrow: a start event alone earns no evidence, and arbitrary
    application prose is never interpreted as a call.  LiveKit's structured ``executing tool``
    and matching ``tools execution completed`` records are stable SDK lifecycle events.  The
    Successful result values are not present in those logs, so they are represented honestly as
    completion evidence rather than fabricated output. Structured ``ToolError while executing
    tool`` records do carry a refusal reason; correlate those with the matching start so a
    truthful refusal is never normalized as success.
    """
    endpoint_names = {
        entry.endpoint.strip("/"): entry.tool
        for entry in (contract.tool_entrypoints if contract is not None else [])
        if entry.endpoint.strip("/")
    }
    contract_tools = (
        {tool.name: tool for tool in contract.tools} if contract is not None else {}
    )
    decoder = json.JSONDecoder()
    starts: list[dict[str, Any]] = []
    completed: set[str] = set()
    refusals: dict[tuple[str, str], str] = {}
    for raw_line in str(output or "").splitlines():
        line = _ANSI.sub("", raw_line)
        marker = "executing tool"
        completion = "tools execution completed"
        record: Any = None
        try:
            structured = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            structured = None
        structured_message = (
            str(structured.get("message") or "") if isinstance(structured, dict) else ""
        )
        refusal = structured_message.startswith("ToolError while executing tool:")
        if isinstance(structured, dict) and (
            structured_message in {marker, completion} or refusal
        ):
            record = structured
            event = "refusal" if refusal else structured_message
        elif marker in line:
            brace = line.find("{", line.find(marker) + len(marker))
            if brace < 0:
                continue
            try:
                record, _ = decoder.raw_decode(line[brace:])
            except (json.JSONDecodeError, TypeError):
                continue
            event = marker
        elif completion in line:
            brace = line.find("{", line.find(completion) + len(completion))
            if brace < 0:
                continue
            try:
                record, _ = decoder.raw_decode(line[brace:])
            except (json.JSONDecodeError, TypeError):
                continue
            event = completion
        else:
            continue
        if not isinstance(record, dict):
            continue
        if event == "refusal":
            speech_id = str(record.get("speech_id") or "")
            function = str(record.get("function") or "").strip("/")
            if speech_id and function:
                refusals[(speech_id, function)] = structured_message.split(":", 1)[
                    -1
                ].strip()
        elif event == marker:
            if not record.get("function"):
                continue
            speech_id = str(record.get("speech_id") or "")
            recorded_name = str(record["function"]).strip("/")
            name = endpoint_names.get(recorded_name, recorded_name)
            arguments: Any = record.get("lk.pii.arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            if contract is not None:
                specification = contract_tools.get(name)
                # LiveKit also logs SDK/workflow functions which are not target tools. Only the
                # contract can make a runtime-log fallback authoritative target evidence.
                if specification is None:
                    continue
                # A speech-level completion has no per-tool result. Do not turn a malformed start
                # into success merely because the surrounding speech completed.
                if any(argument not in arguments for argument in specification.args):
                    continue
            starts.append(
                {
                    "name": name,
                    "recorded_name": recorded_name,
                    "arguments": arguments,
                    "speech_id": speech_id,
                    "at": _log_timestamp(str(record.get("timestamp") or line)),
                }
            )
        elif event == completion:
            if record.get("speech_id"):
                completed.add(str(record["speech_id"]))
    calls: list[Call] = []
    starts_per_speech = {
        speech_id: sum(one["speech_id"] == speech_id for one in starts)
        for speech_id in completed
    }
    for one in starts:
        if not one["speech_id"] or one["speech_id"] not in completed:
            continue
        # A single LiveKit completion can close a batch of tool starts but cannot prove which
        # individual invocation succeeded. Native semantic traces remain authoritative for that
        # case; the bounded-log fallback deliberately emits no ambiguous evidence.
        if starts_per_speech.get(one["speech_id"]) != 1:
            continue
        error = refusals.get(
            (one["speech_id"], one["recorded_name"]), ""
        ) or refusals.get((one["speech_id"], one["name"]), "")
        calls.append(
            Call(
                name=one["name"],
                arguments=one["arguments"],
                result=(
                    error
                    if error
                    else {
                        "evidence": "livekit_runtime_log",
                        "execution": "completed",
                    }
                ),
                ok=not error,
                refused=bool(error),
                error=error,
                at=float(one["at"]),
            )
        )
    return calls


def _log_timestamp(line: str) -> float:
    value = line.strip()
    matched = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", value)
    if matched is not None:
        value = matched.group(1).replace(",", ".")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).timestamp()
    except ValueError:
        return 0.0


def _telemetry_mirror(previous: Call, current: Call) -> bool:
    if previous.name != current.name or previous.arguments != current.arguments:
        return False
    result = previous.result
    if isinstance(result, dict):
        if result.get("execution") == "submitted_agent_runtime":
            return True
        text = str(result.get("result") or "").lower()
    else:
        text = str(result or "").lower()
    return "submitted service has no endpoint" in text


def _timed_exchanges(
    exchanges: list[Any], times: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The conversation with each turn's speech times attached, where they were measured.

    Paired by position, and only when the two agree on how many turns there were. They come
    from the same call but by different routes, so a mismatch means one of them dropped a turn
    -- and pairing them anyway would hang every turn's timing on the wrong words.
    """
    spoken = [{"speaker": turn.speaker, "text": turn.text} for turn in exchanges]
    if len(times) != len(spoken):
        return spoken
    for turn, when in zip(spoken, times, strict=True):
        if when.get("start_time_ms") is None:
            continue
        turn["start_time_ms"] = when["start_time_ms"]
        if when.get("end_time_ms") is not None:
            turn["end_time_ms"] = when["end_time_ms"]
    return spoken


def _keep_tracks(found: list[dict[str, str]], folder: Path) -> list[dict[str, str]]:
    """Copy each recording into this run's folder, keeping the order it was offered in.

    Copied rather than referenced, because the runner's own directory is transient and a run
    that cannot be listened to next week is a run that cannot be shown to anybody.
    """
    import shutil

    folder.mkdir(parents=True, exist_ok=True)
    kept: list[dict[str, str]] = []
    for track in found:
        source = Path(track["path"])
        if not source.exists():
            continue
        landed = folder / f"{track['label'].replace(':', '_')}{source.suffix}"
        try:
            shutil.copyfile(source, landed)
        except OSError:
            continue
        kept.append({"label": track["label"], "path": str(landed)})
    return kept


def _averaged(measured: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each metric's mean over the scenarios that reported it, carrying whether it applied.

    A metric that had nothing to measure scores 1.0, so averaging the lot produces a suite
    summary in which two thirds of the numbers are perfect and none of them mean anything. The
    applicability travels with the average instead of being flattened away, so a reader is never
    shown "browser action safety 1.00" for a suite of phone calls without also being told there
    were no browser actions.
    """
    gathered: dict[str, list[float]] = {}
    applies: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for one in measured:
        for metric in (one or {}).get("metrics") or []:
            name, value = metric.get("name"), metric.get("score")
            if not name or not isinstance(value, (int, float)):
                continue
            gathered.setdefault(name, []).append(float(value))
            # Applicable anywhere is applicable: one scenario exercising a capability is enough
            # to make the number worth reading across the suite.
            applies[name] = applies.get(name, False) or bool(
                metric.get("applicable", True)
            )
            if metric.get("reason") and name not in reasons:
                reasons[name] = str(metric["reason"])
    return [
        {
            "name": name,
            "score": round(sum(values) / len(values), 4),
            "applicable": applies.get(name, True),
            "reason": reasons.get(name, ""),
            "cases": len(values),
        }
        for name, values in sorted(gathered.items())
        if values
    ]
