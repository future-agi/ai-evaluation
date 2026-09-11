from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fi.alk import simulate as public_simulate
from fi.simulate.results import LocalFilesystemResultSink
from fi.simulate.runtime import (
    AgentEndpointSpec,
    EnvironmentSpec,
    RunStatus,
    SimulationSpec,
    SimulatorPolicySpec,
)
from fi.simulate.runtime.runner import SimulationRunner
from fi.simulate.simulation.engines.local_text import LocalTextEngine
from fi.simulate.simulation.models import Persona, Scenario


def _scenario() -> Scenario:
    return Scenario(
        name="runtime-chat",
        dataset=[
            Persona(
                persona={"name": "Morgan"},
                situation="I need a status update.",
                outcome="The status is complete.",
            )
        ],
    )


def _spec(*, timeout: float = 5.0) -> SimulationSpec:
    return SimulationSpec(
        run_id="run_chat_test",
        environment=EnvironmentSpec(
            adapter="chat",
            world_kind="conversation",
            config={"max_turns": 1, "min_turns": 1},
        ),
        target=AgentEndpointSpec(adapter="callable"),
        simulator=SimulatorPolicySpec(adapter="synthetic_user"),
        scenario=_scenario(),
        execution={"timeout": {"run_seconds": timeout}},
    )


def test_runtime_contracts_are_available_from_public_facade() -> None:
    assert public_simulate.SimulationSpec is SimulationSpec
    assert public_simulate.SimulationRunner is SimulationRunner
    assert public_simulate.LocalFilesystemResultSink is LocalFilesystemResultSink


def test_runner_writes_canonical_local_report(tmp_path: Path) -> None:
    async def target(_input):
        return "The status is complete."

    sink = LocalFilesystemResultSink(tmp_path)
    report = asyncio.run(
        SimulationRunner().run(
            _spec(),
            target=target,
            result_sink=sink,
        )
    )

    assert report.status == RunStatus.COMPLETED
    assert report.test_cases[0].result is not None
    assert report.test_cases[0].result.transcript
    assert (tmp_path / report.run_id / "spec.json").exists()
    assert (tmp_path / report.run_id / "plan.json").exists()
    assert (tmp_path / report.run_id / "report.json").exists()


def test_runner_preserves_redacted_persona_failure(caplog) -> None:
    secret = "-".join(("customer", "secret", "value"))

    async def target(_input):
        raise ValueError(secret)

    with caplog.at_level(logging.ERROR, logger="fi.simulate.environments.chat"):
        report = asyncio.run(SimulationRunner().run(_spec(), target=target))

    assert report.status == RunStatus.COMPLETED
    assert report.failure is None
    assert len(report.test_cases) == 1
    case = report.test_cases[0]
    assert case.status.value == "failed"
    assert case.failure is not None
    assert case.failure.code == "chat_persona_failed"
    assert case.failure.details == {"exception_type": "ValueError"}
    assert secret not in report.model_dump_json()
    assert secret not in caplog.text
    assert "ValueError: details redacted" in caplog.text


def test_runner_continues_after_one_persona_crashes() -> None:
    scenario = Scenario(
        name="mixed-chat",
        dataset=[
            Persona(
                persona={"name": name},
                situation="I need a status update.",
                outcome="The status is complete.",
            )
            for name in ("healthy-a", "crash", "healthy-b")
        ],
    )
    spec = _spec().model_copy(update={"scenario": scenario})

    async def target(agent_input):
        if agent_input.persona["name"] == "crash":
            raise RuntimeError("target failed")
        return "The status is complete."

    report = asyncio.run(SimulationRunner().run(spec, target=target))

    assert [case.status.value for case in report.test_cases] == [
        "completed",
        "failed",
        "completed",
    ]
    assert report.test_cases[1].failure.code == "chat_persona_failed"
    assert report.test_cases[0].result.transcript
    assert report.test_cases[2].result.transcript


def test_runner_enforces_run_timeout() -> None:
    async def target(_input):
        await asyncio.sleep(1)
        return "late"

    report = asyncio.run(
        SimulationRunner().run(_spec(timeout=0.01), target=target)
    )

    assert report.status == RunStatus.TIMED_OUT
    assert report.failure is not None
    assert report.failure.code == "simulation_timeout"


def test_local_text_engine_remains_legacy_compatible() -> None:
    async def target(_input):
        return "The status is complete."

    report = asyncio.run(
        LocalTextEngine().run(
            scenario=_scenario(),
            agent_callback=target,
            max_turns=1,
            min_turns=1,
        )
    )

    assert report.results[0].metadata["engine"] == "local_text"
    assert "run_id" not in report.results[0].metadata
