from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from fi.alk.harness import cli
from fi.alk.harness.contract import AgentContract
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.run.conversation import Exchange, Transcript
from fi.alk.harness.scenario import Scenario


@pytest.mark.parametrize(
    "entrypoint,required",
    [
        (entry, required)
        for entry in ("cli", "shared")
        for required in (True, False, None)
    ]
    + [("tool_scenario", True)],
)
@pytest.mark.parametrize("modality", ["chat", "voice"])
@pytest.mark.parametrize("with_calls", [False, True])
def test_resumed_local_simulation_requires_observed_tools(
    tmp_path, monkeypatch, required, modality, with_calls, entrypoint
):
    from fi.alk.harness.run import grade, simulation
    from fi.alk.harness.world.runtime import Call

    job = HarnessJob(
        job_id="resume",
        run_id="resume",
        execution="local",
        source={"kind": "local_repository", "local_path": str(tmp_path)},
        agent={"connector": "auto"},
        require_tool_workflow=bool(required),
    )
    if required is not None:
        (tmp_path / "job.json").write_text(job.model_dump_json())
    contract = AgentContract(
        agent="example",
        conversational=True,
        modality=modality,
        tools=[{"name": "lookup"}],
        tool_entrypoints=[
            {"tool": "lookup", "mode": "import", "module": "api", "callable": "lookup"}
        ],
    )
    scenario = Scenario(name="greeting", instruction="Say hello")
    world = SimpleNamespace(
        calls=[],
        reset=lambda: None,
        close=lambda: None,
        observe=lambda: SimpleNamespace(state={}),
    )
    monkeypatch.setattr(cli, "load", lambda _: contract)
    monkeypatch.setattr(cli, "load_written", lambda _: [scenario])
    monkeypatch.setattr(cli, "_source_root", lambda _: None)
    monkeypatch.setattr("fi.alk.harness.platform.configured", lambda: "disabled")
    monkeypatch.setattr(
        "fi.alk.harness.world.snapshot.require_source_implementation", lambda _: None
    )
    monkeypatch.setattr("fi.alk.harness.world.snapshot.restore", lambda _: world)
    monkeypatch.setattr(
        "fi.alk.harness.folder.apply_setup", lambda *_: SimpleNamespace(ok=True)
    )
    monkeypatch.setattr(
        "fi.alk.harness.folder.check_ready", lambda *_: SimpleNamespace(ok=True)
    )
    monkeypatch.setattr(
        "fi.alk.harness.run.models.for_roles",
        lambda _: {"agent": "a", "user": "u", "judge": "j"},
    )
    monkeypatch.setattr(
        "fi.alk.harness.run.targets.resolve", lambda _: lambda *a, **k: object()
    )
    monkeypatch.setattr(
        "fi.alk.harness.catalogue.load_catalogue",
        lambda _: SimpleNamespace(suite_evals=[]),
    )
    judged = []

    observed_calls = (
        [Call(name="lookup", arguments={}, result="found")] if with_calls else []
    )

    async def converse(*args, **kwargs):
        return Transcript(
            exchanges=[Exchange("agent", "Hello")],
            calls=observed_calls,
            ended="completed",
            spent_usd=0.25,
        )

    monkeypatch.setattr(
        "fi.alk.harness.run.tools.configure_source_voice", lambda *args: None
    )
    monkeypatch.setattr(
        "fi.alk.harness.run.tools.missing_prerequisites", lambda *args: []
    )
    monkeypatch.setattr(
        "fi.alk.harness.run.live.wire",
        lambda *args, **kwargs: (
            world,
            "Greet",
            SimpleNamespace(stop=lambda: None),
            None,
            "",
            False,
        ),
    )

    def place(*args, **kwargs):
        world.calls = list(observed_calls)
        return 0

    monkeypatch.setattr("fi.alk.harness.run.call.place_the_call", place)
    monkeypatch.setattr(
        "fi.alk.harness.run.evidence.newest_report",
        lambda *args, **kwargs: {
            "transcript": "agent: Hello",
            "metadata": {"status": "completed"},
        },
    )
    monkeypatch.setattr(
        simulation, "_keep_tracks", lambda *args: [{"path": "recording.wav"}]
    )

    async def judge(*args, **kwargs):
        judged.append(True)
        return [grade.Judgement("greeted", "conduct", True)], 0

    monkeypatch.setattr("fi.alk.harness.run.converse", converse)
    monkeypatch.setattr(grade, "grade_sub_goals", lambda *args: [])
    monkeypatch.setattr(grade, "judge", judge)
    monkeypatch.setattr(grade, "judge_suite_evals", lambda *args, **kwargs: [])
    monkeypatch.setattr(grade, "reconcile_task_completion", lambda *args: [])
    monkeypatch.setattr(grade, "ungraded_sub_goals", lambda *args: [])
    if entrypoint == "cli":
        status = asyncio.run(
            cli._simulate(
                SimpleNamespace(
                    out=str(tmp_path), name="example", only=None, model=None
                )
            )
        )
        output = tmp_path
    elif entrypoint == "tool_scenario":
        from fi.alk.harness.run import tools

        monkeypatch.setattr(tools, "load_scenarios", lambda _: [scenario])
        monkeypatch.setattr(
            tools, "load_catalogue", lambda _: SimpleNamespace(suite_evals=[])
        )
        monkeypatch.setattr(tools, "load_results", lambda _: [])
        monkeypatch.setattr(tools, "require_source_implementation", lambda _: None)
        server = tools.run_tools(tmp_path, tmp_path, contract=contract)
        answer = asyncio.run(
            next(tool for tool in server.tools if tool.name == "run_scenario").handler(
                {"name": scenario.name}
            )
        )
        status = 1 if answer.get("is_error") else 0
        output = tmp_path
    else:
        output = tmp_path / "separate-results"
        summary = asyncio.run(
            simulation.simulate(
                [scenario],
                contract,
                tmp_path,
                destination=output,
                require_tool_workflow=False,
            )
        )
        status = 1 if summary["unrunnable"] else 0
    result_path = next((output / "runs").glob("*/greeting/result.json"))
    result = json.loads(result_path.read_text())
    missing = required and not with_calls
    assert status == (1 if missing else 0)
    assert judged == ([] if missing else [True])
    assert result["calls"] == len(observed_calls)
    assert result["turns"] == 1
    assert (result_path.parent / "transcript.txt").read_text() == "agent: Hello"
    assert result["spent_usd"] == (0.25 if modality == "chat" else 0)
    if modality == "voice":
        assert result["recording"] == "recording.wav"
        assert result["tracks"] == [{"path": "recording.wav"}]
    if missing:
        assert "evidence_missing" in result["problems"][0]
