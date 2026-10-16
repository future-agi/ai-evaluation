"""A call carries its scenario's sub-actors: faults armed for its tools, voices for its engine."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from test_call_runner import FakeAdapter, _FakeScenario, _context, _postgres_endpoint, _report, _run, _runtime

from fi.alk.harness.bundle_v2 import EvidenceSeam

from fi.alk.harness import call_runner as cr
from fi.alk.harness.cast import Behaviour, Cast, FaultProfile, SubActor, save_cast
from fi.alk.harness.world import GeneratedWorld

_CAST = Cast(
    sub_actors=[
        SubActor(
            id="subactor_bystander",
            name="Bystander",
            role="someone near the caller",
            kind="behavioral",
            channel="audio_mix",
            behaviour=Behaviour(prompt="asks for something the caller did not ask for"),
        ),
        SubActor(
            id="subactor_service",
            name="Service",
            kind="system",
            channel="tool_response",
            depends_on=["book"],
            faults=[FaultProfile(tool="book", kind="error")],
        ),
    ]
)


def _drive(tmp_path: Path, monkeypatch, *, enabled: bool = True, **doc: Any) -> tuple[dict[str, str], list]:
    if enabled:
        monkeypatch.setenv("HARNESS_SUB_ACTORS", "1")
    else:
        monkeypatch.delenv("HARNESS_SUB_ACTORS", raising=False)
    _job, context = _context(tmp_path=tmp_path, evidence_seam=EvidenceSeam.TOOL_TRACE)
    save_cast(context.bundle_dir, _CAST)
    folder = context.bundle_dir / "scenarios" / "k1"
    folder.mkdir(parents=True, exist_ok=True)
    body = {"scenario_key": "k1", "scenario_id": "", "sub_goals": [], "instruction": "Book it.", "persona": None, "fixture": {}, "tests": ""}
    body.update(doc)
    (folder / "scenario.json").write_text(json.dumps(body), encoding="utf-8")
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)

    async def place_call(spec):
        return _report(
            transcript="hi",
            messages=[{"role": "user", "content": "hi"}],
            started_at=started,
            ended_at=started + timedelta(seconds=20),
        )

    armed: list = []
    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call, environ={})
    with patch.object(cr, "_build_spec", wraps=cr._build_spec) as build, patch.object(
        cr, "_arm_faults", side_effect=lambda dsn, rows: armed.append(rows)
    ):
        _run(
            runner,
            _FakeScenario("k1"),
            _runtime(metadata={"livekit_agent_name": "a-w0"}, endpoints={"db": _postgres_endpoint()}),
        )
    return dict(build.call_args.kwargs["environ"]), armed


def test_a_bound_voice_reaches_the_call_and_an_unbound_call_carries_none(tmp_path, monkeypatch):
    environ, _armed = _drive(
        tmp_path, monkeypatch, sub_actors=[{"id": "subactor_bystander", "trigger": {"kind": "mid_journey", "turn": 2}}]
    )
    voices = json.loads(environ[cr.SUB_ACTOR_VOICES_ALIAS])
    assert [one["id"] for one in voices] == ["subactor_bystander"] and voices[0]["trigger"]["turn"] == 2

    environ, _armed = _drive(tmp_path / "next", monkeypatch)
    assert cr.SUB_ACTOR_VOICES_ALIAS not in environ


def test_with_the_switch_off_nothing_is_carried(tmp_path, monkeypatch):
    environ, armed = _drive(tmp_path, monkeypatch, enabled=False, sub_actors=[{"id": "subactor_bystander"}])
    assert cr.SUB_ACTOR_VOICES_ALIAS not in environ
    assert armed == []


def test_the_caller_is_told_who_else_is_there():
    environ = {cr.SUB_ACTOR_VOICES_ALIAS: json.dumps([{"name": "Bystander", "role": "someone near the caller"}])}
    assert "Bystander (someone near the caller)" in cr._with_sub_actors("Book it.", environ)
    assert cr._with_sub_actors("Book it.", {}) == "Book it."


def test_a_traced_fault_is_read_back_onto_its_call():
    assert cr._fault_of("fault:timeout:subactor_service") == ("timeout", "subactor_service")
    assert cr._fault_of("upstream said no") == ("", "")


def test_arming_faults_never_blocks_a_call():
    cr._arm_faults("postgresql://nowhere/none", [{"tool": "book", "kind": "error", "on_call": 1, "latency_ms": 0, "content": "", "sub_actor": "s"}])


class _World(GeneratedWorld):
    name = "w"
    tools = [{"name": "book"}]
    handlers = {"book": "def handle(args, db):\n    return {'booked': True}\n"}


def test_a_world_fails_only_the_call_its_fault_names():
    world = _World(":memory:")
    world.faults = [{"tool": "book", "kind": "error", "on_call": 2, "latency_ms": 0, "content": "", "sub_actor": "subactor_service"}]
    first, second, third = (world.call("book", {}) for _ in range(3))
    assert first.ok and third.ok
    assert not second.ok and second.fault == "error" and second.sub_actor == "subactor_service"
    assert not second.refused


def test_a_content_fault_adds_to_what_the_agent_reads():
    world = _World(":memory:")
    world.faults = [{"tool": "book", "kind": "content", "on_call": 1, "latency_ms": 0, "content": "extra", "sub_actor": "s"}]
    made = world.call("book", {})
    assert made.ok and made.result == {"booked": True, "note": "extra"} and made.fault == "content"


def test_a_world_with_no_faults_is_unchanged():
    world = _World(":memory:")
    made = world.call("book", {})
    assert made.ok and made.result == {"booked": True} and made.fault == ""


def test_a_bound_service_arms_its_faults_for_this_call_only(tmp_path, monkeypatch):
    _environ, armed = _drive(
        tmp_path, monkeypatch, sub_actors=[{"id": "subactor_service", "trigger": {"kind": "on_tool", "tool": "book", "on_call": 2}}]
    )
    assert armed == [[{"tool": "book", "kind": "error", "on_call": 2, "latency_ms": 0, "content": "", "sub_actor": "subactor_service"}]]

    _environ, armed = _drive(tmp_path / "next", monkeypatch)
    assert armed == [[]]
