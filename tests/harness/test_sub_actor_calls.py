"""A call carries its scenario's sub-actors: their voices, for its engine."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from test_call_runner import FakeAdapter, _FakeScenario, _context, _postgres_endpoint, _report, _run, _runtime

from fi.alk.harness.bundle_v2 import EvidenceSeam

from fi.alk.harness import call_runner as cr
from fi.alk.harness.cast import Behaviour, Cast, SubActor, save_cast
from fi.simulate.simulation.sub_actors import with_others

_CAST = Cast(
    sub_actors=[
        SubActor(
            id="subactor_bystander",
            name="Bystander",
            role="someone near the caller",
            channel="audio_mix",
            behaviour=Behaviour(prompt="asks for something the caller did not ask for"),
        ),
    ]
)


def _drive(tmp_path: Path, monkeypatch, *, enabled: bool = True, **doc: Any) -> dict[str, str]:
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

    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call, environ={})
    with patch.object(cr, "_build_spec", wraps=cr._build_spec) as build:
        _run(
            runner,
            _FakeScenario("k1"),
            _runtime(metadata={"livekit_agent_name": "a-w0"}, endpoints={"db": _postgres_endpoint()}),
        )
    return dict(build.call_args.kwargs["environ"])


def test_a_bound_voice_reaches_the_call_and_an_unbound_call_carries_none(tmp_path, monkeypatch):
    environ = _drive(
        tmp_path, monkeypatch, sub_actors=[{"id": "subactor_bystander", "trigger": {"kind": "mid_journey", "turn": 2}}]
    )
    voices = json.loads(environ[cr.SUB_ACTOR_VOICES_ALIAS])
    assert [one["id"] for one in voices] == ["subactor_bystander"] and voices[0]["trigger"]["turn"] == 2

    environ = _drive(tmp_path / "next", monkeypatch)
    assert cr.SUB_ACTOR_VOICES_ALIAS not in environ


def test_with_the_switch_off_nothing_is_carried(tmp_path, monkeypatch):
    environ = _drive(tmp_path, monkeypatch, enabled=False, sub_actors=[{"id": "subactor_bystander"}])
    assert cr.SUB_ACTOR_VOICES_ALIAS not in environ


def test_the_caller_is_told_who_else_is_there():
    voices = [{"name": "Bystander", "role": "someone near the caller", "channel": "audio_mix"}]
    assert "Bystander (someone near the caller)" in with_others("Book it.", voices)
    assert with_others("Book it.", []) == "Book it."


def test_the_caller_is_told_what_to_do_when_someone_takes_the_line():
    assert "say nothing" in with_others("Book it.", [{"name": "Partner", "channel": "audio_handoff"}])
    assert "Give them a turn" in with_others("Order it.", [{"name": "Friend", "channel": "thread_message"}])


def test_turns_count_only_the_person_and_the_agent():
    from fi.alk.harness.run.conversation import Exchange, Transcript

    transcript = Transcript(
        exchanges=[Exchange("customer", "hi"), Exchange("sub_actor", "add one", name="F", sub_actor="s"), Exchange("agent", "ok")]
    )
    assert sum(1 for turn in transcript.exchanges if turn.speaker != "sub_actor") == 2


def test_each_sub_actor_sounds_like_its_own_persona_and_never_like_the_caller():
    caller = {"name": "Dana", "gender": "female", "accent": "American", "languages": ["English"]}
    voices = [
        {"id": "child", "behaviour": {"loudness_db": -12.0}, "persona": {"name": "Child", "gender": "female", "age_group": "18-25", "personality": "Friendly"}},
        {"id": "partner", "behaviour": {}, "persona": {"name": "Alex", "gender": "male", "personality": "Professional"}},
    ]
    voiced = cr._voiced_sub_actors(voices, caller, "cartesia")
    said = [cr.persona_voice(caller, "cartesia")] + [
        voice for one in voiced for voice in (one["behaviour"]["voice"], one["behaviour"]["second_voice"])
    ]
    assert all(said) and len(set(said)) == len(said)
    assert voiced[1]["behaviour"]["emotion"] == cr.persona_emotion(voices[1]["persona"])
    assert voiced[0]["behaviour"]["speed"] == cr.persona_speech_rate(voices[0]["persona"])
    assert -16.0 <= voiced[0]["behaviour"]["loudness_db"] <= -8.0
