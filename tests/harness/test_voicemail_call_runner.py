"""A mailbox reaches the caller's lane the same way the direction does: through the environment."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from test_call_runner import (
    FakeAdapter,
    _FakeScenario,
    _context,
    _report,
    _run,
    _runtime,
)

from fi.alk.harness import call_runner as cr


def _doc(bundle_dir: Path, **extra: Any) -> None:
    folder = bundle_dir / "scenarios" / "k1"
    folder.mkdir(parents=True, exist_ok=True)
    body = {
        "scenario_key": "k1",
        "scenario_id": "",
        "sub_goals": [],
        "instruction": "Leave a message.",
        "persona": None,
        "fixture": {},
        "tests": "",
    }
    body.update(extra)
    (folder / "scenario.json").write_text(json.dumps(body), encoding="utf-8")


def _drive(tmp_path: Path, **doc_fields: Any) -> dict[str, str]:
    """Run one call to completion and hand back the environment it left behind."""
    _job_obj, context = _context(tmp_path=tmp_path)
    _doc(context.bundle_dir, **doc_fields)
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)

    async def place_call(spec):
        return _report(
            transcript="beep",
            messages=[{"role": "user", "content": "beep"}],
            started_at=started,
            ended_at=started + timedelta(seconds=20),
        )

    environ: dict[str, str] = {}
    runner = cr.CallRunnerImpl(
        FakeAdapter(), context, place_call=place_call, environ=environ
    )
    _run(runner, _FakeScenario("k1"), _runtime(metadata={"livekit_agent_name": "a-w0"}))
    return environ


def test_a_voicemail_scenario_marks_its_own_call(tmp_path: Path) -> None:
    environ = _drive(tmp_path, call_direction="outbound", answered_by="voicemail")
    assert environ.get("HARNESS_CALL_DIRECTION") == "outbound"
    assert environ.get("HARNESS_ANSWERED_BY") == "voicemail"


def test_an_ordinary_outbound_scenario_leaves_the_mailbox_unset(tmp_path: Path) -> None:
    environ = _drive(tmp_path, call_direction="outbound", caller_awareness="expecting")
    assert environ.get("HARNESS_CALL_DIRECTION") == "outbound"
    assert "HARNESS_ANSWERED_BY" not in environ


def test_an_inbound_scenario_clears_every_outbound_marking(tmp_path: Path) -> None:
    """One voicemail scenario must not silence the caller of the next call on the same process."""
    _job_obj, context = _context(tmp_path=tmp_path)
    _doc(context.bundle_dir, call_direction="inbound")
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)

    async def place_call(spec):
        return _report(
            transcript="hello",
            messages=[{"role": "user", "content": "hello"}],
            started_at=started,
            ended_at=started + timedelta(seconds=10),
        )

    environ: dict[str, str] = {
        "HARNESS_CALL_DIRECTION": "outbound",
        "HARNESS_CALLER_AWARENESS": "unaware",
        "HARNESS_ANSWERED_BY": "voicemail",
    }
    runner = cr.CallRunnerImpl(
        FakeAdapter(), context, place_call=place_call, environ=environ
    )
    _run(runner, _FakeScenario("k1"), _runtime(metadata={"livekit_agent_name": "a-w0"}))
    assert "HARNESS_CALL_DIRECTION" not in environ
    assert "HARNESS_CALLER_AWARENESS" not in environ
    assert "HARNESS_ANSWERED_BY" not in environ


def test_the_switch_stops_a_mailbox_reaching_the_call(
    tmp_path: Path, monkeypatch
) -> None:
    """A suite written when mailboxes were allowed can be replayed on a run that has turned them
    off, so the scenario on disk still says voicemail and must not silence the call anyway."""
    monkeypatch.setenv("ALK_VOICEMAIL_SCENARIOS", "0")
    environ = _drive(
        tmp_path,
        call_direction="outbound",
        answered_by="voicemail",
        voicemail_style="carrier",
    )
    assert "HARNESS_ANSWERED_BY" not in environ
    assert "HARNESS_VOICEMAIL_STYLE" not in environ
    assert cr.VOICEMAIL_CLIP_ALIAS not in environ
    assert cr.VOICEMAIL_CLIP_TEXT_ALIAS not in environ
    # The direction is not a mailbox concern and still travels.
    assert environ.get("HARNESS_CALL_DIRECTION") == "outbound"
