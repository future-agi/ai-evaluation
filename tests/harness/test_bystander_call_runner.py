"""A second voice reaches the caller's lane the way the mailbox does: through the environment."""

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
        "instruction": "Order the usual.",
        "persona": None,
        "fixture": {},
        "tests": "",
    }
    body.update(extra)
    (folder / "scenario.json").write_text(json.dumps(body), encoding="utf-8")


def _drive(tmp_path: Path, environ: dict[str, str] | None = None, **doc_fields: Any) -> dict[str, str]:
    """Run one call to completion and hand back the environment it left behind."""
    _job_obj, context = _context(tmp_path=tmp_path)
    _doc(context.bundle_dir, **doc_fields)
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)

    async def place_call(spec):
        return _report(
            transcript="hello",
            messages=[{"role": "user", "content": "hello"}],
            started_at=started,
            ended_at=started + timedelta(seconds=20),
        )

    kept = {} if environ is None else environ
    runner = cr.CallRunnerImpl(FakeAdapter(), context, place_call=place_call, environ=kept)
    _run(runner, _FakeScenario("k1"), _runtime(metadata={"livekit_agent_name": "a-w0"}))
    return kept


def test_a_bystander_line_reaches_the_call(tmp_path: Path) -> None:
    environ = _drive(tmp_path, bystander="Mum, are we nearly there")
    assert environ.get("HARNESS_BYSTANDER_LINE") == "Mum, are we nearly there"


def test_a_bystander_is_not_tied_to_direction(tmp_path: Path) -> None:
    """Somebody can talk across the caller whoever placed the call."""
    environ = _drive(
        tmp_path, bystander="Are you nearly done", call_direction="outbound"
    )
    assert environ.get("HARNESS_BYSTANDER_LINE") == "Are you nearly done"
    assert environ.get("HARNESS_CALL_DIRECTION") == "outbound"


def test_a_scenario_with_nobody_else_clears_a_previous_line(tmp_path: Path) -> None:
    """One noisy scenario must not put a child in the back seat of the next call."""
    environ = _drive(tmp_path, {"HARNESS_BYSTANDER_LINE": "left over"})
    assert "HARNESS_BYSTANDER_LINE" not in environ
