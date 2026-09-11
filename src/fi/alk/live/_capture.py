"""Live→fixture demotion (unit 2.7) + the ONE provenance schema.

Imports: stdlib only (plus kit substrate). A captured fixture earns the
``captured_fixture`` class only after: credential-free green replay, a clean
secret scrub, the complete provenance block, and a recorded HUMAN review —
promotion into ``examples/captured/<lane>/`` is a review step, never
automatic. Candidates stay under the run's artifacts dir with
``reviewed: false`` (``captured_fixture_candidate`` is NOT an evidence class).
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._attribution import VERIFICATION_EVENT_TYPE
from ._contract import AGENT_LEARNING_RUN_KIND
from ._stats import LaneRunResult
from ._transcript import read_transcript

# The ONE provenance schema (identical in PRD §4.1 / ARCH §2c / UI-UX §3).
CAPTURE_PROVENANCE_FIELDS = (
    "captured_from_lane",
    "captured_run_id",
    "rung",
    "framework",
    "framework_version",
    "capture_date",
    "transcript_sha256",
    "redaction",
    "reviewed",
    "reviewer",
)

FIXTURE_CAPTURE_INCOMPLETE_FINDING = "fixture_capture_incomplete_transcript"

_CAPTURE_TREE_MARKER = ("examples", "captured")


class CaptureRefusedError(RuntimeError):
    """Demotion refused; carries the structured finding the CLI surfaces."""

    def __init__(self, message: str, *, finding: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.finding = dict(finding)


def _refuse(detail: str) -> CaptureRefusedError:
    return CaptureRefusedError(
        detail,
        finding={
            "type": FIXTURE_CAPTURE_INCOMPLETE_FINDING,
            "level": "error",
            "detail": detail,
        },
    )


def _canonical_transcript_sha256(events: Sequence[Mapping[str, Any]]) -> str:
    """Content hash over the canonical JSONL re-serialization of the events —
    replay-verifiable from the embedded transcript alone."""

    digest = hashlib.sha256()
    for event in events:
        line = json.dumps(dict(event), ensure_ascii=False, default=str) + "\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _scrub_values_found(
    serialized: str, required_env: Sequence[str]
) -> list[str]:
    """Second scan at capture time (ARCH §2c rule 2): any declared env name
    whose CURRENT value appears in the serialized fixture is a scrub hit."""

    hits: list[str] = []
    for name in required_env:
        value = os.environ.get(name)
        if value and value in serialized:
            hits.append(name)
    return hits


def _inside_capture_tree(path: Path) -> bool:
    parts = [part.lower() for part in path.resolve().parts]
    for index in range(len(parts) - 1):
        if (parts[index], parts[index + 1]) == _CAPTURE_TREE_MARKER:
            return True
    return False


def _pick_source_repeat(
    result: LaneRunResult, repeat_index: int | None
) -> Mapping[str, Any]:
    rows = result.per_repeat
    if repeat_index is not None:
        for row in rows:
            if row.get("repeat") == repeat_index:
                return row
        raise ValueError(f"no repeat {repeat_index} in this lane run")
    for row in rows:
        if row.get("passed") and not row.get("quarantined"):
            return row
    raise _refuse(
        "no passing, non-quarantined repeat to capture — a fixture must "
        "replay green, so a non-green run cannot demote"
    )


def capture_to_fixture(
    result: LaneRunResult,
    *,
    output: Path | str,
    reviewed_by: str | None = None,
    scenario: Mapping[str, Any] | None = None,
    repeat_index: int | None = None,
) -> Path:
    """Demote a live run into a credential-free fixture: transcript verbatim
    (already redacted at record time), required_env reduced to NAMES, plus
    the ONE provenance block.

    Without reviewed_by → a CANDIDATE: written under the run's artifacts dir
    (never a gate-scanned tree), evidence_class kept from the source run,
    reviewed=False. With reviewed_by → runs the credential-free replay
    itself, REFUSES to stamp on a non-green replay, then writes
    evidence_class="captured_fixture", reviewed=True, reviewer=<name> — the
    only form legal under examples/captured/<lane>/. Refuses
    (fixture_capture_incomplete_transcript) when transcript.complete is
    False or the scrub finds any credential value."""

    output_path = Path(output)
    if reviewed_by is None and _inside_capture_tree(output_path):
        raise _refuse(
            "candidates never land in the gate-scanned capture tree "
            "(examples/captured/); promotion is a human review step — "
            "pass reviewed_by after review"
        )

    row = _pick_source_repeat(result, repeat_index)
    if not row.get("transcript_complete", False):
        raise _refuse(
            "transcript is truncated (complete: false) — a truncated "
            "transcript can never demote (PRD §4.1)"
        )
    transcript_path = row.get("transcript_path")
    if not transcript_path or not Path(str(transcript_path)).is_file():
        raise _refuse(f"transcript file missing: {transcript_path!r}")
    events = read_transcript(str(transcript_path))
    if not events:
        raise _refuse("transcript is empty — nothing to demote")

    transcript_sha256 = _canonical_transcript_sha256(events)
    capture_block: dict[str, Any] = {
        "captured_from_lane": result.lane,
        "captured_run_id": result.run_id,
        "rung": result.rung,
        "framework": result.framework,
        "framework_version": result.framework_version,
        "capture_date": _datetime.date.today().isoformat(),
        "transcript_sha256": transcript_sha256,
        "redaction": {
            "required_env_names": list(result.required_env),
            "values_found": 0,
        },
        "reviewed": False,
        "reviewer": None,
    }
    payload: dict[str, Any] = {
        "kind": AGENT_LEARNING_RUN_KIND,
        "name": f"captured-{result.lane}-{result.run_id[:8]}",
        "evidence_class": result.evidence_class,  # candidate keeps source class
        "required_env": list(result.required_env),
        "transcript": [dict(event) for event in events],
        "live_lane": {
            "lane": result.lane,
            "rung": result.rung,
            "verdict": result.verdict,
            "captured_repeat": row.get("repeat"),
            "icc": result.icc,
            "divergence_step": result.divergence_step,
        },
        "capture": capture_block,
    }
    if scenario is not None:
        payload["scenario"] = dict(scenario)

    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    scrub_hits = _scrub_values_found(serialized, result.required_env)
    if scrub_hits:
        raise _refuse(
            "credential values found in the capture for declared env names "
            f"{scrub_hits} — fix the substrate redaction and regenerate; "
            "never hand-edit the secret out"
        )

    if reviewed_by is not None:
        replay = replay_fixture_payload(payload)
        if replay["verdict"] != "pass":
            raise _refuse(
                "credential-free replay was not green "
                f"({replay['verdict']}; checks: {replay['checks']}) — "
                "refusing to stamp captured_fixture; nothing written"
            )
        payload["evidence_class"] = "captured_fixture"
        capture_block["reviewed"] = True
        capture_block["reviewer"] = str(reviewed_by)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return output_path


def replay_fixture_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Credential-free replay of a fixture payload: the fixture is data —
    integrity (canonical sha), completeness, redaction residue, and the
    recorded verifier evidence are re-derived; nothing live executes."""

    checks: dict[str, bool] = {}
    events = payload.get("transcript")
    events = events if isinstance(events, list) else []
    checks["transcript_present"] = bool(events)

    capture = payload.get("capture")
    capture = capture if isinstance(capture, Mapping) else {}
    expected_sha = capture.get("transcript_sha256")
    checks["transcript_sha256_match"] = bool(events) and (
        _canonical_transcript_sha256(events) == expected_sha
    )

    verification: bool | None = None
    for event in events:
        if isinstance(event, Mapping) and event.get("type") == VERIFICATION_EVENT_TYPE:
            event_payload = event.get("payload")
            if isinstance(event_payload, Mapping) and "passed" in event_payload:
                verification = bool(event_payload["passed"])
    checks["verification_passed"] = verification is True

    required_env = [
        str(name) for name in payload.get("required_env") or [] if name
    ]
    serialized = json.dumps(dict(payload), ensure_ascii=False, default=str)
    checks["redaction_clean"] = not _scrub_values_found(serialized, required_env)

    return {
        "verdict": "pass" if all(checks.values()) else "fail",
        "evidence_class": payload.get("evidence_class"),
        "checks": checks,
    }


def replay_fixture(path: Path | str) -> dict[str, Any]:
    """Load a fixture file and replay it credential-free (unit 5.4 contract)."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"fixture {path} is not a JSON object")
    return replay_fixture_payload(payload)
