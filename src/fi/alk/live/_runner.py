"""Untrusted-subprocess driver (P3-D1). Imports: stdlib only.

The framework process is an untrusted subprocess: spawned with a scrubbed
env (safe base + the lane's declared ``required_env`` only), no shell, stdio
JSONL IPC (ARCH Decision 2). The worker reads one boot message on stdin and
emits transcript events on stdout, one JSON object per line; stderr is the
``framework_runtime`` attribution evidence channel.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .._paths import project_root
from ..config import API_KEY_ENV_NAMES, SECRET_KEY_ENV_NAMES
from ._transcript import TranscriptRecorder, redact_env_values

# Env vars that may cross into a lane subprocess WITHOUT being declared:
# the bare process-hygiene minimum, nothing identity-bearing. The runner
# additionally injects PYTHONPATH at spawn time when the worker needs the
# kit importable (ARCH §2b Execution).
LANE_SAFE_BASE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")

# Harness identity NEVER crosses into a lane subprocess, even if a lane
# declares it (P3-D1; R§1 #13: red-team lanes must not share harness
# credential space). These are the kit's own key names from config.py.
LANE_BLOCKED_ENV = tuple(API_KEY_ENV_NAMES) + tuple(SECRET_KEY_ENV_NAMES)

# The worker handshake event (stamped on the 'lane' channel; carries
# {framework, framework_version, capability_hash, package_paths}).
READY_EVENT_TYPE = "framework_ready"

_TAIL_CHARS = 2000
_VERSION_CLAUSE = re.compile(r"^(==|!=|>=|<=|>|<)\s*([0-9][0-9A-Za-z.\-_]*)$")


def scrubbed_lane_env(required_env: Sequence[str]) -> dict[str, str]:
    """Build the subprocess env: safe base + declared required_env, minus
    the harness's own keys. Missing required names are simply absent —
    presence checks are the test layer's job (unit 5 credentialed pattern)."""

    env = {k: os.environ[k] for k in LANE_SAFE_BASE_ENV if k in os.environ}
    for name in required_env:
        if name in LANE_BLOCKED_ENV:
            continue  # harness identity never crosses (P3-D1)
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def kit_pythonpath() -> str:
    """The src/ directory that makes ``fi.alk`` importable in a
    worker subprocess (injected at spawn time, ARCH §2b Execution)."""

    return str(project_root(__file__) / "src")


@dataclasses.dataclass
class LaneProcessResult:
    exit_code: int | None
    duration_s: float
    stdout_tail: str          # last 2000 chars, redacted
    stderr_tail: str          # last 2000 chars, redacted
    timed_out: bool


def spawn_lane_subprocess(
    args: Sequence[str],
    *,
    lane: str,
    required_env: Sequence[str],
    cwd: Path,
    timeout_s: float,
    transcript: TranscriptRecorder,
    input_payloads: Sequence[Mapping[str, Any]] = (),
    inject_kit_pythonpath: bool = True,
) -> LaneProcessResult:
    """Run an untrusted framework process (P3-D1): fresh subprocess via
    sys.executable, env = scrubbed_lane_env(required_env) only, no shell,
    cwd inside a tempdir owned by the run. Records spawn/exit/timeout
    events on the 'lane' channel. Never raises on process failure — failure
    classification is attribute_failure()'s job, not an exception path."""

    env = scrubbed_lane_env(required_env)
    if inject_kit_pythonpath:
        env["PYTHONPATH"] = kit_pythonpath()
    argv = [str(arg) for arg in args]
    transcript.record(
        "lane",
        "spawn",
        {"lane": lane, "args": argv, "timeout_s": timeout_s},
    )
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        duration = time.monotonic() - started
        transcript.record("lane", "spawn_error", {"error": str(exc)})
        return LaneProcessResult(
            exit_code=None,
            duration_s=duration,
            stdout_tail="",
            stderr_tail=redact_env_values(str(exc), required_env)[-_TAIL_CHARS:],
            timed_out=False,
        )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _pump_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            stdout_chunks.append(line)
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except ValueError:
                event = None
            if isinstance(event, dict) and "channel" in event and "type" in event:
                payload = event.get("payload")
                transcript.record(
                    str(event.get("channel")),
                    str(event.get("type")),
                    payload if isinstance(payload, dict) else {"value": payload},
                )
            else:
                transcript.record(
                    "lane",
                    "worker_stdout_unparsed",
                    {"line": stripped[:500]},
                )

    def _pump_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_chunks.append(line)

    stdout_thread = threading.Thread(target=_pump_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_pump_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    try:
        if process.stdin is not None:
            for payload in input_payloads:
                process.stdin.write(
                    json.dumps(dict(payload), ensure_ascii=False, default=str)
                    + "\n"
                )
            process.stdin.flush()
            process.stdin.close()
    except (BrokenPipeError, OSError):
        pass  # the worker died before reading its boot — exit/attribution covers it

    timed_out = False
    try:
        exit_code: int | None = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        exit_code = process.returncode
        transcript.record("lane", "timeout", {"timeout_s": timeout_s})
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    duration = time.monotonic() - started
    transcript.record(
        "lane",
        "exit",
        {
            "exit_code": exit_code,
            "duration_s": round(duration, 6),
            "timed_out": timed_out,
        },
    )
    stdout_tail = redact_env_values("".join(stdout_chunks), required_env)
    stderr_tail = redact_env_values("".join(stderr_chunks), required_env)
    return LaneProcessResult(
        exit_code=exit_code,
        duration_s=duration,
        stdout_tail=stdout_tail[-_TAIL_CHARS:],
        stderr_tail=stderr_tail[-_TAIL_CHARS:],
        timed_out=timed_out,
    )


# --- version preflight (unit 2.3) -------------------------------------------


def find_ready_event(
    events: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """First worker handshake event, or None when the worker never got there."""

    for event in events:
        if event.get("type") == READY_EVENT_TYPE:
            return event
    return None


def _version_tuple(version: str) -> tuple[int, ...] | None:
    parts: list[int] = []
    for component in str(version).split("."):
        digits = ""
        for char in component:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else None


def version_ok(framework_version: str | None, requirement: str | None) -> bool:
    """Compare a worker-reported version against a lane's version_requirement.

    Requirement grammar (simplest thing consistent with the architecture):
    comma-separated clauses of ``<op><version>`` with op in
    ``== != >= <= > <``; numeric dotted prefixes compared component-wise.
    No requirement → vacuously ok. Unparseable version/requirement → NOT ok
    (environment trouble voids, never scores)."""

    if requirement is None or not str(requirement).strip():
        return True
    observed = _version_tuple(framework_version or "")
    if observed is None:
        return False
    for clause in str(requirement).split(","):
        clause = clause.strip()
        if not clause:
            continue
        match = _VERSION_CLAUSE.match(clause)
        if not match:
            return False
        op, wanted_raw = match.groups()
        wanted = _version_tuple(wanted_raw)
        if wanted is None:
            return False
        width = max(len(observed), len(wanted))
        left = observed + (0,) * (width - len(observed))
        right = wanted + (0,) * (width - len(wanted))
        if op == "==" and left != right:
            return False
        if op == "!=" and left == right:
            return False
        if op == ">=" and left < right:
            return False
        if op == "<=" and left > right:
            return False
        if op == ">" and left <= right:
            return False
        if op == "<" and left >= right:
            return False
    return True


def version_preflight(
    requirement: str | None,
    ready_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Runs before any scenario turn. On mismatch the row voids lane_infra
    (``framework_version_unsupported``) — environment trouble must never be
    misfiled as framework_runtime robustness evidence or an agent verdict."""

    framework = None
    framework_version = None
    capability_hash = None
    if ready_payload:
        framework = ready_payload.get("framework")
        framework_version = ready_payload.get("framework_version")
        capability_hash = ready_payload.get("capability_hash")
    ok = version_ok(framework_version, requirement)
    preflight: dict[str, Any] = {
        "framework": framework,
        "framework_version": framework_version,
        "capability_hash": capability_hash,
        "version_requirement": requirement,
        "version_ok": ok,
        "void_reason": None,
    }
    if not ok:
        preflight["void_reason"] = (
            "framework_version_unsupported: observed "
            f"{framework_version!r}, required {requirement!r}"
        )
    return preflight


def run_worker_once(
    worker_path: str | Path,
    boot: Mapping[str, Any],
    *,
    lane: str,
    required_env: Sequence[str],
    cwd: Path,
    timeout_s: float,
    transcript: TranscriptRecorder,
    version_requirement: str | None = None,
) -> dict[str, Any]:
    """One repeat = one fresh worker subprocess (process reuse would couple
    repeats and corrupt ICC). Returns the run_once row contract consumed by
    ``_stats.run_repeated``: passed/score/failure_layer/detail/void_reason
    plus step_signature and the version-preflight stanza."""

    from ._attribution import attribute_failure
    from ._stats import step_signature_from_events

    process = spawn_lane_subprocess(
        [sys.executable, str(worker_path)],
        lane=lane,
        required_env=required_env,
        cwd=cwd,
        timeout_s=timeout_s,
        transcript=transcript,
        input_payloads=[boot],
    )
    events = list(transcript.events)
    ready = find_ready_event(events)
    preflight = version_preflight(
        version_requirement,
        ready.get("payload") if ready else None,
    )
    row: dict[str, Any] = {
        "transcript_path": str(transcript.path),
        "step_signature": step_signature_from_events(events),
        "version": preflight,
        "process": {
            "exit_code": process.exit_code,
            "duration_s": round(process.duration_s, 6),
            "timed_out": process.timed_out,
            "stderr_tail": process.stderr_tail,
        },
    }
    for event in events:
        if event.get("type") == "end_state_diff":
            payload = event.get("payload")
            if isinstance(payload, Mapping):
                row["end_state_diff"] = dict(payload)
    if not preflight["version_ok"]:
        row.update(
            passed=None,
            score=None,
            failure_layer="lane_infra",
            void_reason=preflight["void_reason"],
            detail=str(preflight["void_reason"]),
        )
        return row
    attribution = attribute_failure(process, events)
    if attribution is None:
        row.update(passed=True, score=1.0, failure_layer=None, detail="")
        return row
    row.update(
        passed=None if attribution.layer == "lane_infra" else False,
        score=None if attribution.layer == "lane_infra" else 0.0,
        failure_layer=attribution.layer,
        detail=attribution.detail,
    )
    if attribution.layer == "lane_infra":
        row["void_reason"] = attribution.detail
    return row
