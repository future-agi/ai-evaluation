"""Coding-modality bench suite + the artifact-in runner.

A coding suite is a bench-native shape (``agent-learning.bench-suite.v1``): each
task carries an ``instruction``, a held-out ``checks`` oracle (executed against
the candidate, never imported by it), a ``reference_solution`` (the gold, used by
the release gate to prove the verifier accepts a correct answer), and optional
``guards``. This is deliberately distinct from the objective-anchored task
dataset: coding's verdict is "do the held-out tests pass", not a weighted-metric
mean. The two unify at the Result level, not the suite level — exactly the shape
the prior-art survey found (task specs are modality-specific; the Task<->Verifier
coupling and the unified Result are the invariant).

``artifact_in`` control mode scores a *submitted artifact* (candidate code) with
no live agent — the analogue of patch-scoring harnesses. The agent that produced
the artifact is out of scope here; only the held-out oracle decides the verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ._codeexec import run_code_tests
from ._grader import GRADING_COMMAND, run_command_graded

BENCH_SUITE_KIND = "agent-learning.bench-suite.v1"

# A task is graded one of two ways:
#  * "checks" (default, convenience tier): held-out check_* functions import the
#    candidate in-process. Trusted / accidental-gaming only.
#  * "command" (hardened tier): the candidate produces files/output, a held-out
#    grader runs AFTER and emits the verdict via exit code + reward file. Robust
#    against forge + oracle-read; multi-language. See _grader.py.
_CHECKS_FIELDS = ("id", "instruction", "checks", "reference_solution")
_COMMAND_FIELDS = ("id", "instruction", "grader_cmd", "grader_files", "reference_files")


class CodingSuiteError(ValueError):
    """Raised for a malformed coding bench suite."""


def is_bench_suite(obj: Any) -> bool:
    return isinstance(obj, Mapping) and obj.get("kind") == BENCH_SUITE_KIND


def _task_grading(task: Mapping[str, Any]) -> str:
    """The grading mode of a task: 'command' (hardened) or 'checks' (convenience)."""

    mode = task.get("grading")
    if mode in (GRADING_COMMAND, "checks"):
        return str(mode)
    # infer: a grader_cmd ⇒ command-graded; otherwise the legacy checks tier.
    return GRADING_COMMAND if task.get("grader_cmd") else "checks"


def load_coding_suite(obj: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Load + validate a coding bench suite (from a path or an in-memory mapping)."""

    if isinstance(obj, (str, Path)):
        data: Mapping[str, Any] = json.loads(Path(obj).expanduser().read_text("utf-8"))
    else:
        data = obj
    if not is_bench_suite(data):
        raise CodingSuiteError(f"not a {BENCH_SUITE_KIND} suite")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise CodingSuiteError("coding suite has no tasks")
    seen: set[str] = set()
    for i, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise CodingSuiteError(f"task #{i} is not an object")
        required = _COMMAND_FIELDS if _task_grading(task) == GRADING_COMMAND else _CHECKS_FIELDS
        for field in required:
            if not task.get(field):
                raise CodingSuiteError(
                    f"task #{i} ({_task_grading(task)}-graded) missing required field {field!r}"
                )
        tid = str(task["id"])
        if tid in seen:
            raise CodingSuiteError(f"duplicate task id {tid!r}")
        seen.add(tid)
        # Every task must declare at least one guard against reward hacking; the
        # held-out oracle is the primary defence, but the suite must say so.
        guards = task.get("guards") or {}
        if int(guards.get("min_guard_count", 0)) < 1:
            raise CodingSuiteError(
                f"task {tid!r} must declare guards.min_guard_count >= 1 "
                "(the held-out-oracle anti-gaming contract)"
            )
    return dict(data)


def _coding_row(
    task: Mapping[str, Any],
    verdict_obj: Mapping[str, Any],
    *,
    evidence_class: str,
    sandbox: str,
) -> dict[str, Any]:
    result = dict(verdict_obj["result"])
    scalar = result.get("scalar")
    # All-or-nothing: a coding task is resolved only if EVERY held-out check passes.
    verdict = "pass" if scalar is not None and float(scalar) >= 1.0 else "fail"
    return {
        "task_id": str(task["id"]),
        "modality": "coding",
        "world_kind": "code_exec",
        "control_mode": "artifact_in",
        "result": result,
        "verdict": verdict,
        # The candidate code really executed; honest execution_class is executable.
        "execution_class": "executable",
        "evidence_class": evidence_class,
        # executable + any evidence class is never an overclaim (it really ran).
        "overclaim": False,
        "sandbox": sandbox,
        "raw": verdict_obj.get("raw", {}),
    }


def _void_row(task: Mapping[str, Any], reason: str, *, evidence_class: str) -> dict[str, Any]:
    return {
        "task_id": str(task["id"]),
        "modality": "coding",
        "world_kind": "code_exec",
        "control_mode": "artifact_in",
        "result": {
            "scalar": None,
            "components": {},
            "pass_fail": {},
            "explanation": reason,
        },
        "verdict": "void",
        "execution_class": "executable",
        "evidence_class": evidence_class,
        "overclaim": False,
        "error": reason,
    }


def run_coding_artifact_in(
    suite: Mapping[str, Any],
    submission: Mapping[str, Any],
    *,
    sandbox: str = "subprocess",
    evidence_class: str = "captured_fixture",
    max_tasks: int | None = None,
    default_timeout_s: float = 10.0,
) -> list[dict[str, Any]]:
    """Score each task's submitted artifact against its held-out oracle.

    ``submission`` maps ``task_id -> candidate``. For ``checks``-graded tasks the
    candidate is a source string; for ``command``-graded (hardened) tasks it is a
    ``{path: content}`` file map. A task with no submission is recorded ``void``
    (never silently passed); an infra failure is recorded ``void`` too. Pass the
    :func:`reference_submission` to verify the suite itself (what the gate does).
    """

    language = str(suite.get("language", "python"))
    # Honesty: a Docker run executes untrusted candidate code under real
    # isolation -> that is a genuine LIVE event, never a fixture/local class.
    # Force at least live_lane (honor an explicit live_stressed); never downgrade.
    if sandbox == "docker" and evidence_class not in ("live_lane", "live_stressed"):
        evidence_class = "live_lane"
    rows: list[dict[str, Any]] = []
    tasks = suite["tasks"]
    if max_tasks is not None:
        tasks = tasks[: max(0, int(max_tasks))]
    for task in tasks:
        tid = str(task["id"])
        candidate = submission.get(tid)
        if candidate is None:
            rows.append(_void_row(task, "no submission provided", evidence_class=evidence_class))
            continue
        timeout_s = float(task.get("timeout_s", default_timeout_s))
        if _task_grading(task) == GRADING_COMMAND:
            files = candidate if isinstance(candidate, Mapping) else {"solution": str(candidate)}
            verdict_obj = run_command_graded(
                task, {str(k): str(v) for k, v in files.items()},
                sandbox=sandbox, timeout_s=timeout_s,
            )
        else:
            verdict_obj = run_code_tests(
                str(candidate), str(task["checks"]),
                language=language, timeout_s=timeout_s, sandbox=sandbox,
            )
        # An infra/config failure (no Docker daemon, image pull failure, bad
        # sandbox/language) means the lane never ran — record it as VOID, never as
        # a real "fail". Conflating "the daemon was missing" with "the agent was
        # wrong" would silently report a correct agent at 0%.
        if (verdict_obj.get("raw") or {}).get("infra_error"):
            reason = (verdict_obj.get("result") or {}).get("explanation") or "infrastructure error"
            rows.append(_void_row(task, f"infra: {reason}", evidence_class=evidence_class))
            continue
        rows.append(
            _coding_row(task, verdict_obj, evidence_class=evidence_class, sandbox=sandbox)
        )
    return rows


def reference_submission(suite: Mapping[str, Any]) -> dict[str, Any]:
    """The gold submission: every task id -> its reference.

    ``checks``-graded tasks map to a ``reference_solution`` string; ``command``-
    graded tasks map to a ``reference_files`` ``{path: content}`` map.
    """

    out: dict[str, Any] = {}
    for t in suite["tasks"]:
        if _task_grading(t) == GRADING_COMMAND:
            out[str(t["id"])] = {str(k): str(v) for k, v in (t["reference_files"] or {}).items()}
        else:
            out[str(t["id"])] = str(t["reference_solution"])
    return out
