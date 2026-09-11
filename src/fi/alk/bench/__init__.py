"""Unified benchmark harness — the benchmarking front door to the simulation layer.

This module is a *composition facade*, not a new engine. A modern agent benchmark
harness decomposes into five layers — Task / Environment / Agent-adapter /
Verifier / Runner — and the kit already ships them (``tasks`` + worlds + the
framework adapters + ``evals``/``rewardhack`` + the live-lane runner + telemetry).
``bench`` adds the *contract glue* on top:

* **Fixed Task<->Verifier coupling** — a suite carries (or references) its own
  oracle. This is the one part that never varies by modality.
* **Pluggable Environment + Agent-adapter** — the modality (text / tool / coding
  / voice / ...) is a dimension, not a fork.
* **Three control modes** —
    - ``push``         : the harness drives the agent (today's ``run_benchmark``);
                         live for text / tool task datasets;
    - ``artifact_in``  : score a submitted artifact, no live agent; live for coding
                         bench suites (subprocess or opt-in Docker sandbox);
    - ``pull``         : the agent drives a live environment via reset/step
                         (staged; raises ``NotImplementedError`` until it lands).
  Unsupported (suite, control_mode) combinations raise ``BenchError``.
* **A unified ``Result``** ``{scalar, components, pass_fail, explanation}`` that
  every modality's verdict projects into. ``pass_fail`` keys are modality-defined
  (push -> ``{"verdict": bool}``; coding -> ``{check_name: bool}``; void -> ``{}``);
  the portable cross-modality signal is the row-level ``verdict`` + ``result.scalar``.

Honesty primitives are preserved verbatim: every per-task row keeps its
``execution_class`` / ``evidence_class`` and the overclaim tripwire, and the
reward-hack detector still fails a gamed run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .. import tasks

# Re-exported coding-suite helpers (public): callers should use these, not the
# private ``bench._coding`` module. Explicit ``as`` re-export marks them public
# without an ``__all__`` (which would implicitly privatise run_bench /
# run_bench_file / load_bench_suite / modality_for_world_kind / BenchError / ...).
from ._coding import load_coding_suite as load_coding_suite
from ._coding import reference_submission as reference_submission

BENCH_RESULT_KIND = "agent-learning.bench-result.v1"

#: The control modes a bench run can take. ``push`` (text/tool) and ``artifact_in``
#: (coding suites) are live; ``pull`` is staged and raises ``NotImplementedError``.
CONTROL_MODES = ("push", "artifact_in", "pull")

#: Code sandboxes for the ``artifact_in`` coding lane.
SANDBOXES = ("subprocess", "docker")

#: World-kind -> coarse modality label. Kept deliberately small; new worlds map
#: here as they become executable.
_WORLD_KIND_MODALITY = {
    "conversation": "text",
    "tool_api": "tool",
    "code_exec": "coding",
    "browser": "computer_use",
    "computer_use": "computer_use",
    "voice_telephony": "voice",
}


class BenchError(ValueError):
    """Raised for malformed bench suites or invalid harness arguments."""


def modality_for_world_kind(world_kind: str) -> str:
    """Map a world kind to its coarse modality label (``unknown`` if unmapped)."""

    return _WORLD_KIND_MODALITY.get(str(world_kind), "unknown")


def load_bench_suite(suite: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Resolve a bench suite to a compiled TaskDataset.

    Accepts a path (loaded + compiled via :func:`tasks.load_task_dataset`) or an
    already-compiled dataset mapping (returned as a shallow copy). A raw,
    uncompiled mapping is compiled via :func:`tasks.compile_task_dataset` so the
    Goodhart guards are enforced before any run.
    """

    if isinstance(suite, (str, Path)):
        return tasks.load_task_dataset(suite)
    if isinstance(suite, Mapping):
        # A compiled dataset is idempotent under compile; compiling here keeps the
        # guard checks on the Task<->Verifier coupling no matter how it arrived.
        return tasks.compile_task_dataset(suite)
    raise BenchError(
        f"suite must be a path or a dataset mapping, got {type(suite).__name__!r}"
    )


def _project_result(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a ``tasks.run_benchmark`` per-task row into the unified Result.

    The superset shape is synthesised from the strongest external references
    (a metric->number map plus a value+rationale verdict): scalar score,
    per-metric components, pass/fail booleans, and a short explanation.
    """

    metric_averages = row.get("metric_averages") or {}
    components = {
        str(k): float(v)
        for k, v in metric_averages.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    scoring = row.get("scoring") or {}
    return {
        "scalar": row.get("score"),
        "components": components,
        "pass_fail": {"verdict": row.get("verdict") == "pass"},
        "explanation": scoring.get("basis"),
    }


def _bench_row(row: Mapping[str, Any], *, control_mode: str) -> dict[str, Any]:
    world_kind = str(row.get("world_kind") or "")
    out: dict[str, Any] = {
        "task_id": row.get("task_id"),
        "modality": modality_for_world_kind(world_kind),
        "world_kind": world_kind,
        "control_mode": control_mode,
        "result": _project_result(row),
        "verdict": row.get("verdict"),
        "execution_class": row.get("execution_class"),
        "evidence_class": row.get("evidence_class"),
        "overclaim": bool(row.get("overclaim", False)),
    }
    if "rewardhack" in row:
        out["rewardhack"] = row["rewardhack"]
    if "error" in row:
        out["error"] = row["error"]
    return out


def _to_bench_result(result: Mapping[str, Any], *, control_mode: str) -> dict[str, Any]:
    """Re-badge an engine benchmark result under the unified bench contract."""

    per_task = [
        _bench_row(r, control_mode=control_mode)
        for r in result.get("per_task", [])
    ]
    out: dict[str, Any] = {
        "kind": BENCH_RESULT_KIND,
        "control_mode": control_mode,
        "dataset_name": result.get("dataset_name"),
        "dataset_version": result.get("dataset_version"),
        "modalities": sorted({r["modality"] for r in per_task}),
        "per_task": per_task,
        "aggregate": result.get("aggregate"),
    }
    if "telemetry" in result:
        out["telemetry"] = result["telemetry"]
    return out


def _read_suite_obj(
    suite: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(suite, (str, Path)):
        path = Path(suite).expanduser()
        return json.loads(path.read_text("utf-8")), path
    if isinstance(suite, Mapping):
        return dict(suite), None
    raise BenchError(
        f"suite must be a path or a mapping, got {type(suite).__name__!r}"
    )


_LIVE_EVIDENCE_CLASSES = ("live_lane", "live_stressed")


def _coding_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    passed = sum(1 for r in rows if r["verdict"] == "pass")
    voids = sum(1 for r in rows if r["verdict"] == "void")
    scored = n - voids  # void rows (no submission / infra failure) never ran
    scalars = [
        r["result"]["scalar"]
        for r in rows
        if r["result"].get("scalar") is not None
    ]
    evidence_classes = {r.get("evidence_class") for r in rows}

    def _group(key: str) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for r in rows:
            g = out.setdefault(str(r.get(key)), {"count": 0, "passed": 0})
            g["count"] += 1
            if r["verdict"] == "pass":
                g["passed"] += 1
        return out

    return {
        "count": n,
        "passed": passed,
        "void": voids,
        "scored": scored,
        # pass_rate is over SCORED tasks, not all tasks — so an infra failure
        # (e.g. no Docker daemon) that voids every row does NOT read as "0% passed".
        "pass_rate": round(passed / scored, 6) if scored else 0.0,
        "mean_score": round(sum(scalars) / len(scalars), 6) if scalars else 0.0,
        # derived from the actual rows (works for coding, pull/RL, any modality).
        "by_world_kind": _group("world_kind"),
        "by_execution_class": _group("execution_class"),
        "by_modality": _group("modality"),
        # honesty rollup (same 4-key shape as the push aggregate). Not gate-read;
        # row-level honesty is the enforcement primitive.
        "honesty": {
            "evidence_class": next(iter(evidence_classes)) if len(evidence_classes) == 1 else sorted(c for c in evidence_classes if c),
            "fixture_only": all(r.get("evidence_class") not in _LIVE_EVIDENCE_CLASSES for r in rows),
            "any_live": any(r.get("evidence_class") in _LIVE_EVIDENCE_CLASSES for r in rows),
            "any_overclaim": any(bool(r.get("overclaim")) for r in rows),
        },
    }


def _assemble(
    rows: list[dict[str, Any]],
    *,
    control_mode: str,
    name: str | None,
    version: str | None,
    emit_telemetry: bool,
    project_name: str | None,
) -> dict[str, Any]:
    aggregate = _coding_aggregate(rows)
    out: dict[str, Any] = {
        "kind": BENCH_RESULT_KIND,
        "control_mode": control_mode,
        "dataset_name": name,
        "dataset_version": version,
        "modalities": sorted({r["modality"] for r in rows}),
        "per_task": rows,
        "aggregate": aggregate,
    }
    if emit_telemetry:
        from ..telemetry import emit_run

        summary = emit_run(
            kind="bench",
            name=name or "bench",
            metrics={
                "n_tasks": aggregate["count"],
                "pass_rate": aggregate["pass_rate"],
                "mean_score": aggregate["mean_score"],
            },
            verdict="pass" if aggregate["pass_rate"] >= 0.5 else "fail",
            children=[
                (
                    f"task:{r['task_id']}",
                    {"verdict": r.get("verdict"), "score": r["result"].get("scalar")},
                )
                for r in rows
            ],
            project_name=project_name,
        )
        out["telemetry"] = summary.as_dict()
    return out


def _pull_row(
    task: Mapping[str, Any], verdict_obj: Mapping[str, Any], *, evidence_class: str
) -> dict[str, Any]:
    result = dict(verdict_obj["result"])
    raw = verdict_obj.get("raw") or {}
    if raw.get("infra_error"):  # unknown env / bad policy -> the lane never ran
        return {
            "task_id": str(task.get("id")), "modality": "rl", "world_kind": "env",
            "control_mode": "pull", "result": result, "verdict": "void",
            "execution_class": "executable", "evidence_class": evidence_class,
            "overclaim": False, "error": result.get("explanation"), "raw": raw,
        }
    pf = result.get("pass_fail") or {}
    verdict = "pass" if pf.get("goal_reached") else "fail"
    return {
        "task_id": str(task.get("id")),
        "modality": "rl",
        "world_kind": "env",
        "control_mode": "pull",
        "result": result,
        "verdict": verdict,
        "execution_class": "executable",
        "evidence_class": evidence_class,
        "overclaim": False,
        "raw": raw,
    }


def _voice_row(
    task: Mapping[str, Any], verdict_obj: Mapping[str, Any], *, evidence_class: str
) -> dict[str, Any]:
    result = dict(verdict_obj["result"])
    pf = result.get("pass_fail") or {}
    verdict = "pass" if pf.get("voice") else "fail"
    return {
        "task_id": str(task.get("id")),
        "modality": "voice",
        "world_kind": "voice_telephony",
        "control_mode": "artifact_in",
        "result": result,
        "verdict": verdict,
        "execution_class": "executable",
        "evidence_class": evidence_class,
        "overclaim": False,
        "raw": verdict_obj.get("raw", {}),
    }


def _voice_void_row(task: Mapping[str, Any], evidence_class: str) -> dict[str, Any]:
    return {
        "task_id": str(task.get("id")),
        "modality": "voice",
        "world_kind": "voice_telephony",
        "control_mode": "artifact_in",
        "result": {"scalar": None, "components": {}, "pass_fail": {},
                   "explanation": "no transcript submitted"},
        "verdict": "void",
        "execution_class": "executable",
        "evidence_class": evidence_class,
        "overclaim": False,
        "error": "no transcript submitted",
    }


def run_bench(
    suite: Mapping[str, Any] | str | Path,
    agent: Mapping[str, Any] | None = None,
    *,
    control_mode: str = "push",
    submission: Mapping[str, Any] | None = None,
    sandbox: str = "subprocess",
    split: str | None = None,
    max_tasks: int | None = None,
    seed: int = 42,
    evidence_class: str = "captured_fixture",
    detect_reward_hacks: bool = True,
    runner: Any = None,
    emit_telemetry: bool = True,
    project_name: str | None = None,
) -> dict[str, Any]:
    """Run a bench ``suite`` and return a unified ``agent-learning.bench-result.v1``.

    ``control_mode``:
      * ``push`` (default) — the harness drives ``agent`` through a world;
        delegates to :func:`tasks.run_benchmark` (live today for text / tool).
      * ``artifact_in`` — score a ``submission`` (``{task_id: candidate}``) against
        each task's held-out oracle, no live agent. Requires a coding bench suite
        (``agent-learning.bench-suite.v1``). ``sandbox`` selects the executor:
        ``subprocess`` (default) or ``docker`` (15E, hardened isolation).
      * ``pull`` — agent-driven reset/step loop over a live environment (15D).

    Per-task rows carry the unified ``result`` plus the preserved honesty fields
    (``execution_class`` / ``evidence_class`` / ``overclaim``).
    """

    if control_mode not in CONTROL_MODES:
        raise BenchError(
            f"unknown control_mode {control_mode!r}; expected one of {CONTROL_MODES}"
        )

    obj, path = _read_suite_obj(suite)

    from . import _coding

    if _coding.is_bench_suite(obj) and str(obj.get("control")) == "pull":
        # Pull / RL suite: the agent (a policy callable or {"type": reference|noop})
        # drives a simulated environment via reset/step.
        from . import _pull

        if control_mode != "pull":
            raise BenchError(
                f"pull bench suites run under control_mode='pull', not {control_mode!r}"
            )
        if agent is None:
            raise BenchError("pull mode requires an agent (a policy callable or spec)")
        if evidence_class not in tasks._evidence_classes():
            raise BenchError(f"unknown evidence_class {evidence_class!r}")
        task_list = list(obj.get("tasks") or [])
        if max_tasks is not None:
            task_list = task_list[: max(0, int(max_tasks))]
        rows = [
            _pull_row(t, _pull.run_pull(t, agent), evidence_class=evidence_class)
            for t in task_list
        ]
        return _assemble(
            rows, control_mode="pull", name=str(obj.get("name") or ""),
            version=str(obj.get("version") or ""), emit_telemetry=emit_telemetry,
            project_name=project_name,
        )

    if _coding.is_bench_suite(obj) and str(obj.get("control")) == "voice":
        # Voice suite: submit-and-score a voice episode transcript (artifact_in
        # semantics). submission = {task_id: dialogue}. Deterministic verifier.
        from . import _voice

        if control_mode != "artifact_in":
            raise BenchError(
                f"voice bench suites run under control_mode='artifact_in', not {control_mode!r}"
            )
        if submission is None:
            raise BenchError("voice artifact_in requires submission={task_id: dialogue}")
        if evidence_class not in tasks._evidence_classes():
            raise BenchError(f"unknown evidence_class {evidence_class!r}")
        task_list = list(obj.get("tasks") or [])
        if max_tasks is not None:
            task_list = task_list[: max(0, int(max_tasks))]
        rows = []
        for t in task_list:
            tid = str(t.get("id"))
            dialogue = submission.get(tid)
            if dialogue is None:
                rows.append(_voice_void_row(t, evidence_class))
                continue
            vo = _voice.score_voice_episode(
                dialogue, budgets=t.get("budgets"), required_content=t.get("required_content"),
            )
            rows.append(_voice_row(t, vo, evidence_class=evidence_class))
        return _assemble(
            rows, control_mode="artifact_in", name=str(obj.get("name") or ""),
            version=str(obj.get("version") or ""), emit_telemetry=emit_telemetry,
            project_name=project_name,
        )

    if _coding.is_bench_suite(obj):
        coding_suite = _coding.load_coding_suite(obj)
        if control_mode != "artifact_in":
            raise BenchError(
                f"coding bench suites currently run under control_mode='artifact_in', "
                f"not {control_mode!r}"
            )
        if submission is None:
            raise BenchError(
                "artifact_in requires submission={task_id: candidate_source}"
            )
        if sandbox not in SANDBOXES:
            raise BenchError(
                f"unknown sandbox {sandbox!r}; expected one of {SANDBOXES}"
            )
        if evidence_class not in tasks._evidence_classes():
            raise BenchError(
                f"unknown evidence_class {evidence_class!r}; expected one of "
                f"{tuple(tasks._evidence_classes())}"
            )
        rows = _coding.run_coding_artifact_in(
            coding_suite,
            submission,
            sandbox=sandbox,
            evidence_class=evidence_class,
            max_tasks=max_tasks,
        )
        return _assemble(
            rows,
            control_mode="artifact_in",
            name=str(coding_suite.get("name") or ""),
            version=str(coding_suite.get("version") or ""),
            emit_telemetry=emit_telemetry,
            project_name=project_name,
        )

    # --- task-dataset suites (text / tool worlds) ---
    if control_mode == "artifact_in":
        raise BenchError(
            "artifact_in currently requires a coding bench suite "
            "(agent-learning.bench-suite.v1)"
        )
    if control_mode == "pull":
        raise NotImplementedError(
            "control_mode='pull' (agent-driven reset/step over a live environment) "
            "lands in bench step 15D"
        )
    # control_mode == "push"
    if agent is None:
        raise BenchError("push mode requires an agent")
    compiled = tasks.load_task_dataset(path) if path is not None else tasks.compile_task_dataset(obj)
    result = tasks.run_benchmark(
        compiled,
        agent,
        split=split,
        max_tasks=max_tasks,
        seed=seed,
        evidence_class=evidence_class,
        detect_reward_hacks=detect_reward_hacks,
        runner=runner,
        emit_telemetry=emit_telemetry,
        project_name=project_name,
    )
    return _to_bench_result(result, control_mode="push")


def run_bench_file(
    suite_path: str | Path,
    agent: Mapping[str, Any] | None = None,
    *,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience: :func:`run_bench` from a suite file, optionally writing JSON.

    ``agent`` is optional, mirroring :func:`run_bench`: it is required for
    ``push`` but unused for ``artifact_in`` (which takes ``submission=`` via
    ``kwargs``).
    """

    payload = run_bench(Path(suite_path), agent, **kwargs)
    if output_path is not None:
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return payload
