"""First-class Task + TaskDataset — the benchmark surface.

A Task is a THIN COMPOSITION over models that already exist and are content-
addressed (no new invention — empirically validated by spike):

  * the existing typed ``Scenario(kind="task")`` (one effective task row) carries
    goal + verification (fi.simulate.simulation.models);
  * the objective is compiled by ``loss.compile_objective`` VERBATIM, so the
    Goodhart-guard discipline ("There is no override.") holds for a Task exactly
    as for any training loss — a guardless objective is REJECTED here too;
  * the world kind must be a member of ``contract.resolved_world_kinds()`` (the
    frozen closed set + R4 extras) — never widened here;
  * ``execution_class`` is DERIVED from the world kind (+ a fixture flag), never
    asserted above the substrate's truth: ``browser``/``computer_use``/
    ``code_exec``/``voice_telephony`` are TYPED-ONLY in v1 and can be at most
    ``typed_only``; only ``conversation``/``tool_api`` may be ``executable``.
    This is the kit's honesty moat: a typed-only task can NEVER masquerade
    as a live-executed one.

Tasks and datasets are content-addressed (sha256 over the canonical payload minus
``version``), using the same rounding/canonicalization idiom as ``loss.py``. NO
provenance/capture fields are baked in yet (the real-execution fork is deferred;
keeping them out means a later fork decision costs nothing here).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .loss import ObjectiveError, compile_objective

AGENT_LEARNING_TASK_KIND = "agent-learning.task.v1"
AGENT_LEARNING_TASK_DATASET_KIND = "agent-learning.task-dataset.v1"

# closed sets (home here; trinity.py mirrors literals for the gate — GUNA_AXES
# cross-pin pattern; trinity never imports this module).
V1_TASK_DIFFICULTIES = ("easy", "medium", "hard")
V1_TASK_EXECUTION_CLASSES = ("executable", "typed_only", "fixture")


class TaskError(ValueError):
    """Raised when a Task violates the §B1 contract. A ``ValueError`` subclass so
    callers can ``except ValueError`` exactly as for ``ObjectiveError``."""


class TaskDatasetError(TaskError):
    """Raised when a TaskDataset violates the contract."""


# --- canonicalization (the loss.py idiom, factored locally) -----------------
def _round_floats(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, Mapping):
        return {k: _round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(v) for v in value]
    return value


def _content_hash(payload: Mapping[str, Any]) -> str:
    rounded = _round_floats(dict(payload))
    canonical = json.dumps(rounded, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolved_world_kinds() -> tuple[str, ...]:
    """Lazy downward import (the image_loop.py idiom) — the closed set + R4
    extras, never widened here."""
    from fi.simulate.simulation import contract as _contract

    return tuple(_contract.resolved_world_kinds())


def _executable_world_kinds() -> tuple[str, ...]:
    from fi.simulate.simulation import contract as _contract

    return tuple(_contract.EXECUTABLE_WORLD_KINDS_V1)


def derive_execution_class(world_kind: str, *, fixture_only: bool = False) -> str:
    """Derive the HONEST substrate stamp. ``fixture_only`` (a task that only ever
    replays a committed fixture) → ``fixture``; a world kind that EXECUTES in v1
    (``conversation``/``tool_api``) → ``executable``; everything else (the
    typed-only kinds) → ``typed_only``. NEVER returns a class above the
    substrate's real capability."""

    if fixture_only:
        return "fixture"
    if world_kind in _executable_world_kinds():
        return "executable"
    return "typed_only"


def _task_anchor_terms(objective: Mapping[str, Any]) -> list[str]:
    """Terms explicitly flagged as deterministic ground-truth anchors
    (``anchor: true``). The reward-hacking-resistance-by-construction rule below
    requires >= 1; an objective scored only by un-anchored (e.g. judge) terms is
    a Task contract violation — the Task analogue of the image judge-only ban."""

    return [
        str(term.get("eval"))
        for term in (objective.get("evals") or [])
        if isinstance(term, Mapping) and term.get("anchor") is True and term.get("eval")
    ]


def compile_task(payload: Mapping[str, Any]) -> dict:
    """Validate + stamp a Task. Enforces, ON TOP of the verbatim
    ``loss.compile_objective`` Goodhart guard:

      (a) ``scenario.kind == "task"`` with exactly ONE effective task row;
      (b) ``world.kind`` is a member of ``resolved_world_kinds()``;
      (c) ``difficulty`` in ``V1_TASK_DIFFICULTIES``;
      (d) the objective carries >= 1 deterministic ground-truth ANCHOR term
          (``anchor: true``) — reward-hacking-resistance by construction;
      (e) ``execution_class`` is DERIVED, never asserted above the substrate
          (a caller-supplied class that overclaims the world kind is rejected).
    Then delegates objective compilation to ``loss.compile_objective`` VERBATIM."""

    raw = dict(payload)

    task_id = str(raw.get("id") or "").strip()
    if not task_id:
        raise TaskError("task.id is required")
    title = str(raw.get("title") or "").strip()
    if not title:
        raise TaskError("task.title is required")

    # (b) world.kind
    world = dict(raw.get("world") or {})
    world_kind = str(world.get("kind") or "")
    resolved = _resolved_world_kinds()
    if world_kind not in resolved:
        raise TaskError(
            f"task.world.kind {world_kind!r} not in resolved world kinds {resolved}"
        )

    # (c) difficulty
    difficulty = str(raw.get("difficulty") or "medium")
    if difficulty not in V1_TASK_DIFFICULTIES:
        raise TaskError(f"task.difficulty {difficulty!r} not in {V1_TASK_DIFFICULTIES}")

    # (a) scenario.kind == "task", exactly one effective row
    scenario = dict(raw.get("scenario") or {})
    if str(scenario.get("kind") or "") != "task":
        raise TaskError("task.scenario.kind must be 'task'")
    dataset_rows = list(scenario.get("dataset") or [])
    if len(dataset_rows) != 1:
        raise TaskError(
            f"task.scenario must carry exactly one task row; got {len(dataset_rows)}"
        )

    # objective: compile VERBATIM (guards enforced; guardless REJECTED here too)
    objective_payload = raw.get("objective")
    if not isinstance(objective_payload, Mapping):
        raise TaskError("task.objective is required (a declared ObjectiveSpec)")
    try:
        compiled_objective = compile_objective(objective_payload)
    except ObjectiveError as exc:
        # surface as a Task error while preserving the guard message
        raise TaskError(f"task.objective invalid: {exc}") from exc

    # (d) >= 1 deterministic anchor term
    anchors = _task_anchor_terms(objective_payload)
    if not anchors:
        raise TaskError(
            "task.objective must carry >= 1 deterministic ground-truth anchor "
            "term (mark it `anchor: true`); a task scored only by un-anchored "
            "(e.g. judge) terms is reward-hackable by construction"
        )

    # (e) execution_class derived; reject overclaim
    fixture_only = bool(raw.get("fixture_only"))
    derived_class = derive_execution_class(world_kind, fixture_only=fixture_only)
    asserted = raw.get("execution_class")
    if asserted is not None:
        asserted = str(asserted)
        if asserted not in V1_TASK_EXECUTION_CLASSES:
            raise TaskError(
                f"task.execution_class {asserted!r} not in {V1_TASK_EXECUTION_CLASSES}"
            )
        rank = {"fixture": 0, "typed_only": 1, "executable": 2}
        if rank[asserted] > rank[derived_class]:
            raise TaskError(
                f"task.execution_class {asserted!r} overclaims the substrate; "
                f"world.kind {world_kind!r} (fixture_only={fixture_only}) supports "
                f"at most {derived_class!r}"
            )

    compiled: dict[str, Any] = {
        "kind": AGENT_LEARNING_TASK_KIND,
        "id": task_id,
        "title": title,
        "world": {"kind": world_kind, **({"spec": dict(world["spec"])} if world.get("spec") else {})},
        "difficulty": difficulty,
        "tags": [str(t) for t in (raw.get("tags") or [])],
        "scenario": scenario,
        "objective": compiled_objective,
        "anchor_terms": anchors,
        "execution_class": derived_class,
    }
    if raw.get("goal") is not None:
        compiled["goal"] = dict(raw["goal"])
    if raw.get("verification") is not None:
        compiled["verification"] = dict(raw["verification"])
    # environments: mock-tool / world env specs the runner wires into the manifest
    # so a TOOL-USING task (mock_tools + a tool-calling agent) runs through the
    # benchmark/RSI loop, not just direct simulate.
    if raw.get("environments"):
        envs = raw["environments"]
        if not isinstance(envs, Sequence) or isinstance(envs, (str, bytes)):
            raise TaskError("task.environments must be a list of env specs")
        compiled["environments"] = [dict(e) for e in envs if isinstance(e, Mapping)]
    compiled["version"] = _content_hash(
        {k: v for k, v in compiled.items() if k != "version"}
    )
    return compiled


def compile_task_dataset(payload: Mapping[str, Any]) -> dict:
    """Validate + stamp a TaskDataset: every task compiles via ``compile_task``;
    ids are unique; any ``splits`` reference existing ids; coverage by world.kind
    and difficulty is computed. Content-addressed over the compiled tasks."""

    raw = dict(payload)
    name = str(raw.get("name") or "").strip()
    if not name:
        raise TaskDatasetError("dataset.name is required")

    task_payloads = list(raw.get("tasks") or [])
    if not task_payloads:
        raise TaskDatasetError("dataset.tasks must list >= 1 task")

    compiled_tasks: list[dict] = []
    seen_ids: set[str] = set()
    by_world_kind: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for index, task_payload in enumerate(task_payloads, start=1):
        if not isinstance(task_payload, Mapping):
            raise TaskDatasetError(f"dataset.tasks[{index}] must be a mapping")
        try:
            task = compile_task(task_payload)
        except TaskError as exc:
            raise TaskDatasetError(f"dataset.tasks[{index}] invalid: {exc}") from exc
        if task["id"] in seen_ids:
            raise TaskDatasetError(f"duplicate task id {task['id']!r}")
        seen_ids.add(task["id"])
        compiled_tasks.append(task)
        by_world_kind[task["world"]["kind"]] = by_world_kind.get(task["world"]["kind"], 0) + 1
        by_difficulty[task["difficulty"]] = by_difficulty.get(task["difficulty"], 0) + 1

    splits = raw.get("splits")
    normalized_splits: dict[str, list[str]] = {}
    if splits is not None:
        if not isinstance(splits, Mapping):
            raise TaskDatasetError("dataset.splits must be a mapping of name -> [id]")
        for split_name, ids in splits.items():
            id_list = [str(i) for i in (ids or [])]
            missing = [i for i in id_list if i not in seen_ids]
            if missing:
                raise TaskDatasetError(
                    f"dataset.splits[{split_name!r}] references unknown ids {missing}"
                )
            normalized_splits[str(split_name)] = id_list

    compiled: dict[str, Any] = {
        "kind": AGENT_LEARNING_TASK_DATASET_KIND,
        "name": name,
        "tasks": compiled_tasks,
        "coverage": {
            "by_world_kind": dict(sorted(by_world_kind.items())),
            "by_difficulty": dict(sorted(by_difficulty.items())),
            "count": len(compiled_tasks),
        },
    }
    if normalized_splits:
        compiled["splits"] = normalized_splits
    if raw.get("license") is not None:
        compiled["license"] = str(raw["license"])
    compiled["version"] = _content_hash(
        {k: v for k, v in compiled.items() if k != "version"}
    )
    return compiled


def task_world_kinds(dataset: Mapping[str, Any]) -> Sequence[str]:
    """Convenience: the sorted set of world kinds a (compiled) dataset spans."""

    return sorted({task["world"]["kind"] for task in (dataset.get("tasks") or [])})


def load_task_dataset(path: str | Path) -> dict[str, Any]:
    """Load + compile a TaskDataset from a JSON file (the shipped out-of-the-box
    datasets live under ``examples/task_datasets/``)."""

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return compile_task_dataset(payload)


# ===========================================================================
# B2 — the benchmark runner (run an agent across a dataset, honest scoring).
# ===========================================================================
AGENT_LEARNING_BENCHMARK_RESULT_KIND = "agent-learning.benchmark-result.v1"

# the closed evidence-class set is OWNED by live/_contract (frozen 4-tuple); the
# runner never invents one. Non-live execution_classes may NEVER carry a live
# evidence_class — that is the honesty rule the gate's overclaim tripwire checks.
_NON_LIVE_EXECUTION_CLASSES = ("fixture", "typed_only")
_LIVE_EVIDENCE_CLASSES = ("live_lane", "live_stressed")


def _evidence_classes() -> tuple[str, ...]:
    from .live._contract import EVIDENCE_CLASSES

    return tuple(EVIDENCE_CLASSES)


def _resolve_task_list(
    dataset: Mapping[str, Any],
    *,
    split: str | None,
    max_tasks: int | None,
) -> list[dict]:
    tasks_all = list(dataset.get("tasks") or [])
    if split is not None:
        ids = set((dataset.get("splits") or {}).get(split) or [])
        if not ids:
            raise TaskDatasetError(f"benchmark split {split!r} is empty or undefined")
        tasks_all = [t for t in tasks_all if t.get("id") in ids]
    tasks_all = sorted(tasks_all, key=lambda t: str(t.get("id")))  # deterministic order
    if max_tasks is not None:
        tasks_all = tasks_all[: int(max_tasks)]
    return tasks_all


# Canonical objective-anchored scoring lives in ``fi`` (the engine) so the
# optimizer integration can use it WITHOUT importing fi.alk (the
# vendored_engine_boundary). Re-exported here so B1/B2/B6 call sites are unchanged.
from fi.opt._objective_scoring import (  # noqa: E402,F401
    METRIC_ALIASES,
    has_declared_anchor_objective,
    objective_score,
    resolve_metric,
)


def _score_from_result(
    result: Mapping[str, Any],
    *,
    objective: Mapping[str, Any] | None = None,
    threshold: float = 0.5,
) -> tuple[str, float, dict, dict]:
    """Extract (verdict, score, metric_averages, scoring) from a run result.

    The headline ``score`` is the OBJECTIVE score (weighted mean over the task's
    declared terms) when the objective resolves >=1 term — the signal with real
    dynamic range. It falls back to the engine ``evaluation_score`` only when no
    declared term maps to a metric. ``verdict`` is pass iff score >= threshold."""

    summary = result.get("summary") or {}
    metric_averages = dict(summary.get("metric_averages") or {})
    raw_eval = float(summary.get("evaluation_score") or 0.0)

    obj = objective_score(metric_averages, objective or {})
    if obj["score"] is not None:
        score = float(obj["score"])
        basis = "objective"
    else:
        score = raw_eval
        basis = "evaluation_score_fallback"
    verdict = "pass" if score >= threshold else "fail"
    scoring = {
        "basis": basis,
        "raw_evaluation_score": round(raw_eval, 6),
        "threshold": threshold,
        "terms_resolved": obj["terms_resolved"],
        "terms_total": obj["terms_total"],
        "per_term": obj["per_term"],
    }
    return verdict, round(score, 6), metric_averages, scoring


def run_benchmark(
    dataset: Mapping[str, Any],
    agent: Mapping[str, Any],
    *,
    split: str | None = None,
    max_tasks: int | None = None,
    seed: int = 42,
    evidence_class: str = "captured_fixture",
    detect_reward_hacks: bool = False,
    runner: Any = None,
    emit_telemetry: bool = True,
    project_name: str | None = None,
) -> dict[str, Any]:
    """Run one ``agent`` across a (compiled) TaskDataset and return a scored,
    comparable result. Each per-task result is stamped with the task's HONEST
    ``execution_class`` and the run's ``evidence_class`` — and a non-live
    execution_class carrying a live evidence_class is flagged ``overclaim=True``
    (never silently downgraded; the gate asserts none in the fixture lane).

    The actual run goes through the EXISTING engine (``simulate.build_task_run_
    manifest`` -> ``simulate.run_manifest``), proven feasible on conversation/
    tool_api by the execution spike. ``runner`` is an injectable seam (a
    callable ``manifest -> result``) for deterministic testing without the
    engine; when ``None`` the real engine is used.
    """

    if evidence_class not in _evidence_classes():
        raise TaskError(
            f"evidence_class {evidence_class!r} not in {_evidence_classes()}"
        )

    task_list = _resolve_task_list(dataset, split=split, max_tasks=max_tasks)
    if not task_list:
        raise TaskDatasetError("benchmark resolved zero tasks to run")

    per_task: list[dict[str, Any]] = []
    for task in task_list:
        verdict, score, metric_averages, scoring, tool_calls, run_error = _run_one_task(
            task, agent, seed=seed, runner=runner
        )
        execution_class = str(task.get("execution_class") or "typed_only")
        overclaim = (
            execution_class in _NON_LIVE_EXECUTION_CLASSES
            and evidence_class in _LIVE_EVIDENCE_CLASSES
        )
        row: dict[str, Any] = {
            "task_id": str(task.get("id")),
            "world_kind": str(task.get("world", {}).get("kind")),
            "difficulty": str(task.get("difficulty")),
            "verdict": verdict,
            "score": score,
            "execution_class": execution_class,
            "evidence_class": evidence_class,
            "overclaim": bool(overclaim),
            "metric_averages": metric_averages,
            "tool_calls": list(tool_calls or []),
            "scoring": scoring,
        }
        # #3: auto-wire the reward-hack detector — a candidate that GAMES the
        # scorer (e.g. claims completion with zero tool calls on a tool-anchored
        # objective) is FAILED even if objective_score passed (the deterministic
        # anti-gaming anchor objective_score alone misses, e.g. vacuous
        # tool_selection_accuracy=1.0 for an agent that never called a tool).
        if detect_reward_hacks and verdict == "pass":
            from . import rewardhack as _rh

            traj = {"metric_averages": metric_averages, "tool_calls": list(tool_calls or []),
                    "score": score}
            verdict_obj = _rh.score_trajectory(traj, objective=task.get("objective") or {})
            if verdict_obj["hacked"]:
                row["verdict"] = "fail"
                row["rewardhack"] = verdict_obj
                row["score"] = 0.0
        if run_error is not None:
            row["error"] = run_error
        per_task.append(row)

    aggregate = _aggregate(per_task, evidence_class)
    result = {
        "kind": AGENT_LEARNING_BENCHMARK_RESULT_KIND,
        "dataset_version": str(dataset.get("version") or ""),
        "dataset_name": str(dataset.get("name") or ""),
        "split": split,
        "seed": int(seed),
        "agent_kind": str(agent.get("type") or ""),
        "per_task": per_task,
        "aggregate": aggregate,
    }
    if emit_telemetry:
        # W&B/promptfoo dashboard wiring (Phase 14): one run -> root + per-task
        # spans + a dashboard URL when keyed (mode=auto), else a local log. A
        # side-channel — never alters the returned scores.
        from .telemetry import emit_run

        summary = emit_run(
            kind="benchmark",
            name=result["dataset_name"] or "benchmark",
            metrics={
                "n_tasks": aggregate.get("count", 0),
                "pass_rate": aggregate.get("pass_rate", 0.0),
                "mean_score": aggregate.get("mean_score", 0.0),
            },
            verdict="pass" if aggregate.get("pass_rate", 0.0) >= 0.5 else "fail",
            children=[
                (
                    f"task:{r['task_id']}",
                    {
                        "verdict": r.get("verdict"),
                        "score": r.get("score"),
                        "world_kind": r.get("world_kind"),
                    },
                )
                for r in per_task
            ],
            project_name=project_name,
        )
        result["telemetry"] = summary.as_dict()
    return result


def _run_one_task(
    task: Mapping[str, Any],
    agent: Mapping[str, Any],
    *,
    seed: int,
    runner: Any,
) -> tuple[str, float, dict, dict, str | None]:
    scenario = dict(task.get("scenario") or {})
    verification = dict(task.get("verification") or {})
    threshold = float(verification.get("threshold", 0.7))
    objective = dict(task.get("objective") or {})
    # translate the task's verification + scenario row into a REAL evaluation
    # config so the run is scored (not a hollow status='passed' with score 0.0).
    row = (scenario.get("dataset") or [{}])[0] if scenario.get("dataset") else {}
    task_description = str(row.get("situation") or task.get("title") or task.get("id"))
    expected_result = str(row.get("outcome") or "")
    success_criteria = [
        str(c.get("value"))
        for c in (verification.get("checks") or [])
        if isinstance(c, Mapping) and c.get("value") is not None
    ]
    try:
        if runner is not None:
            result = runner(task, agent)
        else:
            import asyncio

            from . import simulate as _simulate

            manifest = _simulate.build_task_run_manifest(
                name=str(task.get("id")),
                agent=dict(agent),
                scenario=scenario,
                task_description=task_description,
                expected_result=expected_result or None,
                success_criteria=success_criteria,
                threshold=threshold,
                environments=list(task.get("environments") or ()),
                auto_execute_tools=True,
                max_turns=int((task.get("world", {}).get("spec") or {}).get("max_turns", 1)),
                simulation_engine=str(
                    (task.get("world", {}).get("spec") or {}).get("engine")
                    or "local_text"
                ),
            )
            # run_manifest is async; the benchmark runner is a synchronous
            # top-level API, so drive the coroutine to completion here.
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                result = asyncio.run(_simulate.run_manifest(manifest))
            else:  # pragma: no cover - benchmark is not called inside a loop
                raise RuntimeError(
                    "run_benchmark cannot run inside an active event loop; "
                    "call it from synchronous code"
                )
        verdict, score, metric_averages, scoring = _score_from_result(
            result, objective=objective, threshold=threshold
        )
        return verdict, score, metric_averages, scoring, _result_tool_calls(result), None
    except Exception as exc:  # noqa: BLE001 — a failed task scores void, never crashes the sweep
        return "void", 0.0, {}, {}, [], f"{type(exc).__name__}: {str(exc)[:160]}"


def _result_tool_calls(result: Mapping[str, Any]) -> list:
    """Extract the agent's tool_calls from a run result (needed by the reward-hack
    detector to measure effort). Shape-tolerant across the runner seam."""
    if not isinstance(result, Mapping):
        return []
    direct = result.get("tool_calls")
    if isinstance(direct, list):
        return direct
    results = (result.get("report") or {}).get("results") if isinstance(result.get("report"), Mapping) else None
    if isinstance(results, list) and results and isinstance(results[0], Mapping):
        return list(results[0].get("tool_calls") or [])
    return []


def _aggregate(per_task: Sequence[Mapping[str, Any]], evidence_class: str) -> dict:
    n = len(per_task)
    passed = sum(1 for r in per_task if r["verdict"] == "pass")
    mean_score = round(sum(float(r["score"]) for r in per_task) / n, 6) if n else 0.0

    def _rollup(key: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in per_task:
            bucket = out.setdefault(str(r[key]), {"count": 0, "passed": 0, "score_sum": 0.0})
            bucket["count"] += 1
            bucket["passed"] += 1 if r["verdict"] == "pass" else 0
            bucket["score_sum"] += float(r["score"])
        for bucket in out.values():
            bucket["mean_score"] = round(bucket.pop("score_sum") / bucket["count"], 6)
        return dict(sorted(out.items()))

    any_live = any(r["evidence_class"] in _LIVE_EVIDENCE_CLASSES for r in per_task)
    any_overclaim = any(r.get("overclaim") for r in per_task)
    return {
        "count": n,
        "passed": passed,
        "pass_rate": round(passed / n, 6) if n else 0.0,
        "mean_score": mean_score,
        "by_world_kind": _rollup("world_kind"),
        "by_difficulty": _rollup("difficulty"),
        "by_execution_class": _rollup("execution_class"),
        "honesty": {
            "evidence_class": evidence_class,
            "fixture_only": not any_live,
            "any_live": any_live,
            "any_overclaim": any_overclaim,
        },
    }


# ===========================================================================
# RSI loop — close the loop: dataset -> optimize -> verify on HELD-OUT.
# ===========================================================================
AGENT_LEARNING_RSI_REPORT_KIND = "agent-learning.rsi-report.v1"


def _apply_candidate(base_agent: Mapping[str, Any], assignment: Mapping[str, Any]) -> dict:
    """Materialize one agent config from the base + a search-space assignment.
    Keys may be dotted ``agent.<field>`` (the kit's whole-agent convention) or a
    bare field; both write onto the agent dict."""

    agent = dict(base_agent)
    for path, value in assignment.items():
        field = str(path).split(".", 1)[1] if str(path).startswith("agent.") else str(path)
        agent[field] = value
    return agent


def _candidate_grid(search_space: Mapping[str, Sequence[Any]], *, cap: int) -> list[dict]:
    """Cartesian product of the (finite) search space, deterministically ordered,
    capped at ``cap`` candidates (logged when truncated)."""

    import itertools

    keys = sorted(search_space)
    value_lists = [list(search_space[k]) for k in keys]
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]
    return grid[: max(1, int(cap))]


def optimize_against_dataset(
    dataset: Mapping[str, Any],
    base_agent: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any]],
    *,
    train_split: str = "train",
    test_split: str = "test",
    max_candidates: int = 16,
    seed: int = 42,
    evidence_class: str = "captured_fixture",
    runner: Any = None,
    emit_telemetry: bool = True,
    project_name: str | None = None,
    detect_reward_hacks: bool = False,
) -> dict[str, Any]:
    """Close the RSI loop: search agent configs, score each on the TRAIN split via
    the (objective-anchored, discriminating) ``run_benchmark``, pick the winner,
    then VERIFY it beats the baseline on the HELD-OUT TEST split.

    The held-out check is the honest RSI guard (the advisor's bar): report lift on
    tasks the optimizer never optimized against, never the metric it climbed. If
    the dataset has no ``splits``, train==test is used and ``held_out`` is False
    (a non-generalization-proving run, flagged as such — no overclaim)."""

    splits = dataset.get("splits") or {}
    has_split = bool(splits.get(train_split)) and bool(splits.get(test_split))
    train = train_split if has_split else None
    test = test_split if has_split else None

    def _score(agent: Mapping[str, Any], split: str | None) -> dict:
        # emit_telemetry=False: the optimizer emits ONE run for the whole search
        # (below), not one per candidate benchmark (Phase 14).
        # detect_reward_hacks: when on, a candidate that GAMES a declared anchor
        # (e.g. claims completion with zero tool calls on a tool-anchored
        # objective) is FAILED — an optimizer that climbs a gameable metric is the
        # textbook reward-hacking failure, so honest config search opts in.
        return run_benchmark(
            dataset, agent, split=split, seed=seed,
            evidence_class=evidence_class, runner=runner, emit_telemetry=False,
            detect_reward_hacks=detect_reward_hacks,
        )["aggregate"]

    candidates = _candidate_grid(search_space, cap=max_candidates)
    truncated = len(_candidate_grid(search_space, cap=10**9)) > len(candidates)

    leaderboard: list[dict[str, Any]] = []
    for idx, assignment in enumerate(candidates):
        agent = _apply_candidate(base_agent, assignment)
        train_agg = _score(agent, train)
        leaderboard.append({
            "candidate_index": idx,
            "assignment": assignment,
            "train_mean_score": train_agg["mean_score"],
            "train_pass_rate": train_agg["pass_rate"],
        })
    leaderboard.sort(key=lambda r: (-r["train_mean_score"], r["candidate_index"]))
    winner = leaderboard[0]
    winner_agent = _apply_candidate(base_agent, winner["assignment"])

    # held-out verification: winner vs baseline (the base_agent) on TEST
    baseline_test = _score(base_agent, test)
    winner_test = _score(winner_agent, test)
    lift = round(winner_test["mean_score"] - baseline_test["mean_score"], 6)

    result = {
        "kind": AGENT_LEARNING_RSI_REPORT_KIND,
        "dataset_version": str(dataset.get("version") or ""),
        "candidates_evaluated": len(candidates),
        "candidates_truncated": truncated,
        "leaderboard": leaderboard,
        "winner": {"assignment": winner["assignment"],
                   "train_mean_score": winner["train_mean_score"]},
        "held_out": {
            "verified": bool(has_split),                    # a real held-out split existed
            "test_split": test_split if has_split else "(train==test; not held-out)",
            "baseline_mean_score": baseline_test["mean_score"],
            "winner_mean_score": winner_test["mean_score"],
            "lift": lift,
            "improved": lift > 0,
        },
    }
    if emit_telemetry:
        # ONE dashboard run for the whole search: root + per-candidate spans (P14).
        from .telemetry import emit_run

        summary = emit_run(
            kind="optimize",
            name=str(dataset.get("name") or "optimize"),
            metrics={
                "candidates": len(candidates),
                "winner_train_score": winner["train_mean_score"],
                "held_out_lift": lift,
                "improved": lift > 0,
            },
            verdict="pass" if lift > 0 else "fail",
            children=[
                (
                    f"candidate:{row['candidate_index']}",
                    {"train_mean_score": row["train_mean_score"],
                     "train_pass_rate": row["train_pass_rate"]},
                )
                for row in leaderboard
            ],
            project_name=project_name,
        )
        result["telemetry"] = summary.as_dict()
    return result


# ===========================================================================
# #2 — real cell_scorer for the 13D practice loop (replaces the all-pass no-op).
# ===========================================================================
def make_cell_scorer(
    *,
    agent: Mapping[str, Any],
    objective: Mapping[str, Any],
    scenario: Mapping[str, Any] | None = None,
    environments: Sequence[Mapping[str, Any]] = (),
    threshold: float = 0.5,
    evidence_class: str = "captured_fixture",
    runner: Any = None,
) -> Any:
    """Build a REAL ``cell_scorer`` for ``practice.run_practice_loop`` that runs the
    agent per cell and scores via ``objective_score`` (engine ``metric_averages`` →
    the ``{scalar, verdict, evidence_class}`` shape the loop consumes).

    Replaces the loop's all-pass no-op default (``scalar 1.0, verdict pass``) so the
    13D practice loop — assess/diagnose/spaced-regression-replay/consolidate —
    measures REAL fitness instead of accepting everything (audit gap #2). Joins
    diagnose-from-trace + no-forgetting (13D) with the discriminating objective
    score. ``runner`` is an injectable seam for deterministic tests."""

    def cell_scorer(cell: Mapping[str, Any]) -> dict[str, Any]:
        sc = dict(scenario or {})
        # the cell's persona overrides the scenario's row when present, so distinct
        # obligation cells exercise distinct conditions (not one fixed run).
        if cell.get("persona") and sc.get("dataset"):
            row0 = dict((sc["dataset"] or [{}])[0])
            row0["persona"] = {"name": str(cell["persona"])[:24]}
            sc = {**sc, "dataset": [row0]}
        if runner is not None:
            result = runner(cell, agent)
        else:
            import asyncio

            from . import simulate as _sim

            row = (sc.get("dataset") or [{}])[0] if sc.get("dataset") else {}
            manifest = _sim.build_task_run_manifest(
                name=str(cell.get("intent") or "cell"),
                agent=dict(agent),
                scenario=sc or None,
                task_description=str(row.get("situation") or cell.get("intent") or "task"),
                expected_result=str(row.get("outcome") or "") or None,
                environments=list(environments),
                auto_execute_tools=True,
                threshold=threshold,
            )
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                result = asyncio.run(_sim.run_manifest(manifest))
            else:  # pragma: no cover
                raise RuntimeError("make_cell_scorer cannot run inside an event loop")

        metrics = (result.get("summary") or {}).get("metric_averages") or {}
        s = objective_score(metrics, objective).get("score")
        scalar = float(s) if s is not None else 0.0
        return {
            "eval": "agent_report",
            "scalar": round(scalar, 6),
            "verdict": "pass" if scalar >= threshold else "fail",
            "evidence_class": evidence_class,
            "metric_averages": metrics,
        }

    return cell_scorer
