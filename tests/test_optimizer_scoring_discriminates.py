"""Bug #2 fix: the flagship optimizer scores candidates on a DECLARED anchor
objective (real dynamic range) instead of the all-metrics-mean evaluation_score —
but ONLY when such an objective is declared (no synthesis), so structural/legacy
manifests are unchanged. Scorer lives in `fi` (engine), re-exported by `tasks`,
so the vendored_engine_boundary holds.
"""

from __future__ import annotations

from fi.alk import optimize, tasks


def _scores(o):
    if isinstance(o, dict):
        if "candidate_id" in o and "score" in o:
            yield o["score"]
        for v in o.values():
            yield from _scores(v)
    elif isinstance(o, list):
        for v in o:
            yield from _scores(v)


def test_declared_anchor_objective_optimization_discriminates() -> None:
    good = {"type": "scripted",
            "content": "Our refund policy is at /help/refunds; refunds within 30 days."}
    bad = {"type": "scripted", "content": "no"}
    m = optimize.build_task_optimization_manifest(
        name="bug2", agent_candidates=[bad, good],
        evaluation_config={"task_description": "Where is the refund policy and the window?",
                           "expected_result": "States policy location + 30-day window.",
                           "success_criteria": ["mentions refund policy", "states a time window"]},
        scenario={"name": "r", "dataset": [{"persona": {"name": "D"},
                  "situation": "Where is the refund policy and the window?",
                  "outcome": "States policy location + 30-day window."}]},
        threshold=0.5, max_turns=1)
    m["objective"] = {"source": "declared", "evals": [
        {"eval": "task_success", "weight": 1.0, "anchor": True},
        {"eval": "goal_progress", "weight": 0.6, "anchor": True}],
        "guards": {"sentinel_rows": [{"id": "s"}], "min_guard_count": 1}}
    res = optimize.optimize_manifest(m)
    distinct = sorted(set(_scores(res)))
    assert len(distinct) >= 2, distinct
    # real dynamic range (was ~0.027 on the all-metrics mean); good beats bad.
    assert max(distinct) - min(distinct) > 0.20, distinct


# --- the scoped guard: no declared objective -> engine score (no regression) --
def test_objective_anchored_score_requires_declared_anchor_objective() -> None:
    from fi.opt.integrations.simulate import _objective_anchored_score

    # has metric_averages but NO declared anchor objective -> None (falls through)
    assert _objective_anchored_score({"summary": {"metric_averages": {"task_completion": 0.2}}}) is None
    assert _objective_anchored_score({"metric_averages": {"task_completion": 0.2},
                                      "objective": {"evals": [{"eval": "x"}]}}) is None  # no anchor
    # declared anchor objective + metrics -> objective-anchored score
    s = _objective_anchored_score({"metric_averages": {"task_completion": 0.2},
                                   "objective": {"evals": [{"eval": "task_success", "anchor": True}]}})
    assert s == 0.2


# --- vendored-engine boundary: scorer is in fi, tasks re-exports it -----------
def test_scorer_lives_in_fi_and_tasks_reexports() -> None:
    from fi.opt import _objective_scoring

    assert tasks.objective_score is _objective_scoring.objective_score
    assert tasks.resolve_metric is _objective_scoring.resolve_metric
    # the fi scorer module must not IMPORT fi.alk (boundary)
    import inspect
    for line in inspect.getsource(_objective_scoring).splitlines():
        stripped = line.strip()
        assert not stripped.startswith(("import fi.alk", "from fi.alk")), stripped
