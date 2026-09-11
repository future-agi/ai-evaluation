"""Code-level RSI — fix a framework agent's actual CODE by run→trace→diagnose→
patch→re-run→keep-if-better-on-held-out.

The general self-improvement model: the `update` ACTION is a CODE EDIT (not a
config patch). The loop runs the framework agent's real source in sim, reads the
trace + the (discriminating, objective-anchored) eval, asks a proposer to PATCH
the source to fix the failure, applies the patch in an ISOLATED workdir, re-runs
on HELD-OUT tasks, and accepts ONLY if held-out improves AND a regression split
does not drop (no-forgetting). Reuses the proven run→eval foundation
(run_benchmark + objective_score); the genuinely new surface is exactly two
things — propose-a-code-edit and sandboxed re-run — both isolated here.

SAFETY (the code_exec verdict-#4 line holds): a code patch is LLM-AUTHORED code
run automatically in a loop. The patched source is written to an isolated temp
workdir and the agent is pointed at it by ABSOLUTE path — a bad mutation NEVER
touches the caller's real file unless the caller explicitly writes back an
accepted patch. Each candidate run is wall-clock bounded. This is for optimizing
a TRUSTED agent's own source; arbitrary untrusted code execution remains the
parked code_exec concern (needs a real sandbox).

OVERCLAIM GUARD: "improved" = held-out bug-class tasks pass on the DETERMINISTIC
anchor (e.g. tool_calls / completion_without_effort) the loop never optimized
directly, AND the regression split is not worse. A code-RSI loop is the most
gameable thing in the kit — never accept on the metric it edited toward.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .tasks import run_benchmark

AGENT_LEARNING_CODE_RSI_REPORT_KIND = "agent-learning.code-rsi-report.v1"

# a proposer maps a diagnosis -> new full source text (or None to give up).
PatchProposer = Callable[[Mapping[str, Any]], "str | None"]


def _agent_for_source(source_path: Path, symbol: str) -> dict[str, Any]:
    """A python-callable agent pointed at a source file by ABSOLUTE path (so the
    patched copy loads, never the caller's original)."""
    return {"type": "python", "callable": f"{source_path.resolve()}:{symbol}"}


def _score_split(
    source_path: Path,
    symbol: str,
    dataset: Mapping[str, Any],
    split: str | None,
    *,
    seed: int,
    runner: Any = None,
) -> dict[str, Any]:
    agent = _agent_for_source(source_path, symbol)
    # detector-aware: a candidate that GAMES the scorer (claims completion with no
    # tool calls on a tool-anchored objective) is FAILED — the deterministic
    # anti-gaming anchor the objective alone misses. This is what makes the
    # no-tool bug detectable and prevents the loop accepting a reward-hacking patch.
    # emit_telemetry=False: the RSI loop emits ONE run (in improve_agent_code),
    # not one per split-score across rounds (Phase 14).
    res = run_benchmark(dataset, agent, split=split, seed=seed,
                        evidence_class="captured_fixture", detect_reward_hacks=True,
                        runner=runner, emit_telemetry=False)
    return res["aggregate"], res["per_task"]


def _failing(per_task: Sequence[Mapping[str, Any]]) -> list[dict]:
    return [dict(r) for r in per_task if r.get("verdict") != "pass"]


def _write_source(workdir: Path, version: int, text: str) -> Path:
    p = workdir / f"agent_v{version}.py"
    p.write_text(text, encoding="utf-8")
    return p


def improve_agent_code(
    *,
    source_text: str,
    symbol: str,
    dataset: Mapping[str, Any],
    propose_patch: PatchProposer,
    objective: Mapping[str, Any],
    train_split: str = "train",
    test_split: str = "test",
    regression_split: str | None = "regression",
    max_rounds: int = 3,
    threshold: float = 0.5,
    seed: int = 42,
    runner: Any = None,
    emit_telemetry: bool = True,
    project_name: str | None = None,
) -> dict[str, Any]:
    """Run the code-level RSI loop on ``source_text`` (a module defining ``symbol``)
    against ``dataset`` (needs ``train``/``test`` splits; ``regression`` optional).

    Returns a report: baseline vs accepted held-out scores, the accepted patch (or
    None), per-round attempts, and the no-forgetting (regression) result. The
    caller decides whether to write an accepted patch back to the real file."""

    splits = dataset.get("splits") or {}
    if not splits.get(train_split) or not splits.get(test_split):
        raise ValueError("dataset needs both train and test splits for code-RSI")
    has_regression = bool(regression_split and splits.get(regression_split))

    with tempfile.TemporaryDirectory(prefix="agent-code-rsi-") as tmp:
        workdir = Path(tmp)
        cur = _write_source(workdir, 0, source_text)

        base_test, _ = _score_split(cur, symbol, dataset, test_split, seed=seed, runner=runner)
        base_reg = None
        if has_regression:
            base_reg, _ = _score_split(cur, symbol, dataset, regression_split, seed=seed, runner=runner)

        rounds: list[dict[str, Any]] = []
        accepted_text: str | None = None
        cur_text = source_text
        prior_attempts: list[dict[str, Any]] = []  # fed back so the loop LEARNS

        train_agg, train_per = _score_split(cur, symbol, dataset, train_split, seed=seed, runner=runner)
        for rnd in range(max_rounds):
            failing = _failing(train_per)
            if not failing:
                rounds.append({"round": rnd, "status": "no_bug_on_train", "train_pass_rate": train_agg["pass_rate"]})
                break

            diagnosis = {
                "current_source": cur_text,
                "symbol": symbol,
                "objective": objective,
                "failing_examples": [
                    {"task_id": f.get("task_id"), "score": f.get("score"),
                     "metric_averages": f.get("metric_averages"),
                     "tool_calls": len(f.get("tool_calls") or []),
                     "rewardhack": f.get("rewardhack"), "error": f.get("error")}
                    for f in failing
                ],
                # the RSI signal: prior rejected patches + WHY (execution errors /
                # no-lift), so the next proposal does not repeat the mistake.
                "prior_attempts": prior_attempts,
                "signal": "failing tasks; check tool use / completion-without-effort",
            }
            new_text = propose_patch(diagnosis)
            if not new_text or new_text == cur_text:
                rounds.append({"round": rnd, "status": "no_patch_proposed"})
                break

            cand = _write_source(workdir, rnd + 1, new_text)
            cand_train, cand_train_per = _score_split(cand, symbol, dataset, train_split, seed=seed, runner=runner)
            cand_test, _ = _score_split(cand, symbol, dataset, test_split, seed=seed, runner=runner)
            cand_reg = None
            if has_regression:
                cand_reg, _ = _score_split(cand, symbol, dataset, regression_split, seed=seed, runner=runner)

            held_out_lift = round(cand_test["mean_score"] - base_test["mean_score"], 6)
            regression_ok = (not has_regression) or (cand_reg["mean_score"] >= base_reg["mean_score"] - 1e-9)
            accept = held_out_lift > 0 and regression_ok
            cand_errors = [r.get("error") for r in cand_train_per if r.get("error")]

            rounds.append({
                "round": rnd, "status": "accepted" if accept else "rejected",
                "train_lift": round(cand_train["mean_score"] - train_agg["mean_score"], 6),
                "held_out_lift": held_out_lift,
                "regression_ok": regression_ok,
                "held_out_baseline": base_test["mean_score"],
                "held_out_candidate": cand_test["mean_score"],
                "candidate_errors": cand_errors[:2],
            })
            if accept:
                accepted_text = new_text
                cur, cur_text = cand, new_text
                base_test = cand_test
                if has_regression:
                    base_reg = cand_reg
                break  # one accepted fix per call (vertical); caller can re-invoke
            # rejected: feed this attempt (source + why it failed) back to the proposer.
            prior_attempts.append({
                "patch_excerpt": new_text[:500],
                "execution_errors": cand_errors[:2],
                "held_out_lift": held_out_lift,
                "reason": ("crashed: " + str(cand_errors[0])) if cand_errors else "no held-out improvement",
            })
            cur, cur_text, train_agg, train_per = cand, new_text, cand_train, cand_train_per

        report = {
            "kind": AGENT_LEARNING_CODE_RSI_REPORT_KIND,
            "fixed": accepted_text is not None,
            "accepted_source": accepted_text,
            "held_out_baseline": base_test["mean_score"] if accepted_text is None else rounds[-1]["held_out_baseline"],
            "held_out_final": base_test["mean_score"],
            "regression_held": (not has_regression) or all(
                r.get("regression_ok", True) for r in rounds if r["status"] in ("accepted", "rejected")
            ),
            "rounds": rounds,
        }
        if emit_telemetry:
            # ONE dashboard run for the whole code-RSI loop: root + per-round spans
            # (P14). Side-channel; never alters the report.
            from .telemetry import emit_run

            lift = round(report["held_out_final"] - report["held_out_baseline"], 6)
            summary = emit_run(
                kind="code-rsi",
                name=symbol,
                metrics={
                    "fixed": report["fixed"],
                    "held_out_baseline": report["held_out_baseline"],
                    "held_out_final": report["held_out_final"],
                    "held_out_lift": lift,
                    "regression_held": report["regression_held"],
                },
                verdict="pass" if report["fixed"] and lift > 0 else "fail",
                children=[
                    (
                        f"round:{r['round']}",
                        {"status": r.get("status"),
                         "held_out_lift": r.get("held_out_lift"),
                         "regression_ok": r.get("regression_ok")},
                    )
                    for r in rounds
                ],
                project_name=project_name,
            )
            report["telemetry"] = summary.as_dict()
        return report


def propose_patch_via_llm(model: str = "gpt-4o-mini") -> PatchProposer:
    """Default proposer: an LLM rewrites the source to fix the diagnosed failure.
    Conditioned on the current source + failing eval + objective; returns the new
    full source (no co-authoring of the fix — the model derives it from the
    trace/eval). Keyed (litellm); credential-free tests use a deterministic
    proposer instead."""

    def _propose(diagnosis: Mapping[str, Any]) -> str | None:
        import re

        import litellm

        prior = diagnosis.get("prior_attempts") or []
        prior_block = (
            "\n\n=== YOUR PRIOR REJECTED ATTEMPTS (do NOT repeat these mistakes) ===\n"
            + str(prior)[:800]
            if prior else ""
        )
        prompt = (
            "You are fixing a Python agent's source code. The agent runs in a "
            "simulation. Each available tool is on `agent_input.tools` as a dict "
            "shaped EITHER {\"name\": str, ...} OR {\"type\":\"function\",\"function\":{\"name\":str}} "
            "— there is NO 'id' key on a tool spec, so read the name defensively "
            "(`t.get('name') or (t.get('function') or {}).get('name')`) and generate "
            "your own call id. The function MUST return a dict "
            "{\"content\": str, \"tool_calls\": [{\"id\": str, \"name\": str, \"arguments\": dict}]}. "
            "It is FAILING because it does not call the available tool (it fabricates "
            "an answer). Rewrite the WHOLE source so it calls the first available "
            "tool by its resolved name with empty arguments and returns that "
            "tool_call. Keep the same function name. Code must run without KeyError. "
            "Return ONLY the new Python source — no prose, no markdown fences."
            + "\n\n=== CURRENT SOURCE ===\n"
            + str(diagnosis.get("current_source", ""))
            + "\n\n=== FAILING EVAL (sample) ===\n"
            + str(diagnosis.get("failing_examples", []))[:600]
            + prior_block
        )
        resp = litellm.completion(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=600,
        )
        text = resp.choices[0].message.content or ""
        text = re.sub(r"^```(?:python)?\n|\n```$", "", text.strip())  # strip fences if any
        return text or None

    return _propose
