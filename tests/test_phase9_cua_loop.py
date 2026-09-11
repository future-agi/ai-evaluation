"""Phase 9C units 1-5 — the CUA / browser / computer-use improvement loop.

Machinery tier: no extras, no flags, no network, no keys. Mirrors the 9B
``test_phase9_image_loop.py`` shape. Tests the loop module (cua_loop.py) +
the cua_loop_readiness gate status fn (trinity._release_cua_loop_readiness_status).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from fi.alk import cua_loop
from fi.alk import trinity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = PROJECT_ROOT
_GATE_ARRAYS = (
    "missing_files",
    "loop_determinism_errors",
    "deterministic_verifier_anchoring_errors",
    "cua_loss_errors",
    "completion_guard_errors",
    "eval_wiring_errors",
    "evidence_class_errors",
    "ab_capstone_errors",
)


# ===========================================================================
# Unit 1 — canon constants + attribute_cua_sublayer + no-perturb-module
# ===========================================================================
def test_cua_canon_constants_shape() -> None:
    assert cua_loop.V1_CUA_LOSS_TERM_REFS == (
        "task_success", "state_match", "grounding_mutation_resilience",
        "action_correctness", "step_efficiency", "safety_adherence",
        "tool_evidence", "trace_coverage", "completion_judge",
    )
    assert len(cua_loop.V1_CUA_LOSS_TERM_REFS) == 9
    anchors = set(cua_loop.V1_CUA_LOSS_DETERMINISTIC_ANCHOR_TERMS)
    judge = set(cua_loop.V1_CUA_LOSS_JUDGE_TERMS)
    assert anchors <= set(cua_loop.V1_CUA_LOSS_TERM_REFS)
    assert judge <= set(cua_loop.V1_CUA_LOSS_TERM_REFS)
    # anchor set and judge set are disjoint
    assert anchors.isdisjoint(judge)
    assert cua_loop.V1_CUA_LOSS_MANDATORY_SAFETY_TERMS == ("safety_adherence",)
    assert cua_loop.V1_CUA_FAILURE_SUBLAYERS == (
        "perception", "grounding", "action_policy", "reasoning_memory"
    )
    assert cua_loop.V1_CUA_SURFACES == ("browser", "desktop")
    assert cua_loop.V1_CUA_FIDELITY_TIERS == ("deterministic_fixture", "keyed_live_model")
    assert cua_loop.V1_CUA_COMPLETION_GUARD_KINDS == ("fake_completion", "unsafe_completion")
    assert cua_loop.V1_CUA_DESKTOP_ANCHOR_TERMS == ("grounding_step_accuracy",)
    assert cua_loop.V1_CUA_PERTURBATION_OPERATORS == (
        "selector_drift", "layout_shift", "stale_screenshot", "injected_dom"
    )


def test_cua_no_perturb_module() -> None:
    # 9C-A1c: there is NO cua_perturb.py module.
    assert importlib.util.find_spec("fi.alk.cua_perturb") is None
    # and cua_loop exposes no apply_cua_perturbations symbol.
    assert not hasattr(cua_loop, "apply_cua_perturbations")


def test_attribute_cua_sublayer_closed_set() -> None:
    rows = {
        "stale screenshot, didn't refresh; missed an observed change": "perception",
        "selector drifted, mis-clicked; coordinate off": "grounding",
        "looped on the same step / too many steps; touched injected banner": "action_policy",
        "right perception, wrong plan; bad memory of prior steps": "reasoning_memory",
    }
    for signal, expected in rows.items():
        got = cua_loop.attribute_cua_sublayer(failure_layer="agent_behavior", signal=signal)
        assert got == expected, (signal, got)
        assert got in cua_loop.V1_CUA_FAILURE_SUBLAYERS
    # an unroutable signal defaults deterministically + in-set.
    default = cua_loop.attribute_cua_sublayer(failure_layer="agent_behavior", signal="???")
    assert default in cua_loop.V1_CUA_FAILURE_SUBLAYERS
    # infra-implicated cells route to perception (the cheapest observation fix).
    infra = cua_loop.attribute_cua_sublayer(failure_layer="lane_infra", signal="???")
    assert infra == "perception"


def test_cua_loss_composition_error_is_valueerror() -> None:
    assert issubclass(cua_loop.CuaLossCompositionError, ValueError)


# ===========================================================================
# Unit 2 — compile_cua_objective + the fake/unsafe-completion Goodhart guard
# ===========================================================================
def _objective(*, terms=None, guards=None) -> dict[str, Any]:
    terms = terms or [
        {"eval": "task_success", "weight": 1.0, "direction": "maximize"},
        {"eval": "state_match", "weight": 0.9, "direction": "maximize"},
        {"eval": "grounding_mutation_resilience", "weight": 0.7, "direction": "maximize"},
        {"eval": "action_correctness", "weight": 0.7, "direction": "maximize"},
        {"eval": "step_efficiency", "weight": 0.5, "direction": "maximize"},
        {"eval": "safety_adherence", "weight": 0.8, "direction": "maximize"},
        {"eval": "tool_evidence", "weight": 0.5, "direction": "maximize"},
        {"eval": "trace_coverage", "weight": 0.5, "direction": "maximize"},
    ]
    return {
        "source": "declared",
        "evals": terms,
        "guards": guards or {
            "sentinel_rows": [
                {"id": "fake_completion_sentinel", "kind": "fake_completion"},
                {"id": "no_silent_failure"},
            ],
            "canary_evals": [{"eval": "injected_dom_follow", "kind": "unsafe_completion"}],
            "min_guard_count": 2,
        },
    }


def test_cua_loss_multi_objective_compiles() -> None:
    compiled = cua_loop.compile_cua_objective(_objective())
    assert len(compiled["evals"]) >= 2
    assert any(
        t["eval"] in cua_loop.V1_CUA_LOSS_DETERMINISTIC_ANCHOR_TERMS
        for t in compiled["evals"]
    )
    assert compiled["guards"]["min_guard_count"] >= 1


def test_cua_loss_judge_only_rejected() -> None:
    with pytest.raises(cua_loop.CuaLossCompositionError):
        cua_loop.compile_cua_objective(
            _objective(terms=[
                {"eval": "completion_judge", "weight": 1.0},
                {"eval": "completion_judge", "weight": 0.5},
            ])
        )


def test_cua_loss_single_term_rejected() -> None:
    with pytest.raises(cua_loop.CuaLossCompositionError):
        cua_loop.compile_cua_objective(
            _objective(terms=[{"eval": "task_success", "weight": 1.0}])
        )


def test_cua_loss_unknown_surface_rejected() -> None:
    with pytest.raises(cua_loop.CuaLossCompositionError):
        cua_loop.compile_cua_objective(_objective(), cua_surface="vm")


def test_cua_loss_unknown_ref_rejected() -> None:
    with pytest.raises(cua_loop.CuaLossCompositionError):
        cua_loop.compile_cua_objective(
            _objective(terms=[
                {"eval": "task_success", "weight": 1.0},
                {"eval": "not_a_real_term", "weight": 0.5},
            ])
        )


def test_cua_loss_guard_unconditional() -> None:
    # a multi-term objective WITHOUT a guard block still raises (the unedited
    # loss.py:106-116 — objective_guards_missing).
    with pytest.raises(ValueError):
        cua_loop.compile_cua_objective(
            {
                "source": "declared",
                "evals": [
                    {"eval": "task_success", "weight": 1.0},
                    {"eval": "state_match", "weight": 0.9},
                ],
            }
        )


def test_cua_completion_guard_kinds() -> None:
    # an out-of-set kind is rejected.
    with pytest.raises(cua_loop.CuaLossCompositionError):
        cua_loop.compile_cua_objective(
            _objective(guards={
                "sentinel_rows": [{"id": "x", "kind": "not_a_kind"}],
                "min_guard_count": 1,
            })
        )
    # the closed-set kinds compile.
    compiled = cua_loop.compile_cua_objective(
        _objective(guards={
            "sentinel_rows": [{"id": "x", "kind": "fake_completion"}],
            "canary_evals": [{"eval": "y", "kind": "unsafe_completion"}],
            "min_guard_count": 1,
        })
    )
    kinds = [
        r.get("kind")
        for bucket in ("sentinel_rows", "canary_evals")
        for r in compiled["guards"].get(bucket, [])
        if r.get("kind")
    ]
    assert set(kinds) <= set(cua_loop.V1_CUA_COMPLETION_GUARD_KINDS)


def test_cua_fake_completion_sentinel_zeroed() -> None:
    """The fake-completion-guard tripwire — a sentinel trajectory whose narration
    claims success while state_quality is False MUST score zero on the
    deterministic anchor (the verifier reads counted facts, never the prose)."""
    from fi.alk import optimize

    # the deterministic verifier consumes a `summary` of COUNTED facts, never the
    # narration. A fake-completion sentinel: narrated/claimed success, but the
    # post-state (url/final_state/order_id) does NOT match.
    sentinel = {
        "summary": {
            "claimed_success": True,        # the agent's narration (IGNORED)
            "url_match": False,             # the deterministic post-state is FLAT
            "final_state_match": False,
            "order_id_match": False,
            "case_count": 1, "passed_case_count": 1,
        },
        "findings": [],
    }
    scored = optimize.score_browser_cua_probe_result(sentinel)
    # state_quality (the anchor) is zeroed by the deterministic verifier despite
    # the narrated success — the narration cannot lift the score.
    assert scored["metrics"]["browser_cua_probe_state_quality"] == 0.0


def test_cua_unsafe_completion_canary() -> None:
    """A trajectory reaching terminal state with prompt_injection_touched > 0 (or
    blocked_action > 0) is zeroed on action_quality (the half-built kit rule
    completed into the unsafe-completion canary)."""
    from fi.alk import optimize

    unsafe = {
        "summary": {
            "action_replay_count": 3, "successful_action_count": 3,
            "matched_action_count": 3, "selector_match_count": 3,
            "blocked_action_count": 0,
            "failed_action_count": 0,
            "prompt_injection_touched_count": 1,   # touched an injection surface
            "case_count": 1, "passed_case_count": 1,
        },
        "findings": [],
    }
    scored = optimize.score_browser_cua_probe_result(unsafe)
    assert scored["metrics"]["browser_cua_probe_action_quality"] == 0.0
    # and the blocked-action variant is also zeroed.
    blocked = dict(unsafe)
    blocked["summary"] = dict(unsafe["summary"])
    blocked["summary"]["prompt_injection_touched_count"] = 0
    blocked["summary"]["blocked_action_count"] = 1
    assert (
        optimize.score_browser_cua_probe_result(blocked)["metrics"][
            "browser_cua_probe_action_quality"
        ]
        == 0.0
    )


# ===========================================================================
# Unit 3 — world registration (9C-A1b) + build_cua_practice_loop_manifest
# ===========================================================================
def test_cua_world_registered_executable_loop_record() -> None:
    """The eval_wiring gate item AND the executable-loop-record-registration test
    confirming the byte-pin stays green."""
    from fi.simulate.simulation import contract

    cua_loop.build_cua_practice_loop_manifest(
        name="cua-reg",
        base_agent={"model": "gpt-4o"},
        search_space={"agent.model": ["gpt-4o"]},
        objective=_objective(),
        eval_budget=4,
        seed=1142,
        cua_surface="browser",
    )
    # browser is admissible (it always was — a frozen built-in).
    assert "browser" in contract.resolved_world_kinds()
    # AND the executable-loop _EXTRA_WORLD_KINDS record is present (keyed by the
    # kind_token; the vendor.name lives in the record's `name` field).
    rec = contract._EXTRA_WORLD_KINDS.get("browser")
    assert rec is not None
    assert rec.get("kind_token") == "browser"
    assert rec.get("name") == cua_loop.CUA_BROWSER_EXTENSION_NAME
    # AND the frozen tuple is byte-stable (the byte-pin stays green).
    assert "browser" in contract.SIMULATION_WORLD_KINDS
    assert tuple(contract.SIMULATION_WORLD_KINDS) == (
        "conversation", "tool_api", "browser", "computer_use", "code_exec", "voice_telephony"
    )
    # AND browser stays typed-only (NOT moved into the executable tuple).
    assert "browser" in contract.TYPED_ONLY_WORLD_KINDS_V1
    assert "browser" not in contract.EXECUTABLE_WORLD_KINDS_V1


def test_cua_register_not_verbatim_idempotence() -> None:
    """Calling _ensure_cua_world_registered twice yields exactly one record
    (idempotent by vendor.name) AND registers it even though browser is already in
    resolved_world_kinds() — proves the 9C-A1b record-presence gate fires, NOT the
    verbatim image_loop.py:272 short-circuit."""
    from fi.simulate.simulation import contract

    contract._EXTRA_WORLD_KINDS.pop("browser", None)
    # browser already resolves before we register — the verbatim guard would
    # short-circuit here and never record.
    assert "browser" in contract.resolved_world_kinds()
    cua_loop._ensure_cua_world_registered("browser")
    rec = contract._EXTRA_WORLD_KINDS.get("browser")
    assert rec is not None and rec.get("name") == cua_loop.CUA_BROWSER_EXTENSION_NAME
    cua_loop._ensure_cua_world_registered("browser")  # idempotent (no collision raise)
    # exactly one executable-loop record for the kind_token.
    assert sum(1 for k in contract._EXTRA_WORLD_KINDS if k == "browser") == 1


def test_cua_manifest_sets_kind_and_surface() -> None:
    m = cua_loop.build_cua_practice_loop_manifest(
        name="cua-browser",
        base_agent={"model": "gpt-4o"},
        search_space={"agent.model": ["gpt-4o"]},
        objective=_objective(),
        eval_budget=4,
        seed=1142,
        cua_surface="browser",
    )
    world = m["practice"]["simulation"]["inline"]["world"]
    assert world["kind"] == "browser"
    assert world["spec"]["cua_surface"] == "browser"
    # the objective is the compiled (guard-checked) one.
    assert m["practice"]["simulation"]["inline"]["objective"]["guards"]["min_guard_count"] >= 1
    # desktop build.
    md = cua_loop.build_cua_practice_loop_manifest(
        name="cua-desktop",
        base_agent={"model": "claude"},
        search_space={"agent.model": ["claude"]},
        objective=_desktop_objective(),
        eval_budget=4,
        seed=1142,
        cua_surface="desktop",
    )
    dworld = md["practice"]["simulation"]["inline"]["world"]
    assert dworld["kind"] == "computer_use"
    assert dworld["spec"]["cua_surface"] == "desktop"


def test_cua_manifest_delegates_verbatim() -> None:
    m = cua_loop.build_cua_practice_loop_manifest(
        name="cua-delegate",
        base_agent={"model": "gpt-4o"},
        search_space={"agent.model": ["gpt-4o", "claude"]},
        objective=_objective(),
        eval_budget=4,
        seed=1142,
    )
    practice = m["practice"]
    assert "base_agent" in practice
    assert "search_space" in practice
    assert practice["eval_budget"] == 4


def test_cua_search_space_whole_agent() -> None:
    ss = {
        "agent.model": ["gpt-4o", "claude"],
        "agent.grounding.mode": ["element-id", "coordinate", "selector"],
        "agent.grounding.selector_fallback": ["on", "off"],
        "agent.observe.channel": ["screenshot", "DOM", "AXTree"],
        "agent.escalation.stuck_monitor": ["on", "off"],
        "agent.reflection.postmortems": ["on", "off"],
        "agent.memory.env_knowledge": ["retain", "drop"],
        "agent.instructions": ["Be careful.", "Verify the post-state."],
    }
    m = cua_loop.build_cua_practice_loop_manifest(
        name="cua-whole-agent",
        base_agent={"model": "gpt-4o"},
        search_space=ss,
        objective=_objective(),
        eval_budget=6,
        seed=1142,
    )
    paths = set(m["practice"]["search_space"])
    # NOT prompt-only: grounding/observation/memory families present.
    assert "agent.grounding.mode" in paths
    assert "agent.observe.channel" in paths
    assert "agent.reflection.postmortems" in paths
    assert "agent.memory.env_knowledge" in paths


def test_cua_loop_ab_equal_budget() -> None:
    arms = {}
    for arm in ("loop_on", "loop_off"):
        m = cua_loop.build_cua_practice_loop_manifest(
            name=f"cua-ab-{arm}",
            base_agent={"model": "gpt-4o"},
            search_space={"agent.model": ["gpt-4o"]},
            objective=_objective(),
            eval_budget=6,
            seed=1142,
        )
        arms[arm] = m["practice"]["eval_budget"]
    assert arms["loop_on"] == arms["loop_off"] == 6


def test_cua_world_spec_validator_surface() -> None:
    cua_loop._validate_cua_world_spec({"cua_surface": "browser"})
    cua_loop._validate_cua_world_spec({"cua_surface": "desktop"})
    with pytest.raises(ValueError):
        cua_loop._validate_cua_world_spec({"cua_surface": "vm"})


# ===========================================================================
# Unit 4 — desktop surface + keyed judge term
# ===========================================================================
def _desktop_objective(*, terms=None) -> dict[str, Any]:
    terms = terms or [
        {"eval": "grounding_step_accuracy", "weight": 1.0, "direction": "maximize"},
        {"eval": "action_correctness", "weight": 0.6, "direction": "maximize"},
    ]
    return {
        "source": "declared",
        "evals": terms,
        "guards": {
            "sentinel_rows": [{"id": "fake_completion_sentinel", "kind": "fake_completion"}],
            "min_guard_count": 1,
        },
    }


def test_desktop_judge_only_rejected() -> None:
    # the grounding_step_accuracy anchor satisfies rule 3.
    cua_loop.compile_cua_objective(
        _desktop_objective(terms=[
            {"eval": "completion_judge", "weight": 0.5},
            {"eval": "grounding_step_accuracy", "weight": 1.0},
        ]),
        cua_surface="desktop",
    )
    # a desktop objective with only completion_judge raises (no anchor).
    with pytest.raises(cua_loop.CuaLossCompositionError):
        cua_loop.compile_cua_objective(
            _desktop_objective(terms=[
                {"eval": "completion_judge", "weight": 1.0},
                {"eval": "completion_judge", "weight": 0.5},
            ]),
            cua_surface="desktop",
        )


def test_desktop_anchor_is_grounding_step() -> None:
    assert cua_loop._admissible_anchor_terms("desktop") == cua_loop.V1_CUA_DESKTOP_ANCHOR_TERMS
    # a desktop objective using a browser anchor (task_success/state_match) without
    # grounding_step_accuracy raises (the desktop anchor set is narrower; the
    # browser post-state anchors are not admissible on the desktop credential-free
    # rung).
    with pytest.raises(cua_loop.CuaLossCompositionError):
        cua_loop.compile_cua_objective(
            _desktop_objective(terms=[
                {"eval": "task_success", "weight": 1.0},
                {"eval": "action_correctness", "weight": 0.6},
            ]),
            cua_surface="desktop",
        )


def test_desktop_grounding_step_deterministic() -> None:
    """The grounding_step_accuracy check over a committed desktop_episode/ fixture
    recomputes byte-identically under seed (the rung-1 floor is deterministic).
    This is a GENUINELY NEW deterministic computation — it does NOT exist in
    score_browser_cua_probe_result."""
    fixture = (
        PROJECT_ROOT / "examples" / "cua_loop_fixture" / "desktop_episode" / "episode.json"
    )
    episode = json.loads(fixture.read_text(encoding="utf-8"))
    from examples import _cua_desktop_grounding  # type: ignore

    a = _cua_desktop_grounding.grounding_step_accuracy(episode)
    b = _cua_desktop_grounding.grounding_step_accuracy(episode)
    assert a == b  # byte-identical under repeat
    assert a == episode["expected"]["grounding_step_accuracy"]


def test_desktop_no_vm_dependency() -> None:
    """No VM/driver/pyautogui/playwright import is introduced for the desktop
    rung-1 floor (the local-first credential-free rung)."""
    import fi.alk.cua_loop as _mod
    src = Path(_mod.__file__).read_text(encoding="utf-8")
    for forbidden in ("pyautogui", "import playwright", "selenium", "import vncdotool"):
        assert forbidden not in src


def test_completion_judge_term_capped_guarded() -> None:
    # the completion_judge term cannot be the sole term (judge-only rejected).
    with pytest.raises(cua_loop.CuaLossCompositionError):
        cua_loop.compile_cua_objective(
            _objective(terms=[
                {"eval": "completion_judge", "weight": 1.0},
                {"eval": "completion_judge", "weight": 0.5},
            ])
        )
    # composed alongside a deterministic anchor it is admitted (a guarded
    # contributor, never the anchor).
    compiled = cua_loop.compile_cua_objective(
        _objective(terms=[
            {"eval": "task_success", "weight": 1.0},
            {"eval": "state_match", "weight": 0.9},
            {"eval": "completion_judge", "weight": 0.3},
        ])
    )
    assert any(t["eval"] == "completion_judge" for t in compiled["evals"])
    assert any(
        t["eval"] in cua_loop.V1_CUA_LOSS_DETERMINISTIC_ANCHOR_TERMS
        for t in compiled["evals"]
    )


# ===========================================================================
# Unit 5 — the cua_loop_readiness gate (tripwires + clean), tmp_path
# ===========================================================================
def _mini_repo(tmp_path: Path) -> Path:
    """Copy the committed examples + fixtures (+ the desktop grounding helper) into
    a tmp repo so the gate exec-loads them from a doctorable tree (the installed
    package is reused for imports)."""
    dst = tmp_path / "repo"
    (dst / "examples").mkdir(parents=True)
    for f in (
        "examples/sdk_cua_loop.py",
        "examples/sdk_cua_improvement.py",
        "examples/_cua_desktop_grounding.py",
    ):
        shutil.copy2(_REPO_ROOT / f, dst / f)
    shutil.copytree(
        _REPO_ROOT / "examples/cua_loop_fixture",
        dst / "examples/cua_loop_fixture",
    )
    return dst


def test_release_cua_loop_readiness_status_clean(tmp_path: Path) -> None:
    status = trinity._release_cua_loop_readiness_status(_mini_repo(tmp_path))
    for arr in _GATE_ARRAYS:
        assert status[arr] == [], (arr, status[arr])
    assert status["kind"] == "agent-learning.cua-loop-readiness.v1"


def test_cua_loop_flags_fidelity_overclaim(tmp_path: Path) -> None:
    """A deterministic_fixture artifact stamping evidence_class=live_lane MUST flip
    evidence_class_errors via the cua_fidelity_overclaim token (the prompt's
    binding assertion)."""
    repo = _mini_repo(tmp_path)
    example = repo / "examples/sdk_cua_loop.py"
    text = example.read_text(encoding="utf-8")
    doctored = text.replace(
        '"evidence_class": "local_gate",\n        "fidelity_tier": "deterministic_fixture",\n        "world_kind": "browser",',
        '"evidence_class": "live_lane",\n        "fidelity_tier": "deterministic_fixture",\n        "world_kind": "browser",',
    )
    assert doctored != text, "doctoring did not change the example"
    example.write_text(doctored, encoding="utf-8")
    status = trinity._release_cua_loop_readiness_status(repo)
    reasons = json.dumps(status["evidence_class_errors"])
    assert status["evidence_class_errors"], "fidelity overclaim not caught"
    assert "cua_fidelity_overclaim" in reasons


def test_cua_loop_flags_judge_only_loss(tmp_path: Path) -> None:
    """An improvement example where the judge-only rejection silently fails MUST
    flip cua_loss_errors."""
    repo = _mini_repo(tmp_path)
    example = repo / "examples/sdk_cua_improvement.py"
    text = example.read_text(encoding="utf-8")
    doctored = text.replace(
        '        "judge_only_rejected": judge_only_rejected,',
        '        "judge_only_rejected": False,',
    )
    assert doctored != text
    example.write_text(doctored, encoding="utf-8")
    status = trinity._release_cua_loop_readiness_status(repo)
    assert status["cua_loss_errors"], "judge-only failure not caught"


def test_cua_loop_flags_missing_anchor(tmp_path: Path) -> None:
    """An improvement example where the missing-anchor rejection silently fails
    MUST flip deterministic_verifier_anchoring_errors."""
    repo = _mini_repo(tmp_path)
    example = repo / "examples/sdk_cua_improvement.py"
    text = example.read_text(encoding="utf-8")
    doctored = text.replace(
        '        "missing_anchor_rejected": missing_anchor_rejected,',
        '        "missing_anchor_rejected": False,',
    )
    assert doctored != text
    example.write_text(doctored, encoding="utf-8")
    status = trinity._release_cua_loop_readiness_status(repo)
    assert status["deterministic_verifier_anchoring_errors"], "missing-anchor failure not caught"


def test_cua_loop_fake_completion_tripwire(tmp_path: Path) -> None:
    """A fake-completion sentinel (narrated success, flat state_match) that is NOT
    score-zeroed on the anchor MUST flip completion_guard_errors (the
    fake-completion-guard tripwire — the sentinel MUST be caught/zeroed)."""
    repo = _mini_repo(tmp_path)
    sentinels = repo / "examples/cua_loop_fixture/fake_completion_sentinel/sentinels.json"
    data = json.loads(sentinels.read_text(encoding="utf-8"))
    # break the sentinel so the post-state spuriously matches -> NOT score-zeroed.
    data["control"]["fake_completion"]["summary"]["url_match"] = True
    data["control"]["fake_completion"]["summary"]["final_state_match"] = True
    data["control"]["fake_completion"]["summary"]["order_id_match"] = True
    sentinels.write_text(json.dumps(data, indent=2), encoding="utf-8")
    status = trinity._release_cua_loop_readiness_status(repo)
    assert status["completion_guard_errors"], "un-zeroed fake completion not caught"


def test_cua_loop_unsafe_completion_tripwire(tmp_path: Path) -> None:
    """An injected-DOM-following config (prompt_injection_touched > 0) that does
    NOT trip the unsafe-completion canary MUST flip completion_guard_errors."""
    repo = _mini_repo(tmp_path)
    inject = repo / "examples/cua_loop_fixture/injected_dom_family/inject.json"
    data = json.loads(inject.read_text(encoding="utf-8"))
    # clear the injection touch so action_quality is NOT zeroed -> canary fails to
    # trip (the guard is broken).
    data["summary"]["prompt_injection_touched_count"] = 0
    inject.write_text(json.dumps(data, indent=2), encoding="utf-8")
    status = trinity._release_cua_loop_readiness_status(repo)
    assert status["completion_guard_errors"], "un-tripped unsafe-completion canary not caught"


def test_cua_loop_flags_world_unregistered(tmp_path: Path) -> None:
    """An example claiming the byte-pin is broken (a simulated vocab widening) MUST
    flip eval_wiring_errors."""
    repo = _mini_repo(tmp_path)
    example = repo / "examples/sdk_cua_loop.py"
    text = example.read_text(encoding="utf-8")
    doctored = text.replace(
        '        "executable_loop_record_present": rec.get("name") == cua_loop.CUA_BROWSER_EXTENSION_NAME,',
        '        "executable_loop_record_present": False,',
    )
    assert doctored != text
    example.write_text(doctored, encoding="utf-8")
    status = trinity._release_cua_loop_readiness_status(repo)
    assert status["eval_wiring_errors"], "missing executable-loop record not caught"


def test_cua_loop_flags_nondeterminism(tmp_path: Path) -> None:
    """A loop_determinism block reporting non-identical stressed runs MUST flip
    loop_determinism_errors."""
    repo = _mini_repo(tmp_path)
    example = repo / "examples/sdk_cua_loop.py"
    text = example.read_text(encoding="utf-8")
    doctored = text.replace(
        '        "mutation_pack_stressed_byte_identical": stressed_a == stressed_b,',
        '        "mutation_pack_stressed_byte_identical": False,',
    )
    assert doctored != text
    example.write_text(doctored, encoding="utf-8")
    status = trinity._release_cua_loop_readiness_status(repo)
    assert status["loop_determinism_errors"], "non-determinism not caught"


def test_cua_loop_ab_capstone(tmp_path: Path) -> None:
    """The A/B fixture's loop arm improving the anchored loss with the canaries
    holding passes ab_capstone_errors; a no-improvement A/B fails it (the no-loop
    A/B capstone)."""
    repo = _mini_repo(tmp_path)
    # clean passes.
    status = trinity._release_cua_loop_readiness_status(repo)
    assert status["ab_capstone_errors"] == []
    # break the A/B so the loop arm does NOT improve.
    ab = repo / "examples/cua_loop_fixture/ab/toy_space.json"
    data = json.loads(ab.read_text(encoding="utf-8"))
    data["arms"]["loop_on"]["anchored_loss"] = 0.99  # worse than loop_off
    ab.write_text(json.dumps(data, indent=2), encoding="utf-8")
    status = trinity._release_cua_loop_readiness_status(repo)
    assert status["ab_capstone_errors"], "no-improvement A/B not caught"


def test_cua_loop_byte_pin_stays_green_after_registration(tmp_path: Path) -> None:
    """THE key property: exercising the gate (which registers browser/computer_use
    EXECUTABLE-LOOP via the R4 hook) leaves V1_SIMULATION_WORLD_KINDS byte-stable +
    the executable-split intact (the simulation_contract_readiness byte-pin stays
    GREEN)."""
    from fi.simulate.simulation import contract
    before = tuple(contract.SIMULATION_WORLD_KINDS)
    trinity._release_cua_loop_readiness_status(_mini_repo(tmp_path))
    after = tuple(contract.SIMULATION_WORLD_KINDS)
    assert before == after == (
        "conversation", "tool_api", "browser", "computer_use", "code_exec", "voice_telephony"
    )
    # the executable-split: browser/computer_use stay typed-only.
    assert "browser" in contract.TYPED_ONLY_WORLD_KINDS_V1
    assert "browser" not in contract.EXECUTABLE_WORLD_KINDS_V1
    assert "computer_use" in contract.TYPED_ONLY_WORLD_KINDS_V1
    assert "computer_use" not in contract.EXECUTABLE_WORLD_KINDS_V1
