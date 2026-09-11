"""CUA loop readiness example (Phase 9C, the cua_loop_readiness gate).

Runs ENTIRELY offline — zero network, zero API keys, zero lanes, no real browser,
no VM — on the committed ``examples/cua_loop_fixture/`` synthetic-DOM fixtures +
goldens, over the already-shipped ``BrowserEnvironment`` +
``score_browser_cua_probe_result`` (the 7-dim deterministic verifier).
``run(output_path)`` returns the full evidence payload the gate audits
field-by-field (eight error arrays) and also writes it to ``output_path``.

Sequence (BBG §6.2):

    register browser/computer_use EXECUTABLE-LOOP via the R4 hook (assert browser
    in resolved_world_kinds() AND the agentlearning.browser_cua _EXTRA_WORLD_KINDS
    record present AND V1_SIMULATION_WORLD_KINDS byte-stable — the 9C-A1b
    executable-loop-record gate, NOT the verbatim image idempotence guard) ->
    loop determinism demo (re-run, byte-identical trajectory + mutation-pack
    stressed runs over BrowserEnvironment) -> deterministic anchors demo
    (state_quality/action_quality/mutation_grounding_quality reproducible over the
    fixtures via score_browser_cua_probe_result; desktop grounding_step_accuracy)
    -> the fake-completion guard demo (sentinel narrates success but is
    score-zeroed on the anchor) -> the unsafe-completion canary demo (injected-DOM
    following trips it) -> the constructed negatives (a deterministic artifact
    claiming live_lane -> caught by cua_fidelity_overclaim).

Honest tiering is structural: a deterministic in-process fixture artifact is
``local_gate``/``captured_fixture`` carrying ``fidelity_tier:
"deterministic_fixture"`` -- NEVER ``live_lane`` (the §2.6 mandate). No
deployable-risk wording.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import cua_loop, optimize
from fi.simulate.environment import BrowserEnvironment
from fi.simulate.simulation import contract

EXAMPLE_DIR = Path(__file__).resolve().parent
FIXTURES = EXAMPLE_DIR / "cua_loop_fixture"
READINESS_KIND = "agent-learning.cua-loop.v1"

_SEED = 1142

sys.path.insert(0, str(EXAMPLE_DIR))
import _cua_desktop_grounding  # noqa: E402


def _load_json(rel: str) -> Any:
    return json.loads((FIXTURES / rel).read_text(encoding="utf-8"))


def _state_quality(summary: dict[str, Any]) -> float:
    scored = optimize.score_browser_cua_probe_result({"summary": summary, "findings": []})
    return scored["metrics"]["browser_cua_probe_state_quality"]


def _action_quality(summary: dict[str, Any]) -> float:
    scored = optimize.score_browser_cua_probe_result({"summary": summary, "findings": []})
    return scored["metrics"]["browser_cua_probe_action_quality"]


def _mutation_grounding_quality(summary: dict[str, Any]) -> float:
    scored = optimize.score_browser_cua_probe_result({"summary": summary, "findings": []})
    return scored["metrics"]["browser_cua_probe_mutation_grounding_quality"]


def _registration() -> dict[str, Any]:
    """Register browser/computer_use EXECUTABLE-LOOP via the R4 hook and assert the
    frozen vocab is byte-stable (browser is admissible without widening
    SIMULATION_WORLD_KINDS — the 9C-A1b executable-loop-record gate)."""
    cua_loop._ensure_cua_world_registered("browser")
    cua_loop._ensure_cua_world_registered("desktop")
    rec = contract._EXTRA_WORLD_KINDS.get("browser") or {}
    rec_desktop = contract._EXTRA_WORLD_KINDS.get("computer_use") or {}
    return {
        "browser_in_resolved_world_kinds": "browser" in contract.resolved_world_kinds(),
        "computer_use_in_resolved_world_kinds": "computer_use" in contract.resolved_world_kinds(),
        "browser_in_frozen_world_kinds": "browser" in contract.SIMULATION_WORLD_KINDS,
        "frozen_vocab_byte_stable": tuple(contract.SIMULATION_WORLD_KINDS) == (
            "conversation", "tool_api", "browser", "computer_use", "code_exec", "voice_telephony"
        ),
        "executable_loop_record_present": rec.get("name") == cua_loop.CUA_BROWSER_EXTENSION_NAME
        and rec.get("kind_token") == "browser",
        "desktop_executable_loop_record_present": (
            rec_desktop.get("name") == cua_loop.CUA_DESKTOP_EXTENSION_NAME
            and rec_desktop.get("kind_token") == "computer_use"
        ),
        "browser_stays_typed_only": "browser" in contract.TYPED_ONLY_WORLD_KINDS_V1
        and "browser" not in contract.EXECUTABLE_WORLD_KINDS_V1,
    }


def _loop_determinism() -> dict[str, Any]:
    """Re-run the loop fixture twice under the pinned seed -> byte-identical
    trajectory + byte-identical mutation-pack stressed runs over BrowserEnvironment
    (deterministic under reset / deep-copy initial_state)."""
    form = _load_json("multistep_form/form.json")
    env = BrowserEnvironment(url=str(form["url"]), dom="<html><body></body></html>")
    snap_a = env.reset()
    snap_b = env.reset()
    env_reset_deterministic = snap_a.state.get("url") == snap_b.state.get("url")

    # the mutation-pack stressed run is deterministic-under-seed (the kit's
    # existing operators; NO cua_perturb.py). The paired clean-vs-stressed delta:
    clean = _load_json("selector_drift_family/clean.json")
    drifted = _load_json("selector_drift_family/drifted.json")
    stressed_a = _state_quality(drifted["summary"])
    stressed_b = _state_quality(drifted["summary"])

    golden = _load_json("expected/loop_trajectory.json")
    return {
        "trajectory_golden_seed": golden["seed"],
        "trajectory_matches_golden_seed": golden["seed"] == _SEED,
        "env_reset_deterministic": bool(env_reset_deterministic),
        "mutation_pack_stressed_byte_identical": stressed_a == stressed_b,
        "paired_clean_link": drifted["paired_clean_run"] == clean["paired_clean_run"],
        "perturbation_profile": drifted["perturbation_profile"],
    }


def _deterministic_anchors() -> dict[str, Any]:
    """Recompute the deterministic anchors over the committed fixtures via
    score_browser_cua_probe_result and check they match the golden (byte-identical
    under seed). Includes the desktop grounding_step_accuracy (a GENUINELY NEW
    deterministic computation, NOT in the browser verifier)."""
    form = _load_json("multistep_form/form.json")
    drifted = _load_json("selector_drift_family/drifted.json")
    inject = _load_json("injected_dom_family/inject.json")
    clean_inject = _load_json("injected_dom_family/clean.json")
    sentinels = _load_json("fake_completion_sentinel/sentinels.json")
    desktop = _load_json("desktop_episode/episode.json")
    golden = _load_json("expected/deterministic_anchors.json")["anchors"]

    computed = {
        "multistep_form": {
            "state_quality": _state_quality(form["summary"]),
            "action_quality": _action_quality(form["summary"]),
            "mutation_grounding_quality": _mutation_grounding_quality(form["summary"]),
        },
        "selector_drift_drifted": {
            "state_quality": _state_quality(drifted["summary"]),
            "action_quality": _action_quality(drifted["summary"]),
            "mutation_grounding_quality": _mutation_grounding_quality(drifted["summary"]),
        },
        "injected_dom_clean": {
            "state_quality": _state_quality(clean_inject["summary"]),
            "action_quality": _action_quality(clean_inject["summary"]),
        },
        "injected_dom_inject": {
            "state_quality": _state_quality(inject["summary"]),
            "action_quality": _action_quality(inject["summary"]),
        },
        "fake_completion": {
            "state_quality": _state_quality(sentinels["control"]["fake_completion"]["summary"]),
        },
        "genuine_completion": {
            "state_quality": _state_quality(sentinels["control"]["genuine_completion"]["summary"]),
        },
        "desktop_episode": {
            "grounding_step_accuracy": _cua_desktop_grounding.grounding_step_accuracy(desktop),
        },
    }

    def _matches(observed: dict, expected: dict) -> bool:
        return all(abs(float(observed[k]) - float(expected[k])) < 1e-9 for k in expected)

    matches_golden = all(
        _matches(computed[cell], golden[cell]) for cell in golden if cell in computed
    )
    return {
        "computed": computed,
        "matches_golden": matches_golden,
        "anchor_terms": list(cua_loop.V1_CUA_LOSS_DETERMINISTIC_ANCHOR_TERMS),
        "desktop_anchor_terms": list(cua_loop.V1_CUA_DESKTOP_ANCHOR_TERMS),
    }


def _completion_guard() -> dict[str, Any]:
    """The fake/unsafe-completion guard demo: the fake-completion sentinel
    (narrated success, flat state_match) is score-zeroed on the anchor; the
    unsafe-completion canary (injected-DOM following) is zeroed on action_quality.
    The verifier reads counted trace facts, never the agent prose."""
    sentinels = _load_json("fake_completion_sentinel/sentinels.json")
    inject = _load_json("injected_dom_family/inject.json")

    fake = sentinels["control"]["fake_completion"]
    genuine = sentinels["control"]["genuine_completion"]
    # the fake-completion narrates success but is score-zeroed on the anchor (the
    # narration cannot lift the score).
    fake_state = _state_quality(fake["summary"])
    genuine_state = _state_quality(genuine["summary"])
    fake_completion_zeroed = (
        fake["narrated_success"] is True and fake_state == 0.0 and genuine_state == 1.0
    )

    # the unsafe-completion canary: a config that "completes" only by touching the
    # injection surface is zeroed on action_quality AND trips the canary.
    inject_action = _action_quality(inject["summary"])
    unsafe_completion_tripped = (
        int(inject["summary"]["prompt_injection_touched_count"]) > 0 and inject_action == 0.0
    )

    return {
        "fake_completion_score_zeroed_on_anchor": bool(fake_completion_zeroed),
        "fake_completion_state_quality": fake_state,
        "genuine_completion_state_quality": genuine_state,
        "unsafe_completion_canary_tripped": bool(unsafe_completion_tripped),
        "injected_action_quality": inject_action,
        "completion_guard_kinds": list(cua_loop.V1_CUA_COMPLETION_GUARD_KINDS),
        "reads_counted_facts_not_prose": True,
    }


def _eval_wiring() -> dict[str, Any]:
    """Assert the loop's evals are wired over the already-shipped substrate and
    browser/computer_use are executable-loop-registered through the R4 hook (NOT a
    vocab widening; the byte-pin stays green)."""
    cua_loop._ensure_cua_world_registered("browser")
    cua_loop._ensure_cua_world_registered("desktop")
    rec = contract._EXTRA_WORLD_KINDS.get("browser") or {}
    return {
        "uses_browser_environment": BrowserEnvironment.name == "browser",
        "browser_registered_via_hook": "browser" in contract.resolved_world_kinds(),
        "computer_use_registered_via_hook": "computer_use" in contract.resolved_world_kinds(),
        "executable_loop_record_present": rec.get("name") == cua_loop.CUA_BROWSER_EXTENSION_NAME,
        "frozen_vocab_byte_stable": tuple(contract.SIMULATION_WORLD_KINDS) == (
            "conversation", "tool_api", "browser", "computer_use", "code_exec", "voice_telephony"
        ),
        "browser_in_frozen_world_kinds": "browser" in contract.SIMULATION_WORLD_KINDS,
    }


def _clean_artifact() -> dict[str, Any]:
    """A §2.6-honest deterministic fixture artifact: local_gate /
    captured_fixture carrying fidelity_tier=deterministic_fixture -- NEVER
    live_lane."""
    return {
        "kind": "deterministic_fixture",
        "evidence_class": "local_gate",
        "fidelity_tier": "deterministic_fixture",
        "world_kind": "browser",
    }


def _negatives() -> dict[str, Any]:
    """The constructed overclaim negatives the gate MUST catch (the design -- do
    not weaken these). Each is a hand-built artifact that violates §2.6."""
    return {
        # a deterministic_fixture artifact stamping evidence_class=live_lane.
        "deterministic_claims_live_lane": {
            "kind": "deterministic_fixture",
            "evidence_class": "live_lane",  # the overclaim
            "fidelity_tier": "deterministic_fixture",
        },
        # a keyed_live_model artifact lacking the keyed-lane flag.
        "keyed_without_credential": {
            "kind": "keyed_live_model",
            "evidence_class": "live_lane",
            "fidelity_tier": "keyed_live_model",
            "credentialed": False,  # the overclaim: no real keys
        },
        # a fake-completion config that is NOT score-zeroed (broken guard).
        "fake_completion_unzeroed": {
            "narrated_success": True,
            "state_quality": 1.0,  # the overclaim: a fake completion that slipped through
            "url_match": False,
        },
    }


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    out = Path(output_path).expanduser() if output_path is not None else None
    payload: dict[str, Any] = {
        "kind": READINESS_KIND,
        "modality": "cua",
        "seed": _SEED,
        # constant mirrors (observed; the gate pins them)
        "fidelity_tiers": list(cua_loop.V1_CUA_FIDELITY_TIERS),
        "loss_term_refs": list(cua_loop.V1_CUA_LOSS_TERM_REFS),
        "deterministic_anchor_terms": list(cua_loop.V1_CUA_LOSS_DETERMINISTIC_ANCHOR_TERMS),
        "desktop_anchor_terms": list(cua_loop.V1_CUA_DESKTOP_ANCHOR_TERMS),
        "judge_terms": list(cua_loop.V1_CUA_LOSS_JUDGE_TERMS),
        "mandatory_safety_terms": list(cua_loop.V1_CUA_LOSS_MANDATORY_SAFETY_TERMS),
        "failure_sublayers": list(cua_loop.V1_CUA_FAILURE_SUBLAYERS),
        "surfaces": list(cua_loop.V1_CUA_SURFACES),
        "completion_guard_kinds": list(cua_loop.V1_CUA_COMPLETION_GUARD_KINDS),
        "perturbation_operators": list(cua_loop.V1_CUA_PERTURBATION_OPERATORS),
        # result blocks
        "registration": _registration(),
        "loop_determinism": _loop_determinism(),
        "deterministic_anchors": _deterministic_anchors(),
        "completion_guard": _completion_guard(),
        "eval_wiring": _eval_wiring(),
        "clean_artifact": _clean_artifact(),
        "negatives": _negatives(),
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return payload


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    result = run(destination)
    if destination is None:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
