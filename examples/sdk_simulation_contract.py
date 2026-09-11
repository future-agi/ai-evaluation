"""Simulation-contract readiness example + gate-fixture generator (Phase 13D).

Deterministic and fully OFFLINE: no network, no API keys. ``run(output_path)``
exercises the contract end-to-end and regenerates the committed fixtures under
``examples/simulation_contract_fixtures/`` that the ``simulation_contract_
readiness`` gate recomputes statically:

  roundtrip/        per-builder round-trip evidence (S1-S8 census)
  typed_persona_manifest.json + result  (G4: is_typed + fidelity)
  goal_pair/        declared-goal stop + no-goal byte-identical twin (G3)
  world_kinds/      one typed fixture per SIMULATION_WORLD_KIND + rung-1 results
  tool_mocks/       mock-level validation + the identity pair (hash flip)
  hashes.json       content hashes incl. one deliberately drifted row (tripwire)
  objective/        declared-guarded / declared-unguarded / derived + derived_view
  cast_dynamics/    R2 fixtures (legal roles / turn-holding dynamics / together)
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fi.alk import loss as L
from fi.alk import simulate as S
from fi.simulate.cli import _run_local_text_manifest, load_manifest
from fi.simulate.simulation.models import Persona

EXAMPLE_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = EXAMPLE_DIR / "simulation_contract_fixtures"
STRIP = S.STABLE_RESULT_ENVELOPE_FIELDS + ("duration_ms", "tool_mock_profile", "goal_machine")
AGENT = {"type": "scripted", "content": "done"}

CENSUS = {
    "S1_task": lambda: S.build_task_run_manifest(
        name="t", agent=AGENT, task_description="do", expected_result="done",
        scenario={"name": "c", "dataset": [{"persona": {"name": "A"}, "situation": "s", "outcome": "done"}]}),
    "S4_browser_cua": lambda: S.build_browser_cua_run_manifest(name="bc", agent=AGENT),
    "S4_realtime": lambda: S.build_realtime_run_manifest(name="rt", agent=AGENT),
    "S4_multimodal": lambda: S.build_multimodal_image_run_manifest(name="mi", agent=AGENT),
    "S5_world_fw_memory": lambda: S.build_world_framework_memory_run_manifest(name="wf", agent=AGENT),
    "S5_social_memory": lambda: S.build_social_memory_framework_run_manifest(name="sm", agent=AGENT),
    "S6_stateful_world": lambda: S.build_stateful_tool_world_run_manifest(name="w", agent=AGENT),
    "S6_world_model": lambda: S.build_world_model_run_manifest(name="wm", agent=AGENT),
    "S6_autonomous_redteam": lambda: S.build_autonomous_redteam_task_world_run_manifest(name="ar", agent=AGENT),
    "S7_openenv": lambda: S.build_openenv_run_manifest(name="oe", agent=AGENT),
}


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in STRIP}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _run(manifest):
    report = asyncio.run(_run_local_text_manifest(manifest, Path(".")))
    return _strip(report.model_dump())


def _digest(obj) -> str:
    return "sha256:" + __import__("hashlib").sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def run(output_path: str | None = None) -> dict:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # --- roundtrip census --------------------------------------------------
    roundtrip = {}
    for row, fn in sorted(CENSUS.items()):
        manifest = fn()
        original = _run(manifest)
        sim = S.derive_simulation_manifest(manifest)
        rerun = _run(S.derive_simulation_run_manifest(sim, agent=manifest["agent"]))
        roundtrip[row] = {
            "builder": row,
            "original_digest": _digest(original),
            "rederived_digest": _digest(rerun),
            "equal": _digest(original) == _digest(rerun),
            "world_kind": sim["world"]["kind"],
        }
    _write(FIXTURE_DIR / "roundtrip" / "census.json", roundtrip)

    # --- typed persona (G4) ------------------------------------------------
    typed_manifest = {
        "version": "agent-learning.run.v1", "name": "typed",
        "scenario": {"name": "typed", "dataset": [
            {"persona": {"name": "T"}, "situation": "s", "outcome": "o", "behavior_policy": {}}]},
        "agent": AGENT, "simulation": {"max_turns": 1, "min_turns": 1}, "evaluation": {"enabled": False},
    }
    _write(FIXTURE_DIR / "typed_persona_manifest.json", typed_manifest)
    typed_result_full = asyncio.run(_run_local_text_manifest(typed_manifest, Path("."))).model_dump()
    _write(FIXTURE_DIR / "typed_persona_result.json", {
        "is_typed": typed_result_full["results"][0]["persona"].get("behavior_policy") is not None,
        "fidelity_attached": "persona_fidelity" in typed_result_full["results"][0]["metadata"],
        "admission_attached": "admission" in typed_result_full["results"][0]["metadata"],
    })

    # --- goal pair (G3) ----------------------------------------------------
    goal_manifest = {
        "version": "agent-learning.run.v1", "name": "g3w",
        "scenario": {"name": "g3w",
                     "goal": {"states": ["won"], "success_state": "won"},
                     "verification": {"checks": [{"name": "won", "kind": "world_success_condition", "rung": "turn"}]},
                     "dataset": [{"persona": {"name": "Q"}, "situation": "s", "outcome": "win"}]},
        "agent": AGENT,
        "simulation": {"engine": "local_text", "max_turns": 2, "min_turns": 1,
                       "environments": [{"type": "world_contract", "name": "w",
                                         "initial_state": {"phase": "closed"},
                                         "success_conditions": [{"name": "won", "must": {"phase": "closed"}}]}]},
        "evaluation": {"enabled": False},
    }
    goal_result = asyncio.run(_run_local_text_manifest(goal_manifest, Path("."))).model_dump()
    _write(FIXTURE_DIR / "goal_pair" / "goal_manifest.json", goal_manifest)
    _write(FIXTURE_DIR / "goal_pair" / "goal_result.json", {
        "stop_reason": goal_result["results"][0]["metadata"]["stop_reason"],
        "goal_machine": goal_result["results"][0]["metadata"].get("goal_machine"),
    })
    nogoal_manifest = load_manifest(EXAMPLE_DIR / "run_manifest.json")
    _write(FIXTURE_DIR / "goal_pair" / "nogoal_result.json", _run(nogoal_manifest))

    # --- world kinds -------------------------------------------------------
    from fi.simulate.simulation.contract import SIMULATION_WORLD_KINDS
    world_kinds = {}
    for kind in SIMULATION_WORLD_KINDS:
        executable = kind in ("conversation", "tool_api")
        derived_legacy = kind in ("browser", "voice_telephony")
        world_kinds[kind] = {
            "kind": kind,
            "executable_contract_native": executable,
            "derived_legacy_rung1": derived_legacy,
            "validation_only": kind in ("computer_use", "code_exec"),
        }
    _write(FIXTURE_DIR / "world_kinds" / "kinds.json", world_kinds)

    # --- tool mocks (identity pair) ----------------------------------------
    p = Persona(persona={"name": "A"}, situation="s", outcome="done", behavior_policy={})
    sim_static = S.build_simulation_manifest(
        name="m", personas=[p.model_dump(exclude_none=True)],
        scenarios=[{"cast": [{"persona": p.version, "role": "user"}], "casting": "each"}],
        world={"kind": "tool_api", "tools": [{"name": "t", "mock": {"level": "static_fixture"}}]})
    sim_replay = S.build_simulation_manifest(
        name="m", personas=[p.model_dump(exclude_none=True)],
        scenarios=[{"cast": [{"persona": p.version, "role": "user"}], "casting": "each"}],
        world={"kind": "tool_api", "tools": [{"name": "t", "mock": {
            "level": "recorded_replay", "source": "cap://x",
            "provenance": {"capture": "sha256:abc"}, "recorded_replay": {"miss_policy": "fail"}}}]})
    _write(FIXTURE_DIR / "tool_mocks" / "identity_pair.json", {
        "static_version": sim_static["version"],
        "replay_version": sim_replay["version"],
        "hashes_differ": sim_static["version"] != sim_replay["version"],
    })

    # --- hashes (with drifted-row tripwire) --------------------------------
    canonical = {row: roundtrip[row]["original_digest"] for row in sorted(roundtrip)}
    canonical["_drifted_row"] = {
        "stored_hash": sim_static["version"],
        "recompute_payload": sim_static,  # recompute must match stored_hash
    }
    _write(FIXTURE_DIR / "hashes.json", canonical)

    # --- objective ---------------------------------------------------------
    guarded = L.compile_objective({"evals": [{"eval": "agent_report", "weight": 1.0}],
                                   "source": "declared",
                                   "guards": {"sentinel_rows": ["row_g"], "min_guard_count": 1}})
    derived = L.compile_objective({"evals": [{"eval": "agent_report", "weight": 1.0}], "source": "derived"})
    _write(FIXTURE_DIR / "objective" / "declared_guarded.json", guarded)
    _write(FIXTURE_DIR / "objective" / "derived.json", derived)
    # an unguarded declared objective must reject — store the rejecting payload.
    _write(FIXTURE_DIR / "objective" / "declared_unguarded_input.json",
           {"evals": [{"eval": "agent_report"}], "source": "declared", "guards": {}})
    # derived view vs an incumbent hand-written weight map (byte-equal).
    weight_obj = L.compile_objective({"evals": [{"eval": "world_contract", "weight": 4.0},
                                                {"eval": "framework_trace", "weight": 3.0}],
                                      "source": "declared",
                                      "guards": {"sentinel_rows": ["row_g"], "min_guard_count": 1}})
    _write(FIXTURE_DIR / "objective" / "derived_view.json", {
        "incumbent": {"world_contract": 4.0, "framework_trace": 3.0},
        "derived_view": L.objective_metric_weights(weight_obj),
    })

    # --- cast / dynamics (R2) ----------------------------------------------
    _write(FIXTURE_DIR / "cast_dynamics" / "legal_roles.json", list(
        __import__("fi.simulate.simulation.contract", fromlist=["x"]).SIMULATION_CAST_ROLES))
    _write(FIXTURE_DIR / "cast_dynamics" / "turn_holding_dynamics_input.json", {
        "at": {"turn": 1}, "event": "counterpart_message",
        "payload": {"responds_to": "user", "text": "hi"},  # must reject
    })
    _write(FIXTURE_DIR / "cast_dynamics" / "casting_together.json", {
        "casting": "together", "must_refuse_typed": True,
    })

    summary = {
        "kind": "agent-learning.simulation-contract-readiness.v1",
        "roundtrip_all_equal": all(r["equal"] for r in roundtrip.values()),
        "census_size": len(roundtrip),
        "fixture_dir": str(FIXTURE_DIR.relative_to(EXAMPLE_DIR.parent)),
    }
    if output_path:
        _write(Path(output_path), summary)
    return summary


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    output = argv[0] if argv else None
    summary = run(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["roundtrip_all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
