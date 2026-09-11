"""Units 15-16 (BBG U15/U16) — the simulation + practice CLI families."""
from __future__ import annotations

import json


from fi.alk import cli
from fi.alk import loss as L
from fi.alk import simulate as S


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def _run_manifest(tmp_path):
    m = S.build_task_run_manifest(
        name="cli", agent={"type": "scripted", "content": "done"},
        task_description="do", expected_result="done",
        scenario={"name": "cli", "dataset": [{"persona": {"name": "A"}, "situation": "s", "outcome": "done"}]},
    )
    return _write(tmp_path, "run.json", m)


# --- simulation family -----------------------------------------------------
def test_simulation_validate_clean(tmp_path, capsys):
    path = _run_manifest(tmp_path)
    # build a valid simulation manifest
    sim = S.derive_simulation_manifest(json.loads(open(path).read()))
    spath = _write(tmp_path, "sim.json", sim)
    rc = cli.main(["simulation", "validate", spath])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["status"] == "valid"


def test_simulation_validate_invalid(tmp_path, capsys):
    bad = {"kind": "agent-learning.simulation.v1", "name": "x",
           "scenarios": [{"cast": [{"persona": "sha256:nope"}]}], "world": {"kind": "conversation"}}
    spath = _write(tmp_path, "bad.json", bad)
    rc = cli.main(["simulation", "validate", spath])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["findings"][0]["type"] == "simulation_contract_invalid"


def test_simulation_lift(tmp_path, capsys):
    path = _run_manifest(tmp_path)
    rc = cli.main(["simulation", "lift", path])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["status"] == "lifted"
    assert out["simulation"]["kind"] == "agent-learning.simulation.v1"
    assert any(f["type"] == "simulation_auto_lifted" for f in out["findings"])


def test_simulation_run(tmp_path, capsys):
    path = _run_manifest(tmp_path)
    sim = S.derive_simulation_manifest(json.loads(open(path).read()))
    spath = _write(tmp_path, "sim.json", sim)
    rc = cli.main(["simulation", "run", spath])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["status"] == "ran"
    assert out["report"]["results"]


def test_simulation_run_refusal_world_kind(tmp_path, capsys):
    # a code_exec simulation refuses contract-native
    p = {"persona": {"name": "A"}, "situation": "s", "outcome": "o", "behavior_policy": {}}
    from fi.simulate.simulation.models import Persona
    ph = Persona(**p).version
    sim = S.build_simulation_manifest(
        name="ce", personas=[p],
        scenarios=[{"cast": [{"persona": ph, "role": "user"}], "casting": "each"}],
        world={"kind": "code_exec"},
    )
    spath = _write(tmp_path, "ce.json", sim)
    rc = cli.main(["simulation", "run", spath])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["findings"][0]["type"] == "world_kind_refusal"


def test_simulation_quiet(tmp_path, capsys):
    path = _run_manifest(tmp_path)
    rc = cli.main(["simulation", "lift", path, "--quiet"])
    assert rc == 0
    assert capsys.readouterr().out == ""


# --- practice family -------------------------------------------------------
def _practice_manifest(tmp_path):
    from fi.alk.optimize import build_practice_loop_manifest
    obj = L.compile_objective({"evals": [{"eval": "agent_report", "weight": 1.0}], "source": "declared",
                              "guards": {"sentinel_rows": ["row_g"], "min_guard_count": 1}})
    sim = {"version": "sha256:v", "inline": {"kind": "agent-learning.simulation.v1", "name": "s",
            "version": "sha256:v", "world": {"kind": "conversation"}, "scenarios": [{"cast": []}],
            "objective": obj}}
    m = build_practice_loop_manifest(name="pl", simulation=sim, base_agent={"provider": "custom", "instructions": "x"},
                                     search_space={"agent.instructions": ["a"]}, eval_budget=20, seed=7, max_rounds=1)
    m["store"] = {"path": str(tmp_path / "records.jsonl")}
    m["practice"]["store"] = {"path": str(tmp_path / "records.jsonl"), "active_cap": 64}
    return _write(tmp_path, "pl.json", m)


def test_practice_run(tmp_path, capsys):
    path = _practice_manifest(tmp_path)
    rc = cli.main(["practice", "run", path])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["status"] == "ran"


def test_practice_ladder_missing_store_refuses(tmp_path, capsys):
    rc = cli.main(["practice", "ladder", "--store", str(tmp_path / "nope.jsonl")])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["findings"][0]["type"] == "consolidation_store_missing"


def test_practice_report_pure_reader(tmp_path, capsys):
    art = _write(tmp_path, "art.json", {"kind": "agent-learning.practice-report.v1", "round": 0})
    rc = cli.main(["practice", "report", art])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["kind"] == "agent-learning.practice-report.v1"


def test_no_train_string_while_gate_red():
    """doctrine #13: no CLI string matches \\btrain(ing|er|ed|s)? while red."""
    import inspect
    import re
    src = inspect.getsource(cli._practice) + inspect.getsource(cli._simulation)
    # the CLI help/payload strings must not contain "train*"
    assert not re.search(r"\btrain(?:ing|er|ed|s)?\b", src, re.IGNORECASE)
