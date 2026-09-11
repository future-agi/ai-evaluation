"""Phase 9C unit 6 — the CUA / browser / computer-use loop CLI front door.

Machinery tier: no extras, no flags, no network, no keys.
"""

from __future__ import annotations

import json
from pathlib import Path

from fi.alk.cli import main


def _manifest() -> dict:
    return {
        "name": "cua-cli-demo",
        "base_agent": {"model": "gpt-4o"},
        "search_space": {
            "agent.model": ["gpt-4o", "claude"],
            "agent.grounding.mode": ["element-id", "selector"],
            "agent.observe.channel": ["screenshot", "DOM"],
        },
        "objective": {
            "source": "declared",
            "evals": [
                {"eval": "task_success", "weight": 1.0, "direction": "maximize"},
                {"eval": "state_match", "weight": 0.9, "direction": "maximize"},
            ],
            "guards": {
                "sentinel_rows": [{"id": "x", "kind": "fake_completion"}],
                "min_guard_count": 1,
            },
        },
        "eval_budget": 4,
        "seed": 1142,
    }


def _write_manifest(tmp_path: Path) -> Path:
    p = tmp_path / "cua_manifest.json"
    p.write_text(json.dumps(_manifest()), encoding="utf-8")
    return p


def test_cli_cua_surface_default_browser(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "out.json"
    code = main(["practice", "cua", str(manifest), "-o", str(out), "--quiet"])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "ran"
    render = payload["cua_render"]
    assert render["cua_surface"] == "browser"
    assert render["world_kind"] == "browser"
    assert render["fidelity_tier"] == "deterministic_fixture"
    # NEVER a judge score on the credential-free path.
    assert "judge_score" not in render
    assert "judge" not in json.dumps(render).lower()


def test_cli_cua_desktop_infra_refuses(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AGENT_LEARNING_CUA_DESKTOP_VM", raising=False)
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "out.json"
    code = main(
        ["practice", "cua", str(manifest), "--cua-surface", "desktop",
         "-o", str(out), "--quiet"]
    )
    assert code == 0  # exit 0 + loud warning (withheld, not a fake number)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "withheld"
    types = [f["type"] for f in payload["findings"]]
    assert "cua_desktop_infra_unavailable" in types


def test_cli_cua_judge_refuses_without_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AGENT_LEARNING_CUA_JUDGE_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    m = _manifest()
    m["objective"]["evals"].append({"eval": "completion_judge", "weight": 0.3})
    p = tmp_path / "judge.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    out = tmp_path / "out.json"
    code = main(["practice", "cua", str(p), "-o", str(out), "--quiet"])
    assert code == 0  # exit 0 + loud warning
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "withheld"
    types = [f["type"] for f in payload["findings"]]
    assert "cua_judge_key_unavailable" in types


def test_cli_cua_fixture_missing_finding(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    code = main(
        ["practice", "cua", str(tmp_path / "nope.json"), "-o", str(out), "--quiet"]
    )
    assert code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    types = [f["type"] for f in payload["findings"]]
    assert "cua_fixture_missing" in types


def test_cli_cua_render_no_judge_on_credential_free_path(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "out.json"
    main(["practice", "cua", str(manifest), "-o", str(out), "--quiet"])
    payload = json.loads(out.read_text(encoding="utf-8"))
    render = payload["cua_render"]
    # only deterministic anchors + the fidelity marker, never a judge score.
    assert set(render["deterministic_anchor_terms"]) == {"task_success", "state_match"}
    assert render["fidelity_tier"] == "deterministic_fixture"


def test_cli_cua_judge_only_loss_refused(tmp_path: Path, monkeypatch) -> None:
    # set a judge key so the keyed-lane withholding does NOT fire first; the
    # compile-time judge-only rejection is what we are asserting here.
    monkeypatch.setenv("AGENT_LEARNING_CUA_JUDGE_KEY", "test-key")
    m = _manifest()
    m["objective"]["evals"] = [
        {"eval": "completion_judge", "weight": 1.0},
        {"eval": "completion_judge", "weight": 0.5},
    ]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    out = tmp_path / "out.json"
    code = main(["practice", "cua", str(p), "-o", str(out), "--quiet"])
    assert code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    types = [f["type"] for f in payload["findings"]]
    assert "cua_surface_unavailable" in types
