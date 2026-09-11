"""Phase 9B unit 6 — the image / multimodal loop CLI front door.

Machinery tier: no extras, no flags, no network, no keys.
"""

from __future__ import annotations

import json
from pathlib import Path

from fi.alk.cli import main


def _manifest() -> dict:
    return {
        "name": "image-cli-demo",
        "base_agent": {"model": "gpt-4o"},
        "search_space": {
            "agent.model": ["gpt-4o", "claude"],
            "image.preprocess.resolution": [256, 512],
            "mmrag.retrieve_images": [True, False],
        },
        "objective": {
            "source": "declared",
            "evals": [
                {"eval": "task_success", "weight": 1.0, "direction": "maximize"},
                {"eval": "ocr_accuracy", "weight": 0.6, "direction": "maximize"},
            ],
            "guards": {
                "sentinel_rows": [{"id": "x", "kind": "perception_bypass"}],
                "min_guard_count": 1,
            },
        },
        "eval_budget": 4,
        "seed": 1142,
    }


def _write_manifest(tmp_path: Path) -> Path:
    p = tmp_path / "image_manifest.json"
    p.write_text(json.dumps(_manifest()), encoding="utf-8")
    return p


def test_cli_image_task_mode_default_understanding(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "out.json"
    code = main(["practice", "image", str(manifest), "-o", str(out), "--quiet"])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "ran"
    render = payload["image_render"]
    assert render["task_mode"] == "understanding"
    assert render["world_kind"] == "image"
    assert render["fidelity_tier"] == "deterministic_fixture"
    # NEVER a judge score on the credential-free path.
    assert "judge_score" not in render
    assert "judge" not in json.dumps(render).lower()


def test_cli_image_generation_refuses_without_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AGENT_LEARNING_IMAGE_JUDGE_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "out.json"
    code = main(
        ["practice", "image", str(manifest), "--task-mode", "generation",
         "-o", str(out), "--quiet"]
    )
    assert code == 0  # exit 0 + loud warning (withheld, not a fake number)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "withheld"
    types = [f["type"] for f in payload["findings"]]
    assert "image_judge_key_unavailable" in types


def test_cli_image_fixture_missing_finding(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    code = main(
        ["practice", "image", str(tmp_path / "nope.json"), "-o", str(out), "--quiet"]
    )
    assert code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    types = [f["type"] for f in payload["findings"]]
    assert "image_fixture_missing" in types


def test_cli_image_render_no_judge_on_credential_free_path(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "out.json"
    main(["practice", "image", str(manifest), "-o", str(out), "--quiet"])
    payload = json.loads(out.read_text(encoding="utf-8"))
    render = payload["image_render"]
    # only deterministic anchors + the fidelity marker, never a judge score.
    assert set(render["deterministic_anchor_terms"]) == {
        "task_success", "ocr_accuracy", "chart_accuracy", "artifact_grounding"
    }
    assert render["fidelity_tier"] == "deterministic_fixture"


def test_cli_image_judge_only_loss_refused(tmp_path: Path) -> None:
    m = _manifest()
    m["objective"]["evals"] = [
        {"eval": "instruction_adherence", "weight": 1.0},
        {"eval": "instruction_adherence", "weight": 0.5},
    ]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    out = tmp_path / "out.json"
    code = main(["practice", "image", str(p), "-o", str(out), "--quiet"])
    assert code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    types = [f["type"] for f in payload["findings"]]
    assert "image_mode_unavailable" in types
