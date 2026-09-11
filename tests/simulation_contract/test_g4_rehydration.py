"""Unit 1 (BBG U1 / ARCH §1.9 G4) — the manifest path re-hydrates ALL
Persona/Scenario typed fields. Regression-first: legacy untyped manifests must
construct byte-identical results; typed manifests must round-trip their layers."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from fi.simulate.cli import _build_scenario, _run_local_text_manifest, load_manifest
from fi.simulate.manifest import ManifestError
from fi.simulate.simulation.models import ScenarioGoal, VerificationSpec

FIXTURES = Path(__file__).parent / "fixtures"
STRIP = ("created_at", "started_at", "completed_at", "duration_s", "timing", "duration_ms")


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in STRIP}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _run(manifest, path):
    report = asyncio.run(_run_local_text_manifest(manifest, path))
    payload = report.model_dump() if hasattr(report, "model_dump") else report
    return _strip(payload)


def test_g4_legacy_manifest_unchanged():
    """The 'old manifests still work' proof: byte-identical to the committed
    pre/post baseline (the legacy-default expressions are byte-identical)."""
    path = Path("examples/run_manifest.json")
    manifest = load_manifest(path)
    observed = _run(manifest, path)
    baseline = json.loads((FIXTURES / "g4_baseline_result.json").read_text())
    assert json.dumps(observed, sort_keys=True) == json.dumps(baseline, sort_keys=True)


def test_g4_typed_persona_rehydrates():
    """A dataset row carrying behavior_policy re-hydrates: is_typed True and the
    engine fires attach_fidelity (persona_fidelity + admission metadata)."""
    manifest = {
        "version": "agent-learning.run.v1",
        "name": "g4-typed",
        "scenario": {
            "name": "g4-typed",
            "dataset": [
                {
                    "persona": {"name": "Tess"},
                    "situation": "Tess needs a typed persona.",
                    "outcome": "The typed layer survives the manifest path.",
                    "behavior_policy": {},
                }
            ],
        },
        "agent": {"type": "scripted", "content": "The typed layer survives the manifest path."},
        "simulation": {"engine": "local_text", "max_turns": 1, "min_turns": 1},
        "evaluation": {"enabled": False},
    }
    scenario = _build_scenario(manifest)
    assert scenario.dataset[0].is_typed is True
    result = _run(manifest, Path("."))
    meta = result["results"][0]["metadata"]
    assert "persona_fidelity" in meta
    assert "admission" in meta


def test_g4_typed_scenario_rehydrates():
    """A scenario block with goal/verification re-hydrates into the typed models."""
    manifest = {
        "version": "agent-learning.run.v1",
        "name": "g4-scenario",
        "scenario": {
            "name": "g4-scenario",
            "kind": "task",
            "goal": {"states": ["resolved"], "success_state": "resolved"},
            "verification": {
                "checks": [{"name": "resolved", "kind": "keyword_fallback", "rung": "turn"}],
                "threshold": 0.7,
            },
            "coverage": {"intents": ["resolve"]},
            "dataset": [
                {
                    "persona": {"name": "Ravi"},
                    "situation": "Ravi files a ticket.",
                    "outcome": "ticket resolved",
                }
            ],
        },
        "agent": {"type": "scripted", "content": "ticket resolved"},
        "simulation": {"engine": "local_text", "max_turns": 1, "min_turns": 1},
        "evaluation": {"enabled": False},
    }
    scenario = _build_scenario(manifest)
    assert isinstance(scenario.goal, ScenarioGoal)
    assert scenario.goal.success_state == "resolved"
    assert isinstance(scenario.verification, VerificationSpec)


def test_g4_invalid_row_names_index():
    """A malformed typed layer raises ManifestError naming the offending row."""
    manifest = {
        "version": "agent-learning.run.v1",
        "name": "g4-bad",
        "scenario": {
            "name": "g4-bad",
            "dataset": [
                {"persona": {"name": "ok"}, "situation": "s", "outcome": "o"},
                {
                    "persona": {"name": "bad"},
                    "situation": "s",
                    "outcome": "o",
                    "behavior_policy": {"disclosure_policy": "not-a-float"},
                },
            ],
        },
        "agent": {"type": "scripted", "content": "x"},
        "simulation": {"engine": "local_text"},
        "evaluation": {"enabled": False},
    }
    with pytest.raises(ManifestError) as exc:
        _build_scenario(manifest)
    assert "scenario.dataset[2]" in str(exc.value)
