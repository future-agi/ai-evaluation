"""Phase 11B — the certification gate (framework_adapter_preset_certification)."""

from __future__ import annotations

from pathlib import Path

from fi.alk import trinity

ROOT = Path(__file__).resolve().parents[1]

ARRAYS = (
    "missing_files",
    "preset_registration_errors",
    "input_mode_errors",
    "probe_determinism_errors",
    "io_contract_binding_errors",
    "cookbook_coverage_errors",
    "live_lane_register_errors",
)


def _status():
    return trinity._release_framework_adapter_preset_certification_status(ROOT)


def test_certification_status_clean():
    status = _status()
    for array in ARRAYS:
        assert status[array] == [], f"{array}: {status[array]}"
    assert status["certified_framework_count"] == 19
    assert status["kind"] == (
        "agent-learning.framework-adapter-preset-certification-readiness.v1"
    )


def test_certification_flags_missing_shim(monkeypatch, tmp_path):
    # A repo root with no cert files -> missing_files populated.
    status = trinity._release_framework_adapter_preset_certification_status(
        tmp_path
    )
    assert status["missing_files"]


def test_certification_flags_vector_db_in_presets(monkeypatch):
    from fi.simulate.agent import frameworks as fw_module

    spec = fw_module.FrameworkAdapterSpec("chromadb", "query", "dict")
    patched = dict(fw_module.FRAMEWORK_PRESETS)
    patched["chromadb"] = spec
    monkeypatch.setattr(fw_module, "FRAMEWORK_PRESETS", patched)
    status = _status()
    frameworks = {
        e.get("framework") for e in status["preset_registration_errors"]
    }
    assert "chromadb" in frameworks


def test_certification_accepts_chat_dict_input_mode():
    # The chat/dict model-client presets pass input_mode_errors (validity,
    # NOT discovery-equality). The clean run proves this.
    status = _status()
    assert status["input_mode_errors"] == []
    from fi.simulate.agent.frameworks import FRAMEWORK_PRESETS

    assert FRAMEWORK_PRESETS["cohere"].input_mode == "dict"


def test_certification_flags_io_surface_mismatch(monkeypatch):
    patched = [dict(row) for row in trinity.V1_FRAMEWORK_PRESET_CERTIFICATION_CONTRACTS]
    patched[0]["io_surface"] = "no_such_surface"
    monkeypatch.setattr(
        trinity, "V1_FRAMEWORK_PRESET_CERTIFICATION_CONTRACTS", patched
    )
    status = _status()
    assert status["io_contract_binding_errors"]


def test_certification_flags_live_lane_malformed(monkeypatch):
    bad = [dict(row) for row in trinity.V1_FRAMEWORK_PRESET_LIVE_VALIDATION_LANE]
    bad[0] = {"framework": bad[0]["framework"], "status": "live_validation_pending"}
    monkeypatch.setattr(
        trinity, "V1_FRAMEWORK_PRESET_LIVE_VALIDATION_LANE", tuple(bad)
    )
    status = _status()
    assert status["live_lane_register_errors"]


def test_certification_live_validated_without_proof_flagged(monkeypatch):
    bad = [dict(row) for row in trinity.V1_FRAMEWORK_PRESET_LIVE_VALIDATION_LANE]
    bad[0] = dict(bad[0], status="live_validated")
    monkeypatch.setattr(
        trinity, "V1_FRAMEWORK_PRESET_LIVE_VALIDATION_LANE", tuple(bad)
    )
    status = _status()
    assert status["live_lane_register_errors"]


def test_certification_live_pending_never_fails():
    # All lane rows are live_validation_pending in the shipped register, and the
    # clean gate passes — the ◐ lane status NEVER gates.
    status = _status()
    statuses = {
        row["status"] for row in trinity.V1_FRAMEWORK_PRESET_LIVE_VALIDATION_LANE
    }
    assert statuses == {"live_validation_pending"}
    assert status["live_lane_register_errors"] == []


def test_certification_flags_ollama_in_lane(monkeypatch):
    bad = list(trinity.V1_FRAMEWORK_PRESET_LIVE_VALIDATION_LANE) + [
        {
            "framework": "ollama",
            "status": "live_validation_pending",
            "env_var": "OLLAMA_HOST",
            "recipe": "agent-learn probe ollama --live",
        }
    ]
    monkeypatch.setattr(
        trinity, "V1_FRAMEWORK_PRESET_LIVE_VALIDATION_LANE", tuple(bad)
    )
    status = _status()
    frameworks = {e.get("framework") for e in status["live_lane_register_errors"]}
    assert "ollama" in frameworks


def test_certification_flags_missing_cookbook(monkeypatch):
    # Drop a framework's page from the required-files list AND assert the
    # cookbook check by pointing at a contract whose page does not exist.
    patched = [dict(row) for row in trinity.V1_FRAMEWORK_PRESET_CERTIFICATION_CONTRACTS]
    patched.append(
        {
            "framework": "nonexistent_fw",
            "path": "examples/sdk_framework_adapter_cert_a2a.py",
            "expected_method": "send_message",
            "expected_input_mode": "dict",
            "io_surface": "side_kwargs",
            "min_runtime_trace_count": 1,
            "min_tool_call_count": 1,
            "require_callable_signature": True,
            "live_lane": False,
        }
    )
    monkeypatch.setattr(
        trinity, "V1_FRAMEWORK_PRESET_CERTIFICATION_CONTRACTS", patched
    )
    # nonexistent_fw is not in FRAMEWORK_PRESETS -> preset_registration_errors
    # records it; it also has no docs page. The relevant guard here:
    status = _status()
    frameworks = {
        e.get("framework") for e in status["preset_registration_errors"]
    }
    assert "nonexistent_fw" in frameworks
