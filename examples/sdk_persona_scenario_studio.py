"""Persona & Scenario Studio readiness example (Phase 7, gate #71).

Runs ENTIRELY on the committed ``examples/persona_library/`` fixtures — zero
network, zero API keys. ``run(output_path)`` returns the full evidence payload
the ``persona_scenario_studio_readiness`` gate audits field-by-field, and also
writes it to ``output_path``. The sequence mirrors the studio lifecycle:

    typed round-trip + legacy upgrade -> behavior-policy compile + per-axis
    realization -> fidelity verdicts (clean pass / drifted quarantined /
    over-acted naturalness-failed) -> calibration lifecycle -> library write +
    coverage + residual -> bias lint (stereotyped fails, clean passes) ->
    vendor import byte-exact parity -> download validation refusals -> a pure
    persona-conditioned red-team manifest (never executed here; the EXECUTABLE
    persona-conditioned campaign evidence lives in the certification example).

No class ever claims population representativeness (2602.18462 hard limit).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import redteam, studio

# Engine-side closed vocabularies — reported as OBSERVED values so the gate can
# pin them to the trinity V1_* constants (import-free duplication per gate
# convention; the example reads the substrate, the gate holds the canon).
from fi.simulate.simulation.behavior_policy import (
    PERSONA_BEHAVIOR_AXES,
    PERSONA_BEHAVIOR_REALIZATION_METRICS,
)
from fi.simulate.simulation.fidelity import (
    PERSONA_FIDELITY_EPIDEMIC_RATE,
    PERSONA_FIDELITY_FLOORS,
    PERSONA_FIDELITY_VERDICTS,
)
from fi.simulate.simulation.models import (
    PERSONA_EVIDENCE_CLASSES,
    PERSONA_TEMPERAMENT_AXES,
    SCENARIO_KINDS,
    Persona,
    Scenario,
    TestCaseResult,
    TestReport,
)
from fi.alk.studio._bias import PERSONA_BIAS_LINT_CHECKS
from fi.alk.studio._calibration import (
    PERSONA_CALIBRATION_PROBES,
    PERSONA_CALIBRATION_STAGES,
)
from fi.alk.studio._coverage import (
    COVERAGE_FORBIDDEN_HEADLINE_KEYS,
    SCENARIO_COVERAGE_AXES,
)
from fi.alk.studio._download import (
    PERSONA_DOWNLOAD_PIN_FIELDS,
    validate_download,
    verify_pin,
)
from fi.alk.studio._library import load_persona
from fi.alk.studio._scan import CONTENT_SCAN_RESULTS, DownloadRejected
from fi.alk.studio._vendor import PERSONA_VENDOR_IMPORT_FORMATS

EXAMPLE_DIR = Path(__file__).resolve().parent
FIXTURES = EXAMPLE_DIR / "persona_library"
READINESS_KIND = "agent-learning.persona-scenario-studio-readiness.v1"


def _load_json(rel: str) -> Any:
    return json.loads((FIXTURES / rel).read_text(encoding="utf-8"))


def _class_contract() -> dict[str, Any]:
    subject = Persona(**_load_json("personas/subject.json"))
    # typed round-trip: dump -> rehydrate -> the content address is stable
    rehydrated = Persona(**subject.model_dump(exclude_none=True))
    typed_roundtrip_stable = (
        subject.is_typed
        and rehydrated.is_typed
        and rehydrated.version == subject.version
        and rehydrated.content_hash() == subject.content_hash()
    )

    legacy_row = _load_json("personas/legacy_row.json")
    upgraded = studio.upgrade_legacy_persona(legacy_row)
    legacy_keys_preserved = all(
        upgraded.persona.get(key) == value
        for key, value in legacy_row["persona"].items()
    )
    legacy_evidence = upgraded.provenance.evidence_class

    scenario = Scenario(**_load_json("scenarios/adversarial.json"))
    scenario_roundtrip_stable = (
        Scenario(**scenario.model_dump(exclude_none=True)).version == scenario.version
    )

    adversarial_requires_arc = False
    try:
        Scenario(name="bad", dataset=[subject.model_dump(exclude_none=True)],
                 kind="adversarial", attack_type="prompt_injection",
                 attack_surface="tool")  # missing escalation arc
    except Exception:
        adversarial_requires_arc = True

    return {
        "typed_roundtrip_stable": typed_roundtrip_stable,
        "legacy_upgraded": upgraded.is_typed is False,
        "legacy_evidence_class": legacy_evidence,
        "legacy_keys_preserved": legacy_keys_preserved,
        "hash_stable": subject.content_hash() == subject.content_hash(),
        "scenario_roundtrip_stable": scenario_roundtrip_stable,
        "adversarial_requires_arc": adversarial_requires_arc,
    }


def _fidelity() -> dict[str, Any]:
    subject = Persona(**_load_json("personas/subject.json"))
    scenario = Scenario(**_load_json("scenarios/adversarial.json"))

    def record_for(name: str) -> dict[str, Any]:
        messages = _load_json(f"transcripts/{name}.json")
        result = TestCaseResult(persona=subject, transcript="", messages=messages)
        studio.attach_fidelity(result, subject, scenario)
        return result

    clean = record_for("clean")
    drifted = record_for("drifted")
    over = record_for("over_acted")
    report = TestReport(results=[clean, drifted, over])
    record_fields = sorted(clean.metadata["persona_fidelity"])

    def view(result: TestCaseResult) -> dict[str, Any]:
        record = result.metadata["persona_fidelity"]
        return {
            "verdict": record["verdict"],
            "admission": result.metadata["admission"],
            "caricature_index": record["naturalness"]["caricature_index"],
            "naturalness": record["naturalness"]["score"],
            "trajectory_len": len(record["drift_trajectory"]),
        }

    user_turns = sum(1 for m in _load_json("transcripts/clean.json") if m["role"] == "user")
    return {
        "record_fields": record_fields,
        "verdicts_seen": sorted(
            {r.metadata["persona_fidelity"]["verdict"] for r in report.results}
        ),
        "clean": view(clean),
        "drifted": view(drifted),
        "over_acted": view(over),
        "admissible_count": len(report.admissible_results()),
        "inconclusive_count": len(report.inconclusive_results()),
        "clean_user_turn_count": user_turns,
        "epidemic_rate": PERSONA_FIDELITY_EPIDEMIC_RATE,
    }


def _calibration(library: Path) -> dict[str, Any]:
    calibratable = Persona(**_load_json("personas/calibratable.json"))
    drift_seed = Persona(**_load_json("personas/drift_seed.json"))
    task = Scenario(**_load_json("scenarios/task.json"))

    ok = studio.calibrate_persona(
        calibratable, library=library, target_class="schema_sampled", scenario=task
    )
    # the seeded-drift fixture forks on REPLAY (retest_jitter): internal/external
    # green, the retest leg red — calibrated WITHOUT a constraining scenario so
    # the divergence is isolated to the replay retest (PRD §4.2 / §9.6 #5).
    red = studio.calibrate_persona(
        drift_seed, library=library, target_class="schema_sampled"
    )
    return {
        "stages": list(ok.get("stages", [])),
        "probes": sorted(ok.get("probes", {})),
        "calibratable": {
            "status": ok.get("status"),
            "verdict": ok.get("verdict"),
            "failed_probe": ok.get("failed_probe"),
            "evidence_class": ok.get("evidence_class"),
            "calibration_ref": ok.get("calibration_ref"),
            "kind": ok.get("kind"),
        },
        "drift_seed": {
            "status": red.get("status"),
            "verdict": red.get("verdict"),
            "failed_probe": red.get("failed_probe"),
            "evidence_class": red.get("evidence_class"),
        },
        "uncalibrated_class": drift_seed.provenance.evidence_class,
    }


def _coverage() -> dict[str, Any]:
    scenarios = [
        Scenario(**_load_json(f"scenarios/coverage_{n}.json")) for n in ("a", "b", "c")
    ]
    report = studio.coverage_report(scenarios)
    axes = {"intents": ["confirm", "reschedule", "cancel"],
            "personas": [scenarios[0].coverage.personas[0]],
            "perturbations": ["typo", "latency", "noise"]}
    residual = studio.residual_uncovered_estimate(scenarios, axes, budget=12, steps=3)

    base = Scenario(**_load_json("scenarios/expansion_base.json"))
    children = studio.expand_scenarios(
        base,
        {"intents": ["a", "b"], "perturbations": ["x", "y"]},
        k=2,
    )
    lineage_ok = bool(children) and all(
        child.parent_version == base.version for child in children
    )
    forbidden_present = [
        key for key in COVERAGE_FORBIDDEN_HEADLINE_KEYS if key in report
    ]
    cells = report["obligation_coverage"]["per_axis"]
    declared_cells = sum(cells[axis]["declared"] for axis in cells)
    return {
        "axes": sorted(report["obligation_coverage"]["per_axis"]),
        "obligation_coverage_rate": report["obligation_coverage"]["rate"],
        "residual_present": "residual_uncovered" in report,
        "plateau_curve": residual["plateau_curve"],
        "plateau_monotone": all(
            residual["plateau_curve"][i] >= residual["plateau_curve"][i + 1]
            for i in range(len(residual["plateau_curve"]) - 1)
        ),
        "forbidden_present": forbidden_present,
        "cells_declared": declared_cells,
        "expansion_lineage_ok": lineage_ok,
        "expansion_child_count": len(children),
    }


def _bias() -> dict[str, Any]:
    stereotyped = [Persona(**row) for row in _load_json("personas/stereotyped_set.json")]
    clean = [Persona(**row) for row in _load_json("personas/clean_set.json")]
    stereo = studio.bias_lint(stereotyped)
    clean_lint = studio.bias_lint(clean)
    return {
        "checks": sorted(stereo["checks"]),
        "stereotyped_status": stereo["status"],
        "clean_status": clean_lint["status"],
        "clean_locales": list(clean_lint["locales_linted"]),
        "stereotyped_failed_checks": sorted(
            name for name, c in stereo["checks"].items() if c["status"] == "fail"
        ),
    }


def _vendor_import() -> dict[str, Any]:
    out: dict[str, Any] = {"formats": list(PERSONA_VENDOR_IMPORT_FORMATS)}
    for fmt, rel in (("vapi", "vendor/vapi_support_rep.txt"),
                     ("retell", "vendor/retell_billing_caller.txt")):
        text = (FIXTURES / rel).read_text(encoding="utf-8")
        persona, goal = studio.import_vendor_persona(text, format=fmt)
        rendered = studio.render_vendor_text(persona)
        out[fmt] = {
            "byte_exact": rendered == text,
            "source_format": persona.provenance.source_format,
            "raw_present": persona.provenance.raw is not None,
            "goal_states": list(goal.states) if goal is not None else [],
            "persona_owns_no_goal": "goal" not in persona.persona
            and "goals" not in persona.persona,
        }
    return out


def _download(library: Path) -> dict[str, Any]:
    clean_payload = _load_json("downloads/clean.json")
    pin = validate_download(clean_payload, source="api.futureagi.com")
    clean_check = verify_pin(clean_payload, pin)

    tampered = _load_json("downloads/tampered.json")
    tampered_check = verify_pin(tampered["payload"], tampered["pin"])

    unpinned = _load_json("downloads/unpinned.json")
    unpinned_check = verify_pin(unpinned["payload"], unpinned["pin"])

    injection_payload = _load_json("downloads/injection.json")
    injection_flagged = False
    refused_in_quarantine = False
    quarantine_unloadable = False
    try:
        validate_download(injection_payload, source="api.futureagi.com")
    except DownloadRejected as rejection:
        injection_flagged = True
        from fi.alk.studio._library import quarantine_payload

        path = quarantine_payload(
            "persona-injection", injection_payload, rejection.findings, library=library
        )
        refused_in_quarantine = "quarantine" in Path(path).parts
        try:
            load_persona(path, library=library)
        except Exception:
            quarantine_unloadable = True

    return {
        "pin_fields": sorted(pin),
        "scan_results": list(CONTENT_SCAN_RESULTS),
        "clean": {"status": clean_check["status"], "scan": pin["content_scan"]["status"],
                  "pin_complete": all(f in pin for f in PERSONA_DOWNLOAD_PIN_FIELDS)},
        "tampered": tampered_check,
        "unpinned": unpinned_check,
        "injection": {
            "flagged": injection_flagged,
            "refused_in_quarantine": refused_in_quarantine,
            "quarantine_unloadable": quarantine_unloadable,
        },
    }


def _persona_conditioned_manifest() -> dict[str, Any]:
    persona = Persona(**_load_json("personas/attack_conditioned.json"))
    scenario = Scenario(**_load_json("scenarios/adversarial.json"))
    manifest = redteam.build_persona_conditioned_redteam_manifest(
        name="studio-persona-conditioned", persona=persona, scenario=scenario,
    )
    return {
        "built": manifest.get("version") == "agent-learning.redteam.v1",
        "attacks": list(manifest["redteam"]["attacks"]),
        "surfaces": list(manifest["redteam"]["surfaces"]),
        "min_turns": manifest["simulation"]["min_turns"],
        "max_turns": manifest["simulation"]["max_turns"],
        "embedded_persona_name": manifest["scenario"]["dataset"][0]["persona"]["name"],
    }


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    library = (
        Path(output_path).expanduser().parent / "persona-library"
        if output_path is not None
        else EXAMPLE_DIR / ".persona-library-demo"
    )

    persona_files = sorted((FIXTURES / "personas").glob("*.json"))
    transcript_files = sorted((FIXTURES / "transcripts").glob("*.json"))

    payload: dict[str, Any] = {
        "kind": READINESS_KIND,
        "representativeness_claim": "none",
        # constant mirrors (observed engine/studio values; the gate pins them)
        "persona_layers": ["identity", "temperament", "behavior_policy",
                           "knowledge", "provenance"],
        "persona_evidence_classes": list(PERSONA_EVIDENCE_CLASSES),
        "persona_temperament_axes": list(PERSONA_TEMPERAMENT_AXES),
        "persona_behavior_axes": list(PERSONA_BEHAVIOR_AXES),
        "persona_behavior_realization_metrics": list(PERSONA_BEHAVIOR_REALIZATION_METRICS),
        "persona_fidelity_verdicts": list(PERSONA_FIDELITY_VERDICTS),
        "persona_fidelity_epidemic_rate": PERSONA_FIDELITY_EPIDEMIC_RATE,
        "persona_fidelity_floors": {k: dict(v) for k, v in PERSONA_FIDELITY_FLOORS.items()},
        "scenario_kinds": list(SCENARIO_KINDS),
        "scenario_coverage_axes": list(SCENARIO_COVERAGE_AXES),
        "scenario_coverage_forbidden_headline_keys": list(COVERAGE_FORBIDDEN_HEADLINE_KEYS),
        "persona_calibration_stages": list(PERSONA_CALIBRATION_STAGES),
        "persona_calibration_probes": list(PERSONA_CALIBRATION_PROBES),
        "persona_content_scan_results": list(CONTENT_SCAN_RESULTS),
        "persona_bias_lint_checks": list(PERSONA_BIAS_LINT_CHECKS),
        "persona_vendor_import_formats": list(PERSONA_VENDOR_IMPORT_FORMATS),
        "persona_download_pin_fields": list(PERSONA_DOWNLOAD_PIN_FIELDS),
        # observed counts
        "fixture_persona_count": len(persona_files),
        "fixture_transcript_count": len(transcript_files),
        # result blocks
        "class_contract": _class_contract(),
        "fidelity": _fidelity(),
        "calibration": _calibration(library),
        "coverage": _coverage(),
        "bias": _bias(),
        "vendor_import": _vendor_import(),
        "download": _download(library),
        "persona_conditioned_manifest": _persona_conditioned_manifest(),
    }
    payload["coverage_cells_declared"] = payload["coverage"]["cells_declared"]

    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return payload


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    result = run(destination)
    if destination is None:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
