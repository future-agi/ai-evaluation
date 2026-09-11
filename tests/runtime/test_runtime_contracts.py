from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fi.simulate.artifacts import ArtifactManifest, ArtifactManifestEntry
from fi.simulate.evidence import EvidenceCapabilities, EvidenceClass, EvidenceSourceSpec
from fi.simulate.results import LocalFilesystemResultSink
from fi.simulate.runtime import (
    AgentEndpointSpec,
    CanonicalEvent,
    EnvironmentSpec,
    FailureStage,
    RunStatus,
    SecretRef,
    SimulationFailure,
    SimulationReport,
    SimulationSpec,
    SimulationTestCaseResult,
    SimulatorPolicySpec,
    TestCaseStatus as CaseStatus,
    derive_artifact_id,
    derive_event_id,
    derive_test_case_id,
)
from fi.simulate.simulation.models import (
    Persona,
    Scenario,
    TestCaseResult as LegacyCaseResult,
    TestReport as LegacyReport,
)


def _persona() -> Persona:
    return Persona(
        persona={"name": "Taylor"},
        situation="I need help.",
        outcome="The problem is resolved.",
    )


def _scenario() -> Scenario:
    return Scenario(name="support", dataset=[_persona()])


def _spec(**target_config: object) -> SimulationSpec:
    return SimulationSpec(
        run_id="run_test",
        environment=EnvironmentSpec(
            adapter="chat",
            world_kind="conversation",
        ),
        target=AgentEndpointSpec(
            adapter="http",
            config=target_config,
            secret_refs={
                "api_key": SecretRef(
                    manager="environment",
                    key="CUSTOMER_AGENT_API_KEY",
                    purpose="target authentication",
                )
            },
        ),
        simulator=SimulatorPolicySpec(adapter="deterministic"),
        scenario=_scenario(),
        evidence={
            "sources": [
                EvidenceSourceSpec(
                    source_id="caller",
                    adapter="caller_observed",
                    evidence_class=EvidenceClass.CALLER_OBSERVED,
                    capabilities=EvidenceCapabilities(transcript=True),
                )
            ]
        },
    )


def test_spec_round_trip_preserves_content_hash() -> None:
    spec = _spec(base_url="https://agent.example.com")

    restored = SimulationSpec.model_validate_json(spec.model_dump_json())

    assert restored == spec
    assert restored.spec_hash == spec.content_hash()


def test_spec_rejects_resolved_secrets() -> None:
    with pytest.raises(ValidationError, match="resolved_secret_forbidden"):
        _spec(api_key="plaintext-secret")


def test_spec_allows_secret_references() -> None:
    spec = _spec()

    assert spec.target.secret_refs["api_key"].key == "CUSTOMER_AGENT_API_KEY"


def test_stable_ids_are_repeatable_and_scoped() -> None:
    case_id = derive_test_case_id("run_a", "persona_a", 0)

    assert case_id == derive_test_case_id("run_a", "persona_a", 0)
    assert case_id != derive_test_case_id("run_b", "persona_a", 0)
    assert derive_event_id(case_id, "target", 1) == derive_event_id(
        case_id, "target", 1
    )
    assert derive_artifact_id(case_id, "agent.wav") != derive_artifact_id(
        case_id, "customer.wav"
    )


def test_artifact_manifest_is_order_independent() -> None:
    entries = [
        ArtifactManifestEntry(
            artifact_id="artifact_b",
            test_case_id="case_a",
            type="audio",
            path="audio/agent.wav",
            checksum="sha256:" + "b" * 64,
            size_bytes=10,
            evidence_class=EvidenceClass.CALLER_OBSERVED,
            evidence_source_id="recorder",
        ),
        ArtifactManifestEntry(
            artifact_id="artifact_a",
            test_case_id="case_a",
            type="audio",
            path="audio/customer.wav",
            checksum="sha256:" + "a" * 64,
            size_bytes=12,
            evidence_class=EvidenceClass.CALLER_OBSERVED,
            evidence_source_id="recorder",
        ),
    ]

    first = ArtifactManifest(run_id="run_a", entries=entries)
    second = ArtifactManifest(run_id="run_a", entries=list(reversed(entries)))

    assert first.manifest_hash == second.manifest_hash


def test_failed_case_requires_typed_failure() -> None:
    with pytest.raises(ValidationError, match="test_case_failure_missing"):
        SimulationTestCaseResult(
            test_case_id="case_a",
            status=CaseStatus.FAILED,
            persona=_persona(),
        )


def test_report_failure_is_metadata_not_transcript() -> None:
    failure = SimulationFailure(
        stage=FailureStage.READINESS,
        code="agent_unavailable",
        message="Target agent did not become ready",
    )
    report = SimulationReport(
        run_id="run_a",
        spec_hash="sha256:" + "0" * 64,
        status=RunStatus.COMPLETED,
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        test_cases=[
            SimulationTestCaseResult(
                test_case_id="case_a",
                status=CaseStatus.AGENT_UNAVAILABLE,
                persona=_persona(),
                failure=failure,
            )
        ],
        artifacts=ArtifactManifest(run_id="run_a"),
    )

    legacy = report.to_legacy()

    assert legacy.results[0].transcript == ""
    assert legacy.results[0].metadata["failure"]["code"] == "agent_unavailable"


def test_legacy_report_conversion_is_additive() -> None:
    persona = _persona()
    legacy = LegacyReport(
        results=[
            LegacyCaseResult(
                persona=persona,
                transcript="user: hi\nassistant: hello",
            )
        ]
    )

    canonical = SimulationReport.from_legacy(
        legacy,
        run_id="run_a",
        spec_hash="sha256:" + "0" * 64,
    )

    assert canonical.test_cases[0].status == CaseStatus.COMPLETED
    assert canonical.to_legacy().results[0].transcript == legacy.results[0].transcript


def test_filesystem_sink_writes_recoverable_run(tmp_path: Path) -> None:
    spec = _spec()
    sink = LocalFilesystemResultSink(tmp_path)
    run_directory = sink.prepare(spec)
    event = CanonicalEvent.create(
        run_id=spec.run_id,
        test_case_id="case_a",
        event_type="session.started",
        source="runtime",
        sequence=0,
    )
    sink.write_event(event)
    report = SimulationReport.from_legacy(
        LegacyReport(
            results=[LegacyCaseResult(persona=_persona(), transcript="complete")]
        ),
        run_id=spec.run_id,
        spec_hash=spec.spec_hash or spec.content_hash(),
    )

    report_path = sink.write_report(report)

    assert SimulationSpec.model_validate_json(
        (run_directory / "spec.json").read_text()
    ) == spec
    assert json.loads((run_directory / "events.jsonl").read_text())["event_id"]
    assert SimulationReport.model_validate_json(report_path.read_text()) == report
    assert (run_directory / "artifacts.json").exists()


def test_filesystem_sink_rejects_unsafe_run_id(tmp_path: Path) -> None:
    spec = _spec().model_copy(update={"run_id": "../escape"})

    with pytest.raises(ValueError, match="run_id_invalid"):
        LocalFilesystemResultSink(tmp_path).prepare(spec)
