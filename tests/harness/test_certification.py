from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from fi.alk.harness.certification import (
    CertificationAuthoring,
    CertificationChecks,
    CertificationGateError,
    CertificationCompiler,
    CertificationRuntime,
    CertificationSource,
    CertificationStatus,
    CheckStatus,
    GenericHarnessArtifactStore,
    HarnessCertification,
    RuntimeValidationEvidence,
    verify_runtime_certification,
)
from fi.alk.harness.repair_controller import RepairHistory
from fi.alk.harness.source_model import SourceModel
from fi.alk.harness.world_ir import WorldIR


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _history() -> RepairHistory:
    return RepairHistory(
        decisions=(),
        results=(),
        compiler_candidates_used=0,
        environment_patches_used=0,
        scenario_patches_used=0,
    )


def _certificate() -> HarnessCertification:
    return HarnessCertification.create(
        status=CertificationStatus.CERTIFIED,
        source=CertificationSource(
            repository="https://example.invalid/agent.git",
            commit="abc123",
            digest=_digest("a"),
            schema_hash=_digest("b"),
        ),
        authoring=CertificationAuthoring(
            contract_hash=_digest("c"),
            world_ir_hash=_digest("d"),
            scenario_set_hash=_digest("e"),
        ),
        compiler=CertificationCompiler(
            version="compiler-v1", bundle_digest=_digest("f")
        ),
        runtime=CertificationRuntime(snapshot="snapshot-r1", validation_attempts=1),
        checks=CertificationChecks(
            static=CheckStatus.PASSED,
            schema_and_seed=CheckStatus.PASSED,
            processes=CheckStatus.PASSED,
            chat_ingress=CheckStatus.PASSED,
            source_invariants=CheckStatus.PASSED,
            scenario_setup_ready="5/5",
            tool_contract="3/3",
            reset_equivalence=CheckStatus.PASSED,
            world_isolation=CheckStatus.PASSED,
        ),
        repairs=_history(),
        limitations=("external side effects were not exercised",),
    )


def _runtime_evidence() -> RuntimeValidationEvidence:
    return RuntimeValidationEvidence(
        source_digest=_digest("a"),
        source_schema_hash=_digest("b"),
        world_ir_hash=_digest("c"),
        compiler_version="compiler-v1",
        bundle_digest=_digest("d"),
        contract_hash=_digest("e"),
        scenario_set_hash=_digest("f"),
        checks=CertificationChecks(schema_and_seed=CheckStatus.PASSED),
        limitations=("tool trajectories not run",),
    )


def test_certificate_is_canonical_and_tamper_evident() -> None:
    certificate = _certificate()
    body = json.loads(certificate.model_dump_json())
    body["checks"]["world_isolation"] = "failed"

    with pytest.raises(ValidationError, match="fingerprint_mismatch"):
        HarnessCertification.model_validate(body)


def test_execution_gate_accepts_only_matching_complete_certificate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime-validation.json"
    path.write_text(_certificate().model_dump_json(), encoding="utf-8")

    assert (
        verify_runtime_certification(
            path,
            bundle_digest=_digest("f"),
            source_digest=_digest("a"),
        ).fingerprint
        == _certificate().fingerprint
    )

    with pytest.raises(CertificationGateError, match="certification_mismatch"):
        verify_runtime_certification(
            path,
            bundle_digest=_digest("0"),
            source_digest=_digest("a"),
        )


def test_execution_gate_rejects_missing_and_incomplete_certificates(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(CertificationGateError, match="certification_missing"):
        verify_runtime_certification(
            missing,
            bundle_digest=_digest("f"),
            source_digest=_digest("a"),
        )

    path = tmp_path / "runtime-validation.json"
    certificate = _certificate()
    incomplete = HarnessCertification.create(
        status=CertificationStatus.CERTIFIED,
        source=certificate.source,
        authoring=certificate.authoring,
        compiler=certificate.compiler,
        runtime=certificate.runtime,
        checks=certificate.checks.model_copy(update={"tool_contract": "2/3"}),
        repairs=certificate.repairs,
        limitations=certificate.limitations,
    )
    path.write_text(incomplete.model_dump_json(), encoding="utf-8")
    with pytest.raises(CertificationGateError, match="certification_incomplete"):
        verify_runtime_certification(
            path,
            bundle_digest=_digest("f"),
            source_digest=_digest("a"),
        )


@pytest.mark.parametrize(
    "check",
    (
        "static",
        "schema_and_seed",
        "processes",
        "source_invariants",
        "reset_equivalence",
        "world_isolation",
    ),
)
def test_execution_gate_rejects_required_check_that_was_not_run(
    tmp_path: Path, check: str
) -> None:
    certificate = _certificate()
    incomplete = HarnessCertification.create(
        status=CertificationStatus.CERTIFIED,
        source=certificate.source,
        authoring=certificate.authoring,
        compiler=certificate.compiler,
        runtime=certificate.runtime,
        checks=certificate.checks.model_copy(update={check: CheckStatus.NOT_RUN}),
        repairs=certificate.repairs,
        limitations=certificate.limitations,
    )
    path = tmp_path / "runtime-validation.json"
    path.write_text(incomplete.model_dump_json(), encoding="utf-8")

    with pytest.raises(CertificationGateError, match=check):
        verify_runtime_certification(
            path,
            bundle_digest=_digest("f"),
            source_digest=_digest("a"),
        )


def test_execution_gate_accepts_explicit_non_applicable_state_checks(
    tmp_path: Path,
) -> None:
    certificate = _certificate()
    external = HarnessCertification.create(
        status=CertificationStatus.CERTIFIED,
        source=certificate.source,
        authoring=certificate.authoring,
        compiler=certificate.compiler,
        runtime=certificate.runtime,
        checks=certificate.checks.model_copy(
            update={
                "schema_and_seed": CheckStatus.NOT_APPLICABLE,
                "source_invariants": CheckStatus.NOT_APPLICABLE,
                "reset_equivalence": CheckStatus.NOT_APPLICABLE,
                "world_isolation": CheckStatus.NOT_APPLICABLE,
            }
        ),
        repairs=certificate.repairs,
        limitations=("provider state is external to the harness",),
    )

    path = tmp_path / "runtime-validation.json"
    path.write_text(external.model_dump_json(), encoding="utf-8")

    assert (
        verify_runtime_certification(
            path,
            bundle_digest=_digest("f"),
            source_digest=_digest("a"),
        ).fingerprint
        == external.fingerprint
    )


def test_execution_gate_rejects_non_applicable_checks_without_limitation(
    tmp_path: Path,
) -> None:
    certificate = _certificate()
    incomplete = HarnessCertification.create(
        status=CertificationStatus.CERTIFIED,
        source=certificate.source,
        authoring=certificate.authoring,
        compiler=certificate.compiler,
        runtime=certificate.runtime,
        checks=certificate.checks.model_copy(
            update={"schema_and_seed": CheckStatus.NOT_APPLICABLE}
        ),
        repairs=certificate.repairs,
    )
    path = tmp_path / "runtime-validation.json"
    path.write_text(incomplete.model_dump_json(), encoding="utf-8")

    with pytest.raises(
        CertificationGateError, match="not_applicable_without_limitation"
    ):
        verify_runtime_certification(
            path,
            bundle_digest=_digest("f"),
            source_digest=_digest("a"),
        )


def test_certified_status_cannot_hide_diagnostics() -> None:
    body = _certificate().model_dump(mode="python")
    body["diagnostics"] = [
        {
            "schema_version": "futureagi.harness-diagnostic.v1",
            "stage": "validating_environment",
            "domain": "environment",
            "component": "seed",
            "code": "schema_type_mismatch",
            "owner": "compiler",
            "retryable": False,
            "repair_strategy": "recompile_with_discovered_type",
            "redacted_message": "type mismatch",
            "evidence_refs": [],
            "fingerprint": _digest("0"),
        }
    ]

    with pytest.raises(ValidationError):
        HarnessCertification.model_validate(body)


def test_artifact_store_round_trips_and_uses_private_atomic_files(
    tmp_path: Path,
) -> None:
    source = SourceModel.create(source_digest=_digest("a"), engine="postgres")
    world = WorldIR.create(source_model_fingerprint=source.fingerprint, tables=())
    store = GenericHarnessArtifactStore(tmp_path / "generic")
    source_schema, world_schema = store.write_contract_schemas()

    source_path = store.write_source_model(source)
    world_path = store.write_world_ir(world)
    history_path = store.write_repair_history(_history())
    evidence_path = store.write_runtime_evidence(_runtime_evidence())
    certificate_path = store.write_certification(_certificate())

    assert store.read_source_model() == source
    assert json.loads(source_schema.read_text())["$defs"]["SourceTable"]
    assert json.loads(world_schema.read_text())["$defs"]["WorldValue"]
    assert stat.S_IMODE(source_schema.stat().st_mode) == 0o600
    assert stat.S_IMODE(world_schema.stat().st_mode) == 0o600
    assert store.read_world_ir() == world
    assert store.read_repair_history() == _history()
    assert store.read_runtime_evidence() == _runtime_evidence()
    assert store.read_certification() == _certificate()
    for path in (
        source_path,
        world_path,
        history_path,
        evidence_path,
        certificate_path,
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list((tmp_path / "generic").glob(".*.json.*"))


def test_artifact_store_refuses_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "generic"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (root / GenericHarnessArtifactStore.SOURCE_MODEL).symlink_to(outside)

    with pytest.raises(ValueError, match="symlink_forbidden"):
        GenericHarnessArtifactStore(root).read_source_model()


@pytest.mark.parametrize(
    "status", [CheckStatus.NOT_RUN, CheckStatus.FAILED, CheckStatus.NOT_APPLICABLE]
)
def test_chat_certificate_requires_explicit_pass(tmp_path, status):
    certificate = _certificate()
    altered = HarnessCertification.create(
        status=certificate.status,
        source=certificate.source,
        authoring=certificate.authoring,
        compiler=certificate.compiler,
        runtime=certificate.runtime,
        checks=certificate.checks.model_copy(update={"chat_ingress": status}),
        repairs=certificate.repairs,
        limitations=certificate.limitations,
    )
    path = tmp_path / "certificate.json"
    path.write_text(altered.model_dump_json())
    with pytest.raises(CertificationGateError, match="chat_ingress"):
        verify_runtime_certification(
            path, bundle_digest=_digest("f"), source_digest=_digest("a")
        )
    if status is CheckStatus.NOT_APPLICABLE:
        assert verify_runtime_certification(
            path,
            bundle_digest=_digest("f"),
            source_digest=_digest("a"),
            chat_ingress_required=False,
        )


def test_missing_chat_check_defaults_to_not_run():
    assert CertificationChecks.model_validate({}).chat_ingress is CheckStatus.NOT_RUN


@pytest.mark.parametrize(
    "mode,modality,required",
    [("repo", "chat", True), ("repo", "voice", False), ("external", "chat", False)],
)
def test_chat_applicability_uses_bound_contract_and_provider_mode(
    tmp_path, mode, modality, required
):
    from types import SimpleNamespace
    from fi.alk.harness.certification import requires_chat_ingress
    from fi.alk.harness.job import SourceKind, ProviderExecutionMode

    (tmp_path / "contract.json").write_text(json.dumps({"modality": modality}))
    job = SimpleNamespace(
        source=SimpleNamespace(
            kind=SourceKind.PROVIDER
            if mode == "external"
            else SourceKind.LOCAL_REPOSITORY
        ),
        agent=SimpleNamespace(mode=ProviderExecutionMode.CONNECT_ONLY),
    )
    assert requires_chat_ingress(job, tmp_path) is required
