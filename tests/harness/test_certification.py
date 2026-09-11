from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from fi.alk.harness.certification import (
    CertificationAuthoring,
    CertificationChecks,
    CertificationCompiler,
    CertificationRuntime,
    CertificationSource,
    CertificationStatus,
    CheckStatus,
    GenericHarnessArtifactStore,
    HarnessCertification,
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
            source_invariants=CheckStatus.PASSED,
            scenario_setup_ready="5/5",
            tool_contract="3/3",
            reset_equivalence=CheckStatus.PASSED,
            world_isolation=CheckStatus.PASSED,
        ),
        repairs=_history(),
        limitations=("external side effects were not exercised",),
    )


def test_certificate_is_canonical_and_tamper_evident() -> None:
    certificate = _certificate()
    body = json.loads(certificate.model_dump_json())
    body["checks"]["world_isolation"] = "failed"

    with pytest.raises(ValidationError, match="fingerprint_mismatch"):
        HarnessCertification.model_validate(body)


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

    source_path = store.write_source_model(source)
    world_path = store.write_world_ir(world)
    history_path = store.write_repair_history(_history())
    certificate_path = store.write_certification(_certificate())

    assert store.read_source_model() == source
    assert store.read_world_ir() == world
    assert store.read_repair_history() == _history()
    assert store.read_certification() == _certificate()
    for path in (source_path, world_path, history_path, certificate_path):
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
