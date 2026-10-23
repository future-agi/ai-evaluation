import asyncio
import json
import shutil

import pytest

from fi.alk.harness import (
    authoring_runtime_validation,
    bundle_author_v2,
    compose_integration,
)
from fi.alk.harness.certification import (
    CertificationAuthoring,
    CertificationChecks,
    CertificationCompiler,
    CertificationRuntime,
    CertificationSource,
    CertificationStatus,
    CheckStatus,
    HarnessCertification,
    RepairHistory,
)
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.process_preflight import PreflightError


@pytest.fixture
def native_bundle(tmp_path, monkeypatch):
    monkeypatch.delenv("ALK_TRUSTED_COMPOSE_SOCKET", raising=False)
    monkeypatch.setenv("ALK_COMPOSE_BROKER_SOCKET", "/private/alk/broker.sock")
    source = tmp_path / "source"
    source.mkdir()
    (source / "compose.yaml").write_text(
        "services:\n  app:\n    image: node:22\n    ports: ['3000:3000']\n"
    )
    authoring = tmp_path / "authoring"
    scenario = authoring / "scenarios" / "one"
    (scenario / "checks").mkdir(parents=True)
    (scenario / "scenario.json").write_text(
        json.dumps({"name": "one", "instruction": "hello", "sub_goals": ["response"]})
    )
    for name in ("setup", "ready"):
        (scenario / f"{name}.py").write_text(f"def {name}(world):\n    return None\n")
    (scenario / "checks" / "response.py").write_text(
        "def check(world, calls):\n    return None\n"
    )
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "modality": "chat",
                "runtime": {
                    "language": "javascript",
                    "execution": "compose",
                    "compose_file": "compose.yaml",
                    "workdir": ".",
                    "service": "app",
                    "command": ["node", "server.js"],
                    "interface": {
                        "kind": "http",
                        "protocol": "json_template",
                        "port": 3000,
                        "path": "/chat",
                        "request_template": {"message": "{message}"},
                        "response_path": "response",
                    },
                },
            }
        )
    )
    job = HarnessJob.model_validate(
        {
            "job_id": "broker",
            "run_id": "broker",
            "execution": "local",
            "source": {"kind": "local_repository", "local_path": str(source)},
            "agent": {"connector": "auto"},
            "scenario_count": 1,
            "runtime": {"cpu_units": 2, "memory_mb": 4096, "parallelism": 1},
        }
    )
    output = tmp_path / "bundle"
    manifest = bundle_author_v2.author_bundle_v2(
        source=source, job=job, authoring=authoring, output=output
    )
    return source, authoring, job, output, manifest


def test_broker_only_validation_reaches_native_validator(
    native_bundle, tmp_path, monkeypatch
):
    source, authoring, job, _, manifest = native_bundle
    secrets = tmp_path / "secrets.json"
    secrets.write_text("{}")
    called = []

    async def native(**kwargs):
        assert kwargs["manifest"].digest == manifest.digest
        called.append(kwargs["manifest"].runtime.kind.value)
        return 1

    monkeypatch.setattr(compose_integration, "validate_native", native)
    assert (
        asyncio.run(
            authoring_runtime_validation.validate_once(
                job,
                source,
                authoring,
                secrets_path=secrets,
                local_runtime=True,
            )
        )
        == 1
    )
    assert called == ["compose"]


@pytest.mark.parametrize("enabled", [True, False])
def test_certified_compose_reuse_requires_either_optin(
    native_bundle, tmp_path, monkeypatch, enabled
):
    source, authoring, job, output, manifest = native_bundle
    artifact = authoring / "generic-harness"
    shutil.copytree(output, artifact / "certified-bundle")
    digest = authoring_runtime_validation._artifact_digest
    certificate = HarnessCertification.create(
        status=CertificationStatus.CERTIFIED,
        limitations=(
            "Native application state is opaque; schema, invariants and value-level reset/isolation checks are not applicable.",
        ),
        source=CertificationSource(
            digest=bundle_author_v2.source_fingerprint(source),
            schema_hash="sha256:" + "a" * 64,
        ),
        authoring=CertificationAuthoring(
            contract_hash=digest(authoring / "contract.json"),
            world_ir_hash="sha256:" + "b" * 64,
            scenario_set_hash=digest(authoring / "scenarios"),
        ),
        compiler=CertificationCompiler(
            version="native-compose-observation-v1", bundle_digest=manifest.digest
        ),
        runtime=CertificationRuntime(snapshot="test", validation_attempts=1),
        checks=CertificationChecks(
            static=CheckStatus.PASSED,
            schema_and_seed=CheckStatus.NOT_APPLICABLE,
            processes=CheckStatus.PASSED,
            chat_ingress=CheckStatus.PASSED,
            source_invariants=CheckStatus.NOT_APPLICABLE,
            scenario_setup_ready="1/1",
            tool_contract="0/0",
            action_behavior="0/0",
            reset_equivalence=CheckStatus.NOT_APPLICABLE,
            world_isolation=CheckStatus.NOT_APPLICABLE,
        ),
        repairs=RepairHistory(
            decisions=(),
            results=(),
            compiler_candidates_used=0,
            environment_patches_used=0,
            scenario_patches_used=0,
        ),
    )
    (authoring / "runtime-validation.json").write_text(certificate.model_dump_json())
    if not enabled:
        monkeypatch.delenv("ALK_COMPOSE_BROKER_SOCKET")
    kwargs = dict(
        source=source, job=job, authoring=authoring, output=tmp_path / "reused"
    )
    if enabled:
        reused = bundle_author_v2._reuse_certified_bundle(**kwargs)
        assert reused.digest == manifest.digest
    else:
        with pytest.raises(PreflightError, match="compose_not_hosted"):
            bundle_author_v2._reuse_certified_bundle(**kwargs)
