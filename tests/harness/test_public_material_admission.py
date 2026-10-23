import asyncio
import hashlib
import json

import pytest
from pydantic import ValidationError

from fi.alk.harness.bundle import BundleError, _inspect_file
from fi.alk.harness.bundle_v2 import BundleProvenanceV2
from fi.alk.harness.compose_integration import validate_source
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.provision import source_fingerprint
from fi.alk.harness.public_material import PublicMaterialAdmission

TOKEN = b"sk-publicDemonstrationValueNotCredential"


def approval(source, relative, token=TOKEN):
    return PublicMaterialAdmission(
        source_digest=source_fingerprint(source),
        path=relative,
        file_sha256=hashlib.sha256((source / relative).read_bytes()).hexdigest(),
        pattern="sk_token",
        match_sha256=hashlib.sha256(token).hexdigest(),
        reason="Reviewed public UI demonstration value.",
        evidence="https://example.org/source/pinned-revision",
    )


@pytest.mark.parametrize(
    "relative", ["web/components/Input.jsx", "docs/tutorial/sample.txt"]
)
def test_review_admits_only_exact_source_file_match(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b'placeholder="' + TOKEN + b'"')
    with pytest.raises(BundleError, match="secret_material"):
        validate_source(tmp_path)
    admission = approval(tmp_path, relative)
    validate_source(tmp_path, [admission])
    assert TOKEN.decode() not in admission.model_dump_json()
    # A separate source change invalidates the operator review too.
    (tmp_path / "other.txt").write_text("changed")
    with pytest.raises(ValueError, match="source_changed"):
        validate_source(tmp_path, [admission])


@pytest.mark.parametrize(
    "change",
    [
        "extra_match",
        "file_changed",
        "missing_match",
        "private_key",
        "forbidden_filename",
    ],
)
def test_review_does_not_disable_other_findings(tmp_path, change):
    relative = "example.txt"
    path = tmp_path / relative
    path.write_bytes(TOKEN)
    admission = approval(tmp_path, relative)
    if change == "extra_match":
        path.write_bytes(TOKEN + b" sk-anotherUnreviewedValue1234567890")
        admission = approval(tmp_path, relative)
    elif change == "file_changed":
        path.write_bytes(TOKEN + b" changed")
        admission = admission.model_copy(
            update={"source_digest": source_fingerprint(tmp_path)}
        )
    elif change == "missing_match":
        path.write_text("ordinary text")
        admission = approval(tmp_path, relative)
    elif change == "private_key":
        path.write_bytes(TOKEN + b"\n-----BEGIN PRIVATE KEY-----\nREALBODY")
        admission = approval(tmp_path, relative)
    else:
        path.rename(tmp_path / ".env")
        admission = approval(tmp_path, ".env")
    with pytest.raises(BundleError):
        validate_source(tmp_path, [admission])


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/absolute",
        "a/../b",
        "./a",
        "a\\b",
        "a//b",
        "C:/x",
        ".",
        "file\x00",
    ],
)
def test_unsafe_admission_path_rejected(tmp_path, path):
    (tmp_path / "file.txt").write_bytes(TOKEN)
    raw = approval(tmp_path, "file.txt").model_dump()
    raw["path"] = path
    with pytest.raises(ValidationError, match="path_unsafe"):
        PublicMaterialAdmission.model_validate(raw)


def test_policy_cannot_authorize_private_key_or_persist_raw_match(tmp_path):
    (tmp_path / "file.txt").write_bytes(TOKEN)
    raw = approval(tmp_path, "file.txt").model_dump()
    for field, value in [
        ("pattern", "private_key"),
        ("reason", TOKEN.decode()),
        ("evidence", TOKEN.decode()),
    ]:
        with pytest.raises(ValidationError):
            PublicMaterialAdmission.model_validate({**raw, field: value})


def test_job_authority_is_local_operator_only(tmp_path):
    (tmp_path / "file.txt").write_bytes(TOKEN)
    raw = dict(
        job_id="test",
        run_id="test",
        execution="local",
        source={"kind": "local_repository", "local_path": str(tmp_path)},
        agent={"connector": "auto"},
        public_material_admissions=[approval(tmp_path, "file.txt").model_dump()],
    )
    assert len(HarnessJob.model_validate(raw).public_material_admissions) == 1
    with pytest.raises(ValidationError, match="local_operator_required"):
        HarnessJob.model_validate({**raw, "execution": "hosted"})


def test_provenance_cannot_transplant_admission(tmp_path):
    (tmp_path / "file.txt").write_bytes(TOKEN)
    admission = approval(tmp_path, "file.txt")
    with pytest.raises(ValidationError, match="source_changed"):
        BundleProvenanceV2(
            source_kind="local",
            source_digest="0" * 64,
            public_material_admissions=[admission],
        )


def test_scanner_checks_match_across_old_chunk_boundary(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b" " * (1024 * 1024 - 12) + TOKEN)
    with pytest.raises(BundleError, match="secret_material"):
        _inspect_file(tmp_path, path)
    validate_source(tmp_path, [approval(tmp_path, "sample.txt")])


@pytest.mark.parametrize("fixture_only", [False, True])
def test_compile_seal_load_and_provision_preserve_review(
    tmp_path, monkeypatch, fixture_only, resolved_compose_images
):
    from fi.alk.harness.bundle_author_v2 import author_bundle_v2
    from fi.alk.harness.bundle_v2 import load_bundle_v2, seal_bundle_v2
    from fi.alk.harness.certified_compose_runtime import CertifiedComposeRuntimeProvider
    from fi.alk.harness.process_preflight import preflight_bundle

    source = tmp_path / "source"
    source.mkdir()
    (source / "public.txt").write_bytes(TOKEN)
    (source / "compose.yaml").write_text(
        "services:\n  app:\n    image: example/chat:1\n"
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
                    },
                },
            }
        )
    )
    admission = approval(source, "public.txt")
    field = "public_material_admissions"
    if fixture_only:
        from fi.alk.harness.public_material import PublicTestFixtureAdmission

        (source / "public.txt").write_bytes(b"-----BEGIN PRIVATE KEY----- fixed test")
        admission = PublicTestFixtureAdmission.model_validate(
            {
                key: value
                for key, value in approval(source, "public.txt").model_dump().items()
                if key not in {"pattern", "match_sha256"}
            }
        )
        field = "public_test_fixture_admissions"
    job = HarnessJob.model_validate(
        dict(
            job_id="test",
            run_id="test",
            execution="local",
            source={"kind": "local_repository", "local_path": str(source)},
            agent={"connector": "auto"},
            scenario_count=1,
            **{field: [admission]},
        )
    )
    monkeypatch.setenv("ALK_TRUSTED_COMPOSE_SOCKET", "/private/test.sock")
    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source, job=job, authoring=authoring, output=output
    )
    loaded = load_bundle_v2(output)
    assert getattr(loaded.provenance, field) == [admission]
    assert seal_bundle_v2(loaded) == manifest.digest
    preflight_bundle(output, loaded, parallelism=1, secret_refs={}, allow_compose=True)
    provider = CertifiedComposeRuntimeProvider(socket_path=tmp_path / "socket")
    # Source verification precedes all Docker access, including reuse of a running world.
    (source / "public.txt").write_bytes(TOKEN + b" changed")
    monkeypatch.setattr(
        provider, "_check_socket", lambda: pytest.fail("Docker reached")
    )
    with pytest.raises(ValueError, match="source_changed"):
        asyncio.run(
            provider.provision(
                loaded, source=source, bundle_dir=output, work_directory=tmp_path
            )
        )
    getattr(loaded.provenance, field)[0] = admission.model_copy(
        update={"reason": "changed review"}
    )
    assert seal_bundle_v2(loaded) != manifest.digest


@pytest.mark.parametrize("fixture_only", [False, True, None])
def test_reset_rejects_changed_source_before_copy(
    tmp_path, monkeypatch, fixture_only
):
    from types import SimpleNamespace
    from fi.alk.harness.certified_compose_runtime import CertifiedComposeRuntimeProvider

    source = tmp_path / "submitted"
    source.mkdir()
    (source / "public.txt").write_bytes(TOKEN)
    admission = approval(source, "public.txt")
    field = "public_material_admissions"
    if fixture_only:
        from fi.alk.harness.public_material import PublicTestFixtureAdmission

        (source / "public.txt").write_bytes(b"-----BEGIN PRIVATE KEY----- fixed test")
        admission = PublicTestFixtureAdmission.model_validate(
            {
                key: value
                for key, value in approval(source, "public.txt").model_dump().items()
                if key not in {"pattern", "match_sha256"}
            }
        )
        field = "public_test_fixture_admissions"
    manifest = SimpleNamespace(
        provenance=BundleProvenanceV2(
            source_kind="local",
            source_digest=admission.source_digest,
            **({field: [admission]} if fixture_only is not None else {}),
        ),
        capabilities={},
    )
    world_dir = tmp_path / "world"
    (world_dir / "source").mkdir(parents=True)
    runtime = SimpleNamespace(world_index=0)
    provider = CertifiedComposeRuntimeProvider(socket_path=tmp_path / "socket")
    provider._source = source
    provider._document = {"services": {"app": {"image": "example/app:1"}}}
    provider._worlds[0] = runtime, world_dir, manifest
    monkeypatch.setattr(provider, "_stop", lambda *args: None)
    (source / "public.txt").write_bytes(TOKEN + b" changed")
    with pytest.raises(ValueError, match="source_changed"):
        asyncio.run(provider.reset(runtime, work_directory=tmp_path))
    assert not (world_dir / "source").exists()
