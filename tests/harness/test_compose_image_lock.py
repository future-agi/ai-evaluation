import copy

import pytest

from fi.alk.harness.compose_image_lock import (
    ResolvedComposeImage,
    image_identity,
    lock_document,
    selected_images,
)


DIGEST = "sha256:" + "a" * 64


def inspect(reference="registry.example/team/chat:stable", **changes):
    value = {
        "RepoDigests": [f"registry.example/team/chat@{DIGEST}"],
        "Os": "linux",
        "Architecture": "amd64",
        "Config": {
            "Labels": {
                "org.opencontainers.image.source": "https://github.com/team/chat",
                "org.opencontainers.image.revision": "b" * 40,
                "org.opencontainers.image.version": "1.2.3",
            }
        },
    }
    value.update(changes)
    return image_identity(
        service="chat",
        reference=reference,
        platform="linux/amd64",
        inspected=value,
        repository="team/chat",
        revision="b" * 40,
        application=True,
        source_declared_digest=True,
    )


def test_identity_is_digest_bound_and_records_source_evidence():
    item = inspect()
    assert item.immutable_reference == f"registry.example/team/chat@{DIGEST}"
    assert item.correspondence == "source_revision"
    assert item.version == "1.2.3"


@pytest.mark.parametrize(
    "change",
    [
        {"RepoDigests": ["registry.example/team/chat@sha256:bad"]},
        {"RepoDigests": [f"registry.example/unrelated@{DIGEST}"]},
        {"Architecture": "arm64"},
        {
            "Config": {
                "Labels": {
                    "org.opencontainers.image.source": "https://github.com/other/project"
                }
            }
        },
        {"Config": {"Labels": {"org.opencontainers.image.revision": "c" * 40}}},
        {"Config": {"Labels": {}}},
    ],
)
def test_invalid_or_unproved_application_identity_fails(change):
    with pytest.raises(ValueError, match="compose_image_"):
        inspect(**change)


def test_source_declared_digest_does_not_claim_commit_correspondence():
    item = inspect(
        reference=f"registry.example/team/chat@{DIGEST}", Config={"Labels": {}}
    )
    assert item.correspondence == "source_declared_digest"
    assert item.source_revision is None


def test_source_selection_cannot_pull_an_arbitrary_image(tmp_path):
    (tmp_path / "compose.yaml").write_text(
        "services:\n  chat:\n    image: registry.example/team/chat:stable\n"
    )
    runtime = {
        "execution": "compose",
        "compose_file": "compose.yaml",
        "service": "chat",
    }
    document = {"services": {"chat": {"image": "registry.example/other/agent:latest"}}}
    with pytest.raises(ValueError, match="compose_image_source_selection_mismatch"):
        selected_images(tmp_path, runtime, document, control_service="chat")


def test_lock_preserves_builds_and_reuses_digest_after_tag_changes():
    original = {
        "services": {
            "chat": {"image": "registry.example/team/chat:stable"},
            "worker": {"build": ".", "image": "worker:local"},
        }
    }
    document = copy.deepcopy(original)
    item = inspect()
    lock_document(document, [item])
    assert document["services"]["chat"]["image"].endswith(DIGEST)
    assert document["services"]["worker"] == original["services"]["worker"]
    assert original["services"]["chat"]["image"].endswith(":stable")
    for _ in range(3):
        assert (
            copy.deepcopy(document)["services"]["chat"]["image"]
            == item.immutable_reference
        )
    with pytest.raises(ValueError, match="compose_image_lock_selection_mismatch"):
        lock_document(document, [item])


def test_evidence_cannot_contain_invalid_immutable_reference():
    with pytest.raises(ValueError):
        ResolvedComposeImage(
            service="chat",
            declared_reference="chat:latest",
            immutable_reference="chat:latest",
            platform="linux/amd64",
            correspondence="dependency",
        )


@pytest.mark.parametrize("generated_digest", [False, True])
def test_resolution_uses_only_bounded_docker_operations_without_credentials(
    tmp_path, monkeypatch, generated_digest
):
    import json
    from types import SimpleNamespace
    from fi.alk.harness.certified_compose_runtime import CertifiedComposeRuntimeProvider

    reference = f"registry.example/team/chat@{DIGEST}"
    declared = "${IMAGE}" if generated_digest else reference
    (tmp_path / "compose.yaml").write_text(
        f"services:\n  chat:\n    image: {declared}\n"
    )
    runtime = {
        "execution": "compose",
        "compose_file": "compose.yaml",
        "service": "chat",
    }
    if generated_digest:
        runtime["config_files"] = {".env": {"values": {"IMAGE": reference}}}
    document = json.dumps({"services": {"chat": {"image": reference}}})
    owner = CertifiedComposeRuntimeProvider(socket_path=tmp_path / "docker.sock")
    monkeypatch.setattr(owner, "_check_socket", lambda: None)
    monkeypatch.setenv("CONTROLLER_TOKEN", "must-not-reach-docker")
    calls = []

    def run(command, environment, project, operation, **kwargs):
        calls.append(command)
        assert "CONTROLLER_TOKEN" not in environment
        assert set(environment) == {"PATH", "DOCKER_HOST", "DOCKER_CONFIG"}
        assert kwargs["private_output"] is True
        assert 0 < kwargs["timeout"] <= 600
        if command[1] == "info":
            return json.dumps({"OSType": "linux", "Architecture": "x86_64"})
        if command[2] == "pull":
            return "download complete"
        return json.dumps(
            [{"RepoDigests": [reference], "Os": "linux", "Architecture": "amd64"}]
        )

    monkeypatch.setattr(owner, "_run_command", run)
    manifest = SimpleNamespace(
        runtime=SimpleNamespace(control_service="chat"),
        provenance=SimpleNamespace(repository=None, commit=None),
    )
    if generated_digest:
        with pytest.raises(
            ValueError, match="compose_image_source_correspondence_missing"
        ):
            owner.resolve_images(
                manifest=manifest, document=document, runtime=runtime, source=tmp_path
            )
    else:
        identities = owner.resolve_images(
            manifest=manifest, document=document, runtime=runtime, source=tmp_path
        )
        assert identities[0].immutable_reference == reference
    assert [command[1:3] for command in calls] == [
        ["info", "--format"],
        ["image", "pull"],
        ["image", "inspect"],
    ]
    assert "run" not in str(calls) and "build" not in str(calls)


def test_resolution_timeout_has_no_tag_or_build_fallback(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from fi.alk.harness import certified_compose_runtime as module

    reference = f"registry.example/team/chat@{DIGEST}"
    (tmp_path / "compose.yaml").write_text(
        f"services:\n  chat:\n    image: {reference}\n"
    )
    owner = module.CertifiedComposeRuntimeProvider(socket_path=tmp_path / "docker.sock")
    monkeypatch.setattr(owner, "_check_socket", lambda: None)
    times = iter([0, 601])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        owner,
        "_run_command",
        lambda *a, **k: pytest.fail("Docker must not run after deadline"),
    )
    with pytest.raises(
        module.ComposeRuntimeError, match="compose_image_resolution_timeout"
    ):
        owner.resolve_images(
            manifest=SimpleNamespace(runtime=SimpleNamespace(control_service="chat")),
            document=json.dumps({"services": {"chat": {"image": reference}}}),
            runtime={
                "execution": "compose",
                "compose_file": "compose.yaml",
                "service": "chat",
            },
            source=tmp_path,
        )


@pytest.mark.parametrize(
    "reference",
    [
        "--privileged",
        "https://registry/image",
        "user:password@registry/image",
        "image\nsecret",
        "image@sha256:123",
        "repo/../image",
    ],
)
def test_invalid_reference_never_reaches_docker(reference):
    from fi.alk.harness.compose_image_lock import image_repository

    with pytest.raises(ValueError, match="compose_image_"):
        image_repository(reference)


def test_generated_digest_does_not_prove_source_ownership():
    with pytest.raises(ValueError, match="compose_image_source_correspondence_missing"):
        image_identity(
            service="chat",
            reference=f"registry.example/team/chat@{DIGEST}",
            platform="linux/amd64",
            inspected={
                "RepoDigests": [f"registry.example/team/chat@{DIGEST}"],
                "Os": "linux",
                "Architecture": "amd64",
            },
            repository="team/chat",
            revision="b" * 40,
            application=True,
            source_declared_digest=False,
        )


def test_digest_change_changes_sealed_bundle_identity(
    tmp_path, resolved_compose_images, monkeypatch
):
    import json
    from fi.alk.harness.compose_integration import compile_bundle
    from fi.alk.harness.job import HarnessJob
    from fi.alk.harness.certified_compose_runtime import CertifiedComposeRuntimeProvider

    source = tmp_path / "source"
    source.mkdir()
    (source / "compose.yaml").write_text(
        "services:\n  chat:\n    image: example/chat:latest\n"
    )
    authoring = tmp_path / "authoring"
    authoring.mkdir()
    contract = {
        "modality": "chat",
        "runtime": {
            "execution": "compose",
            "compose_file": "compose.yaml",
            "service": "chat",
            "interface": {
                "kind": "http",
                "protocol": "json_template",
                "port": 8000,
                "path": "/chat",
            },
        },
    }
    (authoring / "contract.json").write_text(json.dumps(contract))
    job = HarnessJob.model_validate(
        {
            "job_id": "image-lock",
            "run_id": "image-lock",
            "execution": "local",
            "source": {"kind": "local_repository", "local_path": str(source)},
            "agent": {"connector": "auto"},
            "scenario_count": 1,
            "runtime": {"cpu_units": 2, "memory_mb": 4096, "parallelism": 1},
        }
    )
    monkeypatch.setenv("ALK_TRUSTED_COMPOSE_SOCKET", "/private/alk/docker.sock")
    output = tmp_path / "bundle"
    first = compile_bundle(
        source=source,
        job=job,
        authoring=authoring,
        output=output,
        contract=contract,
        include_scenarios=False,
    )
    original = (output / "compose.json").read_bytes()
    previous = CertifiedComposeRuntimeProvider.resolve_images

    def retargeted(self, **kwargs):
        return [
            item.model_copy(
                update={
                    "immutable_reference": "docker.io/example/chat@sha256:" + "f" * 64
                }
            )
            for item in previous(self, **kwargs)
        ]

    monkeypatch.setattr(CertifiedComposeRuntimeProvider, "resolve_images", retargeted)
    second = compile_bundle(
        source=source,
        job=job,
        authoring=authoring,
        output=output,
        contract=contract,
        include_scenarios=False,
    )
    assert first.digest != second.digest
    assert original != (output / "compose.json").read_bytes()
    assert "example/chat:latest" not in (output / "compose.json").read_text()
    evidence = json.loads((output / "image-lock.json").read_text())
    assert evidence["images"][0]["declared_reference"] == "example/chat:latest"
    assert evidence["images"][0]["immutable_reference"].endswith("f" * 64)
