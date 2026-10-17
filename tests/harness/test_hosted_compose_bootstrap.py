import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from fi.alk.harness import hosted_compose_bootstrap as module
from fi.alk.harness.job import HarnessJob
from fi.alk.harness.provision import source_fingerprint


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "work" / "source"
    source.mkdir(parents=True)
    (source / "app.py").write_text("print('hello')\n")
    secrets = tmp_path / "run" / "secrets.json"
    secrets.parent.mkdir()
    secrets.write_text(json.dumps({"TARGET": "target-only", "SIM": "never-forward"}))
    secrets.chmod(0o600)
    job = HarnessJob.model_validate(
        {
            "job_id": "job-1",
            "run_id": "run-1",
            "execution": "hosted",
            "runtime": {"isolation": "dedicated_vm"},
            "source": {
                "kind": "github",
                "repository": "org/repo",
                "visibility": "public",
                "commit_sha": "a" * 40,
            },
            "agent": {
                "connector": "custom",
                "secret_refs": {
                    "TARGET": {
                        "manager": "platform-vault",
                        "key": "target",
                        "purpose": "target_provider",
                    },
                    "SIM": {
                        "manager": "platform-config",
                        "key": "sim",
                        "purpose": "simulator_provider",
                    },
                },
            },
        }
    )
    job_path = source.parent / "job.json"
    job_path.write_text(job.model_dump_json())
    job_path.chmod(0o600)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    profile = module.HostedComposeProfile(
        source=str(source),
        job=str(job_path),
        secrets=str(secrets),
        private_root=str(private),
        initialize_socket=str(tmp_path / "initialize.sock"),
        broker_socket=str(tmp_path / "broker.sock"),
        docker_socket=str(tmp_path / "docker.sock"),
        artifacts=str(tmp_path / "artifacts"),
        client_uid=os.getuid(),
        client_gid=os.getgid(),
    )
    monkeypatch.setattr(module.os, "chown", lambda *a, **k: None)
    monkeypatch.setattr(module, "validate_config", lambda config: None)
    request = module.InitializeRequest(
        operation="initialize",
        job_id=job.job_id,
        source_digest=source_fingerprint(source),
    )
    return profile, job, request


def test_registration_freezes_source_and_only_target_values(setup):
    profile, job, request = setup
    registration = module._Registration(profile)
    result = registration.initialize(request)
    assert result == {"initialized": True, "source_digest": request.source_digest}
    config = registration.broker.config
    assert config.target_configuration == {"TARGET": "target-only"}
    assert config.source_repository == job.source.repository
    assert config.source_revision == job.source.commit_sha
    frozen = Path(config.source) / "app.py"
    assert frozen.read_text() == "print('hello')\n"
    assert stat.S_IMODE(frozen.stat().st_mode) == 0o444
    Path(profile.source, "app.py").write_text("changed")
    Path(profile.secrets).unlink()
    assert registration.initialize(request) == result
    assert frozen.read_text() == "print('hello')\n"
    with pytest.raises(module.BrokerError, match="registration_conflict"):
        registration.initialize(request.model_copy(update={"source_digest": "0" * 64}))
    with pytest.raises(module.BrokerError, match="registration_conflict"):
        registration.initialize(request.model_copy(update={"job_id": "other"}))


@pytest.mark.parametrize(
    ("relative", "content", "error"),
    [
        (".npmrc", "registry=https://registry.npmjs.org/\n", "bundle_secret_file_forbidden"),
        (
            "tests/credential_fixture.py",
            "FAKE_TOKEN = 'ghp_" + "x" * 36 + "'\n",
            "bundle_secret_material_detected",
        ),
    ],
)
def test_raw_source_content_is_admitted_at_provisioning_not_bootstrap(
    setup, tmp_path, relative, content, error
):
    from fi.alk.harness.bundle import BundleError
    from fi.alk.harness.certified_compose_runtime import CertifiedComposeRuntimeProvider

    profile, job, request = setup
    original = Path(profile.source) / relative
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_text(content)
    request = request.model_copy(
        update={"source_digest": source_fingerprint(Path(profile.source))}
    )
    registration = module._Registration(profile)
    registration.initialize(request)
    frozen = Path(registration.broker.config.source)
    assert (frozen / relative).read_text() == content
    assert stat.S_IMODE((frozen / relative).stat().st_mode) == 0o444
    assert source_fingerprint(frozen) == request.source_digest

    provider = CertifiedComposeRuntimeProvider(socket_path=Path(profile.docker_socket))
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "compose.json").write_text(json.dumps({
        "services": {"app": {"image": "example/app:1", "volumes": [f"./{relative}:/fixture"]}}
    }))
    bundle = SimpleNamespace(
        runtime=SimpleNamespace(document="compose.json"),
        capabilities={},
        provenance=SimpleNamespace(
            source_digest=request.source_digest,
            public_material_admissions=job.public_material_admissions,
            public_test_fixture_admissions=job.public_test_fixture_admissions,
        )
    )
    with pytest.raises(BundleError, match=error):
        provider._provision(bundle, frozen, bundle_dir, frozen, 1)
    assert provider._worlds == {}


@pytest.mark.parametrize("attack", ["symlink", "hardlink", "fifo", "directory_symlink"])
def test_source_escape_never_reads_external_bytes(setup, attack):
    profile, _, request = setup
    source = Path(profile.source)
    sentinel = source.parent / "private-sentinel"
    sentinel.write_text("SECRET_SENTINEL")
    if attack == "symlink":
        (source / "escape").symlink_to(sentinel)
    elif attack == "directory_symlink":
        (source / "escape").symlink_to(source.parent, target_is_directory=True)
    elif attack == "hardlink":
        os.link(sentinel, source / "escape")
    else:
        os.mkfifo(source / "escape")
    registration = module._Registration(profile)
    with pytest.raises(module.BrokerError) as error:
        registration.initialize(request)
    assert "SECRET_SENTINEL" not in str(error.value)
    assert registration.broker is None
    assert not list(Path(profile.private_root).glob("source*"))


def test_wrong_digest_removes_unpublished_snapshot(setup):
    profile, _, request = setup
    registration = module._Registration(profile)
    with pytest.raises(module.BrokerError, match="source_digest_mismatch"):
        registration.initialize(request.model_copy(update={"source_digest": "0" * 64}))
    assert registration.broker is None
    assert list(Path(profile.private_root).iterdir()) == []


@pytest.mark.parametrize(
    "bad", [None, {}, {"TARGET": 42}, {"TARGET": {"nested": "secret"}}]
)
def test_required_target_credential_is_strict(setup, bad):
    profile, _, request = setup
    if bad is None:
        Path(profile.secrets).unlink()
    else:
        Path(profile.secrets).write_text(json.dumps(bad))
    with pytest.raises(module.BrokerError):
        module._Registration(profile).initialize(request)
    assert list(Path(profile.private_root).iterdir()) == []


def test_no_target_alias_needs_no_secret_file(setup):
    profile, job, request = setup
    job.agent.secret_refs.clear()
    Path(profile.job).write_text(job.model_dump_json())
    Path(profile.secrets).unlink()
    registration = module._Registration(profile)
    registration.initialize(request)
    assert registration.broker.config.target_configuration == {}


@pytest.mark.parametrize("name", ["job", "secrets"])
@pytest.mark.parametrize("attack", ["symlink", "hardlink", "writable"])
def test_fixed_control_files_cannot_be_redirected(setup, name, attack):
    profile, _, request = setup
    path = Path(getattr(profile, name))
    if attack == "writable":
        path.chmod(0o666)
    else:
        saved = path.with_suffix(".saved")
        path.rename(saved)
        if attack == "symlink":
            path.symlink_to(saved)
        else:
            os.link(saved, path)
    with pytest.raises(module.BrokerError):
        module._Registration(profile).initialize(request)


def test_process_profile_absence_is_noop(setup, monkeypatch, tmp_path):
    profile, job, _ = setup
    monkeypatch.setattr(module, "_PROFILE_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(module, "_INITIALIZE_SOCKET", tmp_path / "absent.sock")
    assert module.initialize_hosted_compose(job, Path(profile.source)) is None


@pytest.mark.parametrize("directory", [False, True])
def test_source_symlink_swap_during_open_is_rejected(setup, monkeypatch, directory):
    profile, _, request = setup
    source = Path(profile.source)
    sentinel = source.parent / "root-private"
    sentinel.mkdir()
    (sentinel / "app.py").write_text("SECRET_SENTINEL")
    if directory:
        (source / "nested").mkdir()
        (source / "nested" / "inner").write_text("safe")
    victim = source / ("nested" if directory else "app.py")
    original_open = os.open
    attacked = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal attacked
        if path == victim.name and kwargs.get("dir_fd") is not None and not attacked:
            attacked = True
            victim.rename(victim.with_name(victim.name + ".old"))
            victim.symlink_to(
                sentinel if directory else sentinel / "app.py",
                target_is_directory=directory,
            )
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", racing_open)
    with pytest.raises(module.BrokerError) as error:
        module._Registration(profile).initialize(request)
    assert attacked
    assert "SECRET_SENTINEL" not in str(error.value)
    assert list(Path(profile.private_root).iterdir()) == []


def test_source_mutation_during_read_is_rejected(setup, monkeypatch):
    profile, _, request = setup
    victim = Path(profile.source) / "app.py"
    original_read = os.read
    attacked = False

    def racing_read(fd, size):
        nonlocal attacked
        result = original_read(fd, size)
        if result and not attacked:
            attacked = True
            victim.write_text("changed during capture")
        return result

    monkeypatch.setattr(module.os, "read", racing_read)
    with pytest.raises(module.BrokerError, match="source_changed"):
        module._Registration(profile).initialize(request)
    assert attacked
    assert list(Path(profile.private_root).iterdir()) == []


def test_contained_source_link_is_preserved(setup):
    profile, _, request = setup
    source = Path(profile.source)
    (source / "linked.py").symlink_to("app.py")
    request = request.model_copy(update={"source_digest": source_fingerprint(source)})
    registration = module._Registration(profile)
    registration.initialize(request)
    frozen = Path(registration.broker.config.source)
    assert (frozen / "linked.py").readlink() == Path("app.py")
    assert (frozen / "linked.py").read_text() == (frozen / "app.py").read_text()


@pytest.mark.parametrize(
    "limit,value",
    [("max_bytes", 1), ("max_file_bytes", 1), ("max_entries", 1), ("max_depth", 1)],
)
def test_source_limits_fail_without_publication(setup, limit, value):
    profile, _, request = setup
    source = Path(profile.source)
    (source / "deep" / "deeper").mkdir(parents=True)
    (source / "deep" / "deeper" / "payload").write_text("extra")
    profile = profile.model_copy(update={limit: value})
    with pytest.raises(module.BrokerError):
        module._Registration(profile).initialize(request)
    assert list(Path(profile.private_root).iterdir()) == []


def test_failed_registration_can_retry_but_attempts_are_bounded(setup):
    profile, _, request = setup
    registration = module._Registration(profile.model_copy(update={"max_failures": 2}))
    bad = request.model_copy(update={"source_digest": "0" * 64})
    for _ in range(2):
        with pytest.raises(module.BrokerError, match="source_digest_mismatch"):
            registration.initialize(bad)
    with pytest.raises(module.BrokerError, match="registration_exhausted"):
        registration.initialize(request)
    retry = module._Registration(profile)
    with pytest.raises(module.BrokerError):
        retry.initialize(bad)
    assert retry.initialize(request)["initialized"] is True


def test_control_parent_symlink_is_rejected(setup):
    profile, _, request = setup
    parent = Path(profile.secrets).parent
    saved = parent.with_name("saved-run")
    parent.rename(saved)
    parent.symlink_to(saved, target_is_directory=True)
    with pytest.raises(module.BrokerError):
        module._Registration(profile).initialize(request)


def test_job_identity_is_checked_before_source_capture(setup):
    profile, _, request = setup
    with pytest.raises(module.BrokerError, match="job_identity_mismatch"):
        module._Registration(profile).initialize(
            request.model_copy(update={"job_id": "other"})
        )
    assert list(Path(profile.private_root).iterdir()) == []


def test_source_ownership_is_required(setup):
    profile, _, request = setup
    profile = profile.model_copy(update={"client_uid": profile.client_uid + 1})
    with pytest.raises(module.BrokerError, match="control_file_invalid"):
        module._Registration(profile).initialize(request)


def test_declared_profile_rejects_missing_socket_and_wrong_source(setup, monkeypatch):
    profile, job, _ = setup
    monkeypatch.setattr(module, "_load_profile", lambda: profile)
    with pytest.raises(module.BrokerError, match="fixed_source_required"):
        module.initialize_hosted_compose(job, Path("/elsewhere"))
    with pytest.raises(module.BrokerError):
        module.initialize_hosted_compose(job, Path(profile.source))


def test_missing_profile_with_socket_is_not_process_fallback(
    setup, monkeypatch, tmp_path
):
    profile, job, _ = setup
    path = tmp_path / "unexpected.sock"
    path.touch()
    monkeypatch.setattr(module, "_PROFILE_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(module, "_INITIALIZE_SOCKET", path)
    with pytest.raises(module.BrokerError, match="profile_required"):
        module.initialize_hosted_compose(job, Path(profile.source))


def test_registration_rejects_duplicate_secret_keys(setup):
    profile, _, request = setup
    Path(profile.secrets).write_text('{"TARGET":"one","TARGET":"two"}')
    with pytest.raises(module.BrokerError, match="duplicate_json_key"):
        module._Registration(profile).initialize(request)


@pytest.mark.parametrize("uid", [0, 2001])
def test_client_verifies_root_peer_before_sending_identity(setup, monkeypatch, uid):
    import socket
    import struct

    profile, job, request = setup
    checked = []
    monkeypatch.setattr(module, "_load_profile", lambda: profile)
    monkeypatch.setattr(module, "_trusted_socket", checked.append)
    monkeypatch.setattr(socket, "SO_PEERCRED", 17, raising=False)
    reply = json.dumps(
        {"result": {"initialized": True, "source_digest": request.source_digest}}
    ).encode()

    class Connection:
        def __init__(self):
            self.received = struct.pack("!I", len(reply)) + reply
            self.sent = b""

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def settimeout(self, value):
            assert value > 0

        def connect(self, path):
            assert path == profile.initialize_socket

        def getsockopt(self, *_):
            return struct.pack("3i", 123, uid, 2000)

        def sendall(self, payload):
            self.sent += payload

        def recv(self, size):
            part, self.received = self.received[:size], self.received[size:]
            return part

    connection = Connection()
    monkeypatch.setattr(socket, "socket", lambda *a: connection)
    if uid:
        with pytest.raises(module.BrokerError, match="root_peer_required"):
            module.initialize_hosted_compose(job, Path(profile.source))
        assert connection.sent == b""
    else:
        assert module.initialize_hosted_compose(job, Path(profile.source)) == Path(
            profile.broker_socket
        )
        assert checked == [Path(profile.initialize_socket), Path(profile.broker_socket)]
        assert json.loads(connection.sent[4:]) == request.model_dump()
        assert b"target-only" not in connection.sent
        assert b"never-forward" not in connection.sent


def test_socket_parent_must_be_root_controlled(tmp_path):
    with pytest.raises(module.BrokerError, match="root_owned_path_required"):
        module._trusted_socket(tmp_path / "socket")


@pytest.mark.parametrize("uid,mode", [(2000, 0o620), (0, 0o627)])
def test_socket_owner_and_private_mode_are_required(tmp_path, monkeypatch, uid, mode):
    from types import SimpleNamespace

    monkeypatch.setattr(module, "_root_controlled", lambda *a, **k: None)
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda *a: SimpleNamespace(st_uid=uid, st_mode=stat.S_IFSOCK | mode),
    )
    with pytest.raises(module.BrokerError, match="private_socket_required"):
        module._trusted_socket(tmp_path / "socket")


def test_untrusted_profile_cannot_enable_compose(setup, monkeypatch, tmp_path):
    profile, job, _ = setup
    path = tmp_path / "profile.json"
    path.write_text(profile.model_dump_json())
    monkeypatch.setattr(module, "_PROFILE_PATH", path)
    with pytest.raises(module.BrokerError, match="root_owned_path_required"):
        module.initialize_hosted_compose(job, Path(profile.source))


def test_receive_deadline_cannot_be_extended_by_partial_writes(monkeypatch):
    from types import SimpleNamespace

    now = iter([1.0, 2.0, 4.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(now))
    timeouts = []
    connection = SimpleNamespace(settimeout=timeouts.append, recv=lambda count: b"\0")
    with pytest.raises(module.BrokerError, match="receive_deadline"):
        module._receive_before(connection, 3.0)
    assert timeouts == [2.0, 1.0]


def test_source_owner_mismatch_is_rejected_after_job_capture(setup, monkeypatch):
    from types import SimpleNamespace

    profile, _, request = setup
    original = os.stat

    def wrong_owner(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if path == "app.py" and kwargs.get("dir_fd") is not None:
            return SimpleNamespace(st_uid=profile.client_uid + 1)
        return info

    monkeypatch.setattr(module.os, "stat", wrong_owner)
    with pytest.raises(module.BrokerError, match="source_owner_invalid"):
        module._Registration(profile).initialize(request)
    assert list(Path(profile.private_root).iterdir()) == []


def test_secret_with_wrong_purpose_is_never_forwarded(setup):
    profile, job, request = setup
    reference = job.agent.secret_refs["TARGET"]
    reference.purpose = "simulator_provider"
    reference.manager = "platform-config"
    Path(profile.job).write_text(job.model_dump_json())
    registration = module._Registration(profile)
    registration.initialize(request)
    assert registration.broker.config.target_configuration == {}


def test_failed_broker_validation_does_not_poison_retry(setup, monkeypatch):
    profile, _, request = setup

    def fail(_):
        raise module.BrokerError("compose_broker_test_failure")

    monkeypatch.setattr(module, "validate_config", fail)
    registration = module._Registration(profile)
    with pytest.raises(module.BrokerError, match="compose_broker_test_failure"):
        registration.initialize(request)
    assert list(Path(profile.private_root).iterdir()) == []
    assert not Path(profile.artifacts).exists()
    monkeypatch.setattr(module, "validate_config", lambda config: None)
    assert registration.initialize(request)["initialized"] is True


def test_public_error_code_preserves_only_safe_compose_codes():
    assert module._public_error_code(
        RuntimeError("compose_resource_exhausted: private detail")
    ) == "compose_resource_exhausted"
    assert module._public_error_code(
        RuntimeError("private failure: secret")
    ) == "compose_bootstrap_request_failed"


def test_admission_for_different_source_fails_before_broker_creation(setup):
    from fi.alk.harness.public_material import PublicMaterialAdmission

    profile, job, request = setup
    job.public_material_admissions = [
        PublicMaterialAdmission(
            source_digest="0" * 64,
            path="app.py",
            file_sha256="0" * 64,
            pattern="github_token",
            match_sha256="0" * 64,
            reason="Reviewed public fixture",
            evidence="Public repository file",
        )
    ]
    Path(profile.job).write_text(job.model_dump_json())
    registration = module._Registration(profile)
    with pytest.raises(module.BrokerError):
        registration.initialize(request)
    assert registration.broker is None
    assert list(Path(profile.private_root).iterdir()) == []


def test_expired_registration_cannot_copy_source(setup, monkeypatch):
    profile, _, request = setup
    registration = module._Registration(profile)
    monkeypatch.setattr(module.time, "monotonic", lambda: registration.deadline + 1)
    with pytest.raises(module.BrokerError, match="registration_exhausted"):
        registration.initialize(request)
    assert list(Path(profile.private_root).iterdir()) == []


def test_existing_snapshot_is_never_removed_by_failed_registration(setup):
    profile, _, request = setup
    previous = Path(profile.private_root) / "source"
    previous.mkdir()
    marker = previous / "previous-job.txt"
    marker.write_text("preserve this snapshot")
    with pytest.raises(module.BrokerError):
        module._Registration(profile).initialize(request)
    assert marker.read_text() == "preserve this snapshot"
    assert list(Path(profile.private_root).iterdir()) == [previous]
