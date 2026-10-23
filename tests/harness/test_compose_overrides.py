import json
import os
import shutil
import subprocess
import uuid

import pytest

from fi.alk.harness.contract import Runtime
from fi.alk.harness.compose_merge import merge_compose
from fi.alk.harness.runtime_selection import select_runtime
from fi.alk.harness.certified_compose_runtime import (
    normalize_compose,
    ComposeRuntimeError,
)


def files(root, image="example/image:1"):
    base = root / "base.yaml"
    overlay = root / "small.yaml"
    base.write_text(f"""services:
  gateway:
    image: {image}
    command: [sleep, '120']
    environment: [MODE=large, UNUSED_ENDPOINT=http://optional:8080]
    depends_on: [ledger, optional]
  ledger:
    image: {image}
    command: [sleep, '120']
  optional:
    image: {image}
    command: [sleep, '120']
""")
    overlay.write_text("""services:
  gateway:
    environment: [MODE=small]
    depends_on:
      optional: {condition: service_started, required: false}
  optional:
    profiles: [analytics]
""")
    return base, overlay


def selection(root, **extra):
    return select_runtime(
        root,
        {
            "execution": "compose",
            "compose_file": "base.yaml",
            "compose_overrides": ["small.yaml"],
            "service": "gateway",
            **extra,
        },
    )


@pytest.fixture
def cli():
    if not shutil.which("docker"):
        pytest.skip("native Compose CLI unavailable")
    if subprocess.run(["docker", "compose", "version"], capture_output=True).returncode:
        pytest.skip("native Compose CLI unavailable")


def test_native_merge_and_optional_profile_selection(tmp_path, cli, monkeypatch):
    base, overlay = files(tmp_path)
    original = [p.read_bytes() for p in [base, overlay]]
    monkeypatch.setenv("COMPOSE_PROFILES", "analytics")
    monkeypatch.setenv("MODE", "ambient-private")
    (tmp_path / ".env").write_text("MODE=dotenv-private\n")
    merged = merge_compose(tmp_path, [base, overlay])
    assert "MODE=small" in merged["services"]["gateway"]["environment"]
    assert "ambient-private" not in json.dumps(merged)
    assert "dotenv-private" not in json.dumps(merged)
    selected = selection(tmp_path)
    assert list(selected.services) == ["ledger", "gateway"]
    assert list(selected.services["gateway"]["depends_on"]) == ["ledger"]
    assert set(selection(tmp_path, additional_services=["optional"]).services) == {
        "ledger",
        "gateway",
        "optional",
    }
    assert [p.read_bytes() for p in [base, overlay]] == original


def test_native_override_order_and_reset_tags(tmp_path, cli):
    base, overlay = files(tmp_path)
    third = tmp_path / "last.yaml"
    third.write_text("services:\n  gateway:\n    environment: !override [MODE=last]\n")
    merged = merge_compose(tmp_path, [base, overlay, third])
    assert merged["services"]["gateway"]["environment"] == ["MODE=last"]
    reverse = merge_compose(tmp_path, [base, third, overlay])
    assert reverse["services"]["gateway"]["environment"] == ["MODE=small"]


@pytest.mark.parametrize(
    "body",
    [
        "include: /etc/secret.yaml\nservices: {}",
        "services:\n  app:\n    extends: {file: /etc/secret.yaml, service: app}",
        "services:\n  app:\n    env_file: /etc/secret.env",
        "services:\n  app:\n    env_file: [{path: ../outside.env}]",
        "services:\n  app:\n    label_file: /etc/secret.env",
        "secrets: {token: {file: /etc/secret}}\nservices: {}",
    ],
)
def test_external_read_rejected_before_cli(tmp_path, monkeypatch, body):
    base, overlay = files(tmp_path)
    overlay.write_text(body)

    def forbidden(*args, **kwargs):
        pytest.fail("CLI invoked before input admission")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(ValueError):
        merge_compose(tmp_path, [base, overlay])


def test_symlink_escape_and_size_rejected_before_cli(tmp_path, monkeypatch):
    base, overlay = files(tmp_path)
    overlay.unlink()
    overlay.symlink_to("/etc/hosts")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("CLI invoked"))
    with pytest.raises(ValueError, match="outside_source"):
        merge_compose(tmp_path, [base, overlay])
    overlay.unlink()
    overlay.write_text("#" * (2 * 1024 * 1024))
    with pytest.raises(ValueError, match="input_too_large"):
        merge_compose(tmp_path, [base, overlay])


@pytest.mark.parametrize(
    "overrides",
    [
        ["base.yaml"],
        ["small.yaml"] * 2,
        ["../other.yaml"],
        ["/etc/file"],
        [""],
        ["a"] * 9,
    ],
)
def test_override_contract_rejects_bad_paths_and_counts(overrides):
    with pytest.raises(ValueError):
        Runtime(
            execution="compose",
            compose_file="base.yaml",
            service="gateway",
            compose_overrides=overrides,
        )


def test_single_file_optional_unprofiled_and_required_profiled_stay_selected(tmp_path):
    base, _ = files(tmp_path)
    base.write_text(
        base.read_text()
        .replace(
            "depends_on: [ledger, optional]",
            "depends_on:\n      ledger: {condition: service_started, required: false}\n      optional: {condition: service_started, required: true}",
        )
        .replace("  optional:\n", "  optional:\n    profiles: [analytics]\n")
    )
    result = select_runtime(
        tmp_path,
        {"execution": "compose", "compose_file": "base.yaml", "service": "gateway"},
    )
    assert set(result.services) == {"gateway", "ledger", "optional"}


def test_merged_privilege_still_rejected(tmp_path, cli):
    files(tmp_path)
    (tmp_path / "small.yaml").write_text(
        "services:\n  gateway:\n    privileged: true\n"
    )
    selected = selection(tmp_path)
    with pytest.raises(ComposeRuntimeError):
        normalize_compose({"services": selected.services}, tmp_path, capabilities={})


def test_live_overlay_starts_only_selected_two_services(tmp_path, cli):
    image = os.environ.get("ALK_TEST_COMPOSE_IMAGE")
    if not image:
        pytest.skip("requires explicitly selected cached Docker image")
    files(tmp_path, image)
    selected = selection(tmp_path)
    document = normalize_compose(
        {"services": selected.services}, tmp_path, capabilities={}
    )
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(document))
    project = "alk-overlay-" + uuid.uuid4().hex[:10]

    def compose(*args):
        return subprocess.run(
            ["docker", "compose", "-p", project, "-f", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout

    try:
        compose("up", "-d", "--pull", "never")
        assert set(compose("ps", "--services", "--status", "running").split()) == {
            "gateway",
            "ledger",
        }
        assert compose("exec", "-T", "gateway", "printenv", "MODE").strip() == "small"
    finally:
        compose("down", "--volumes", "--remove-orphans")


def test_overlay_compiles_and_seals_through_native_bundle(tmp_path, cli, monkeypatch):
    from fi.alk.harness.bundle_author_v2 import author_bundle_v2
    from fi.alk.harness.job import HarnessJob
    from fi.alk.harness.process_preflight import preflight_bundle

    source = tmp_path / "source"
    source.mkdir()
    files(source)
    authoring = tmp_path / "authoring"
    scenario = authoring / "scenarios" / "one"
    (scenario / "checks").mkdir(parents=True)
    (scenario / "scenario.json").write_text(
        json.dumps({"name": "one", "instruction": "hello", "sub_goals": ["response"]})
    )
    for name in ["setup", "ready"]:
        (scenario / f"{name}.py").write_text(f"def {name}(world):\n    return None\n")
    (scenario / "checks/response.py").write_text(
        'def check(world,calls):\n    return "no result"\n'
    )
    (authoring / "contract.json").write_text(
        json.dumps(
            {
                "modality": "chat",
                "runtime": {
                    "execution": "compose",
                    "compose_file": "base.yaml",
                    "compose_overrides": ["small.yaml"],
                    "service": "gateway",
                    "interface": {
                        "kind": "http",
                        "protocol": "json_template",
                        "port": 8080,
                        "path": "/chat",
                    },
                },
            }
        )
    )
    job = HarnessJob.model_validate(
        {
            "job_id": "overlay",
            "run_id": "overlay",
            "execution": "local",
            "source": {"kind": "local_repository", "local_path": str(source)},
            "agent": {"connector": "auto"},
            "scenario_count": 1,
            "runtime": {"cpu_units": 2, "memory_mb": 4096, "parallelism": 1},
        }
    )
    monkeypatch.setenv("ALK_TRUSTED_COMPOSE_SOCKET", "/private/alk/docker.sock")
    output = tmp_path / "bundle"
    manifest = author_bundle_v2(
        source=source, job=job, authoring=authoring, output=output
    )
    preflight_bundle(
        output, manifest, parallelism=1, secret_refs={}, allow_compose=True
    )
    document = json.loads((output / "compose.json").read_text())
    assert set(document["services"]) == {"ledger", "gateway"}
    assert "MODE=small" in document["services"]["gateway"]["environment"]
    assert manifest.runtime.control_service == "gateway"
    assert (source / "small.yaml").read_text().startswith("services:")


@pytest.mark.parametrize(
    "field",
    [
        "extends: {file: /etc/host.yaml, service: x}",
        "env_file: /etc/private.env",
        "label_file: /etc/private.env",
    ],
)
def test_yaml_anchor_cannot_hide_file_reads(tmp_path, monkeypatch, field):
    base, overlay = files(tmp_path)
    overlay.write_text(
        "x-shared: &shared\n  " + field + "\nservices:\n  gateway:\n    <<: *shared\n"
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("CLI invoked"))
    with pytest.raises(ValueError):
        merge_compose(tmp_path, [base, overlay])


@pytest.mark.parametrize(
    "body",
    [
        "x-common: &common\n  include: /etc/host.yaml\n<<: *common\nservices: {}\n",
        "x-loop: &loop\n  <<: *loop\nservices:\n  gateway:\n    <<: *loop\n",
    ],
)
def test_top_level_anchor_and_recursive_merge_rejected_before_cli(
    tmp_path, monkeypatch, body
):
    base, overlay = files(tmp_path)
    overlay.write_text(body)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("CLI invoked"))
    with pytest.raises(ValueError):
        merge_compose(tmp_path, [base, overlay])


@pytest.mark.parametrize("activation", ["root", "public"])
def test_optional_dependency_in_active_profile_is_preserved(tmp_path, cli, activation):
    files(tmp_path)
    if activation == "root":
        path = tmp_path / "small.yaml"
        path.write_text(
            path.read_text().replace(
                "  gateway:\n", "  gateway:\n    profiles: [analytics]\n"
            )
        )
    selected = select_runtime(
        tmp_path,
        {
            "execution": "compose",
            "compose_file": "base.yaml",
            "compose_overrides": ["small.yaml"],
            "service": "gateway",
        },
        public_environment={"COMPOSE_PROFILES": "analytics"}
        if activation == "public"
        else {},
    )
    assert set(selected.services) == {"gateway", "ledger", "optional"}
    assert "optional" in selected.services["gateway"]["depends_on"]


@pytest.mark.parametrize("activation", ["root", "public"])
def test_live_native_active_profile_matches_selection(tmp_path, cli, activation):
    image = os.environ.get("ALK_TEST_COMPOSE_IMAGE")
    if not image:
        pytest.skip("requires explicitly selected cached Docker image")
    base, overlay = files(tmp_path, image)
    if activation == "root":
        overlay.write_text(
            overlay.read_text().replace(
                "  gateway:\n", "  gateway:\n    profiles: [analytics]\n"
            )
        )
    public = {"COMPOSE_PROFILES": "analytics"} if activation == "public" else {}
    selected = select_runtime(
        tmp_path,
        {
            "execution": "compose",
            "compose_file": "base.yaml",
            "compose_overrides": ["small.yaml"],
            "service": "gateway",
        },
        public_environment=public,
    )
    project = "alk-profile-" + uuid.uuid4().hex[:10]
    env = {key: value for key, value in os.environ.items() if key != "COMPOSE_PROFILES"}
    env.update(public)

    def compose(*args):
        return subprocess.run(
            [
                "docker",
                "compose",
                "--env-file",
                "/dev/null",
                "-p",
                project,
                "-f",
                str(base),
                "-f",
                str(overlay),
                *args,
            ],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout

    try:
        compose("up", "-d", "--pull", "never", "gateway")
        assert (
            set(compose("ps", "--services", "--status", "running").split())
            == set(selected.services)
            == {
                "gateway",
                "ledger",
                "optional",
            }
        )
    finally:
        compose("down", "--volumes", "--remove-orphans")


@pytest.mark.parametrize(
    "options", [{"nocopy": True}, {"subpath": "nested"}, [], None, False]
)
def test_named_volume_nonempty_or_malformed_options_remain_rejected(tmp_path, options):
    document = {
        "services": {
            "gateway": {
                "image": "example/image:1",
                "volumes": [
                    {
                        "type": "volume",
                        "source": "storage",
                        "target": "/data",
                        "volume": options,
                    }
                ],
            }
        },
        "volumes": {"storage": {}},
    }
    with pytest.raises(ComposeRuntimeError, match="mount_fields_forbidden"):
        normalize_compose(document, tmp_path, capabilities={})


def test_native_plain_named_volume_normalizes_like_short_syntax(tmp_path, cli):
    base, overlay = files(tmp_path)
    overlay.write_text(
        "services:\n  gateway:\n    volumes: [storage:/data]\nvolumes:\n  storage: {}\n"
    )
    merged = merge_compose(tmp_path, [base, overlay])
    assert merged["services"]["gateway"]["volumes"] == [
        {
            "type": "volume",
            "source": "storage",
            "target": "/data",
            "volume": {},
        }
    ]
    normalized = normalize_compose(merged, tmp_path, capabilities={})
    merged["services"]["gateway"]["volumes"] = ["storage:/data"]
    short = normalize_compose(merged, tmp_path, capabilities={})
    native_mount = normalized["services"]["gateway"]["volumes"][0]
    short_mount = short["services"]["gateway"]["volumes"][0]
    assert native_mount == {"type": "volume", "source": "storage", "target": "/data"}
    assert native_mount.get("read_only", False) is short_mount.pop("read_only") is False
    assert native_mount == short_mount
