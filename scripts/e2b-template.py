#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["e2b==2.37.1"]
# ///
"""Build, publish, and certify the complete hosted ALK E2B template.

E2B's Dockerfile parser does not support the multi-stage Dockerfile.hosted.
This publisher therefore lets Docker BuildKit build the authoritative image,
pushes that image by immutable digest, imports the digest into E2B, and then
certifies every capability declared by hosted-snapshot/catalog.json.

Usage from a clean ALK checkout:

    docker login
    uv run scripts/e2b-template.py \
      --env-file /path/to/platform/.env \
      --image-repository futureagi/alk-hosted-runtime \
      --template-name alk-hosted-production

For a private registry, also export E2B_REGISTRY_USERNAME and
E2B_REGISTRY_PASSWORD. Docker push authentication continues to come from the
normal Docker credential store; registry credentials are passed only to E2B so
its builder can pull the immutable image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

E2B_SDK_VERSION = "2.37.1"
DEFAULT_IMAGE_REPOSITORY = "futureagi/alk-hosted-runtime"
DEFAULT_TEMPLATE_NAME = "alk-hosted-production"
DEFAULT_CPU_COUNT = 4
DEFAULT_MEMORY_MB = 8192
DEFAULT_DISK_GB = 10
DEFAULT_CERTIFICATION_TTL_SECONDS = 900

PLATFORM_BOOTSTRAP_COMMAND = (
    "install -d -o svc-control -g svc-control -m 0700 "
    "/run/futureagi /run/user/2000 && "
    "rm -f /usr/local/bin/python && "
    "printf '#!/bin/sh\\nexec /opt/alk-venv/bin/python \"$@\"\\n' "
    "> /usr/local/bin/python && chmod 0755 /usr/local/bin/python && "
    "ln -sfn /opt/alk-venv/bin/pip /usr/local/bin/pip && "
    "if [ -x /opt/alk-venv/bin/uv ]; then "
    "ln -sfn /opt/alk-venv/bin/uv /usr/local/bin/uv; fi && "
    "if [ -x /opt/alk-venv/bin/uvx ]; then "
    "ln -sfn /opt/alk-venv/bin/uvx /usr/local/bin/uvx; fi && "
    "test -x /usr/local/bin/uv && test -x /usr/local/bin/uvx"
)
HOSTED_RUNTIME_ENV = {
    "PATH": (
        "/opt/alk-venv/bin:/usr/lib/postgresql/16/bin:/opt/erlang/bin:"
        "/opt/rabbitmq/sbin:/opt/node22/bin:/usr/local/sbin:/usr/local/bin:"
        "/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PIP_DEFAULT_TIMEOUT": "300",
    "PIP_RETRIES": "10",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
    "PIP_CERT": "/etc/ssl/certs/ca-certificates.crt",
    "PYTHONDONTWRITEBYTECODE": "1",
    "XDG_RUNTIME_DIR": "/run/user/2000",
}


@dataclass(frozen=True)
class PublishedImage:
    tag: str
    digest: str

    @property
    def reference(self) -> str:
        return f"{self.tag.rsplit(':', 1)[0]}@{self.digest}"


@dataclass(frozen=True)
class CertificationResult:
    sandbox_id: str
    disk_gb: int
    checks: tuple[str, ...]


def _run(
    command: list[str],
    *,
    cwd: Path,
    capture: bool = False,
) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def _git(repo: Path, *args: str) -> str:
    return _run(["git", *args], cwd=repo, capture=True)


def _load_selected_env(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"environment file not found: {path}")
    allowed = {
        "E2B_API_KEY",
        "E2B_REGISTRY_USERNAME",
        "E2B_REGISTRY_PASSWORD",
    }
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        name, value = line.split("=", 1)
        name = name.strip()
        if name not in allowed or name in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[name] = value


def _load_catalog(path: Path) -> dict[str, Any]:
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"invalid hosted catalog {path}: {exc}") from exc
    if catalog.get("schema_version") != "futureagi.hosted-snapshot-catalog.v1":
        raise SystemExit("unsupported hosted catalog schema")
    runtimes = catalog.get("runtimes")
    engines = catalog.get("engines")
    binaries = catalog.get("binaries")
    if not isinstance(runtimes, dict) or not isinstance(engines, dict):
        raise SystemExit("hosted catalog must declare runtimes and engines")
    if not isinstance(binaries, list):
        raise SystemExit("hosted catalog must declare binaries")
    supported_engines = {"postgres", "redis", "rabbitmq"}
    unknown = set(engines) - supported_engines
    if unknown:
        raise SystemExit(
            "certifier has no behavioral check for catalog engines: "
            + ", ".join(sorted(unknown))
        )
    return catalog


def _assert_clean_checkout(repo: Path, *, allow_dirty: bool) -> None:
    dirty = _git(repo, "status", "--porcelain")
    if dirty and not allow_dirty:
        raise SystemExit(
            "refusing to publish from a dirty tracked checkout; commit the release "
            "or pass --allow-dirty for an explicit local experiment"
        )


def _parse_image_digest(metadata_path: Path) -> str:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = str(metadata.get("containerimage.digest") or "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise SystemExit(
            "BuildKit did not return an immutable containerimage.digest; "
            f"metadata keys={sorted(metadata)}"
        )
    return digest


def build_and_push_image(
    *,
    repo: Path,
    repository: str,
    image_tag: str,
    source_revision: str,
    no_cache: bool,
) -> PublishedImage:
    tag = f"{repository}:{image_tag}"
    with tempfile.TemporaryDirectory(prefix="alk-e2b-image-") as temporary:
        metadata_path = Path(temporary) / "metadata.json"
        command = [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64",
            "--file",
            str(repo / "Dockerfile.hosted"),
            "--build-arg",
            f"ALK_HOSTED_SOURCE_REVISION={source_revision}",
            "--tag",
            tag,
            "--metadata-file",
            str(metadata_path),
            "--provenance=false",
            "--sbom=false",
            "--push",
        ]
        if no_cache:
            command.append("--no-cache")
        command.append(str(repo))
        _run(command, cwd=repo)
        digest = _parse_image_digest(metadata_path)
    print(f"IMAGE_TAG={tag}")
    print(f"IMAGE_DIGEST={digest}")
    print(f"IMAGE_REFERENCE={repository}@{digest}")
    return PublishedImage(tag=tag, digest=digest)


def _build_log(entry: Any) -> None:
    message = getattr(entry, "message", None)
    print(message if message is not None else entry, flush=True)


def publish_e2b_template(
    *,
    image_reference: str,
    template_name: str,
    template_alias: str | None,
    cpu_count: int,
    memory_mb: int,
    source_revision: str,
    api_key: str,
    skip_cache: bool,
):
    from e2b import Template

    registry_username = os.environ.get("E2B_REGISTRY_USERNAME") or None
    registry_password = os.environ.get("E2B_REGISTRY_PASSWORD") or None
    if bool(registry_username) != bool(registry_password):
        raise SystemExit(
            "E2B_REGISTRY_USERNAME and E2B_REGISTRY_PASSWORD must be supplied together"
        )
    image_kwargs: dict[str, str] = {}
    if registry_username and registry_password:
        image_kwargs = {
            "username": registry_username,
            "password": registry_password,
        }
    template = (
        Template()
        .from_image(image_reference, **image_kwargs)
        .set_envs(HOSTED_RUNTIME_ENV)
        .set_user("svc-control")
        .set_workdir("/work")
    )
    build = Template.build(
        template,
        name=template_name,
        alias=template_alias,
        tags=["hosted-production", f"alk-{source_revision[:12]}"],
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        skip_cache=skip_cache,
        api_key=api_key,
        on_build_logs=_build_log,
    )
    if not build.build_id or not build.name:
        raise SystemExit(f"E2B returned an incomplete build result: {build!r}")
    reference = f"{build.name}:{build.build_id}"
    print(f"E2B_TEMPLATE_NAME={build.name}")
    print(f"E2B_TEMPLATE_BUILD_ID={build.build_id}")
    print(f"E2B_TEMPLATE_REFERENCE={reference}")
    return build, reference


def _sandbox_command(
    sandbox: Any,
    command: str,
    *,
    label: str,
    user: str = "svc-control",
    timeout: int = 120,
) -> str:
    result = sandbox.commands.run(command, user=user, timeout=timeout)
    exit_code = int(getattr(result, "exit_code", 1))
    stdout = str(getattr(result, "stdout", "") or "")
    stderr = str(getattr(result, "stderr", "") or "")
    if exit_code != 0:
        tail = "\n".join(part for part in (stdout[-2000:], stderr[-2000:]) if part)
        raise RuntimeError(f"certification failed [{label}] exit={exit_code}\n{tail}")
    print(f"CERTIFIED {label}")
    return stdout


def _base_checks(catalog: dict[str, Any], *, compose_profile: bool = False) -> list[tuple[str, str]]:
    checks: list[tuple[str, str]] = [
        (
            "control-layout",
            (
                "test \"$(stat -c '%a:%U:%G' /run/futureagi)\" = "
                "'700:svc-control:svc-control' && "
                "test \"$(stat -c '%a:%U:%G' /run/user/2000)\" = "
                "'700:svc-control:svc-control' && "
                "test \"$(stat -c '%a:%U:%G' /work)\" = "
                "'755:svc-control:svc-control' && "
                "test \"$(stat -c '%a:%U:%G' /work/artifacts)\" = "
                "'755:svc-control:svc-control'"
            ),
        ),
        (
            "service-users",
            (
                'test "$(id -u svc-control)" = 2000 && '
                'test "$(id -u svc-agent)" = 2001 && '
                'test "$(id -u svc-tools)" = 2002 && '
                'test "$(id -u svc-data)" = 2003'
            ),
        ),
        (
            "alk-import",
            (
                "python -c 'import fi.alk.harness.hosted_entrypoint, "
                "fi.alk.harness.bundle_author_v2'"
            ),
        ),
        (
            "no-container-runtime",
            (
                "test ! -e /var/run/docker.sock && ! command -v docker && "
                "! command -v dockerd && ! command -v docker-compose"
            ),
        ),
        (
            "no-heavy-ml-packages",
            """python - <<'PY'
from importlib.metadata import distributions

blocked = {
    "sentence-transformers",
    "tensorflow",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
}
installed = {
    str(distribution.metadata["Name"]).lower()
    for distribution in distributions()
    if distribution.metadata["Name"]
}
present = sorted(blocked & installed)
if present:
    raise SystemExit("forbidden heavyweight ML packages: " + ", ".join(present))
PY""",
        ),
    ]
    for version in catalog["runtimes"].get("python", []):
        token = re.sub(r"[^0-9.]", "", str(version))
        if not re.fullmatch(r"\d+\.\d+", token):
            raise SystemExit(f"invalid Python catalog version: {version!r}")
        checks.append(
            (
                f"python-{token}",
                (
                    f"python{token} -c 'import encodings,sys; "
                    f"assert sys.version_info[:2] == ({token.replace('.', ', ')})' && "
                    f"python{token} -m venv /tmp/cert-python-{token} && "
                    f"/tmp/cert-python-{token}/bin/python -c 'import encodings' && "
                    f"rm -rf /tmp/cert-python-{token}"
                ),
            )
        )
    for version in catalog["runtimes"].get("node", []):
        major = str(version)
        if not major.isdigit():
            raise SystemExit(f"invalid Node catalog version: {version!r}")
        checks.append(
            (
                f"node-{major}",
                (
                    f'node{major} -e \'if (process.versions.node.split(".")[0] !== '
                    f'"{major}") process.exit(1)\' && npm{major} --version'
                ),
            )
        )
    for binary in catalog["binaries"]:
        name = str(binary)
        if not re.fullmatch(r"[a-zA-Z0-9._+-]+", name):
            raise SystemExit(f"invalid binary catalog entry: {binary!r}")
        version_flag = "-version" if name == "ffmpeg" else "--version"
        checks.append((f"binary-{name}", f"command -v {name} && {name} {version_flag}"))
    if compose_profile:
        checks = [(label, command) for label, command in checks if label != "no-container-runtime"]
        checks.append(("hosted-compose-profile", """python - <<'PY_CHECK'
import os, socket, stat
from pathlib import Path
from fi.alk.harness.hosted_compose_bootstrap import _load_profile
profile = _load_profile()
assert profile is not None
assert os.getuid() == profile.client_uid
for path in Path("/usr/local").rglob("*"):
    info = path.lstat()
    assert info.st_uid == 0, str(path)
    if not stat.S_ISLNK(info.st_mode):
        assert not info.st_mode & 0o022, (str(path), oct(stat.S_IMODE(info.st_mode)))
for raw in (
    "/usr/local/bin", "/usr/local/sbin", "/usr/bin", "/usr/sbin", "/bin", "/sbin",
    "/opt/alk-venv/bin/python", "/usr/bin/docker", "/usr/bin/dockerd",
    "/usr/bin/containerd", "/usr/bin/runc", "/usr/bin/timeout",
    "/usr/libexec/docker/cli-plugins/docker-compose",
):
    lexical = Path(raw)
    resolved = lexical.resolve(strict=True)
    for path in {lexical, resolved, *lexical.parents, *resolved.parents}:
        info = path.stat()
        assert info.st_uid == 0 and not info.st_mode & 0o022, (str(path), info.st_uid, oct(stat.S_IMODE(info.st_mode)))
assert stat.S_ISSOCK(Path(profile.initialize_socket).stat().st_mode)
try:
    connection = socket.socket(socket.AF_UNIX)
    connection.connect(profile.docker_socket)
except PermissionError:
    pass
else:
    raise AssertionError("controller reached private Docker socket")
assert not Path('/var/run/docker.sock').exists()
PY_CHECK"""))
    return checks


def _postgres_check(version: str) -> str:
    return f"""
set -eu
test "$(postgres --version | sed -E 's/.* ([0-9]+).*/\\1/')" = {version}
root=$(mktemp -d /tmp/alk-postgres-cert.XXXXXX)
cleanup() {{
  /usr/lib/postgresql/{version}/bin/pg_ctl -D "$root/data" -m immediate stop >/dev/null 2>&1 || true
  rm -rf "$root"
}}
trap cleanup EXIT
initdb -D "$root/data" --no-locale --encoding=UTF8 >/dev/null
postgres -D "$root/data" -k "$root" -p 55432 >"$root/postgres.log" 2>&1 &
for _ in $(seq 1 60); do
  pg_isready -h "$root" -p 55432 >/dev/null 2>&1 && break
  sleep 0.25
done
pg_isready -h "$root" -p 55432 >/dev/null
psql -h "$root" -p 55432 -d postgres -Atqc 'select 1' | grep -qx 1
cleanup
trap - EXIT
""".strip()


def _redis_check(version: str) -> str:
    major = version.split(".", 1)[0]
    return f"""
set -eu
redis-server --version | grep -Eq 'v={re.escape(major)}\\.'
root=$(mktemp -d /tmp/alk-redis-cert.XXXXXX)
cleanup() {{
  redis-cli -h 127.0.0.1 -p 56379 shutdown nosave >/dev/null 2>&1 || true
  rm -rf "$root"
}}
trap cleanup EXIT
redis-server --bind 127.0.0.1 --port 56379 --dir "$root" --daemonize yes
for _ in $(seq 1 40); do
  redis-cli -h 127.0.0.1 -p 56379 ping 2>/dev/null | grep -qx PONG && break
  sleep 0.25
done
redis-cli -h 127.0.0.1 -p 56379 ping | grep -qx PONG
cleanup
trap - EXIT
""".strip()


def _rabbitmq_check(version: str) -> str:
    major_minor = ".".join(version.split(".")[:2])
    return f"""
export PATH=/opt/erlang/bin:/opt/rabbitmq/sbin:/usr/local/bin:/usr/bin:/bin
set -eu
rabbitmqctl version | grep -Eq '^{re.escape(major_minor)}\\.'
root=$(mktemp -d /tmp/alk-rabbitmq-cert.XXXXXX)
cleanup() {{
  RABBITMQ_NODENAME=alk_cert@localhost rabbitmqctl -q stop >/dev/null 2>&1 || true
  rm -rf "$root"
}}
trap cleanup EXIT
mkdir -p "$root/home" "$root/mnesia" "$root/log"
printf '%s\n' '[rabbitmq_management].' > "$root/enabled_plugins"
cat > "$root/rabbitmq.conf" <<'EOF'
listeners.tcp.default = 55672
management.tcp.port = 55673
default_user = harness
default_pass = harness-cert
loopback_users.guest = false
EOF
export HOME="$root/home"
export RABBITMQ_NODENAME=alk_cert@localhost
export RABBITMQ_NODE_PORT=55672
export RABBITMQ_DIST_PORT=55674
export RABBITMQ_CONFIG_FILE="$root/rabbitmq"
export RABBITMQ_ENABLED_PLUGINS_FILE="$root/enabled_plugins"
export RABBITMQ_MNESIA_BASE="$root/mnesia"
export RABBITMQ_LOG_BASE="$root/log"
export RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS='+S 2:2 +SDcpu 1 +SDio 1'
rabbitmq-server -detached
for _ in $(seq 1 120); do
  rabbitmq-diagnostics -q ping >/dev/null 2>&1 && break
  sleep 0.5
done
rabbitmq-diagnostics -q ping >/dev/null
rabbitmq-plugins list -e -m | grep -qx rabbitmq_management
cleanup
trap - EXIT
""".strip()


def certify_template(
    *,
    reference: str,
    catalog: dict[str, Any],
    catalog_path: Path,
    api_key: str,
    ttl_seconds: int,
    required_disk_gb: int,
    compose_profile: bool = False,
) -> CertificationResult:
    from e2b import Sandbox

    sandbox = Sandbox.create(
        template=reference,
        timeout=ttl_seconds,
        secure=True,
        allow_internet_access=False,
        api_key=api_key,
        request_timeout=300,
    )
    checks: list[str] = []
    try:
        _sandbox_command(
            sandbox,
            PLATFORM_BOOTSTRAP_COMMAND,
            label="platform-bootstrap",
            user="root",
            timeout=60,
        )
        checks.append("platform-bootstrap")
        _sandbox_command(
            sandbox,
            "uv --version && uvx --version",
            label="uv-toolchain",
        )
        checks.append("uv-toolchain")

        expected_catalog = catalog_path.read_bytes()
        actual_catalog = sandbox.files.read(
            "/opt/alk/hosted-snapshot-catalog.json",
            format="bytes",
            user="svc-control",
        )
        if bytes(actual_catalog) != expected_catalog:
            raise RuntimeError(
                "certification failed [catalog]: template catalog differs from checkout"
            )
        expected_digest = hashlib.sha256(expected_catalog).hexdigest()
        print(f"CERTIFIED catalog sha256:{expected_digest}")
        checks.append("catalog")
        _sandbox_command(
            sandbox,
            "command -v uv >/dev/null && command -v uvx >/dev/null && "
            "uv --version && uvx --version",
            label="binary-uv",
        )
        checks.append("binary-uv")

        for label, command in _base_checks(catalog, compose_profile=compose_profile):
            _sandbox_command(sandbox, command, label=label)
            checks.append(label)

        engine_checks = {
            "postgres": _postgres_check,
            "redis": _redis_check,
            "rabbitmq": _rabbitmq_check,
        }
        for engine, declaration in sorted(catalog["engines"].items()):
            version = str((declaration or {}).get("version") or "")
            if not version:
                raise RuntimeError(f"catalog engine {engine!r} has no version")
            _sandbox_command(
                sandbox,
                engine_checks[engine](version),
                label=f"engine-{engine}-{version}",
                timeout=180,
            )
            checks.append(f"engine-{engine}")

        disk_path = "/tmp/futureagi-certification-disk-capacity-kib"
        _sandbox_command(
            sandbox,
            f"df -Pk /work | awk 'NR == 2 {{print $2}}' > {disk_path}",
            label="disk-capacity",
        )
        disk_output = sandbox.files.read(disk_path, user="svc-control")
        disk_kib = int(str(disk_output).strip())
        disk_gb = disk_kib // (1024 * 1024)
        if disk_kib < required_disk_gb * 1024 * 1024:
            raise RuntimeError(
                f"certification failed [disk-capacity]: {disk_gb} GiB capacity, "
                f"{required_disk_gb} GiB required"
            )
        checks.append("disk-capacity")
        sandbox_id = str(sandbox.sandbox_id)
    finally:
        sandbox.kill()

    time.sleep(2)
    paginator = Sandbox.list(api_key=api_key, limit=100)
    active_ids = {str(item.sandbox_id) for item in paginator.next_items()}
    if sandbox_id in active_ids:
        raise RuntimeError(
            f"certification sandbox {sandbox_id} still exists after kill"
        )
    print(f"CERTIFIED cleanup sandbox={sandbox_id}")
    checks.append("cleanup")
    return CertificationResult(
        sandbox_id=sandbox_id,
        disk_gb=disk_gb,
        checks=tuple(checks),
    )


def certify_template_with_retries(
    *,
    attempts: int,
    **kwargs: Any,
) -> CertificationResult:
    from e2b import TimeoutException

    if attempts < 1:
        raise SystemExit("--certification-attempts must be at least 1")
    for attempt in range(1, attempts + 1):
        try:
            return certify_template(**kwargs)
        except TimeoutException:
            if attempt == attempts:
                raise
            delay = attempt * 5
            print(
                f"E2B certification transport timed out on attempt {attempt}/{attempts}; "
                f"retrying with a fresh sandbox in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--image-repository",
        default=os.environ.get("ALK_E2B_IMAGE_REPOSITORY", DEFAULT_IMAGE_REPOSITORY),
    )
    parser.add_argument("--image-tag")
    parser.add_argument(
        "--image-reference",
        help="Existing immutable image@sha256 reference; skips Docker build/push",
    )
    parser.add_argument(
        "--template-reference",
        help="Existing immutable template-name:build-id; skips E2B template build",
    )
    parser.add_argument(
        "--template-name",
        default=os.environ.get("E2B_TEMPLATE_NAME", DEFAULT_TEMPLATE_NAME),
    )
    parser.add_argument(
        "--template-alias", default=os.environ.get("E2B_TEMPLATE_ALIAS")
    )
    parser.add_argument("--cpu-count", type=int, default=DEFAULT_CPU_COUNT)
    parser.add_argument("--memory-mb", type=int, default=DEFAULT_MEMORY_MB)
    parser.add_argument("--required-disk-gb", type=int, default=DEFAULT_DISK_GB)
    parser.add_argument(
        "--certification-ttl-seconds",
        type=int,
        default=DEFAULT_CERTIFICATION_TTL_SECONDS,
    )
    parser.add_argument("--certification-attempts", type=int, default=3)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Release JSON path (default: dist/e2b-template-<git-sha>.json)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo = Path(__file__).resolve().parent.parent
    dockerfile = repo / "Dockerfile.hosted"
    catalog_path = repo / "hosted-snapshot" / "catalog.json"
    if not dockerfile.is_file():
        raise SystemExit(f"missing {dockerfile}")
    catalog = _load_catalog(catalog_path)
    if args.env_file:
        _load_selected_env(args.env_file.expanduser().resolve())
    source_revision = _git(repo, "rev-parse", "HEAD")
    _assert_clean_checkout(repo, allow_dirty=args.allow_dirty)
    image_tag = args.image_tag or source_revision

    print(f"ALK_SOURCE_REVISION={source_revision}")
    print(f"DOCKERFILE={dockerfile}")
    print(f"CATALOG={catalog_path}")
    print(f"IMAGE_REPOSITORY={args.image_repository}")
    print(f"IMAGE_TAG={image_tag}")
    print(f"E2B_TEMPLATE_NAME={args.template_name}")
    print(f"E2B_CPU_COUNT={args.cpu_count}")
    print(f"E2B_MEMORY_MB={args.memory_mb}")
    print(f"REQUIRED_DISK_GB={args.required_disk_gb}")

    if args.dry_run:
        print("DRY_RUN ok: validated checkout, Dockerfile, catalog, and release inputs")
        return 0

    api_key = os.environ.get("E2B_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("E2B_API_KEY is required (or use --env-file)")

    if args.image_reference:
        if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", args.image_reference):
            raise SystemExit(
                "--image-reference must be immutable image@sha256:<64 hex>"
            )
        image_reference = args.image_reference
        image_tag_value = ""
        image_digest = args.image_reference.rsplit("@", 1)[1]
    elif args.template_reference:
        image_reference = ""
        image_tag_value = ""
        image_digest = ""
    else:
        image = build_and_push_image(
            repo=repo,
            repository=args.image_repository,
            image_tag=image_tag,
            source_revision=source_revision,
            no_cache=args.no_cache,
        )
        image_reference = image.reference
        image_tag_value = image.tag
        image_digest = image.digest

    if args.template_reference:
        matched = re.fullmatch(
            r"(?P<name>[^:]+):(?P<build_id>[0-9a-f]{8}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
            args.template_reference,
        )
        if matched is None:
            raise SystemExit(
                "--template-reference must be immutable template-name:<build UUID>"
            )
        template_reference = args.template_reference
        template_name = matched.group("name")
        template_build_id = matched.group("build_id")
        template_alias = ""
    else:
        build, template_reference = publish_e2b_template(
            image_reference=image_reference,
            template_name=args.template_name,
            template_alias=args.template_alias,
            cpu_count=args.cpu_count,
            memory_mb=args.memory_mb,
            source_revision=source_revision,
            api_key=api_key,
            skip_cache=args.no_cache,
        )
        template_name = build.name
        template_build_id = build.build_id
        template_alias = build.alias
    certification = certify_template_with_retries(
        attempts=args.certification_attempts,
        reference=template_reference,
        catalog=catalog,
        catalog_path=catalog_path,
        api_key=api_key,
        ttl_seconds=args.certification_ttl_seconds,
        required_disk_gb=args.required_disk_gb,
    )

    output = args.output or repo / "dist" / f"e2b-template-{source_revision}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    release = {
        "schema_version": "futureagi.e2b-template-release.v1",
        "alk_source_revision": source_revision,
        "image_tag": image_tag_value,
        "image_digest": image_digest,
        "image_reference": image_reference,
        "template_name": template_name,
        "template_alias": template_alias,
        "template_build_id": template_build_id,
        "template_reference": template_reference,
        "cpu_count": args.cpu_count,
        "memory_mb": args.memory_mb,
        "verified_disk_gb": certification.disk_gb,
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "certification_checks": list(certification.checks),
        "certification_sandbox_id": certification.sandbox_id,
    }
    output.write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("---")
    print(f"ALK_E2B_TEMPLATE_REFERENCE={template_reference}")
    print(f"ALK_E2B_TEMPLATE_BUILD_ID={template_build_id}")
    print(f"ALK_E2B_TEMPLATE_CPU_UNITS={args.cpu_count}")
    print(f"ALK_E2B_TEMPLATE_MEMORY_MB={args.memory_mb}")
    print(f"ALK_E2B_TEMPLATE_DISK_GB={certification.disk_gb}")
    print("ALK_E2B_MAX_TTL_SECONDS=<SET_TO_E2B_PLAN_LIMIT>")
    print(f"RELEASE_METADATA={output}")
    print("---")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
