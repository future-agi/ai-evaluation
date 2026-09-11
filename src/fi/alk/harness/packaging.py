"""Deterministic packaging admission for submitted agent repositories.

This pass never executes source or invents a runtime.  It finds packaging the repository already
ships and catches common, expensive failures before Docker is started.  The selected component is
also explicit, which matters for monorepositories containing several unrelated example agents.
"""

from __future__ import annotations

import json
import re
import shlex
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
import yaml


class PackagingKind(str, Enum):
    COMPOSE = "compose"
    DOCKERFILE = "dockerfile"


class PackagingFinding(BaseModel):
    code: str
    message: str
    blocking: bool = True


class PackagingCandidate(BaseModel):
    path: str
    kind: PackagingKind
    findings: list[PackagingFinding] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    runtime_candidates: list[str] = Field(default_factory=list)
    runtime_source_roots: list[str] = Field(default_factory=list)

    @property
    def viable(self) -> bool:
        return not any(item.blocking for item in self.findings)


class PackagingManifest(BaseModel):
    source_root: str
    ready: bool
    selected_path: str | None = None
    selected_kind: PackagingKind | None = None
    agent_runtime_packaged: bool = False
    candidates: list[PackagingCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


_IGNORED = {".git", ".venv", "node_modules", "vendor", "build", "dist", "artifacts"}
_COMPOSE_NAMES = {
    "compose.yml",
    "compose.yaml",
    "docker-compose.yml",
    "docker-compose.yaml",
}
_BIND_SOURCE = re.compile(r"--mount=type=bind,[^\n]*?source=([^,\s\\]+)")
_HOST_MOUNT = re.compile(
    r"(?m)^\s*-\s*(?P<source>(?:/|~|\.|\$HOME|\$\{HOME\})[^:\n]*):(?P<target>/[^:\n]+)"
)


def inspect_packaging(
    root: str | Path,
    *,
    max_depth: int = 4,
    external_environment: bool = False,
) -> PackagingManifest:
    """Find and validate existing container packaging without running Docker."""
    source = Path(root).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"packaging_source_missing: {source}")

    candidates: list[PackagingCandidate] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if len(relative.parts) > max_depth or any(
            part in _IGNORED for part in relative.parts
        ):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if _is_compose_file(path.name):
            candidates.append(
                _compose_candidate(
                    source,
                    path,
                    external_environment=external_environment,
                )
            )
        elif _is_dockerfile(path.name):
            candidates.append(_dockerfile_candidate(source, path))

    viable = [item for item in candidates if item.viable]
    selected: PackagingCandidate | None = None
    root_compose = [
        item
        for item in viable
        if item.kind is PackagingKind.COMPOSE and "/" not in item.path
    ]
    root_dockerfiles = [
        item
        for item in viable
        if item.kind is PackagingKind.DOCKERFILE and "/" not in item.path
    ]
    development_compose = any(
        finding.code == "compose_development_configuration"
        for item in root_compose
        for finding in item.findings
    )
    if len(root_compose) == 1 and not (
        development_compose and len(root_dockerfiles) == 1
    ):
        selected = root_compose[0]
    elif len(root_dockerfiles) == 1:
        selected = root_dockerfiles[0]
    elif len(viable) == 1:
        selected = viable[0]

    notes: list[str] = []
    if not candidates:
        # This is not inherently invalid: a remote-provider agent or a genuinely in-process
        # agent may need no container runtime. The understanding stage decides that later.
        notes.append(
            "repository ships neither Compose nor a Dockerfile; runtime admission depends on "
            "whether agent understanding finds external infrastructure"
        )
    elif not viable:
        notes.append("all discovered packaging has blocking preflight findings")
    elif selected is None:
        notes.append(
            "multiple runnable components were found; select the agent subdirectory or packaging path"
        )
    return PackagingManifest(
        source_root=str(source),
        ready=selected is not None or not candidates,
        selected_path=selected.path if selected else None,
        selected_kind=selected.kind if selected else None,
        agent_runtime_packaged=(
            selected is not None
            and (
                selected.kind is PackagingKind.DOCKERFILE
                or bool(selected.runtime_candidates)
            )
        ),
        candidates=candidates,
        notes=notes,
    )


def _is_dockerfile(name: str) -> bool:
    """Accept Dockerfile variants without treating adjacent metadata as images."""
    if name == "Dockerfile":
        return True
    if not name.startswith("Dockerfile."):
        return False
    return not name.endswith((".dockerignore", ".md", ".txt"))


def _is_compose_file(name: str) -> bool:
    """Recognize standard and explicitly named Compose variants."""
    lowered = name.lower()
    if lowered in _COMPOSE_NAMES:
        return True
    if not lowered.endswith((".yml", ".yaml")):
        return False
    return lowered.startswith(("compose.", "docker-compose.")) and not any(
        marker in lowered for marker in (".example.", ".sample.", ".bak.")
    )


def _dockerfile_candidate(root: Path, path: Path) -> PackagingCandidate:
    content = path.read_text(encoding="utf-8", errors="replace")
    logical = content.replace("\\\n", " ")
    findings: list[PackagingFinding] = []
    if path.parent != root:
        return PackagingCandidate(
            path=path.relative_to(root).as_posix(),
            kind=PackagingKind.DOCKERFILE,
            findings=[
                PackagingFinding(
                    code="dockerfile_context_selection_required",
                    message="Nested Dockerfile requires an explicit component root/build context",
                    blocking=False,
                )
            ],
        )
    sources = [match.group(1) for match in _BIND_SOURCE.finditer(logical)]
    sources.extend(_dockerfile_copy_sources(logical))
    for raw in sorted(set(sources)):
        value = raw.strip("\"'")
        if not value or value in {".", "./"}:
            continue
        if any(token in value for token in ("$", "*", "?", "[")):
            continue
        if value.startswith(("http://", "https://")):
            continue
        target = (root / value.removeprefix("./")).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            findings.append(
                PackagingFinding(
                    code="dockerfile_source_outside_repository",
                    message=f"Dockerfile references source outside the submitted root: {value}",
                )
            )
            continue
        if not target.exists():
            findings.append(
                PackagingFinding(
                    code="dockerfile_build_input_missing",
                    message=f"Dockerfile requires missing build input: {value}",
                )
            )
    return PackagingCandidate(
        path=path.relative_to(root).as_posix(),
        kind=PackagingKind.DOCKERFILE,
        findings=findings,
    )


def _dockerfile_copy_sources(content: str) -> list[str]:
    """Return every local source from shell- and JSON-form COPY/ADD instructions."""
    sources: list[str] = []
    for raw_line in content.splitlines():
        match = re.match(r"^\s*(COPY|ADD)\s+(.+)$", raw_line, re.IGNORECASE)
        if not match:
            continue
        arguments = match.group(2).strip()
        if re.search(r"(?:^|\s)--from(?:=|\s)", arguments):
            continue
        while arguments.startswith("--"):
            try:
                option, arguments = arguments.split(None, 1)
            except ValueError:
                arguments = ""
                break
            # Options with a separate value consume that value as well.
            if "=" not in option and option.lower() in {
                "--chown",
                "--chmod",
                "--exclude",
            }:
                try:
                    _, arguments = arguments.split(None, 1)
                except ValueError:
                    arguments = ""
                    break
        if not arguments:
            continue
        if arguments.startswith("["):
            try:
                values = json.loads(arguments)
            except json.JSONDecodeError:
                continue
            if isinstance(values, list) and len(values) >= 2:
                sources.extend(str(value) for value in values[:-1])
            continue
        try:
            values = shlex.split(arguments, comments=True)
        except ValueError:
            continue
        if len(values) >= 2:
            sources.extend(values[:-1])
    return sources


def _compose_candidate(
    root: Path,
    path: Path,
    *,
    external_environment: bool = False,
) -> PackagingCandidate:
    content = path.read_text(encoding="utf-8", errors="replace")
    findings: list[PackagingFinding] = []
    services: dict[str, object] = {}
    try:
        document = yaml.safe_load(content) or {}
        raw_services = (
            document.get("services", {}) if isinstance(document, dict) else {}
        )
        if isinstance(raw_services, dict):
            services = {str(name): value for name, value in raw_services.items()}
    except yaml.YAMLError as exc:
        findings.append(
            PackagingFinding(
                code="compose_yaml_invalid",
                message=f"Compose YAML cannot be parsed: {exc}",
            )
        )
    for service_name, raw_service in services.items():
        if not isinstance(raw_service, dict):
            continue
        env_files = raw_service.get("env_file") or []
        if isinstance(env_files, (str, dict)):
            env_files = [env_files]
        for raw_env_file in env_files:
            optional = (
                isinstance(raw_env_file, dict)
                and raw_env_file.get("required") is False
            )
            value = (
                str(raw_env_file.get("path") or "")
                if isinstance(raw_env_file, dict)
                else str(raw_env_file)
            )
            if not value or "$" in value:
                continue
            candidate = (path.parent / value).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                findings.append(
                    PackagingFinding(
                        code="compose_env_file_outside_repository",
                        message=f"{service_name} reads an env file outside the repository: {value}",
                    )
                )
                continue
            if not candidate.is_file() and not optional:
                findings.append(
                    PackagingFinding(
                        code="compose_env_file_missing",
                        message=(
                            f"{service_name} uses uploaded environment values instead of "
                            f"repository secret file: {value}"
                            if external_environment
                            else f"{service_name} requires missing env file: {value}"
                        ),
                        blocking=not external_environment,
                    )
                )
    if services and all(
        isinstance(service, dict)
        and not service.get("image")
        and not service.get("build")
        for service in services.values()
    ):
        findings.append(
            PackagingFinding(
                code="compose_override_fragment",
                message=(
                    "Compose file only overrides existing services and cannot run "
                    "as a standalone environment"
                ),
            )
        )
    for match in _HOST_MOUNT.finditer(content):
        findings.append(
            PackagingFinding(
                code="compose_host_bind_mount",
                message=f"Compose depends on host path {match.group('source')}",
                # Local execution may deliberately use a repository-owned mount. A hosted
                # provider must turn this finding into policy admission or mount a secret ref.
                blocking=False,
            )
        )
    development_signals = []
    if re.search(r"(?m)^\s*(?:tty|stdin_open)\s*:\s*true\s*$", content, re.IGNORECASE):
        development_signals.append("interactive terminal")
    if re.search(r"(?m)^\s*container_name\s*:", content):
        development_signals.append("fixed container name")
    if re.search(r"(?m)^\s*-\s*['\"]?\d{2,5}-\d{2,5}:\d{2,5}-\d{2,5}", content):
        development_signals.append("broad published port range")
    if development_signals:
        findings.append(
            PackagingFinding(
                code="compose_development_configuration",
                message="Compose appears development-oriented: "
                + ", ".join(development_signals),
                blocking=False,
            )
        )
    if re.search(r"(?m)^\s*privileged\s*:\s*true\s*$", content, re.IGNORECASE):
        findings.append(
            PackagingFinding(
                code="compose_privileged",
                message="Compose requests privileged container execution",
            )
        )
    if re.search(
        r"(?m)^\s*(?:network_mode|pid|ipc)\s*:\s*['\"]?host['\"]?\s*$", content
    ):
        findings.append(
            PackagingFinding(
                code="compose_host_namespace",
                message="Compose requests a host namespace",
            )
        )
    runtime_candidates = _compose_runtime_candidates(services)
    return PackagingCandidate(
        path=path.relative_to(root).as_posix(),
        kind=PackagingKind.COMPOSE,
        findings=findings,
        services=sorted(services),
        runtime_candidates=runtime_candidates,
        runtime_source_roots=_compose_runtime_source_roots(
            root, path, services, runtime_candidates
        ),
    )


def _compose_runtime_source_roots(
    root: Path,
    compose_path: Path,
    services: dict[str, object],
    runtime_candidates: list[str],
) -> list[str]:
    """Return submitted paths that can affect the selected application runtime."""
    paths: set[str] = set()
    for name in runtime_candidates:
        raw_service = services.get(name)
        service = raw_service if isinstance(raw_service, dict) else {}
        build = service.get("build")
        context = (
            str(build.get("context") or ".")
            if isinstance(build, dict)
            else str(build or "")
        )
        if (
            context
            and "$" not in context
            and not context.startswith(("http://", "https://"))
        ):
            candidate = (compose_path.parent / context).resolve()
            try:
                relative = candidate.relative_to(root)
            except ValueError:
                continue
            if candidate.exists():
                paths.add(relative.as_posix() or ".")
        env_files = service.get("env_file") or []
        if isinstance(env_files, (str, dict)):
            env_files = [env_files]
        for raw_env_file in env_files:
            value = (
                str(raw_env_file.get("path") or "")
                if isinstance(raw_env_file, dict)
                else str(raw_env_file)
            )
            if not value or "$" in value:
                continue
            candidate = (compose_path.parent / value).resolve()
            try:
                relative = candidate.relative_to(root)
            except ValueError:
                continue
            if candidate.is_file():
                paths.add(relative.as_posix())
    return sorted(paths)


def _compose_runtime_candidates(services: dict[str, object]) -> list[str]:
    """Identify application services without treating databases/admin UIs as agents."""
    infrastructure = {
        "postgres",
        "postgresql",
        "mysql",
        "mariadb",
        "redis",
        "clickhouse",
        "mongodb",
        "mongo",
        "rabbitmq",
        "kafka",
        "nats",
        "minio",
        "elasticsearch",
        "opensearch",
        "qdrant",
        "neo4j",
    }
    administration = {"pgadmin", "redis-commander", "adminer", "grafana", "kibana"}
    application_roles = {
        "api",
        "backend",
        "server",
        "voice-server",
        "voice_server",
        "orchestrator",
        "runtime",
    }
    candidates: list[str] = []
    for name, raw_service in services.items():
        service = raw_service if isinstance(raw_service, dict) else {}
        image = str(service.get("image") or "").lower()
        haystack = f"{name} {image}".lower()
        if name.lower() in administration or any(
            word in haystack for word in administration
        ):
            continue
        if name.lower() in application_roles:
            candidates.append(name)
            continue
        if any(word in name.lower() for word in ("agent", "worker", "bot", "app")):
            candidates.append(name)
            continue
        known_infrastructure = any(word in haystack for word in infrastructure)
        if service.get("build") and not known_infrastructure:
            candidates.append(name)
    return sorted(candidates)


__all__ = [
    "PackagingCandidate",
    "PackagingFinding",
    "PackagingKind",
    "PackagingManifest",
    "inspect_packaging",
]
