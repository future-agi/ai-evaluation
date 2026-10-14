from __future__ import annotations

import re
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from .contract import Runtime


class RuntimeSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class NativeSelection:
    workdir: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class ComposeSelection:
    compose_file: Path
    control_service: str
    services: dict[str, Any]


def select_runtime(
    root: Path,
    runtime: Runtime | dict[str, Any],
    *,
    public_environment: dict[str, str] | None = None,
) -> NativeSelection | ComposeSelection:
    raw = runtime.model_dump() if isinstance(runtime, Runtime) else runtime
    interface = raw.get("interface") or {}
    runtime = Runtime.model_validate(
        {key: value for key, value in raw.items() if key != "interface"}
    )
    root = root.resolve()
    workdir = (root / (runtime.workdir or ".")).resolve()
    if not workdir.is_relative_to(root) or not workdir.is_dir():
        raise RuntimeSelectionError("runtime_workdir_invalid")
    if runtime.execution == "native":
        return NativeSelection(workdir, tuple(runtime.command))
    if runtime.compose_file:
        compose = (root / runtime.compose_file).resolve()
        if not compose.is_relative_to(root) or not compose.is_file():
            raise RuntimeSelectionError("runtime_compose_file_invalid")
    else:
        matches = [
            root / name
            for name in (
                "compose.yaml",
                "compose.yml",
                "docker-compose.yaml",
                "docker-compose.yml",
            )
            if (root / name).is_file()
        ]
        if len(matches) > 1:
            raise RuntimeSelectionError(
                "compose_ambiguous: " + ", ".join(p.name for p in matches)
            )
        if not matches:
            return NativeSelection(workdir, tuple(runtime.command))
        compose = matches[0]
    compose = compose.resolve()
    if not compose.is_relative_to(root):
        raise RuntimeSelectionError("runtime_compose_file_invalid")
    try:
        body = yaml.safe_load(compose.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeSelectionError("compose_invalid") from exc
    if not isinstance(body, dict) or not isinstance(body.get("services"), dict):
        raise RuntimeSelectionError("compose_invalid: services must be an object")
    services = {
        name: dict(service)
        for name, service in body["services"].items()
        if isinstance(service, dict)
    }
    if len(services) != len(body["services"]):
        raise RuntimeSelectionError("compose_invalid: each service must be an object")
    control = runtime.service
    if control or (
        runtime.command
        and interface
        and interface.get("kind") == "http"
        and interface.get("port")
    ):
        control, services = _runtime_compose_services(
            compose.parent,
            services,
            str(workdir.relative_to(compose.parent))
            if workdir.is_relative_to(compose.parent)
            else str(workdir),
            int(interface.get("port") or 0),
            control,
            public_environment or {},
        )
    services = {
        name: {
            key: _interpolate_structure(value, public_environment or {})
            if key in {"build", "image", "ports", "depends_on"}
            else value
            for key, value in service.items()
        }
        for name, service in services.items()
    }
    return ComposeSelection(compose, control, services)


def _runtime_compose_services(
    root: Path,
    services: dict[str, Any],
    workdir: str,
    port: int,
    control: str,
    public_environment: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    """Select a declared HTTP runtime and retain its submitted dependency graph."""
    component = (root / workdir).resolve()
    matches = []
    for name, service in services.items():
        if control:
            break
        if not isinstance(service, dict) or not service.get("build"):
            continue
        build = _interpolate_structure(service["build"], public_environment)
        context = build.get("context", ".") if isinstance(build, dict) else build
        if (root / str(context)).resolve() != component:
            continue
        ports = _interpolate_structure(service.get("ports") or [], public_environment)
        targets = [
            str(value.get("target", ""))
            if isinstance(value, dict)
            else str(value).split("/")[0].rsplit(":", 1)[-1]
            for value in ports
            if (
                value.get("protocol", "tcp") == "tcp"
                if isinstance(value, dict)
                else "/" not in str(value) or str(value).endswith("/tcp")
            )
        ]
        if str(port) in targets:
            matches.append(name)
    if not control and len(matches) != 1:
        raise RuntimeSelectionError(
            "control_service_ambiguous: declared runtime must match exactly one "
            f"Compose build context and container port; found {matches}"
        )
    control = control or matches[0]
    selected: set[str] = set()
    pending = [control]
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        service = services.get(name)
        if not isinstance(service, dict):
            raise RuntimeSelectionError(f"compose_dependency_missing: {name}")
        unsupported = [
            key
            for key in ("links", "volumes_from", "network_mode", "ipc", "pid")
            if service.get(key)
            and (
                key in {"links", "volumes_from"}
                or str(service[key]).startswith("service:")
            )
        ]
        if unsupported:
            raise RuntimeSelectionError(
                f"compose_dependency_unsupported: {name} uses {', '.join(unsupported)}"
            )
        dependencies = _interpolate_structure(
            service.get("depends_on") or [], public_environment
        )
        service["depends_on"] = dependencies
        if not isinstance(dependencies, (list, dict)) or not all(
            isinstance(dependency, str) for dependency in dependencies
        ):
            raise RuntimeSelectionError(f"compose_dependency_invalid: {name}")
        selected.add(name)
        pending.extend(dependencies)
    omitted = set(services) - selected
    for name in selected:
        environment = services[name].get("environment") or {}
        values = (
            environment.values()
            if isinstance(environment, dict)
            else (str(item).partition("=")[2] for item in environment)
        )
        for value in values:
            value = str(value or "")
            try:
                hostname = urlsplit(value if "://" in value else "//" + value).hostname
            except ValueError:
                continue
            if hostname in omitted:
                raise RuntimeSelectionError(
                    f"compose_dependency_undeclared: {name} environment references {hostname}; "
                    "declare it in depends_on"
                )
    try:
        order = tuple(
            TopologicalSorter(
                {
                    name: list(services[name].get("depends_on") or [])
                    for name in services
                    if name in selected
                }
            ).static_order()
        )
    except CycleError as exc:
        raise RuntimeSelectionError("compose_dependency_cycle") from exc
    return control, {name: services[name] for name in order}


def _interpolate_structure(value: Any, public_environment: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [_interpolate_structure(item, public_environment) for item in value]
    if isinstance(value, dict):
        return {
            _interpolate_structure(key, public_environment): _interpolate_structure(
                item, public_environment
            )
            for key, item in value.items()
        }
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, operator, default = match.groups()
        supplied = public_environment.get(name)
        if supplied is not None and (supplied or not (operator or "").startswith(":")):
            return supplied
        if operator in {"-", ":-"}:
            return default
        raise RuntimeSelectionError(f"compose_structural_configuration_missing: {name}")

    result = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-?])(.*?))?\}", replace, value)
    if "$" in result:
        raise RuntimeSelectionError("compose_structural_interpolation_unsupported")
    return result
