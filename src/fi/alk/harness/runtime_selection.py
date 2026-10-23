from __future__ import annotations

import re
import os
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
    document: dict[str, Any]
    source_files: tuple[Path, ...]


def select_runtime(
    root: Path,
    runtime: Runtime | dict[str, Any],
    *,
    public_environment: dict[str, str] | None = None,
    validate_environment_dependencies: bool = True,
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
    if runtime.compose_overrides:
        from .compose_merge import merge_compose

        try:
            body = merge_compose(
                root,
                [compose, *(root / name for name in runtime.compose_overrides)],
                public_environment=public_environment,
            )
        except ValueError as exc:
            raise RuntimeSelectionError(str(exc)) from None
        source_files = (compose, *((root / name).resolve() for name in runtime.compose_overrides))
        service_bases = {name: compose.parent for name in body["services"]}
    else:
        body, service_bases, source_files = _load_compose(root, compose, public_environment or {})
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
            validate_environment_dependencies,
            service_bases,
            additional_services=runtime.additional_services,
        )
    for name in runtime.compose_image_services:
        if name not in services:
            raise RuntimeSelectionError(f"compose_image_service_unselected: {name}")
        services[name].pop("build", None)
    services = {
        name: {
            key: _interpolate_structure(value, public_environment or {})
            if key in {"build", "image", "ports", "depends_on", "deploy", "volumes"}
            and not (runtime.compose_overrides and key == "volumes")
            else value
            for key, value in service.items()
        }
        for name, service in services.items()
    }
    for name in runtime.compose_image_services:
        image = services[name].get("image")
        if (
            not isinstance(image, str)
            or not image.strip()
            or "$" in image
            or any(char.isspace() for char in image)
        ):
            raise RuntimeSelectionError(f"compose_image_reference_invalid: {name}")
    for name, service in services.items():
        if service_bases[name] != compose.parent:
            _rebase_service(
                service,
                service_bases[name],
                compose.parent,
                root,
                public_environment or {},
            )
    return ComposeSelection(compose, control, services, body, source_files)


def _runtime_compose_services(
    root: Path,
    services: dict[str, Any],
    workdir: str,
    port: int,
    control: str,
    public_environment: dict[str, str],
    validate_environment_dependencies: bool = True,
    service_bases: dict[str, Path] | None = None,
    *,
    additional_services: list[str] | None = None,
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
        if (
            (service_bases or {}).get(name, root) / str(context)
        ).resolve() != component:
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
    inactive_optional: dict[str, set[str]] = {}
    roots = {control, *(additional_services or [])}

    def profiles(service):
        values = _interpolate_structure(
            service.get("profiles") or [], public_environment
        )
        if (
            not isinstance(values, list)
            or len(values) > 32
            or any(
                not isinstance(value, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value)
                for value in values
            )
        ):
            raise RuntimeSelectionError("compose_profiles_invalid")
        return set(values)

    active_profiles = set(
        filter(None, public_environment.get("COMPOSE_PROFILES", "").split(","))
    )
    if len(active_profiles) > 32 or any(
        value != "*" and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value)
        for value in active_profiles
    ):
        raise RuntimeSelectionError("compose_profiles_invalid")
    for name in roots:
        if isinstance(services.get(name), dict):
            active_profiles.update(profiles(services[name]))
    pending = [control, *(additional_services or [])]
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
        for dependency in dependencies:
            options = dependencies[dependency] if isinstance(dependencies, dict) else {}
            target = services.get(dependency, {})
            dependency_profiles = profiles(target)
            if (
                isinstance(options, dict)
                and options.get("required") is False
                and dependency_profiles
                and not dependency_profiles.intersection(active_profiles)
                and "*" not in active_profiles
                and dependency not in roots
            ):
                inactive_optional.setdefault(name, set()).add(dependency)
                continue
            pending.append(dependency)
    for name in selected:
        dependencies = services[name].get("depends_on")
        if isinstance(dependencies, dict):
            services[name]["depends_on"] = {
                key: value for key, value in dependencies.items() if key in selected
            }
    omitted = set(services) - selected
    for name in selected if validate_environment_dependencies else ():
        environment = services[name].get("environment") or {}
        values = (
            environment.values()
            if isinstance(environment, dict)
            else (str(item).partition("=")[2] for item in environment)
        )
        for value in values:
            try:
                value = _interpolate_structure(str(value or ""), public_environment)
            except RuntimeSelectionError:
                continue
            try:
                hostname = urlsplit(value if "://" in value else "//" + value).hostname
            except ValueError:
                continue
            if hostname in omitted and hostname not in inactive_optional.get(
                name, set()
            ):
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

    if len(value) > 65536:
        raise RuntimeSelectionError("compose_structural_interpolation_limit")
    result = []
    position = 0
    while position < len(value):
        if value[position] != "$":
            result.append(value[position])
            position += 1
            continue
        if value.startswith("$$", position):
            result.append("$")
            position += 2
            continue
        if value.startswith("${", position):
            start = position + 2
            end, depth = start, 1
            while end < len(value) and depth:
                if value.startswith("${", end):
                    depth += 1
                    if depth > 32:
                        raise RuntimeSelectionError(
                            "compose_structural_interpolation_limit"
                        )
                    end += 2
                elif value[end] == "}":
                    depth -= 1
                    end += 1
                else:
                    end += 1
            if depth:
                raise RuntimeSelectionError(
                    "compose_structural_interpolation_unsupported"
                )
            expression = value[start : end - 1]
            match = re.fullmatch(
                r"([A-Za-z_][A-Za-z0-9_]*)(?:(:?[?+\-])(.*))?", expression, re.S
            )
            if not match:
                raise RuntimeSelectionError(
                    "compose_structural_interpolation_unsupported"
                )
            name, operator, default = match.groups()
            supplied = public_environment.get(name)
            present = supplied is not None and (
                bool(supplied) or not (operator or "").startswith(":")
            )
            if operator in {"+", ":+"}:
                replacement = (
                    _interpolate_structure(default, public_environment)
                    if present
                    else ""
                )
            elif present:
                replacement = supplied
            elif operator in {"-", ":-"}:
                replacement = _interpolate_structure(default, public_environment)
            else:
                raise RuntimeSelectionError(
                    f"compose_structural_configuration_missing: {name}"
                )
            result.append(replacement)
            position = end
        else:
            match = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*)", value[position:])
            if not match:
                raise RuntimeSelectionError(
                    "compose_structural_interpolation_unsupported"
                )
            name = match.group(1)
            if name not in public_environment:
                raise RuntimeSelectionError(
                    f"compose_structural_configuration_missing: {name}"
                )
            result.append(public_environment[name])
            position += len(match.group(0))
    return "".join(result)


def _load_compose(
    root: Path, compose: Path, environment: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Path], tuple[Path, ...]]:
    seen = set()
    active = set()
    service_bases = {}
    total_bytes = 0

    def load(path):
        nonlocal total_bytes
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeSelectionError("compose_include_path_invalid")
        if path in active:
            raise RuntimeSelectionError("compose_include_cycle")
        if path in seen:
            raise RuntimeSelectionError("compose_include_duplicate")
        seen.add(path)
        active.add(path)
        if len(seen) > 32 or len(active) > 16:
            raise RuntimeSelectionError("compose_include_limit")
        total_bytes += path.stat().st_size
        if total_bytes > 2 * 1024 * 1024:
            raise RuntimeSelectionError("compose_include_limit")
        try:
            body = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeSelectionError("compose_invalid") from exc
        if not isinstance(body, dict):
            raise RuntimeSelectionError("compose_invalid")
        includes = body.pop("include", [])
        if not isinstance(includes, list):
            raise RuntimeSelectionError("compose_include_invalid")
        for key in ("services", "volumes", "networks"):
            if key in body and not isinstance(body[key], dict):
                raise RuntimeSelectionError("compose_invalid")
        for name, service in body.get("services", {}).items():
            if not isinstance(service, dict):
                raise RuntimeSelectionError("compose_invalid")
            service_bases[name] = path.parent
        for include in includes:
            if not isinstance(include, str):
                raise RuntimeSelectionError("compose_include_unsupported")
            include = _interpolate_structure(include, environment)
            if Path(include).is_absolute() or "://" in include:
                raise RuntimeSelectionError("compose_include_path_invalid")
            child = load(path.parent / include)
            for key, value in child.items():
                if key in {"name", "version"} or key.startswith("x-"):
                    continue
                if key not in {"services", "volumes", "networks"}:
                    raise RuntimeSelectionError("compose_include_top_level_unsupported")
                target = body.setdefault(key, {})
                if target.keys() & value.keys():
                    raise RuntimeSelectionError(
                        f"compose_include_resource_conflict: {key}"
                    )
                target.update(value)
        active.remove(path)
        return body

    document = load(compose)
    return document, service_bases, tuple(sorted(seen))


def _rebase_service(service, base, destination, root, environment):
    def relative(value, *, interpolate=True):
        if interpolate:
            value = _interpolate_structure(value, environment)
        if (
            not isinstance(value, str)
            or Path(value).is_absolute()
            or value.startswith("~")
        ):
            raise RuntimeSelectionError("compose_source_path_invalid")
        path = (base / value).resolve()
        if not path.is_relative_to(root):
            raise RuntimeSelectionError("compose_source_path_escape")
        return "./" + os.path.relpath(path, destination)

    build = service.get("build")
    if build:
        build = {"context": build} if isinstance(build, str) else build
        build["context"] = relative(build.get("context", "."), interpolate=False)
        service["build"] = build
    envfiles = service.get("env_file", [])
    envfiles = [envfiles] if isinstance(envfiles, str) else envfiles
    for index, entry in enumerate(envfiles):
        if isinstance(entry, str):
            envfiles[index] = relative(entry)
        elif isinstance(entry, dict) and "path" in entry:
            entry["path"] = relative(entry["path"])
        else:
            raise RuntimeSelectionError("compose_env_file_invalid")
    if envfiles:
        service["env_file"] = envfiles
    mounts = service.get("volumes", [])
    for index, mount in enumerate(mounts):
        if isinstance(mount, str):
            parts = mount.split(":")
            if parts[0].startswith((".", "/", "~")):
                parts[0] = relative(parts[0], interpolate=False)
                mounts[index] = ":".join(parts)
        elif isinstance(mount, dict) and mount.get("type") == "bind":
            mount["source"] = relative(mount["source"], interpolate=False)
