"""Opt-in Compose execution inside a trusted, isolated sandbox daemon."""

from __future__ import annotations

import asyncio
import copy
import json
import ipaddress
import os
import signal
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import time
import uuid

import yaml

from .bundle_v2 import EnvironmentBundleV2, RuntimeKindV2
from .process_runtime import (
    EnvironmentRuntime,
    RuntimeEndpoint,
    RuntimeState,
    default_capability_prober,
)


class ComposeRuntimeError(RuntimeError):
    pass


def _bundle_context_excludes(bundle: EnvironmentBundleV2) -> tuple[str, ...]:
    value = getattr(bundle, "metadata", {}).get("context_excludes") or []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ComposeRuntimeError("compose_source_context_exclude_invalid")
    return tuple(value)


_SERVICE_KEYS = {
    "image",
    "platform",
    "build",
    "command",
    "entrypoint",
    "environment",
    "working_dir",
    "user",
    "depends_on",
    "healthcheck",
    "volumes",
    "ports",
    "expose",
    "restart",
    "init",
    "stop_grace_period",
    "stop_signal",
    "read_only",
    "tmpfs",
    "shm_size",
    "ulimits",
    "mem_limit",
    "cpus",
    "networks",
    "hostname",
    "labels",
    "env_file",
    "x-alk-secret-files",
    "logging",
    "extra_hosts",
    "container_name",
    "deploy",
    "cap_drop",
    "security_opt",
}
_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def _local_service_options(service: dict) -> str | None:
    alias = service.pop("container_name", None)
    if alias is not None and (not isinstance(alias, str) or not _NAME.fullmatch(alias)):
        raise ComposeRuntimeError("compose_container_name_invalid")
    logging = service.get("logging")
    if logging is None:
        logging = {"driver": "json-file"}
    if logging is not None:
        if (
            not isinstance(logging, dict)
            or set(logging) - {"driver", "options"}
            or logging.get("driver", "json-file") not in {"json-file", "local"}
        ):
            raise ComposeRuntimeError("compose_logging_driver_forbidden")
        options = logging.get("options", {})
        if not isinstance(options, dict) or set(options) - {"max-size", "max-file"}:
            raise ComposeRuntimeError("compose_logging_options_invalid")
        size, count = (
            str(options.get("max-size", "20m")),
            str(options.get("max-file", "3")),
        )
        if not re.fullmatch(r"(?:[1-9]|[1-9][0-9]|100)m", size) or not re.fullmatch(
            r"(?:[1-9]|10)", count
        ):
            raise ComposeRuntimeError("compose_logging_limits_invalid")
        service["logging"] = {
            "driver": logging.get("driver", "json-file"),
            "options": {"max-size": size, "max-file": count},
        }
    hosts = service.get("extra_hosts", {})
    if isinstance(hosts, list):
        pairs = [
            item.replace("=", ":", 1).partition(":")
            for item in hosts
            if isinstance(item, str)
        ]
        if len(pairs) != len(hosts) or any(not separator for _, separator, _ in pairs):
            raise ComposeRuntimeError("compose_extra_hosts_invalid")
        hosts = {name: address for name, _, address in pairs}
        if len(hosts) != len(pairs):
            raise ComposeRuntimeError("compose_extra_hosts_duplicate")
    if not isinstance(hosts, dict) or len(hosts) > 32:
        raise ComposeRuntimeError("compose_extra_hosts_invalid")
    for name, address in hosts.items():
        if (
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or not isinstance(address, str)
        ):
            raise ComposeRuntimeError("compose_extra_hosts_invalid")
        if address != "host-gateway":
            try:
                literal = (
                    address[1:-1]
                    if address.startswith("[") and address.endswith("]")
                    else address
                )
                hosts[name] = str(ipaddress.ip_address(literal))
            except ValueError:
                raise ComposeRuntimeError("compose_extra_hosts_invalid") from None
    if hosts:
        service["extra_hosts"] = hosts
    return alias


def _within(root: Path, value: str, *, exists: bool = True) -> Path:
    if not isinstance(value, str) or not value or "$" in value:
        raise ComposeRuntimeError("compose_path_invalid")
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root.resolve()) or (
        exists and not candidate.exists()
    ):
        raise ComposeRuntimeError("compose_path_outside_source_or_missing")
    return candidate


def _mount_destination(value) -> PurePosixPath:
    if not isinstance(value, str):
        raise ComposeRuntimeError("compose_mount_target_invalid")
    target = PurePosixPath(value)
    if not target.is_absolute() or ".." in target.parts:
        raise ComposeRuntimeError("compose_mount_target_invalid")
    return target


def normalize_compose(document: dict, source: Path, *, capabilities: dict) -> dict:
    """Close the host-control surface before Docker sees any repository configuration."""
    if not isinstance(document, dict) or set(document) - {
        "services",
        "volumes",
        "networks",
        "version",
        "name",
        "x-alk-config-files",
        "x-alk-directories",
    }:
        raise ComposeRuntimeError("compose_top_level_unsupported")
    result = copy.deepcopy(document)
    result.pop("name", None)
    result.pop("version", None)
    from .contract import (
        RuntimeConfigFile,
        validate_runtime_resource_paths,
        runtime_public_environment,
    )

    resources = result.get("x-alk-config-files", {})
    directories = result.get("x-alk-directories", [])
    validate_runtime_resource_paths(resources, directories)
    runtime_public_environment({"config_files": resources})
    resources = {
        path: RuntimeConfigFile.model_validate(value)
        for path, value in resources.items()
    }
    declared = {*resources, *directories}
    for value in declared:
        path = source / value
        for ancestor in [path, *path.parents]:
            if ancestor == source:
                break
            if ancestor.is_symlink():
                raise ComposeRuntimeError("compose_resource_symlink_forbidden")
        if path.exists() and (
            not path.is_file() if value in resources else not path.is_dir()
        ):
            raise ComposeRuntimeError("compose_resource_type_invalid")
        if value in resources and path.exists() and path.stat().st_mode & 0o111:
            raise ComposeRuntimeError("compose_config_executable_forbidden")
    consumed = set()
    services = result.get("services")
    if not isinstance(services, dict) or not services:
        raise ComposeRuntimeError("compose_services_required")
    for kind in ("volumes", "networks"):
        entries = result.get(kind, {})
        if not isinstance(entries, dict):
            raise ComposeRuntimeError(f"compose_{kind}_invalid")
        for name, spec in entries.items():
            if not _NAME.fullmatch(name) or (spec and set(spec) - {"driver"}):
                raise ComposeRuntimeError(f"compose_{kind}_host_control_forbidden")
            if spec and spec.get(
                "driver", "local" if kind == "volumes" else "bridge"
            ) != ("local" if kind == "volumes" else "bridge"):
                raise ComposeRuntimeError(f"compose_{kind}_driver_forbidden")
    for name, service in services.items():
        if not _NAME.fullmatch(name) or not isinstance(service, dict):
            raise ComposeRuntimeError("compose_service_invalid")
        if set(service) - _SERVICE_KEYS:
            raise ComposeRuntimeError(
                f"compose_service_fields_forbidden: {name}: {','.join(sorted(set(service) - _SERVICE_KEYS))}"
            )
        if "platform" in service:
            from .compose_image_lock import validate_image_platform

            validate_image_platform(service["platform"])
        dropped = service.get("cap_drop", [])
        if not isinstance(dropped, list) or any(
            not isinstance(value, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", value)
            for value in dropped
        ):
            raise ComposeRuntimeError("compose_cap_drop_invalid")
        security = service.get("security_opt", [])
        if not isinstance(security, list) or any(
            value not in ("no-new-privileges:true", "no-new-privileges=true")
            for value in security
        ):
            raise ComposeRuntimeError("compose_service_fields_forbidden: security_opt")
        if "deploy" in service:
            deploy = service["deploy"]
            if not isinstance(deploy, dict) or set(deploy) != {"replicas"}:
                raise ComposeRuntimeError(
                    f"compose_deploy_fields_unsupported: {name}: only replicas is supported"
                )
            replicas = deploy["replicas"]
            if not (
                (type(replicas) is int and replicas == 1)
                or (isinstance(replicas, str) and replicas == "1")
            ):
                raise ComposeRuntimeError(
                    f"compose_replicas_unsupported: {name}: exactly one replica is supported"
                )
            deploy["replicas"] = 1
        scoped_alias = _local_service_options(service)
        if not service.get("image") and not service.get("build"):
            raise ComposeRuntimeError("compose_service_image_or_build_required")
        build = service.get("build")
        if build:
            build = {"context": build} if isinstance(build, str) else build
            if not isinstance(build, dict) or set(build) - {
                "context",
                "dockerfile",
                "target",
                "args",
            }:
                raise ComposeRuntimeError("compose_build_fields_forbidden")
            args = build.get("args", {})
            if isinstance(args, list):
                if any(not isinstance(item, str) or "=" not in item for item in args):
                    raise ComposeRuntimeError("compose_build_passthrough_forbidden")
                args = dict(item.split("=", 1) for item in args)
            if not isinstance(args, dict) or any(
                value is None for value in args.values()
            ):
                raise ComposeRuntimeError("compose_build_passthrough_forbidden")
            build["args"] = args
            context = _within(source, build.get("context", "."))
            dockerfile = _within(context, build.get("dockerfile", "Dockerfile"))
            if not dockerfile.is_file():
                raise ComposeRuntimeError("compose_dockerfile_missing")
            if dockerfile.relative_to(source).as_posix() in declared:
                raise ComposeRuntimeError("compose_resource_build_input_forbidden")
            build["context"] = str(context)
            build["dockerfile"] = str(dockerfile.relative_to(context))
            service["build"] = build
        environment = service.get("environment", {})
        if isinstance(environment, list):
            if any(
                not isinstance(item, str) or "=" not in item for item in environment
            ):
                raise ComposeRuntimeError("compose_environment_passthrough_forbidden")
            environment = dict(item.split("=", 1) for item in environment)
        if not isinstance(environment, dict) or any(
            value is None for value in environment.values()
        ):
            raise ComposeRuntimeError("compose_environment_passthrough_forbidden")
        service["environment"] = environment
        env_files = service.get("env_file", [])
        if isinstance(env_files, str):
            env_files = [env_files]
        if not isinstance(env_files, list):
            raise ComposeRuntimeError("compose_env_file_invalid")
        contained_files = []
        for entry in env_files:
            entry = {"path": entry} if isinstance(entry, str) else entry
            if not isinstance(entry, dict) or set(entry) - {"path", "required"}:
                raise ComposeRuntimeError("compose_env_file_invalid")
            required = entry.get("required", True)
            if (
                not isinstance(required, bool)
                or Path(entry.get("path", "")).is_absolute()
            ):
                raise ComposeRuntimeError("compose_env_file_invalid")
            candidate = _within(source, entry.get("path", ""), exists=False)
            resource = candidate.relative_to(source.resolve()).as_posix()
            if resource in resources:
                consumed.add(resource)
                contained_files.append({"path": str(candidate), "required": True})
                continue
            path = _within(source, entry.get("path", ""), exists=required)
            if not path.exists() and not required:
                continue
            if not path.is_file() or path.stat().st_size > 65536:
                raise ComposeRuntimeError("compose_env_file_invalid")
            contained_files.append({"path": str(path), "required": True})
        service["env_file"] = contained_files
        mounts = []
        for mount in service.get("volumes", []):
            if isinstance(mount, str):
                parts = mount.split(":")
                if len(parts) not in (2, 3) or (
                    len(parts) == 3 and parts[2] not in ("ro", "rw")
                ):
                    raise ComposeRuntimeError("compose_mount_invalid")
                origin, target = parts[:2]
                mount = {
                    "type": "bind" if origin.startswith((".", "/", "~")) else "volume",
                    "source": origin,
                    "target": target,
                    "read_only": len(parts) == 3 and parts[2] == "ro",
                }
            # Native Compose expands plain named mounts with an empty options object.
            if (
                isinstance(mount, dict)
                and mount.get("type") == "volume"
                and mount.get("volume") == {}
            ):
                mount = {key: value for key, value in mount.items() if key != "volume"}
            if not isinstance(mount, dict) or set(mount) - {
                "type",
                "source",
                "target",
                "read_only",
                "bind",
            }:
                raise ComposeRuntimeError("compose_mount_fields_forbidden")
            if "read_only" in mount and type(mount["read_only"]) is not bool:
                raise ComposeRuntimeError("compose_mount_read_only_invalid")
            if "bind" in mount:
                options = mount["bind"]
                if (
                    mount.get("type") != "bind"
                    or not isinstance(options, dict)
                    or set(options) - {"create_host_path"}
                    or (
                        "create_host_path" in options
                        and type(options["create_host_path"]) is not bool
                    )
                ):
                    raise ComposeRuntimeError("compose_bind_options_forbidden")
            target = _mount_destination(mount.get("target", ""))
            if mount.get("type") == "bind":
                origin = mount.get("source", "")
                if Path(origin).is_absolute() or origin.startswith("~"):
                    raise ComposeRuntimeError("compose_host_bind_forbidden")
                path = _within(
                    source,
                    origin,
                    exists=mount.get("bind", {}).get("create_host_path") is False,
                )
                resource = path.relative_to(source.resolve()).as_posix()
                if resource in declared:
                    consumed.add(resource)
                else:
                    path = _within(source, origin)
                if path.is_socket() or path.is_block_device() or path.is_char_device():
                    raise ComposeRuntimeError("compose_special_file_forbidden")
                mount["source"] = str(path)
            elif mount.get("type") == "volume":
                if mount.get("source") not in result.get("volumes", {}):
                    raise ComposeRuntimeError("compose_volume_undeclared")
            else:
                raise ComposeRuntimeError("compose_mount_type_forbidden")
            mounts.append(mount)
        service["volumes"] = mounts
        from .contract import RuntimeSecretFile

        bindings = service.get("x-alk-secret-files", [])
        if not isinstance(bindings, list) or len(bindings) > 32:
            raise ComposeRuntimeError("compose_secret_files_invalid")
        destinations = [PurePosixPath(item["target"]) for item in mounts]
        if "tmpfs" in service:
            entries = service["tmpfs"]
            entries = [entries] if isinstance(entries, str) else entries
            if not isinstance(entries, list) or len(entries) > 32:
                raise ComposeRuntimeError("compose_tmpfs_invalid")
            for entry in entries:
                if not isinstance(entry, str) or len(entry) > 4096:
                    raise ComposeRuntimeError("compose_tmpfs_invalid")
                path, separator, options = entry.partition(":")
                target = _mount_destination(path)
                if (
                    str(target) != path
                    or str(target) == "/"
                    or any(char in path for char in ("$", "\\"))
                    or any(ord(char) < 32 or ord(char) == 127 for char in path)
                ):
                    raise ComposeRuntimeError("compose_tmpfs_path_invalid")
                if separator:
                    seen = set()
                    for option in options.split(","):
                        key, equals, value = option.partition("=")
                        valid = (
                            key == "mode"
                            and re.fullmatch(r"[0-7]{3,4}", value)
                            or key == "size"
                            and re.fullmatch(r"[1-9][0-9]{0,9}[kKmMgG]?", value)
                        )
                        if not equals or key in seen or not valid:
                            raise ComposeRuntimeError(
                                "compose_tmpfs_options_unsupported"
                            )
                        seen.add(key)
                if any(
                    target.is_relative_to(old) or old.is_relative_to(target)
                    for old in destinations
                ):
                    raise ComposeRuntimeError("compose_tmpfs_mount_overlap")
                destinations.append(target)
            service["tmpfs"] = entries
        environment_names = set(environment)
        for raw in bindings:
            binding = RuntimeSecretFile.model_validate(raw)
            target = PurePosixPath(binding.destination)
            if any(
                target.is_relative_to(old) or old.is_relative_to(target)
                for old in destinations
            ):
                raise ComposeRuntimeError("compose_secret_file_mount_overlap")
            destinations.append(target)
            if binding.environment_variable:
                if binding.environment_variable in environment_names:
                    raise ComposeRuntimeError(
                        "compose_secret_file_environment_conflict"
                    )
                environment_names.add(binding.environment_variable)
        networks = service.get("networks", [])
        if not isinstance(networks, (list, dict)) or any(
            n not in result.get("networks", {}) for n in networks
        ):
            raise ComposeRuntimeError("compose_network_invalid")
        if isinstance(networks, dict):
            for options in networks.values():
                if options is not None and (
                    not isinstance(options, dict)
                    or set(options) - {"aliases"}
                    or not isinstance(options.get("aliases", []), list)
                    or any(
                        not isinstance(alias, str) or not _NAME.fullmatch(alias)
                        for alias in options.get("aliases", [])
                    )
                ):
                    raise ComposeRuntimeError("compose_network_options_invalid")
        if scoped_alias:
            if not networks:
                result.setdefault("networks", {}).setdefault("default", {})
                networks = ["default"]
            networks = (
                {name: dict(options or {}) for name, options in networks.items()}
                if isinstance(networks, dict)
                else {name: {} for name in networks}
            )
            for options in networks.values():
                options["aliases"] = list(
                    dict.fromkeys([*options.get("aliases", []), scoped_alias])
                )
            service["networks"] = networks
        service["ports"] = []
        service["security_opt"] = ["no-new-privileges:true"]
        service["restart"] = "no"
    if declared - consumed:
        raise ComposeRuntimeError(
            "compose_resource_unused: " + ", ".join(sorted(declared - consumed))
        )
    for capability in capabilities.values():
        if capability.service not in services or capability.container_port is None:
            raise ComposeRuntimeError("compose_capability_unresolved")
        ports = services[capability.service]["ports"]
        port = {
            "target": capability.container_port,
            "host_ip": "127.0.0.1",
            "published": "0",
            "protocol": "tcp",
        }
        if port not in ports:
            ports.append(port)
    return result


def validate_source_tree(source: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink() and (
            path.readlink().is_absolute()
            or not path.resolve().is_relative_to(source.resolve())
        ):
            raise ComposeRuntimeError("compose_source_symlink_escape")
        if not path.is_symlink() and not path.is_file() and not path.is_dir():
            raise ComposeRuntimeError("compose_source_special_file")


def _own_directory(path: Path, uid: int, gid: int) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for _, directories, files, current in os.fwalk(
            ".", dir_fd=descriptor, follow_symlinks=False
        ):
            os.fchown(current, uid, gid)
            for name in [*directories, *files]:
                info = os.stat(name, dir_fd=current, follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    continue
                if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                    raise ComposeRuntimeError("compose_directory_special_file")
                if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                    raise ComposeRuntimeError("compose_directory_hardlink_forbidden")
                os.chown(name, uid, gid, dir_fd=current, follow_symlinks=False)
    finally:
        os.close(descriptor)


class CertifiedComposeRuntimeProvider:
    name = "certified-compose"

    def __init__(
        self,
        *,
        socket_path: Path,
        configuration: dict[str, str] | None = None,
        artifact_root: Path | None = None,
    ):
        self.socket_path = socket_path
        self.configuration = dict(configuration or {})
        if any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or key in {"PATH", "HOME", "LD_PRELOAD", "LD_LIBRARY_PATH"}
            or key.startswith(("DOCKER_", "COMPOSE_", "BUILDKIT_", "BUILDX_"))
            for key in self.configuration
        ):
            raise ComposeRuntimeError("compose_reserved_configuration")
        self.artifact_root = artifact_root
        self._worlds: dict[
            int, tuple[EnvironmentRuntime, Path, EnvironmentBundleV2]
        ] = {}
        self._source: Path | None = None
        self._document: dict | None = None
        self._redactions = tuple(
            sorted(
                {
                    encoded
                    for value in self.configuration.values()
                    if value
                    for encoded in (
                        value,
                        json.dumps(value, ensure_ascii=False)[1:-1],
                        json.dumps(value.replace("$", "$$"), ensure_ascii=False)[1:-1],
                    )
                },
                key=len,
                reverse=True,
            )
        )

    def _check_socket(self):
        path = self.socket_path
        if path.is_symlink() or path.parent.is_symlink():
            raise ComposeRuntimeError("compose_private_socket_required")
        info, parent = path.stat(), path.parent.stat()
        if (
            not stat.S_ISSOCK(info.st_mode)
            or info.st_uid != 0
            or parent.st_uid != 0
            or info.st_mode & 0o077
            or parent.st_mode & 0o077
        ):
            raise ComposeRuntimeError("compose_private_socket_required")

    def _command(
        self,
        directory: Path,
        project: str,
        *args: str,
        timeout: float = 600,
        private_output: bool = False,
    ) -> str:
        environment = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "DOCKER_HOST": f"unix://{self.socket_path}",
            "DOCKER_CONFIG": str(directory / "docker-config"),
            "COMPOSE_DISABLE_ENV_FILE": "1",
        }
        command = [
            "docker",
            "compose",
            "--env-file",
            str(directory / "configuration.env"),
            "--project-name",
            project,
            "--project-directory",
            str(directory / "source"),
            "-f",
            str(directory / "compose.json"),
            *args,
        ]
        return self._run_command(
            command, environment, project, args[0], timeout=timeout,
            private_output=private_output,
        )

    def _run_command(self, command, environment, project, operation, *, timeout, private_output):
        timed_out = False
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=output,
                stderr=subprocess.DEVNULL if private_output else subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            finally:
                size = output.seek(0, os.SEEK_END)
                output.seek(max(0, size - 2 * 1024 * 1024))
                text = output.read().decode("utf-8", errors="replace")
                redacted = "[private output omitted]" if private_output else text
                for value in self._redactions:
                    redacted = redacted.replace(value, "[REDACTED]")
                if self.artifact_root:
                    self.artifact_root.mkdir(parents=True, exist_ok=True)
                    with (self.artifact_root / f"{project}.log").open("a") as log:
                        log.write(
                            f"\ncompose {operation} exit={process.returncode} timeout={timed_out}\n{redacted}\n"
                        )
        if timed_out:
            raise ComposeRuntimeError(f"compose_command_timeout: {operation}")
        if process.returncode:
            if "no space left on device" in text.casefold():
                raise ComposeRuntimeError("compose_resource_exhausted")
            raise ComposeRuntimeError(
                f"compose_command_failed: {operation} exit={process.returncode}"
            )
        if private_output and size > 2 * 1024 * 1024:
            raise ComposeRuntimeError("compose_configuration_output_too_large")
        return text

    def resolve_images(self, *, manifest, document, runtime, source):
        from .compose_image_lock import image_identity, selected_images, validate_image_platform

        self._check_socket()
        images = selected_images(source, runtime, json.loads(document), control_service=manifest.runtime.control_service)
        from .runtime_selection import select_runtime, ComposeSelection

        try:
            source_defaults = select_runtime(
                source, runtime, public_environment={}, validate_environment_dependencies=False,
            )
        except ValueError:
            source_defaults = None
        result = []
        deadline = time.monotonic() + 600
        with tempfile.TemporaryDirectory(prefix="alk-image-resolution-") as scratch:
            environment = {
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "DOCKER_HOST": f"unix://{self.socket_path}",
                "DOCKER_CONFIG": scratch,
            }

            def docker(*args):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ComposeRuntimeError("compose_image_resolution_timeout")
                return self._run_command(
                    ["docker", *args], environment, "image-resolution", args[0],
                    timeout=remaining, private_output=True,
                )

            host = json.loads(docker("info", "--format", "{{json .}}"))
            architecture = {"x86_64": "amd64", "aarch64": "arm64"}.get(
                host.get("Architecture"), host.get("Architecture")
            )
            host_platform = validate_image_platform(f"{host.get('OSType')}/{architecture}")
            for name, service in images.items():
                platform = service.get("platform") or host_platform
                if platform.split("/")[:2] != host_platform.split("/")[:2]:
                    raise ComposeRuntimeError("compose_image_host_platform_mismatch")
                reference = service["image"]
                docker("image", "pull", "--platform", platform, reference)
                inspected = json.loads(docker("image", "inspect", reference))
                if not isinstance(inspected, list) or len(inspected) != 1 or not isinstance(inspected[0], dict):
                    raise ComposeRuntimeError("compose_image_inspection_invalid")
                item = image_identity(
                    service=name, reference=reference, platform=platform, inspected=inspected[0],
                    repository=manifest.provenance.repository, revision=manifest.provenance.commit,
                    application=name == manifest.runtime.control_service,
                    source_declared_digest=(
                        isinstance(source_defaults, ComposeSelection)
                        and source_defaults.services.get(name, {}).get("image") == reference
                    ),
                )
                result.append(item)
        return result

    async def provision(
        self,
        bundle: EnvironmentBundleV2,
        *,
        source: Path,
        bundle_dir: Path,
        work_directory: Path,
        contract=None,
        instances: int = 1,
    ) -> list[EnvironmentRuntime]:
        return await asyncio.to_thread(
            self._provision, bundle, source, bundle_dir, work_directory, instances
        )

    def _provision(self, bundle, source, bundle_dir, work_directory, instances):
        from .compose_integration import validate_source
        from .compose_source import source_inputs

        document = yaml.safe_load(
            _within(bundle_dir, bundle.runtime.document).read_text()
        )
        inputs = source_inputs(
            source,
            document,
            capabilities=bundle.capabilities,
            context_excludes=_bundle_context_excludes(bundle),
        )
        validate_source(
            source, bundle.provenance.public_material_admissions,
            fixtures=bundle.provenance.public_test_fixture_admissions,
            source_digest=bundle.provenance.source_digest, paths=inputs.files,
        )
        self._check_socket()
        if bundle.runtime.kind is not RuntimeKindV2.COMPOSE:
            raise ComposeRuntimeError("compose_bundle_required")
        if bundle.seed and bundle.seed.stores:
            raise ComposeRuntimeError("compose_native_seed_required")
        if instances < 1:
            raise ComposeRuntimeError("compose_instances_invalid")
        if self._worlds:
            if any(
                runtime.bundle_digest != bundle.digest
                for runtime, _, _ in self._worlds.values()
            ):
                raise ComposeRuntimeError("compose_bundle_changed")
            for runtime, directory, manifest in self._worlds.values():
                if runtime.state is not RuntimeState.READY:
                    self._stop(runtime, directory)
                    shutil.rmtree(directory / "source")
                    runtime.state = RuntimeState.PREPARING
                    self._start(runtime, directory, manifest)
            return [entry[0] for entry in self._worlds.values()]
        self._source = source
        self._document = document
        try:
            for index in range(instances):
                directory = Path(
                    tempfile.mkdtemp(prefix=f"compose-w{index}-", dir=work_directory)
                )
                directory.chmod(0o700)
                runtime = EnvironmentRuntime(
                    runtime_id=f"alk-{uuid.uuid4().hex}",
                    world_index=index,
                    bundle_digest=bundle.digest,
                    state=RuntimeState.PREPARING,
                    metadata={"observation_mode": "native"},
                )
                self._worlds[index] = runtime, directory, bundle
                self._start(runtime, directory, bundle)
        except BaseException:
            self._close()
            raise
        return [entry[0] for entry in self._worlds.values()]

    def _start(self, runtime, directory, bundle):
        from .compose_source import source_inputs
        from .compose_integration import validate_source

        inputs = source_inputs(
            self._source,
            self._document,
            capabilities=bundle.capabilities,
            context_excludes=_bundle_context_excludes(bundle),
        )
        validate_source(
            self._source, bundle.provenance.public_material_admissions,
            fixtures=bundle.provenance.public_test_fixture_admissions,
            source_digest=bundle.provenance.source_digest, paths=inputs.files,
        )
        inputs.copy_to(directory / "source")
        document = normalize_compose(
            self._document, directory / "source", capabilities=bundle.capabilities
        )
        resources = document.pop("x-alk-config-files", {})
        directories = document.pop("x-alk-directories", [])
        for value in directories:
            path = directory / "source" / value
            path.mkdir(parents=True, exist_ok=True)
        resource_values = {}
        resource_bind_values = {}
        public_values = {}
        for path, resource in resources.items():
            values = dict(resource.get("values", {}))
            for key, value in values.items():
                if key in public_values and public_values[key] != value:
                    raise ComposeRuntimeError("compose_config_public_value_conflict")
                public_values[key] = value
            for key, alias in resource.get("secret_aliases", {}).items():
                value = self.configuration.get(alias)
                if (
                    not value
                    or len(value.encode()) > 65536
                    or any(c in value for c in "\x00\n\r")
                ):
                    raise ComposeRuntimeError(
                        "compose_config_secret_missing_or_invalid"
                    )
                values[key] = value
            resource_values[path] = (
                "\n".join(
                    key + "=" + json.dumps(value.replace("$", "$$"), ensure_ascii=False)
                    for key, value in sorted(values.items())
                )
                + "\n"
            )
            if len(resource_values[path].encode()) > 65536:
                raise ComposeRuntimeError("compose_config_file_oversized")
            resource_bind_values[path] = (
                "\n".join(
                    key + "=" + json.dumps(value, ensure_ascii=False)
                    for key, value in sorted(values.items())
                )
                + "\n"
            )
        config_paths = {
            path: directory / f"config-{index}.env"
            for index, path in enumerate(resources)
        }
        resource_mounts = {}
        directory_mounts = {}
        for name, service in document["services"].items():
            resource_mounts[name] = []
            directory_mounts[name] = []
            retained = []
            for mount in service["volumes"]:
                relative = (
                    Path(mount["source"]).relative_to(directory / "source").as_posix()
                    if mount["type"] == "bind"
                    else None
                )
                if relative in resource_values:
                    if "${" in resource_bind_values[relative]:
                        raise ComposeRuntimeError(
                            "compose_bound_dotenv_interpolation_forbidden"
                        )
                    from .contract import RuntimeSecretFile

                    RuntimeSecretFile(alias="CONFIG", destination=mount["target"])
                    resource_mounts[name].append(
                        {
                            "destination": mount["target"],
                            "content": resource_bind_values[relative],
                            "read_only": mount.get("read_only", False),
                        }
                    )
                else:
                    retained.append(mount)
                    if relative in directories:
                        directory_mounts[name].append(mount)
            service["volumes"] = retained
            for entry in service["env_file"]:
                relative = (
                    Path(entry["path"]).relative_to(directory / "source").as_posix()
                )
                if relative in resource_values:
                    private = config_paths[relative]
                    private.write_text(resource_values[relative])
                    private.chmod(0o600)
                    entry["path"] = str(private)
        bindings = {
            name: service.pop("x-alk-secret-files", [])
            for name, service in document["services"].items()
        }
        file_aliases = {item["alias"] for items in bindings.values() for item in items}
        for alias in file_aliases:
            value = self.configuration.get(alias)
            if not value or len(value.encode()) > 1024 * 1024:
                raise ComposeRuntimeError(
                    "compose_secret_file_value_missing_or_oversized"
                )
            if f"${{{alias}" in json.dumps(document) or f"${alias}" in json.dumps(
                document
            ):
                raise ComposeRuntimeError(
                    "compose_secret_file_raw_interpolation_forbidden"
                )
        if (
            sum(len(self.configuration[alias].encode()) for alias in file_aliases)
            > 8 * 1024 * 1024
        ):
            raise ComposeRuntimeError("compose_secret_file_total_too_large")
        for service in document["services"].values():
            for alias in file_aliases:
                service["environment"].pop(alias, None)
        configuration = "\n".join(
            key + "=" + json.dumps(value.replace("$", "$$"), ensure_ascii=False)
            for key, value in {**public_values, **self.configuration}.items()
            if key not in file_aliases
        )
        (directory / "configuration.env").write_text(configuration + "\n")
        (directory / "configuration.env").chmod(0o600)
        redactions = set(self._redactions)
        public_log_values = set(public_values.values())
        env_index = 0
        for service in document["services"].values():
            build = service.get("build", {})
            if build and any(
                f"${{{key}" in json.dumps(build) or f"${key}" in json.dumps(build)
                for key in self.configuration
            ):
                raise ComposeRuntimeError("compose_build_configuration_forbidden")
            redactions.update(
                str(value)
                for value in service["environment"].values()
                if value and str(value) not in public_log_values
            )
            for entry in service["env_file"]:
                snapshot = directory / f"service-{env_index}.env"
                env_index += 1
                snapshot.write_bytes(Path(entry["path"]).read_bytes())
                snapshot.chmod(0o600)
                entry["path"] = str(snapshot)
        (directory / "compose.json").write_text(json.dumps(document))
        (directory / "compose.json").chmod(0o600)
        resolved_text = self._command(
            directory,
            runtime.runtime_id,
            "config",
            "--format",
            "json",
            private_output=True,
        )
        try:
            resolved = json.loads(resolved_text)
            for service in resolved["services"].values():
                if set(service.get("environment", {})) & file_aliases:
                    raise ComposeRuntimeError(
                        "compose_secret_file_raw_environment_forbidden"
                    )
                redactions.update(
                    str(value)
                    for value in service.get("environment", {}).values()
                    if value and str(value) not in public_log_values
                )
        except (ValueError, KeyError, TypeError, AttributeError):
            raise ComposeRuntimeError("compose_configuration_invalid") from None
        self._redactions = tuple(sorted(redactions, key=len, reverse=True))
        (directory / "configured").touch(mode=0o600)
        for name, items in resource_mounts.items():
            bindings[name].extend(items)
        if any(bindings.values()) or any(directory_mounts.values()):
            self._mount_secret_files(
                directory, runtime.runtime_id, document, bindings, directory_mounts
            )
        self._command(
            directory,
            runtime.runtime_id,
            "up",
            "--detach",
            "--build",
            "--wait",
            "--wait-timeout",
            "300",
            timeout=1800,
        )
        self._command(directory, runtime.runtime_id, "images", "--format", "json")
        endpoints = {}
        for slug, cap in bundle.capabilities.items():
            output = self._command(
                directory,
                runtime.runtime_id,
                "port",
                cap.service,
                str(cap.container_port),
            ).strip()
            if not re.fullmatch(r"127\.0\.0\.1:[0-9]+", output):
                raise ComposeRuntimeError("compose_published_port_invalid")
            scheme = (
                "http"
                if cap.protocol.value in {"http", "mcp", "livekit"}
                else cap.protocol.value
            )
            endpoints[slug] = RuntimeEndpoint(
                capability=slug,
                protocol=cap.protocol.value,
                address=f"{scheme}://{output}",
                configuration_name=cap.configuration_name,
            )
        runtime.endpoints = endpoints
        for probe in bundle.readiness:
            deadline = time.monotonic() + probe.timeout_seconds
            while not self._probe(runtime, bundle, probe):
                if time.monotonic() >= deadline:
                    raise ComposeRuntimeError(
                        f"compose_readiness_failed: {probe.capability}"
                    )
                time.sleep(probe.interval_seconds)
        runtime.state = RuntimeState.READY

    def _mount_secret_files(
        self, directory, project, document, bindings, directory_mounts=None
    ):
        directory_mounts = directory_mounts or {}
        directory_owners = {}
        # Probe images before mounting credentials; the application entrypoint never runs.
        for name, items in bindings.items():
            if not items and not directory_mounts.get(name):
                continue
            if document["services"][name].get("build"):
                self._command(directory, project, "build", name, timeout=1800)
            probe = 'for p do while [ "$p" != / ]; do [ ! -L "$p" ] || exit 45; p=${p%/*}; [ -n "$p" ] || p=/; done; done; id -u; id -g'
            probe_document = copy.deepcopy(document)
            for probe_service in probe_document["services"].values():
                probe_service["environment"] = {}
                probe_service["env_file"] = []
            (directory / "compose.json").write_text(json.dumps(probe_document))
            identity = (
                self._command(
                    directory,
                    project,
                    "run",
                    "--rm",
                    "--no-deps",
                    "-T",
                    "--entrypoint",
                    "/bin/sh",
                    name,
                    "-c",
                    probe,
                    "alk-file-probe",
                    *[item["destination"] for item in items],
                    *[item["target"] for item in directory_mounts.get(name, [])],
                    private_output=True,
                )
                .strip()
                .splitlines()
            )
            if len(identity) != 2 or any(not value.isdecimal() for value in identity):
                raise ComposeRuntimeError("compose_secret_file_owner_unresolved")
            uid, gid = map(int, identity)
            if max(uid, gid) >= 2**32 - 1:
                raise ComposeRuntimeError("compose_secret_file_owner_invalid")
            for mount in directory_mounts.get(name, []):
                path = Path(mount["source"])
                if path in directory_owners and directory_owners[path] != (uid, gid):
                    raise ComposeRuntimeError("compose_directory_owner_conflict")
                directory_owners[path] = (uid, gid)
                _own_directory(path, uid, gid)
            private = directory / "secret-files" / name
            private.mkdir(mode=0o700, parents=True, exist_ok=True)
            for index, item in enumerate(items):
                path = private / str(index)
                path.unlink(missing_ok=True)
                fd = os.open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
                )
                try:
                    with os.fdopen(fd, "wb") as output:
                        output.write(
                            (
                                item["content"]
                                if "content" in item
                                else self.configuration[item["alias"]]
                            ).encode()
                        )
                    os.chown(path, uid, gid)
                except BaseException:
                    path.unlink(missing_ok=True)
                    raise
                service = document["services"][name]
                service["volumes"].append(
                    {
                        "type": "bind",
                        "source": str(path),
                        "target": item["destination"],
                        "read_only": (
                            item.get("read_only", True) if "content" in item else True
                        ),
                    }
                )
                if item.get("environment_variable"):
                    service["environment"][item["environment_variable"]] = item[
                        "destination"
                    ]
        (directory / "compose.json").write_text(json.dumps(document))

    def _probe(self, runtime, bundle, probe):
        endpoint = runtime.endpoints[probe.capability]
        port = int(endpoint.address.rsplit(":", 1)[1])
        return default_capability_prober(
            protocol=bundle.capabilities[probe.capability].protocol,
            host="127.0.0.1",
            port=port,
            path=probe.path,
        )

    def _stop(self, runtime, directory):
        if (directory / "configured").exists():
            try:
                self._command(
                    directory,
                    runtime.runtime_id,
                    "logs",
                    "--no-color",
                    "--tail",
                    "2000",
                    timeout=30,
                )
            finally:
                self._command(
                    directory,
                    runtime.runtime_id,
                    "down",
                    "--volumes",
                    "--remove-orphans",
                    "--timeout",
                    "10",
                    timeout=120,
                )
        (directory / "configured").unlink(missing_ok=True)
        runtime.state = RuntimeState.STOPPED

    async def reset(
        self, runtime, *, work_directory: Path, keep_processes: bool = False
    ):
        def reset():
            current, directory, bundle = self._worlds[runtime.world_index]
            self._stop(current, directory)
            shutil.rmtree(directory / "source")
            current.state = RuntimeState.PREPARING
            self._start(current, directory, bundle)

        await asyncio.to_thread(reset)

    async def healthy(self, runtime, *, work_directory: Path) -> bool:
        _, directory, bundle = self._worlds[runtime.world_index]

        def check():
            try:
                running = self._command(
                    directory,
                    runtime.runtime_id,
                    "ps",
                    "--status",
                    "running",
                    "--services",
                ).splitlines()
                valid = set(running) == set(self._document["services"]) and all(
                    self._probe(runtime, bundle, probe) for probe in bundle.readiness
                )
            except (ComposeRuntimeError, OSError, subprocess.TimeoutExpired):
                valid = False
            runtime.state = RuntimeState.READY if valid else RuntimeState.UNHEALTHY
            return valid

        return await asyncio.to_thread(check)

    def _close(self):
        errors = []
        for runtime, directory, _ in self._worlds.values():
            try:
                self._stop(runtime, directory)
                shutil.rmtree(directory)
            except Exception as error:
                errors.append(error)
        if errors:
            raise ComposeRuntimeError("compose_cleanup_failed") from errors[0]
        self._worlds.clear()

    async def close(self, *, work_directory: Path):
        await asyncio.to_thread(self._close)
