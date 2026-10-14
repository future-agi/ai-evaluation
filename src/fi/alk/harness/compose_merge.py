"""Native Compose overlay merging without engine access or ambient configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import yaml

LIMIT = 2 * 1024 * 1024
COMPOSE_LOCATIONS = (
    "/usr/libexec/docker/cli-plugins/docker-compose",
    "/usr/lib/docker/cli-plugins/docker-compose",
    "/usr/local/libexec/docker/cli-plugins/docker-compose",
    "/usr/local/lib/docker/cli-plugins/docker-compose",
    "/Applications/Docker.app/Contents/Resources/cli-plugins/docker-compose",
)


def _merge_size(node, active, memo):
    """Bound YAML anchor expansion before SafeLoader flattens inspected mappings."""
    if id(node) in memo:
        return memo[id(node)]
    if id(node) in active or len(active) >= 32:
        raise ValueError("compose_overlay_yaml_merge_cycle_or_depth")
    if not isinstance(node, yaml.MappingNode):
        raise ValueError("compose_overlay_mapping_required")
    active.add(id(node))
    size = 0
    for key, value in node.value:
        if key.tag == "tag:yaml.org,2002:merge":
            entries = value.value if isinstance(value, yaml.SequenceNode) else [value]
            for entry in entries:
                size += _merge_size(entry, active, memo)
        else:
            size += 1
        if size > 8192:
            raise ValueError("compose_overlay_yaml_merge_too_large")
    active.remove(id(node))
    memo[id(node)] = size
    return size


def _mapping(node, loader):
    if not isinstance(node, yaml.MappingNode):
        raise ValueError("compose_overlay_mapping_required")
    _merge_size(node, set(), {})
    # Only YAML anchors are inspected here; native Compose still owns !override/!reset.
    loader.flatten_mapping(node)
    return {
        key.value: value
        for key, value in node.value
        if isinstance(key, yaml.ScalarNode)
    }


def _admit(path: Path, root: Path, base: Path):
    """Check file-reading features before the native parser can follow them."""
    loader = yaml.SafeLoader(path.read_text())
    try:
        document = _mapping(loader.get_single_node(), loader)
        if set(document) & {"include", "configs", "secrets"}:
            raise ValueError("compose_overlay_file_loading_unsupported")
        services = (
            _mapping(document["services"], loader) if "services" in document else {}
        )
        for service in services.values():
            fields = _mapping(service, loader)
            if set(fields) & {"extends", "label_file"}:
                raise ValueError("compose_overlay_file_loading_unsupported")
            if "env_file" not in fields:
                continue
            node = fields["env_file"]
            entries = node.value if isinstance(node, yaml.SequenceNode) else [node]
            for entry in entries:
                if isinstance(entry, yaml.MappingNode):
                    entry = _mapping(entry, loader).get("path")
                if not isinstance(entry, yaml.ScalarNode):
                    raise ValueError("compose_overlay_env_file_invalid")
                value = entry.value
                if (
                    not value
                    or "$" in value
                    or "\\" in value
                    or Path(value).is_absolute()
                    or not (base / value).resolve().is_relative_to(root)
                ):
                    raise ValueError("compose_overlay_env_file_outside_source")
    except (yaml.YAMLError, RecursionError) as exc:
        raise ValueError("compose_overlay_yaml_invalid") from exc
    finally:
        loader.dispose()


def merge_compose(root: Path, paths: list[Path]) -> dict:
    root = root.resolve()
    resolved = [path.resolve() for path in paths]
    if not 2 <= len(resolved) <= 9 or len(set(resolved)) != len(resolved):
        raise ValueError("compose_overlay_files_invalid")
    if any(not path.is_relative_to(root) or not path.is_file() for path in resolved):
        raise ValueError("compose_overlay_path_outside_source_or_missing")
    if sum(path.stat().st_size for path in resolved) > LIMIT:
        raise ValueError("compose_overlay_input_too_large")
    for path in resolved:
        _admit(path, root, resolved[0].parent)
    binary = next(
        (
            path
            for path in COMPOSE_LOCATIONS
            if Path(path).is_file() and os.access(path, os.X_OK)
        ),
        None,
    )
    if binary is None:
        raise ValueError("compose_overlay_cli_unavailable")
    command = [
        binary,
        "--env-file",
        os.devnull,
        "--project-name",
        "alk-merge",
        "--project-directory",
        str(resolved[0].parent),
    ]
    for path in resolved:
        command.extend(["-f", str(path)])
    command.extend(
        [
            "config",
            "--format",
            "json",
            "--no-interpolate",
            "--no-env-resolution",
            "--no-path-resolution",
            "--no-normalize",
        ]
    )
    with (
        tempfile.TemporaryDirectory(prefix="alk-compose-config-") as config,
        tempfile.TemporaryFile() as output,
    ):
        process = subprocess.Popen(
            command,
            cwd=root,
            env={
                "PATH": os.defpath,
                "COMPOSE_DISABLE_ENV_FILE": "1",
                "DOCKER_CONFIG": config,
            },
            stdout=output,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 30
        try:
            while process.poll() is None:
                if os.fstat(output.fileno()).st_size > LIMIT:
                    raise ValueError("compose_overlay_output_too_large")
                if time.monotonic() >= deadline:
                    raise ValueError("compose_overlay_cli_timeout")
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise ValueError("compose_overlay_cli_failed")
            output.seek(0)
            raw = output.read(LIMIT + 1)
            if len(raw) > LIMIT:
                raise ValueError("compose_overlay_output_too_large")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    try:
        document = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise ValueError("compose_overlay_output_invalid") from exc
    if not isinstance(document, dict) or not isinstance(document.get("services"), dict):
        raise ValueError("compose_overlay_output_invalid")
    for key in ("volumes", "networks"):
        if document.get(key) is None:
            document.pop(key, None)
    return document
