"""Relocate build-local launchers after copying a world runtime tree."""

from __future__ import annotations

import os
from pathlib import Path


def prepare_runtime_environment(root: Path) -> dict[str, str]:
    """Keep build-downloaded assets inside the tree copied into each world."""
    state = root / ".alk-runtime"
    home = state / "home"
    cache = state / "cache"
    huggingface = cache / "huggingface"
    env = {
        "HOME": str(home),
        "XDG_CACHE_HOME": str(cache),
        "HF_HOME": str(huggingface),
        "HF_HUB_CACHE": str(huggingface / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(huggingface / "hub"),
    }
    for value in env.values():
        directory = Path(value)
        if not directory.resolve().is_relative_to(root.resolve()):
            raise OSError("runtime cache directory escapes the runtime tree")
        directory.mkdir(parents=True, exist_ok=True)
    return env


def relocate_runtime_files(build: Path, runtime: Path) -> None:
    """Redirect internal absolute links and virtualenv launchers to the private copy.

    System interpreter links stay shared. This is path isolation for cooperative
    processes, not a filesystem security boundary between processes sharing a UID.
    """
    old = str(build.resolve())
    new = str(runtime.resolve())
    for path in runtime.rglob("*"):
        if path.is_symlink():
            target = os.readlink(path)
            if target == old or target.startswith(old + os.sep):
                path.unlink()
                path.symlink_to(new + target[len(old) :])
        elif path.is_file() and (
            path.parent.name == "bin"
            or path.suffix == ".pth"
            or (
                path.name.startswith("__editable__")
                and path.name.endswith("_finder.py")
            )
        ):
            if path.stat().st_size > 1024 * 1024:
                continue
            data = path.read_bytes()
            if b"\0" not in data and old.encode() in data:
                path.write_bytes(data.replace(old.encode(), new.encode()))


def materialize_config_resources(root, config_files, directories, secrets):
    """Write declared dotenv files only into a private runtime copy, never a build."""
    import json

    from .contract import validate_runtime_resource_paths

    validate_runtime_resource_paths(config_files, directories)
    root = root.resolve()
    rendered = {}
    environment = {}
    for relative in [*directories, *config_files]:
        path = root / relative
        if any(
            parent.is_symlink() for parent in [path, *path.parents] if parent != root
        ):
            raise OSError("runtime resource path contains a symlink")
        if not path.resolve().is_relative_to(root):
            raise OSError("runtime resource path escapes the runtime tree")
        if relative in config_files and path.exists():
            raise OSError("runtime config would overwrite a source file")
    for relative, resource in config_files.items():
        values = dict(resource.values)
        for name, alias in resource.secret_aliases.items():
            value = secrets.get(alias)
            if (
                not value
                or len(value.encode()) > 65536
                or any(c in value for c in "\x00\n\r")
            ):
                raise OSError("runtime config secret missing or invalid")
            values[name] = value
        for name, value in values.items():
            if "${" in value:
                raise OSError("runtime config dotenv interpolation forbidden")
            if name in environment and environment[name] != value:
                raise OSError("runtime config values conflict")
            environment[name] = value
        text = (
            "\n".join(
                key + "=" + json.dumps(value, ensure_ascii=False)
                for key, value in sorted(values.items())
            )
            + "\n"
        )
        if len(text.encode()) > 65536:
            raise OSError("runtime config file oversized")
        rendered[relative] = text
    for relative in directories:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative, text in rendered.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
    return environment
