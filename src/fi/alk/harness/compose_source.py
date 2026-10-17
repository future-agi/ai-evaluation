"""Derive the filesystem inputs consumed by a validated Compose deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(frozen=True)
class SourceInputs:
    source: Path
    entries: tuple[Path, ...]

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(
            self.source / path for path in self.entries
            if not (self.source / path).is_symlink() and (self.source / path).is_file()
        )

    def copy_to(self, destination: Path) -> None:
        destination.mkdir()
        directories = []
        links = []
        for relative in self.entries:
            source = self.source / relative
            target = destination / relative
            if source.is_symlink():
                links.append((source, target))
            elif source.is_dir():
                target.mkdir(exist_ok=True)
                directories.append((source, target))
            else:
                shutil.copy2(source, target)
        for source, target in links:
            target.symlink_to(source.readlink())
        for source, target in reversed(directories):
            shutil.copystat(source, target)
        shutil.copystat(self.source, destination)


def source_inputs(
    source: Path,
    document: dict,
    *,
    capabilities: dict,
    context_excludes: tuple[str, ...] = (),
) -> SourceInputs:
    from .certified_compose_runtime import (
        ComposeRuntimeError, normalize_compose, validate_source_tree,
    )
    from .generated_runtime import _context_path_excluded

    source = source.resolve()
    for value in context_excludes:
        if not isinstance(value, str):
            raise ComposeRuntimeError("compose_source_context_exclude_invalid")
        relative = Path(value)
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in value
            or "$" in value
            or any(ord(char) < 32 for char in value)
            or any(char in value for char in "?[]")
        ):
            raise ComposeRuntimeError("compose_source_context_exclude_invalid")
    validate_source_tree(source)
    normalized = normalize_compose(document, source, capabilities=capabilities)
    roots = set()
    for name, service in normalized["services"].items():
        original = document["services"][name]
        if service.get("build"):
            build = original["build"]
            roots.add(build if isinstance(build, str) else build.get("context", "."))
        for entry in service["env_file"]:
            roots.add(entry["path"])
        env_files = original.get("env_file", [])
        for entry in [env_files] if isinstance(env_files, str) else env_files:
            value = entry if isinstance(entry, str) else entry["path"]
            if (source / value).exists():
                roots.add(value)
        for mount in original.get("volumes", []):
            if isinstance(mount, str):
                origin = mount.split(":", 1)[0]
                if origin.startswith((".", "/", "~")):
                    roots.add(origin)
            elif mount.get("type") == "bind":
                roots.add(mount["source"])

    entries = set()
    expanded = set()
    active = set()

    def add(path: Path, descend=True):
        try:
            relative = path.relative_to(source)
        except ValueError:
            relative = Path()
        if relative.parts and _context_path_excluded(relative, context_excludes):
            return
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(source)
        except (OSError, RuntimeError, ValueError):
            raise ComposeRuntimeError("compose_source_symlink_escape_or_cycle") from None
        if relative.parts:
            parent = source
            for part in relative.parts[:-1]:
                if part == "..":
                    parent = parent.parent
                else:
                    ancestor = parent / part
                    if ancestor.is_symlink():
                        add(ancestor, False)
                    parent = ancestor.resolve(strict=True)
                if not parent.is_relative_to(source):
                    raise ComposeRuntimeError("compose_source_symlink_escape_or_cycle")
            path = parent.parent if relative.name == ".." else parent / relative.name
            relative = path.relative_to(source)
        if path == source:
            if not descend:
                return
        else:
            add(path.parent, False)
            entries.add(relative)
        if path.is_symlink():
            add(path.parent / path.readlink(), descend)
        elif path.is_dir() and descend:
            if resolved in active:
                raise ComposeRuntimeError("compose_source_symlink_cycle")
            if resolved in expanded:
                return
            active.add(resolved)
            for child in sorted(path.iterdir()):
                add(child)
            active.remove(resolved)
            expanded.add(resolved)

    for value in sorted(roots):
        path = source / value
        if path.exists() or path.is_symlink():
            relative = path.resolve().relative_to(source)
            if _context_path_excluded(relative, context_excludes):
                raise ComposeRuntimeError("compose_source_required_path_excluded")
            add(path)
    return SourceInputs(source, tuple(sorted(entries, key=lambda p: (len(p.parts), p.as_posix()))))
