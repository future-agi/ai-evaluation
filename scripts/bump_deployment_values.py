#!/usr/bin/env python3
"""Pin a certified E2B template in deployment values files.

Rewrites the hosted-template settings in each values file to match a release
written by scripts/e2b-template.py. Lines are edited in place so every other
byte of a file is kept, and nothing is written unless every file passes.

Usage:

    python3 scripts/bump_deployment_values.py \
      --release dist/e2b-template-release.json \
      us/gcp/deployment/values.yaml eu/gcp/deployment/values.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

RELEASE_SCHEMA = "futureagi.e2b-template-release.v1"
TEMPLATE_KEYS = (
    "ALK_E2B_TEMPLATE_REFERENCE",
    "ALK_E2B_TEMPLATE_BUILD_ID",
    "ALK_E2B_TEMPLATE_CPU_UNITS",
    "ALK_E2B_TEMPLATE_MEMORY_MB",
    "ALK_E2B_TEMPLATE_DISK_GB",
)
PARALLEL_KEY = "HARNESS_PARALLEL_SNAPSHOT_DIGESTS"

_ALL_KEYS = (*TEMPLATE_KEYS, PARALLEL_KEY)
_BUILD_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_TEMPLATE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_KEY_START = re.compile(r"^[ \t]*(?P<key>" + "|".join(_ALL_KEYS) + r")[ \t]*:")
_KEY_LINE = re.compile(
    r"^(?P<head>[ \t]*(?:" + "|".join(_ALL_KEYS) + r")[ \t]*:[ \t]*)"
    r"(?P<value>\"[^\"\\]*\"|'[^']*'|[^\s\"'#|>&*!{\[][^\s#]*)"
    r"(?P<tail>[ \t]*|[ \t]+#.*)$"
)


class BumpError(ValueError):
    """The release or a values file is not safe to act on."""


@dataclass(frozen=True)
class TemplatePin:
    name: str
    reference: str
    build_id: str
    cpu_units: int
    memory_mb: int
    disk_gb: int


@dataclass(frozen=True)
class _KeyLine:
    index: int
    head: str
    value: str
    tail: str
    ending: str


def _positive_int(release: dict, field: str) -> int:
    value = release.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BumpError(
            f"template release field {field} must be a whole number above zero"
        )
    return value


def load_release(path: Path) -> TemplatePin:
    try:
        release = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BumpError(f"cannot read the template release {path}") from exc
    if not isinstance(release, dict):
        raise BumpError("template release must be a JSON object")
    if release.get("schema_version") != RELEASE_SCHEMA:
        raise BumpError(f"template release must have schema_version {RELEASE_SCHEMA}")
    name = release.get("template_name")
    build_id = release.get("template_build_id")
    reference = release.get("template_reference")
    if not isinstance(name, str) or not _TEMPLATE_NAME.fullmatch(name):
        raise BumpError(
            "template release field template_name is not a valid template name"
        )
    if not isinstance(build_id, str) or not _BUILD_ID.fullmatch(build_id):
        raise BumpError("template release field template_build_id is not a build UUID")
    if reference != f"{name}:{build_id}":
        raise BumpError(
            "template release field template_reference must be template_name:template_build_id"
        )
    return TemplatePin(
        name=name,
        reference=reference,
        build_id=build_id,
        cpu_units=_positive_int(release, "cpu_count"),
        memory_mb=_positive_int(release, "memory_mb"),
        disk_gb=_positive_int(release, "verified_disk_gb"),
    )


def _unquote(value: str) -> str:
    return value[1:-1] if value[0] in {'"', "'"} else value


def _scan(lines: list[str], label: str) -> dict[str, list[_KeyLine]]:
    found: dict[str, list[_KeyLine]] = {key: [] for key in _ALL_KEYS}
    for index, raw in enumerate(lines):
        ending = "\r" if raw.endswith("\r") else ""
        body = raw[: len(raw) - len(ending)]
        start = _KEY_START.match(body)
        if start is None:
            continue
        key = start.group("key")
        match = _KEY_LINE.match(body)
        if match is None:
            raise BumpError(
                f"{label}: line {index + 1} sets {key} in a form that cannot be rewritten safely"
            )
        found[key].append(
            _KeyLine(
                index,
                match.group("head"),
                match.group("value"),
                match.group("tail"),
                ending,
            )
        )
    return found


def _swap_build(value: str, old_build: str, new_build: str) -> str | None:
    entries = value.split(",")
    if not old_build or old_build not in (entry.strip() for entry in entries):
        return None
    swapped: list[str] = []
    for entry in entries:
        if entry.strip() == old_build:
            entry = new_build
        if entry.strip() == new_build and new_build in (
            kept.strip() for kept in swapped
        ):
            continue
        swapped.append(entry)
    return ",".join(swapped)


def bump_text(text: str, pin: TemplatePin, *, label: str) -> tuple[str, list[str]]:
    """Return the text with the template pinned, and one line per setting that changed."""
    lines = text.split("\n")
    found = _scan(lines, label)
    counts = {key: len(found[key]) for key in TEMPLATE_KEYS}
    if min(counts.values()) == 0 or len(set(counts.values())) != 1:
        detail = ", ".join(f"{key} x{count}" for key, count in counts.items())
        raise BumpError(
            f"{label}: every template setting must appear the same number of times, "
            f"at least once; found {detail}"
        )
    old_builds = {_unquote(line.value) for line in found["ALK_E2B_TEMPLATE_BUILD_ID"]}
    if len(old_builds) != 1:
        raise BumpError(
            f"{label}: ALK_E2B_TEMPLATE_BUILD_ID differs between blocks; "
            "pin every block to one template by hand first"
        )
    old_build = old_builds.pop()

    wanted = {
        "ALK_E2B_TEMPLATE_REFERENCE": pin.reference,
        "ALK_E2B_TEMPLATE_BUILD_ID": pin.build_id,
        "ALK_E2B_TEMPLATE_CPU_UNITS": str(pin.cpu_units),
        "ALK_E2B_TEMPLATE_MEMORY_MB": str(pin.memory_mb),
        "ALK_E2B_TEMPLATE_DISK_GB": str(pin.disk_gb),
    }
    changes: list[str] = []

    def rewrite(line: _KeyLine, value: str) -> bool:
        updated = f'{line.head}"{value}"{line.tail}{line.ending}'
        if updated == lines[line.index]:
            return False
        lines[line.index] = updated
        return True

    for key in TEMPLATE_KEYS:
        changed = sum(rewrite(line, wanted[key]) for line in found[key])
        if changed:
            changes.append(f"{key}: {changed} line(s) set to {wanted[key]}")
    swapped_lines = 0
    for line in found[PARALLEL_KEY]:
        swapped = _swap_build(_unquote(line.value), old_build, pin.build_id)
        if swapped is None:
            continue
        if '"' in swapped or "\\" in swapped:
            raise BumpError(
                f"{label}: line {line.index + 1} sets {PARALLEL_KEY} in a form that "
                "cannot be rewritten safely"
            )
        swapped_lines += rewrite(line, swapped)
    if swapped_lines:
        changes.append(
            f"{PARALLEL_KEY}: {swapped_lines} line(s) now list {pin.build_id}"
        )

    after = _scan(lines, label)
    for key in TEMPLATE_KEYS:
        values = [_unquote(line.value) for line in after[key]]
        if values != [wanted[key]] * counts[key]:
            raise BumpError(
                f"{label}: {key} was not pinned on every line; nothing was written"
            )
    return "\n".join(lines), changes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        type=Path,
        required=True,
        help="Release JSON written by scripts/e2b-template.py",
    )
    parser.add_argument("values_files", type=Path, nargs="+", metavar="values.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    planned: list[tuple[Path, str, str, list[str]]] = []
    try:
        pin = load_release(args.release)
        for path in args.values_files:
            try:
                original = path.read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise BumpError(f"cannot read {path} as UTF-8 text") from exc
            updated, changes = bump_text(original, pin, label=str(path))
            planned.append((path, original, updated, changes))
    except BumpError as exc:
        raise SystemExit(f"error: {exc}") from exc

    for path, original, updated, changes in planned:
        if updated == original:
            print(f"{path}: already pinned")
            continue
        path.write_bytes(updated.encode("utf-8"))
        print(f"{path}: pinned {pin.reference}")
        for change in changes:
            print(f"  {change}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
