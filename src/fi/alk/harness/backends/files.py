"""Read-only file tools for backends that have no host CLI behind them.

Claude Code brings its own Read, Glob and Grep. Any other loop that is granted those names gets
these: same names, same core arguments, same read-only stance. They exist so a stage's skill
text ("read the repository, never write to it") means the same thing on every backend.

Write access is not implemented on purpose. The harness's own artifacts go through its tools,
which validate them; the agent under test is somebody's real repository. A backend that wants
write tools is asking to skip the gates, and the answer is no.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from .base import ToolSpec

MAX_READ_LINES = 2000
MAX_LINE_CHARS = 2000
MAX_MATCHES = 200
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}


def _ok(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _error(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def _resolved(raw: str, cwd: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else cwd / path


def _walk(root: Path):
    """Every file under root, skipping the trees nobody means when they say 'the repo'."""
    stack = [root]
    while stack:
        folder = stack.pop()
        try:
            entries = sorted(folder.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in _SKIP_DIRS:
                    stack.append(entry)
            else:
                yield entry


def file_tools(cwd: str | None) -> list[ToolSpec]:
    """Read, Glob and Grep rooted at the session's working directory."""
    base = Path(cwd) if cwd else Path.cwd()

    async def read(args: dict) -> dict:
        raw = str(args.get("file_path") or args.get("path") or "")
        if not raw:
            return _error("file_path is required")
        path = _resolved(raw, base)
        if not path.is_file():
            return _error(f"{path} is not a file that exists")
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return _error(f"could not read {path}: {exc}")
        offset = max(int(args.get("offset") or 1), 1)
        limit = min(int(args.get("limit") or MAX_READ_LINES), MAX_READ_LINES)
        window = lines[offset - 1 : offset - 1 + limit]
        numbered = "\n".join(
            f"{offset + index}\t{line[:MAX_LINE_CHARS]}"
            for index, line in enumerate(window)
        )
        remaining = len(lines) - (offset - 1 + len(window))
        if remaining > 0:
            numbered += f"\n... ({remaining} more lines; call again with offset={offset + len(window)})"
        return _ok(numbered or "(empty file)")

    async def glob(args: dict) -> dict:
        pattern = str(args.get("pattern") or "")
        if not pattern:
            return _error("pattern is required")
        root = _resolved(str(args.get("path") or "."), base)
        if not root.is_dir():
            return _error(f"{root} is not a directory that exists")
        matches = []
        for entry in _walk(root):
            relative = str(entry.relative_to(root))
            if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(
                entry.name, pattern
            ):
                matches.append(str(entry))
                if len(matches) >= MAX_MATCHES:
                    break
        return _ok("\n".join(matches) or f"no files match {pattern!r} under {root}")

    async def grep(args: dict) -> dict:
        pattern = str(args.get("pattern") or "")
        if not pattern:
            return _error("pattern is required")
        try:
            expression = re.compile(pattern)
        except re.error as exc:
            return _error(f"invalid regular expression: {exc}")
        root = _resolved(str(args.get("path") or "."), base)
        wanted = str(args.get("glob") or "")
        hits: list[str] = []
        targets = [root] if root.is_file() else list(_walk(root)) if root.is_dir() else []
        if not targets:
            return _error(f"{root} does not exist")
        for entry in targets:
            if wanted and not fnmatch.fnmatch(entry.name, wanted):
                continue
            try:
                text = entry.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if expression.search(line):
                    hits.append(f"{entry}:{number}:{line[:400]}")
                    if len(hits) >= MAX_MATCHES:
                        break
            if len(hits) >= MAX_MATCHES:
                break
        return _ok("\n".join(hits) or f"no lines match {pattern!r}")

    return [
        ToolSpec(
            name="Read",
            description=(
                "Read a file. Arguments: file_path (absolute or relative to the working "
                "directory), optional offset (1-based first line) and limit (line count)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["file_path"],
            },
            handler=read,
        ),
        ToolSpec(
            name="Glob",
            description=(
                "Find files by name pattern, e.g. **/*.py. Arguments: pattern, optional path "
                "to search under."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
            handler=glob,
        ),
        ToolSpec(
            name="Grep",
            description=(
                "Search file contents with a regular expression. Arguments: pattern, optional "
                "path (file or directory) and glob to filter file names."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                },
                "required": ["pattern"],
            },
            handler=grep,
        ),
    ]
