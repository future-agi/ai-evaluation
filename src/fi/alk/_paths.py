from __future__ import annotations

from pathlib import Path


def project_root(start: str | Path) -> Path:
    path = Path(start).expanduser().resolve()
    current = path.parent if path.is_file() else path
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "fi" / "alk"
        ).is_dir():
            return candidate
    raise RuntimeError(f"agent_learning_kit_project_root_not_found: {path}")
