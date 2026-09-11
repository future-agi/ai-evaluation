from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

LEGACY_SOURCE_TREES = (
    (
        WORKSPACE_ROOT / "agent-opt" / "src" / "fi" / "opt",
        PROJECT_ROOT / "src" / "fi" / "opt",
        {
            Path("components.py"),
            Path("__init__.py"),
            Path("integrations/simulate.py"),
            Path("optimizers/agent.py"),
        },
    ),
    (
        WORKSPACE_ROOT / "simulate-sdk" / "fi" / "simulate",
        PROJECT_ROOT / "src" / "fi" / "simulate",
        {
            Path("agent/frameworks.py"),
            Path("cli.py"),
            Path("environment.py"),
            Path("manifest.py"),
            Path("suite.py"),
        },
    ),
    (
        WORKSPACE_ROOT / "ai-evaluation" / "python" / "fi" / "evals",
        PROJECT_ROOT / "src" / "fi" / "evals",
        {Path("metrics/agents/report.py")},
    ),
    (
        WORKSPACE_ROOT / "ai-evaluation" / "python" / "fi" / "cli",
        PROJECT_ROOT / "src" / "fi" / "cli",
        set(),
    ),
)


def test_legacy_sdk_source_trees_are_moved_into_agent_learning_kit():
    missing_legacy = [
        str(legacy.relative_to(WORKSPACE_ROOT))
        for legacy, _, _ in LEGACY_SOURCE_TREES
        if not legacy.exists()
    ]
    if missing_legacy:
        pytest.skip(
            "legacy SDK repos are not present beside agent-learning-kit: "
            + ", ".join(missing_legacy)
        )

    missing_files: list[str] = []
    unexpected_drift: list[str] = []
    for legacy_root, unified_root, allowed_drift in LEGACY_SOURCE_TREES:
        for relative_path in _source_files(legacy_root):
            unified_path = unified_root / relative_path
            legacy_label = str(legacy_root.relative_to(WORKSPACE_ROOT) / relative_path)
            unified_label = str(unified_root.relative_to(PROJECT_ROOT) / relative_path)
            if not unified_path.exists():
                missing_files.append(f"{legacy_label} -> {unified_label}")
                continue
            if relative_path in allowed_drift:
                continue
            if (legacy_root / relative_path).read_bytes() != unified_path.read_bytes():
                unexpected_drift.append(f"{legacy_label} -> {unified_label}")

    assert missing_files == []
    assert unexpected_drift == []


def _source_files(root: Path) -> list[Path]:
    return sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
