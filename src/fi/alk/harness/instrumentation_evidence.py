from __future__ import annotations

import ast
import os
from pathlib import Path

_SKIPPED_DIRECTORIES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "site-packages",
    "venv",
}
_OTEL_SDK_TRACE = "opentelemetry.sdk.trace"
_OTEL_EXPORTER = "opentelemetry.exporter."
_OTEL_GRPC_EXPORTER = "opentelemetry.exporter.otlp.proto.grpc"


def detect_instrumentation(source_root: Path | None) -> str:
    if source_root is None:
        return "none"
    imported = _python_imports(source_root.resolve())
    if any(
        name.split(".")[0] == "fi_instrumentation"
        or name.split(".")[0].startswith("traceai_")
        for name in imported
    ):
        return "traceai"
    if not any(_within(name, _OTEL_SDK_TRACE) for name in imported):
        return "none"
    exporters = {name for name in imported if name.startswith(_OTEL_EXPORTER)}
    leaves = {
        name
        for name in exporters
        if not any(other.startswith(name + ".") for other in exporters)
    }
    if leaves and all(_within(name, _OTEL_GRPC_EXPORTER) for name in leaves):
        return "none"
    return "opentelemetry"


def _python_imports(root: Path) -> set[str]:
    imported: set[str] = set()
    local: set[str] = set()
    for directory, subdirectories, files in os.walk(root):
        if Path(directory) != root and any(
            name.endswith(".dist-info") for name in subdirectories
        ):
            subdirectories[:] = []
            continue
        subdirectories[:] = [
            name for name in subdirectories if name not in _SKIPPED_DIRECTORIES
        ]
        if Path(directory) != root and "__init__.py" in files:
            local.add(Path(directory).name)
        for file_name in files:
            if not file_name.endswith(".py"):
                continue
            local.add(file_name[: -len(".py")])
            try:
                tree = ast.parse(
                    (Path(directory) / file_name).read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, SyntaxError, ValueError):
                continue
            imported.update(_imports(tree))
    return {name for name in imported if name.split(".")[0] not in local}


def _imports(node: ast.AST) -> set[str]:
    found: set[str] = set()
    if isinstance(node, ast.Import):
        found.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
        found.add(node.module)
        found.update(f"{node.module}.{alias.name}" for alias in node.names)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.If) and _is_type_checking(child.test):
            for branch in child.orelse:
                found.update(_imports(branch))
            continue
        found.update(_imports(child))
    return found


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _within(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")
