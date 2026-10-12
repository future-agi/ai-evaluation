"""Reject demonstrably non-executable Python tool examples without importing source.

This is a negative-evidence gate, not a general static registration detector. Dynamic tools,
external packages and non-Python implementations still require runtime proof.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

from .contract import AgentContract, Origin, ToolSource


def _module_path(root: Path, module_name: str) -> Path | None:
    parts = module_name.split(".")
    if not parts or not all(part.isidentifier() for part in parts):
        return None
    module = Path(*parts)
    candidates = [
        base / relative
        for base in (root, root / "src")
        for relative in (module.with_suffix(".py"), module / "__init__.py")
    ]
    return next(
        (
            path
            for path in candidates
            if path.is_file() and path.resolve().is_relative_to(root)
        ),
        None,
    )


def _callable_node(tree: ast.AST, dotted_name: str) -> ast.AST | None:
    parts = dotted_name.split(".")
    nodes = list(getattr(tree, "body", []))
    found: ast.AST | None = None
    for part in parts:
        found = next(
            (
                node
                for node in nodes
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                and node.name == part
            ),
            None,
        )
        if found is None:
            return None
        nodes = list(getattr(found, "body", []))
    return found


def attach_tool_sources(contract: AgentContract, source: Path) -> AgentContract:
    """Attach exact Python source to resolvable submitted tool entrypoints."""

    root = source.resolve()
    tools = {tool.name: tool for tool in contract.tools}
    for entry in contract.tool_entrypoints:
        if (
            entry.mode not in {"import", "construct"}
            or not entry.module
            or not entry.callable
        ):
            continue
        path = _module_path(root, entry.module)
        if path is None:
            continue
        try:
            text = path.read_text(encoding="utf-8")
            node = _callable_node(ast.parse(text), entry.callable)
        except (UnicodeError, SyntaxError):
            continue
        if node is None or not getattr(node, "end_lineno", None):
            continue
        decorators = getattr(node, "decorator_list", [])
        start = min(
            [node.lineno, *(decorator.lineno for decorator in decorators)],
        )
        end = int(node.end_lineno)
        relative = path.resolve().relative_to(root).as_posix()
        origin = Origin(kind="code", path=relative, line=start)
        entry.origin = origin
        entry.implementation_origin = "source"
        entry.source = ToolSource(
            path=relative,
            line_start=start,
            line_end=end,
            code="\n".join(text.splitlines()[start - 1 : end]),
        )
        tool = tools.get(entry.tool)
        if tool is not None:
            tool.origin = origin
    return contract


def tool_evidence_problems(contract: AgentContract, source: Path) -> list[str]:
    root = source.resolve()
    problems = []
    for entry in contract.tool_entrypoints:
        if entry.mode not in {"import", "construct"} or not entry.module:
            continue
        path = _module_path(root, entry.module)
        if path is None:
            continue
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
            comments = [
                token.string
                for token in tokenize.generate_tokens(io.StringIO(text).readline)
                if token.type == tokenize.COMMENT
            ]
        except (UnicodeError, SyntaxError, tokenize.TokenError):
            continue  # The actual source interpreter/package validator owns these failures.
        name = entry.callable.rsplit(".", 1)[-1]
        if not name.isidentifier():
            continue
        # Assignments/imports may legitimately export a callable without a function definition.
        bindings = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        bindings.update(
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
        bindings.update(
            node.asname or node.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.alias)
        )
        commented = any(
            re.search(r"\b(?:async\s+)?def\s+" + re.escape(name) + r"\s*\(", line)
            for line in comments
        )
        if commented and name not in bindings:
            problems.append(
                f"tool[{entry.tool}]:commented-only-entrypoint:{entry.module}.{entry.callable} — "
                "the named callable exists only in comments, not executable Python. Remove "
                "the example tool or supply its actual runtime binding. An agent with no "
                "custom tools must use tools=[]; do not invent a tool or backing data."
            )
    return problems
