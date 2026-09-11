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

from .contract import AgentContract


def tool_evidence_problems(contract: AgentContract, source: Path) -> list[str]:
    root = source.resolve()
    problems = []
    for entry in contract.tool_entrypoints:
        if entry.mode not in {"import", "construct"} or not entry.module:
            continue
        parts = entry.module.split(".")
        if not all(part.isidentifier() for part in parts):
            continue
        module = Path(*parts)
        candidates = [
            base / relative
            for base in (root, root / "src")
            for relative in (module.with_suffix(".py"), module / "__init__.py")
        ]
        path = next(
            (p for p in candidates if p.is_file() and p.resolve().is_relative_to(root)),
            None,
        )
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
