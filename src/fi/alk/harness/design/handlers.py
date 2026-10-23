"""Checks on a tool handler the model wrote, before it joins a design's world.

Handlers run in the process that holds the model's credentials (`GeneratedWorld.call` execs
them), so a designed handler may import only a small standard-library allow-list and may not use
the builtins or dunder attributes that reach outside the world. Then it is called on a copy of the
world: its sample must succeed with an effect that matches `mutates_state`, a missing required
argument must be refused rather than crash, and every argument must change what it does.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from typing import Any

from ..releases import ToolDefinition
from ..world.runtime import Call, GeneratedWorld

ALLOWED_IMPORTS = frozenset(
    {
        "collections",
        "datetime",
        "decimal",
        "fractions",
        "functools",
        "itertools",
        "json",
        "math",
        "re",
        "statistics",
        "string",
    }
)
_BANNED_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "open",
        "quit",
        "setattr",
        "vars",
    }
)


def source_problems(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    problems: set[str] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.add("a relative import")
            modules = [node.module or ""]
        problems.update(
            f"imports {module}"
            for module in modules
            if module.split(".")[0] not in ALLOWED_IMPORTS
        )
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            problems.add(f"uses {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.add(f"reaches {node.attr}")
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "handle"
        for node in tree.body
    ):
        problems.add("defines no handle(args, db)")
    return sorted(problems)


def _trial(
    world: GeneratedWorld, name: str, source: str, arguments: Mapping[str, Any]
) -> tuple[Call, tuple[Any, ...]]:
    """One call with this source on the world, which is then put back as it was; the call, and
    everything it did, comparably."""
    saved, calls, previous = (
        world.checkpoint(),
        list(world.calls),
        world.handlers.get(name),
    )
    world.handlers[name] = source
    try:
        call = world.call(name, dict(arguments))
        after = world.state()
    finally:
        world.revert(saved)
        world.calls = calls
        if previous is None:
            world.handlers.pop(name, None)
        else:
            world.handlers[name] = previous
    done = (
        call.ok,
        call.refused,
        call.error,
        json.dumps(call.result, sort_keys=True, default=str),
        json.dumps(after, sort_keys=True, default=str),
    )
    return call, done


def _other(value: Any, schema: Mapping[str, Any]) -> Any:
    choices = [one for one in schema.get("enum") or [] if one != value]
    if choices:
        return choices[0]
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return f"{value}-other"
    return None


def behaviour_problems(
    world: GeneratedWorld,
    definition: ToolDefinition,
    source: str,
    sample: Mapping[str, Any],
) -> list[str]:
    """What calling the handler on a copy of the world shows is wrong with it."""
    untouched = json.dumps(world.state(), sort_keys=True, default=str)
    call, done = _trial(world, definition.name, source, sample)
    if call.refused:
        return [f"the sample call was refused: {call.error}. Give arguments it accepts"]
    if not call.ok:
        return [f"the sample call crashed: {call.error}"]
    problems: list[str] = []
    changed = done[-1] != untouched
    if changed != definition.mutates_state:
        problems.append(
            f"mutates_state is {str(definition.mutates_state).lower()} but the sample call "
            + ("changed the world" if changed else "changed nothing")
        )
    properties = definition.parameters.get("properties") or {}
    for argument in definition.parameters.get("required") or []:
        without = {key: value for key, value in sample.items() if key != argument}
        missing, _ = _trial(world, definition.name, source, without)
        if missing.ok:
            problems.append(
                f"accepts a call without {argument}, which it requires: refuse it, or make "
                "it optional"
            )
        elif not missing.refused:
            problems.append(
                f"crashes without {argument} ({missing.error}): raise ToolError instead"
            )
    for argument, value in sample.items():
        other = _other(value, properties.get(argument) or {})
        if other is None:
            continue
        _, changed_once = _trial(
            world, definition.name, source, {**sample, argument: other}
        )
        if changed_once == done:
            problems.append(
                f"ignores {argument}: {other!r} instead of {value!r} changes nothing it "
                "returns or writes"
            )
    return problems
