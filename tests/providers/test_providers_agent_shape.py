from __future__ import annotations

import ast
import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from fi.alk.providers.agent import Agent
from fi.alk.providers.setup_hash import compute_setup_hash

AUTHORED = Path(__file__).parent / "fixtures" / "authored"


def _valid_agent() -> dict:
    return json.loads((AUTHORED / "voice_source.json").read_text())["agent"]


def _hosted(agent: dict) -> None:
    agent["kind"] = "hosted"
    agent["provider_ref"] = None


def _telephony_unsorted(agent: dict) -> None:
    agent["telephony"] = [
        {"number": "+15550100", "direction_supported": ["outbound", "inbound"]}
    ]


INVALID: dict[str, Callable[[dict], None]] = {
    "unknown_top_level_key": lambda agent: agent.update(owner="me"),
    "unknown_nested_key": lambda agent: agent["behaviour"][0].update(owner="me"),
    "hosted_without_provider_ref": _hosted,
    "source_with_provider_ref": lambda agent: agent.update(
        provider_ref={"provider": "vapi", "id": "a1"}
    ),
    "empty_behaviour": lambda agent: agent.update(behaviour=[]),
    "repeated_unit_id": lambda agent: agent["behaviour"].append(
        copy.deepcopy(agent["behaviour"][0])
    ),
    "unit_id_pattern": lambda agent: agent["behaviour"][0].update(id="Main Unit"),
    "unit_tool_not_in_tools": lambda agent: agent["behaviour"][0]["tools"].append(
        "missing_tool"
    ),
    "supports_unsorted": lambda agent: agent.update(supports=["outbound", "inbound"]),
    "supports_repeated": lambda agent: agent.update(supports=["inbound", "inbound"]),
    "telephony_direction_unsorted": _telephony_unsorted,
    "unknown_type": lambda agent: agent.update(type="email"),
    "unknown_instrumentation": lambda agent: agent.update(instrumentation="langfuse"),
}


def test_the_base_document_is_valid() -> None:
    Agent.model_validate(_valid_agent())


@pytest.mark.parametrize("mutate", INVALID.values(), ids=INVALID.keys())
def test_rejects_each_invalid_document(mutate: Callable[[dict], None]) -> None:
    agent = _valid_agent()
    mutate(agent)
    agent["setup_hash"] = compute_setup_hash(agent)

    with pytest.raises(ValidationError):
        Agent.model_validate(agent)


def test_rejects_a_setup_hash_that_does_not_match() -> None:
    agent = _valid_agent()
    agent["name"] = "renamed after hashing"

    with pytest.raises(ValidationError, match="setup_hash_mismatch"):
        Agent.model_validate(agent)


FORBIDDEN_IMPORTS = ("django", "fi.simulate", "fi.alk.harness", "fi.alk.episode")


def _imported_names(source: Path) -> set[str]:
    package = ["fi", "alk", "providers"]
    names: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module.split(".") if node.module else []
            if node.level:
                base = package[: len(package) - node.level + 1] + base
            names.add(".".join(base))
            names.update(".".join([*base, alias.name]) for alias in node.names)
    return names


def test_no_import_of_simulate_harness_or_episode() -> None:
    providers = Path(__file__).parents[2] / "src" / "fi" / "alk" / "providers"
    sources = sorted(providers.glob("*.py"))

    forbidden = {
        f"{source.name}: {name}"
        for source in sources
        for name in _imported_names(source)
        for module in FORBIDDEN_IMPORTS
        if name == module or name.startswith(module + ".")
    }

    assert {source.name for source in sources} >= {"agent.py", "setup_hash.py"}
    assert not forbidden
