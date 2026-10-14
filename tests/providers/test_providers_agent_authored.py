from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fi.alk.harness.contract import AgentContract
from fi.alk.providers.agent import AgentBuildError, build_authored_agent
from fi.alk.providers.setup_hash import compute_setup_hash

AUTHORED = Path(__file__).parent / "fixtures" / "authored"


def _contract(**fields: object) -> dict:
    return AgentContract.model_validate({"agent": "test-agent", **fields}).model_dump(
        mode="json"
    )


def _only_tool(
    *, spec: dict | None = None, entrypoints: list[dict] | None = None
) -> dict:
    contract = _contract(
        tools=[{"name": "lookup", **(spec or {})}], tool_entrypoints=entrypoints or []
    )
    return build_authored_agent(contract)["tools"][0]


@pytest.mark.parametrize(
    "case",
    [
        "chat_source",
        "voice_source",
        "no_prompt",
        "no_tools",
        "service_tool",
        "traceai_instrumented",
        "opentelemetry_instrumented",
    ],
)
def test_recorded_contracts_give_expected_agent(case: str) -> None:
    recorded = json.loads((AUTHORED / f"{case}.json").read_text())

    agent = build_authored_agent(recorded["contract"])

    assert agent == recorded["agent"]
    assert agent["kind"] == "source"
    assert agent["provider_ref"] is None
    assert [unit["shape"] for unit in agent["behaviour"]] == ["external"]


@pytest.mark.parametrize(
    ("entrypoints", "expected_kind"),
    [
        ([{"tool": "lookup", "mode": "service"}], "http"),
        ([{"tool": "lookup", "mode": "import"}], "builtin"),
        ([{"tool": "lookup", "mode": "construct"}], "builtin"),
        ([{"tool": "lookup", "mode": "unreachable"}], "unknown"),
        ([{"tool": "lookup", "mode": "grpc"}], "unknown"),
        ([{"tool": "other", "mode": "service"}], "unknown"),
        ([], "unknown"),
        (
            [
                {"tool": "lookup", "mode": "service"},
                {"tool": "lookup", "mode": "import"},
            ],
            "http",
        ),
    ],
    ids=[
        "service",
        "import",
        "construct",
        "unreachable",
        "unknown_word",
        "other_tool",
        "no_entry",
        "first_entry_wins",
    ],
)
def test_authored_tool_kind(entrypoints: list[dict], expected_kind: str) -> None:
    tool = _only_tool(
        entrypoints=[{**entry, "endpoint": "/lookup"} for entry in entrypoints]
    )

    assert tool["kind"] == expected_kind
    assert tool["endpoint"] is None


def test_authored_tool_parameters() -> None:
    assert _only_tool(spec={"args": []})["parameters"] is None
    assert _only_tool(
        spec={
            "args": ["order_id", "region"],
            "arg_values": {"region": ["eu", "us"], "order_id": "A-1"},
        }
    )["parameters"] == {
        "type": "object",
        "properties": {"order_id": {}, "region": {"enum": ["eu", "us"]}},
        "required": ["order_id", "region"],
    }


@pytest.mark.parametrize("modality", ["sms", "", ["voice", "chat"]])
def test_authored_rejects_modality(modality: object) -> None:
    with pytest.raises(AgentBuildError) as raised:
        build_authored_agent(_contract(modality=modality))

    assert str(raised.value) == "agent_invalid"
    assert raised.value.detail == "modality"


def test_authored_is_pure_and_hash_is_stable_for_one_document() -> None:
    contract = json.loads((AUTHORED / "voice_source.json").read_text())["contract"]
    before = copy.deepcopy(contract)

    first = build_authored_agent(contract)
    second = build_authored_agent(contract)

    assert contract == before
    assert first == second
    assert first["setup_hash"] == compute_setup_hash(first)


def test_authored_reads_the_redacted_document() -> None:
    agent = build_authored_agent(
        _contract(
            tools=[
                {
                    "name": "login",
                    "args": ["user", "password"],
                    "arg_values": {"password": ["hunter2", "s3cret"]},
                }
            ],
            dependencies=[{"name": "db", "reached": {"dsn_env": "ORDERS_DSN"}}],
            data_schema={"accounts": {"api_key": "sk-live-1", "columns": ["id"]}},
            base_environment={"accounts": [{"id": 1}]},
        )
    )
    stored = agent["extension"]["contract"]

    assert agent["tools"][0]["parameters"]["properties"]["password"] == {}
    assert stored["tools"][0]["arg_values"]["password"] == "[redacted]"
    assert stored["dependencies"][0]["reached"]["dsn_env"] == "ORDERS_DSN"
    assert stored["data_schema"] == {
        "accounts": {"api_key": "[redacted]", "columns": ["id"]}
    }
    assert "base_environment" not in stored
    assert "hunter2" not in json.dumps(agent)
