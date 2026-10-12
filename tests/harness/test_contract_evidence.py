import asyncio
import hashlib
import json

import jsonschema
import pytest

from fi.alk.harness.contract import (
    AgentContract,
    ContractItem,
    DataStore,
    StoreConfiguration,
    is_data_free_conversation,
)
from fi.alk.harness.provider_import import safe_provider_profile
from fi.alk.harness.tools import accept_contract, contract_tools


def submit(tmp_path, payload, **context):
    server = contract_tools(tmp_path / "out", **context)
    tool = next(tool for tool in server.tools if tool.name == "submit_contract")
    jsonschema.validate(payload, tool.input_schema)
    result = asyncio.run(tool.handler(payload))
    if result.get("is_error") and "submit again as is" in result["content"][0]["text"]:
        result = asyncio.run(tool.handler(payload))
    return result


def body(**changes):
    return {
        "agent": "support",
        "domain": {"label": "Customer Support"},
        "system_prompt_excerpt": "Support customers.",
        "tools": [],
        "hard_constraints": [{"text": "Be truthful", "origin": {"kind": "prompt"}}],
        "real_use_cases": [{"text": "Answer questions", "origin": {"kind": "prompt"}}],
    } | changes


def saved(tmp_path):
    return json.loads((tmp_path / "out/contract.json").read_text())


def test_source_evidence_survives_real_submit_with_legacy_fields(tmp_path):
    prompt = "You are Claire, a support agent.\nBe truthful.\n"
    (tmp_path / "prompt.txt").write_text(prompt)
    code = 'def look_up(db):\n    return db.execute("SELECT id FROM customers").fetchall()\n'
    (tmp_path / "agent.py").write_text(code)
    payload = body(
        system_prompt=prompt,
        system_prompt_origin={"kind": "file", "path": "prompt.txt"},
        hard_constraints=[
            {
                "text": "Be truthful",
                "short": "Truthful",
                "origin": {"kind": "file", "path": "prompt.txt"},
                "enforced_by": ["prompt"],
            }
        ],
        tools=[
            {
                "name": "look_up",
                "origin": {"kind": "file", "path": "agent.py"},
                "effect": "Reads customers",
                "effect_kind": "read",
                "effect_evidence": {
                    "origin": {"kind": "file", "path": "agent.py", "line": 2},
                    "excerpt": 'db.execute("SELECT id FROM customers")',
                },
            }
        ],
        tool_entrypoints=[
            {
                "tool": "look_up",
                "mode": "import",
                "module": "agent",
                "callable": "look_up",
            },
            {
                "tool": "archive",
                "mode": "unreachable",
                "notes": "No source implementation was supplied.",
            },
        ],
        actors=[
            {
                "name": "Claire",
                "role": "support agent",
                "origin": {"kind": "file", "path": "prompt.txt"},
            }
        ],
        data_store={
            "kind": "sqlite",
            "location": "external_db",
            "configured_by": {"source": "environment", "ref": "DATABASE_URL"},
        },
        runtime_dependencies=[
            {"name": "model", "kind": "inference", "used_by": ["support"]}
        ],
    )
    result = submit(tmp_path, payload, source_root=tmp_path)
    assert not result.get("is_error"), result
    contract = saved(tmp_path)
    assert contract["system_prompt"] == prompt
    assert (
        contract["system_prompt_origin"]["sha256"]
        == hashlib.sha256(prompt.encode()).hexdigest()
    )
    assert contract["hard_constraints"] == ["Be truthful"]
    assert contract["hard_constraint_details"][0]["short"] == "Truthful"
    assert contract["tools"][0]["effect"] == "Reads customers"
    assert contract["tools"][0]["effect_kind"] == "read"
    assert contract["tool_entrypoints"][0]["source"]["code"] == code.rstrip("\n")
    assert contract["tool_entrypoints"][0]["implementation_origin"] == "source"
    assert contract["data_store"]["configured_by"] == "DATABASE_URL"
    assert contract["data_store"]["configuration"] == {
        "source": "environment",
        "ref": "DATABASE_URL",
        "section": None,
    }
    assert contract["runtime_dependencies"][0]["used_by"] == ["support"]
    assert contract["amendment_details"] == [
        {
            "subject": "archive",
            "kind": "unreachable_tool",
            "reason": "No source implementation was supplied.",
        }
    ]
    assert contract["actors"][0]["name"] == "Claire"


@pytest.mark.parametrize("provider", ["vapi", "retell"])
def test_provider_prompt_and_function_are_grounded_and_scrubbed(tmp_path, provider):
    prompt = "You are Claire.\n Help the caller.\n"
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": ["a", "b"]},
            "api_key": {"type": "string", "description": "Credential reference"},
        },
        "required": ["id"],
        "additionalProperties": False,
    }
    if provider == "vapi":
        profile = {
            "provider": provider,
            "model": {
                "messages": [{"role": "system", "content": prompt}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "lookup", "parameters": parameters},
                        "server": {
                            "url": "https://user:secret@example.com/tool?token=secret"
                        },
                    },
                    {"type": "endCall"},
                ],
            },
        }
        prompt_field, builtin = "model.messages.0.content", "endCall"
    else:
        profile = {
            "provider": provider,
            "general_prompt": prompt,
            "general_tools": [
                {
                    "type": "custom",
                    "name": "lookup",
                    "url": "https://user:secret@example.com/tool?token=secret",
                    "parameters": parameters,
                },
                {"type": "end_call", "name": "finish"},
            ],
        }
        prompt_field, builtin = "general_prompt", "finish"
    payload = body(
        tools=[
            {"name": "lookup", "origin": {"kind": "tool_list"}},
            {"name": builtin, "origin": {"kind": "platform_builtin"}},
        ],
        system_prompt="Model summary",
    )
    result = submit(tmp_path, payload, provider_profile=profile)
    assert not result.get("is_error"), result
    contract = saved(tmp_path)
    assert contract["system_prompt"] == prompt
    assert contract["system_prompt_origin"]["field"] == prompt_field
    assert (
        contract["system_prompt_origin"]["sha256"]
        == hashlib.sha256(prompt.encode()).hexdigest()
    )
    assert contract["tools"][0]["effect_kind"] == "external"
    assert contract["tools"][1]["effect_kind"] == "control"
    assert contract["tool_entrypoints"] == []
    assert contract["tool_entrypoint_details"][0]["provider_function"] == {
        "url": "https://example.com/tool",
        "parameters": parameters,
    }
    assert (
        contract["tool_entrypoint_details"][1]["implementation_origin"]
        == "provider_builtin"
    )
    assert "secret" not in json.dumps(contract)
    assert contract["call_direction"] == "inbound"


def test_retell_flow_tools_are_grounded_through_named_conversation_nodes(tmp_path):
    profile = {
        "provider": "retell",
        "conversation_flow": {
            "global_prompt": "Help the caller.",
            "nodes": [
                {
                    "type": "conversation",
                    "name": "Opening",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "lookup",
                            "url": "https://example.com/tool",
                        }
                    ],
                }
            ],
        },
    }
    result = submit(
        tmp_path, body(tools=[{"name": "lookup"}]), provider_profile=profile
    )
    assert not result.get("is_error"), result
    contract = saved(tmp_path)
    assert (
        contract["system_prompt_origin"]["field"] == "conversation_flow.global_prompt"
    )
    assert (
        contract["tools"][0]["origin"]["field"] == "conversation_flow.nodes.0.tools.0"
    )
    assert contract["tools"][0]["effect_kind"] == "external"


@pytest.mark.parametrize("rich_amendment", [False, True])
def test_provider_inventory_overrides_unreachable_display_claim_without_changing_runtime(
    tmp_path,
    rich_amendment,
):
    payload = body(
        tools=[{"name": "finish"}],
        tool_entrypoints=[
            {"tool": "finish", "mode": "unreachable", "notes": "No local code"}
        ],
        amendments=["finish: No local code", "Keep this unrelated observation"],
        amendment_details=[
            {"subject": "finish", "kind": "unreachable_tool", "reason": "No local code"}
        ]
        if rich_amendment
        else [],
    )
    result = submit(
        tmp_path,
        payload,
        provider_profile={
            "provider": "retell",
            "general_tools": [{"type": "end_call", "name": "finish"}],
        },
    )
    assert not result.get("is_error"), result
    contract = saved(tmp_path)
    assert contract["tool_entrypoints"][0]["mode"] == "unreachable"
    assert (
        contract["tool_entrypoint_details"][0]["implementation_origin"]
        == "provider_builtin"
    )
    assert contract["amendments"] == ["Keep this unrelated observation"]
    assert contract["amendment_details"] == []


@pytest.mark.parametrize(
    "profile",
    [
        {
            "provider": "vapi",
            "model": {
                "messages": [
                    {"role": "system", "content": "First"},
                    {"role": "system", "content": "Second"},
                ]
            },
        },
        {"provider": "retell", "general_prompt": "[redacted]"},
        safe_provider_profile(
            {
                "provider": "retell",
                "general_prompt": "https://user:secret@example.com/prompt?token=secret",
            }
        ),
    ],
)
def test_unavailable_provider_prompt_does_not_promote_excerpt(tmp_path, profile):
    assert not submit(
        tmp_path, body(system_prompt="summary"), provider_profile=profile
    ).get("is_error")
    assert saved(tmp_path)["system_prompt"] == ""
    assert saved(tmp_path)["system_prompt_origin"] is None


def test_source_prompt_literal_is_verified_without_importing(tmp_path):
    prompt = "First\nSecond\n"
    (tmp_path / "agent.py").write_text(
        f"raise RuntimeError('do not execute')\nPROMPT = {prompt!r}\n"
    )
    result = submit(
        tmp_path,
        body(
            system_prompt=prompt,
            system_prompt_origin={"kind": "code", "path": "agent.py", "line": 2},
        ),
        source_root=tmp_path,
    )
    assert not result.get("is_error"), result
    assert saved(tmp_path)["system_prompt"] == prompt


def test_dynamic_prompt_fragment_is_not_published_as_full_prompt(tmp_path):
    (tmp_path / "agent.py").write_text('PROMPT = "Be helpful " + customer_context\n')
    result = accept_contract(
        body(
            system_prompt="Be helpful ",
            system_prompt_origin={"kind": "file", "path": "agent.py"},
        ),
        tmp_path / "out",
        source_root=tmp_path,
    )
    assert result["is_error"]
    assert "exact source string literal" in result["content"][0]["text"]


def test_conflicting_schema_companion_is_rejected():
    with pytest.raises(ValueError, match="columns conflict"):
        AgentContract.model_validate(
            body(
                data_schema={"customers": {"id": "INTEGER"}},
                data_schema_columns={"customers": [{"name": "id", "type": "TEXT"}]},
            )
        )


@pytest.mark.parametrize("path", ["../outside.txt", "escape.txt"])
def test_prompt_cannot_escape_source(tmp_path, path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (source / "escape.txt").symlink_to(outside)
    result = accept_contract(
        body(
            system_prompt="private", system_prompt_origin={"kind": "file", "path": path}
        ),
        tmp_path / "out",
        source_root=source,
    )
    assert result["is_error"]
    assert "inside the submitted source" in result["content"][0]["text"]
    assert not (tmp_path / "out/contract.json").exists()


def test_invented_effect_evidence_is_rejected(tmp_path):
    (tmp_path / "agent.py").write_text("def get_customer(): return None\n")
    payload = body(
        tools=[
            {
                "name": "get_customer",
                "effect_kind": "read",
                "effect_evidence": {
                    "origin": {"kind": "file", "path": "agent.py"},
                    "excerpt": "SELECT * FROM customers",
                },
            }
        ]
    )
    result = accept_contract(payload, tmp_path / "out", source_root=tmp_path)
    assert result["is_error"]
    assert "does not match its source" in result["content"][0]["text"]


def test_legacy_effect_and_missing_prompt_evidence_stay_unclassified(tmp_path):
    result = accept_contract(
        body(
            tools=[{"name": "get_customer", "effect": "look up a customer"}],
            system_prompt_excerpt="An excerpt",
        ),
        tmp_path / "out",
    )
    assert not result.get("is_error")
    assert saved(tmp_path)["tools"][0]["effect_kind"] is None
    assert saved(tmp_path)["system_prompt"] == ""


def test_rich_companions_cannot_disagree_with_legacy_text():
    with pytest.raises(ValueError, match="conflict"):
        AgentContract.model_validate(
            body(hard_constraints=["Do A"], hard_constraint_details=[{"text": "Do B"}])
        )
    contract = AgentContract.model_validate(
        body(hard_constraints=[], hard_constraint_details=[{"text": "Do A"}])
    )
    assert contract.hard_constraints == ["Do A"]


def test_display_store_details_do_not_change_runtime_data_free_check():
    contract = AgentContract(
        agent="chat",
        real_use_cases=["Chat"],
        data_store={"kind": "none", "location": "none"},
    )
    assert is_data_free_conversation(contract)


def test_typed_companions_derive_legacy_values():
    contract = AgentContract(hard_constraint_details=[ContractItem(text="Be truthful")])
    assert contract.hard_constraints == ["Be truthful"]
    store = DataStore(
        configuration=StoreConfiguration(source="environment", ref="DATABASE_URL")
    )
    assert store.configured_by == "DATABASE_URL"
    with pytest.raises(ValueError, match="conflicting datastore configuration"):
        DataStore(
            configured_by="OTHER_DATABASE_URL",
            configuration={"source": "environment", "ref": "DATABASE_URL"},
        )


def test_empty_native_column_type_stays_empty():
    contract = AgentContract(
        data_schema={"items": {"columns": [{"name": "value", "type": ""}]}}
    )
    assert contract.data_schema == {"items": {"value": ""}}
    assert contract.data_schema_columns["items"][0].type == ""


def test_provider_schema_redacts_secret_defaults_but_keeps_the_declaration():
    profile = safe_provider_profile(
        {
            "general_tools": [
                {
                    "parameters": {
                        "type": "object",
                        "$ref": "https://user:private-value@example.com/schema?signed=private-value#private-value",
                        "properties": {
                            "api_key": {"type": "string", "default": "private-value"},
                        },
                    }
                }
            ],
            "api_key": "private-value",
        }
    )
    assert profile["api_key"] == "[redacted]"
    assert profile["general_tools"][0]["parameters"]["properties"]["api_key"] == {
        "type": "string",
        "default": "[redacted]",
    }
    assert (
        profile["general_tools"][0]["parameters"]["$ref"]
        == "https://example.com/schema"
    )


def test_embedded_provider_prompt_credentials_are_not_published_or_hashed(tmp_path):
    prompt = "Read https://user:password@example.com/account?signature=private-value before helping."
    profile = safe_provider_profile({"provider": "retell", "general_prompt": prompt})
    assert (
        profile["general_prompt"] == "Read https://example.com/account before helping."
    )
    assert profile["unavailable_verbatim_fields"] == ["general_prompt"]
    result = submit(
        tmp_path, body(system_prompt_excerpt=prompt), provider_profile=profile
    )
    assert not result.get("is_error"), result
    contract = saved(tmp_path)
    assert contract["system_prompt"] == ""
    assert contract["system_prompt_origin"] is None
    assert "password" not in json.dumps(contract)
    assert "private-value" not in json.dumps(contract)


def test_unverified_rule_provenance_and_enforcement_are_omitted(tmp_path):
    result = accept_contract(
        body(
            hard_constraints=[
                {
                    "text": "Never charge twice",
                    "origin": {"kind": "file"},
                    "enforced_by": ["code", "prompt", "judge"],
                    "sub_goal": "nonexistent",
                }
            ]
        ),
        tmp_path / "out",
    )
    assert not result.get("is_error"), result
    item = saved(tmp_path)["hard_constraint_details"][0]
    assert item["origin"] is None
    assert item["enforced_by"] == []
    assert item["sub_goal"] == ""
    assert saved(tmp_path)["hard_constraints"] == ["Never charge twice"]


def test_rule_provenance_cannot_point_to_an_unrelated_file(tmp_path):
    (tmp_path / "prompt.txt").write_text("Help the caller")
    result = accept_contract(
        body(
            hard_constraints=[
                {
                    "text": "Never charge twice",
                    "origin": {"kind": "file", "path": "prompt.txt"},
                    "enforced_by": ["code", "prompt"],
                }
            ]
        ),
        tmp_path / "out",
        source_root=tmp_path,
    )
    assert not result.get("is_error"), result
    item = saved(tmp_path)["hard_constraint_details"][0]
    assert item["origin"] is None
    assert item["enforced_by"] == []


def test_code_and_judge_enforcement_need_actual_guards_and_catalogue(tmp_path):
    from fi.alk.harness.catalogue import Catalogue, SubGoal, save_catalogue

    (tmp_path / "agent.py").write_text(
        'def charge():\n    raise ValueError("Never charge twice")\n'
    )
    save_catalogue(
        Catalogue(
            sub_goals=[
                SubGoal(
                    name="truthful",
                    what="Be truthful",
                    judged="Was the agent truthful?",
                )
            ]
        ),
        tmp_path / "out",
    )
    result = accept_contract(
        body(
            hard_constraints=[
                {
                    "text": "Never charge twice",
                    "origin": {"kind": "code", "path": "agent.py", "line": 2},
                    "enforced_by": ["code"],
                },
                {
                    "text": "Be truthful",
                    "enforced_by": ["judge"],
                    "sub_goal": "truthful",
                },
                {
                    "text": "Always refund",
                    "enforced_by": ["judge"],
                    "sub_goal": "truthful",
                },
            ]
        ),
        tmp_path / "out",
        source_root=tmp_path,
    )
    assert not result.get("is_error"), result
    items = saved(tmp_path)["hard_constraint_details"]
    assert items[0]["enforced_by"] == ["code"]
    assert items[1]["enforced_by"] == ["judge"]
    assert items[1]["sub_goal"] == "truthful"
    assert items[2]["enforced_by"] == []
    assert items[2]["sub_goal"] == ""


def test_contradictory_amendments_are_rejected_before_saving(tmp_path):
    result = accept_contract(
        body(
            amendments=["lookup: It is reachable"],
            amendment_details=[
                {
                    "subject": "lookup",
                    "kind": "unreachable_tool",
                    "reason": "It is unreachable",
                }
            ],
        ),
        tmp_path / "out",
    )
    assert result["is_error"]
    assert "amendment details conflict" in result["content"][0]["text"]
    assert not (tmp_path / "out/contract.json").exists()
