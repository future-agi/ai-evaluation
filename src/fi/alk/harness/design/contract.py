"""The `contract.json` a design's builder session reads.

The scenario tooling, the world's probes and the writers all read an `AgentContract`, and it has
to pass `validate_contract` (an agent name, use cases, unique tool names). A design has no agent
to read one from, so it is derived: the tools from `tools.json` with their types and enum values,
the use cases and one-liner from the approved proposal, the agent rules as hard constraints, and
the template's rows as the base data. Attached servers' tools stay out of it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..contract import AgentContract, DataStore, ToolSpec
from ..releases import (
    _ARG_TYPES,
    POLICY,
    TOOLS,
    RuleClass,
    ToolDefinition,
    load_policy,
    load_tools,
)
from .workspace import Design

CONTRACT = "contract.json"


def _spec(tool: ToolDefinition) -> ToolSpec:
    properties = {
        argument: schema
        for argument, schema in (tool.parameters.get("properties") or {}).items()
        if isinstance(schema, dict)
    }
    return ToolSpec(
        name=tool.name,
        description=tool.description,
        args=list(properties),
        arg_types={
            argument: _ARG_TYPES.get(str(schema.get("type")), "str")
            for argument, schema in properties.items()
        },
        arg_values={
            argument: schema["enum"]
            for argument, schema in properties.items()
            if isinstance(schema.get("enum"), list)
        },
    )


def design_contract(workspace: Path, design: Design) -> AgentContract:
    tools = load_tools(workspace) if (workspace / TOOLS).is_file() else []
    rules = (
        [
            rule.text
            for rule in load_policy(workspace).rules
            if rule.rule_class is RuleClass.AGENT_POLICY
        ]
        if (workspace / POLICY).is_file()
        else []
    )
    proposal = design.proposal
    return AgentContract(
        agent=design.name,
        one_liner=proposal.one_liner if proposal else "",
        modality=design.modality,
        conversational=True,
        hard_constraints=rules,
        tools=[_spec(tool) for tool in tools],
        base_environment=dict(design.base_environment),
        data_store=DataStore(kind="sqlite"),
        real_use_cases=list(proposal.use_cases) if proposal else [],
        implementation="present",
    )


def write_contract(workspace: Path, design: Design) -> AgentContract:
    contract = design_contract(workspace, design)
    (workspace / CONTRACT).write_text(
        json.dumps(contract.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    return contract
