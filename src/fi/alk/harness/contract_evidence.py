"""Ground display evidence in the submitted source and captured provider profile."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contract import (
    AgentContract,
    Amendment,
    EffectEvidence,
    Origin,
    ProviderFunction,
    ToolEntry,
)
from .provider_import import redact_profile_urls, safe_provider_profile


def _file_text(root: Path, origin: Origin) -> str:
    path = (root / origin.path).resolve()
    if not origin.path or not path.is_relative_to(root.resolve()):
        raise ValueError("evidence path must stay inside the submitted source")
    return path.read_bytes().decode("utf-8")


def _field(profile: Mapping[str, Any], field: str) -> Any:
    value: Any = profile
    for part in field.split("."):
        if isinstance(value, Mapping):
            value = value.get(part)
        elif isinstance(value, list) and part.isdecimal() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def _evidence_text(
    origin: Origin, root: Path | None, profile: Mapping[str, Any]
) -> str:
    if (
        origin.kind
        in {"provider_field", "system_prompt", "tool_list", "platform_builtin"}
        and not origin.field
    ):
        raise ValueError("provider evidence requires an exact field")
    if origin.kind in {"file", "code", "doc", "config"} and not origin.path:
        raise ValueError("file evidence requires an exact path")
    if origin.path and root is not None:
        text = _file_text(root, origin)
        if origin.line is not None and origin.line > len(text.splitlines()):
            raise ValueError("evidence line is outside the source file")
        return text
    if origin.field:
        value = _field(profile, origin.field)
        if value is not None:
            return (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )
    raise ValueError("evidence needs an available source file or provider field")


def _source_prompt(contract: AgentContract, root: Path | None) -> None:
    origin = contract.system_prompt_origin
    if root is None or origin is None or not origin.path:
        contract.system_prompt = ""
        contract.system_prompt_origin = None
        return
    if origin.kind not in {"file", "code", "prompt", "doc", "config"}:
        raise ValueError("source prompt requires a file origin")
    text = _file_text(root, origin)
    prompt = contract.system_prompt
    if Path(origin.path).suffix == ".py":
        tree = ast.parse(text)
        literals = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) or (
                isinstance(node, ast.keyword)
                and node.arg
                in {"instructions", "system_prompt", "system_message", "prompt"}
            ):
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    if origin.line is None or value.lineno == origin.line:
                        literals.append(value)
        if not prompt or not any(node.value == prompt for node in literals):
            raise ValueError("system_prompt is not an exact source string literal")
    elif prompt and prompt != text:
        raise ValueError("system_prompt does not match the complete prompt file")
    else:
        prompt = text
    contract.system_prompt = prompt
    if redact_profile_urls(prompt) != prompt:
        contract.system_prompt = ""
        contract.system_prompt_origin = None
        return
    contract.system_prompt_origin = Origin(
        kind="file",
        path=origin.path,
        line=origin.line,
        field=origin.field,
        sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )


def _provider_tools(profile: Mapping[str, Any]):
    def walk(value: Any, path: str, *, tool_inventory: bool = False):
        if isinstance(value, list):
            for index, item in enumerate(value):
                yield from walk(item, f"{path}.{index}", tool_inventory=tool_inventory)
        elif isinstance(value, Mapping):
            if tool_inventory and isinstance(value.get("type"), str):
                yield path, value
                return
            for key, item in value.items():
                if isinstance(item, (Mapping, list)):
                    yield from walk(
                        item,
                        f"{path}.{key}" if path else str(key),
                        tool_inventory=key in {"tools", "general_tools", "tool"},
                    )

    if profile.get("provider") == "vapi":
        model = profile.get("model") or {}
        if isinstance(model, Mapping):
            yield from walk(model.get("tools"), "model.tools", tool_inventory=True)
        yield from walk(
            profile.get("reusable_tools"), "reusable_tools", tool_inventory=True
        )
    elif profile.get("provider") == "retell":
        for key in ("general_tools", "states", "conversation_flow"):
            yield from walk(
                profile.get(key), key, tool_inventory=key == "general_tools"
            )


_CONTROL_TOOLS = {
    "endCall",
    "end_call",
    "transferCall",
    "transfer_call",
    "handoff",
    "pressDigit",
    "press_digit",
}


def _provider_facts(contract: AgentContract, profile: Mapping[str, Any]) -> None:
    prompts: list[tuple[str, str]] = []
    if profile.get("provider") == "retell":
        value = profile.get("general_prompt")
        if isinstance(value, str):
            prompts.append(("general_prompt", value))
        elif isinstance(profile.get("conversation_flow"), Mapping):
            value = profile["conversation_flow"].get("global_prompt")
            if isinstance(value, str):
                prompts.append(("conversation_flow.global_prompt", value))
    elif profile.get("provider") == "vapi":
        model = profile.get("model")
        messages = model.get("messages", []) if isinstance(model, Mapping) else []
        for index, message in enumerate(messages):
            if isinstance(message, Mapping) and message.get("role") == "system":
                if isinstance(message.get("content"), str):
                    prompts.append(
                        (f"model.messages.{index}.content", message["content"])
                    )
                else:
                    prompts.append((f"model.messages.{index}.content", ""))
    contract.system_prompt = ""
    contract.system_prompt_origin = None
    if len(prompts) == 1:
        field, prompt = prompts[0]
        if (
            prompt
            and "[redacted]" not in prompt
            and field not in profile.get("unavailable_verbatim_fields", [])
        ):
            contract.system_prompt = prompt
            contract.system_prompt_origin = Origin(
                kind="provider_field",
                field=field,
                sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            )
    entries = {entry.tool: entry for entry in contract.tool_entrypoints}
    inventory: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for field, definition in _provider_tools(profile):
        function = definition.get("function")
        name = (
            function.get("name")
            if isinstance(function, Mapping)
            else definition.get("name")
        )
        name = name or definition.get("type")
        inventory.setdefault(str(name), []).append((field, definition))
    for tool in contract.tools:
        matches = inventory.get(tool.name, [])
        if len(matches) != 1:
            continue
        field, definition = matches[0]
        custom = definition.get("type") in {"function", "custom"}
        tool.origin = Origin(
            kind="tool_list" if custom else "platform_builtin", field=field
        )
        entry = entries.get(tool.name)
        if entry is None:
            entry = ToolEntry(tool=tool.name, mode="provider")
            contract.tool_entrypoint_details.append(entry)
        if entry.source is None:
            entry.implementation_origin = (
                "provider_custom_function" if custom else "provider_builtin"
            )
            entry.origin = tool.origin
        if custom:
            function = definition.get("function") or definition
            if not isinstance(function, Mapping):
                raise ValueError(
                    f"tool[{tool.name}]: provider function must be an object"
                )
            server = definition.get("server") or {}
            url = definition.get("url") or (
                server.get("url") if isinstance(server, Mapping) else None
            )
            entry.provider_function = ProviderFunction(
                url=url, parameters=function.get("parameters")
            )
            if url:
                tool.effect_kind = "external"
                tool.effect_evidence = EffectEvidence(
                    origin=tool.origin,
                    excerpt=json.dumps(definition, ensure_ascii=False),
                )
        elif definition.get("type") in _CONTROL_TOOLS:
            tool.effect_kind = "control"
            tool.effect_evidence = EffectEvidence(
                origin=tool.origin, excerpt=json.dumps(definition, ensure_ascii=False)
            )


def ground_contract(
    contract: AgentContract,
    *,
    source_root: Path | None = None,
    provider_profile: Mapping[str, Any] | None = None,
    catalogue: Mapping[str, Any] | None = None,
) -> AgentContract:
    profile = safe_provider_profile(provider_profile or {})
    contract.tool_entrypoint_details = []
    for entry in contract.tool_entrypoints:
        entry.provider_function = None
        entry.implementation_origin = None
    for amendment in contract.amendment_details:
        if f"{amendment.subject}: {amendment.reason}" not in contract.amendments:
            raise ValueError("amendment details conflict with legacy amendments")
    if profile.get("provider") in {"vapi", "retell"}:
        _provider_facts(contract, profile)
    elif contract.system_prompt or contract.system_prompt_origin:
        _source_prompt(contract, source_root)
    contract.system_prompt_excerpt = redact_profile_urls(contract.system_prompt_excerpt)

    for item in (*contract.hard_constraint_details, *contract.real_use_case_details):
        if set(item.enforced_by) - {"code", "prompt", "judge"}:
            raise ValueError("enforced_by must name code, prompt, or judge")
        origin = item.origin
        text = ""
        if origin is not None:
            if (
                origin.kind
                in {"provider_field", "system_prompt", "tool_list", "platform_builtin"}
                and not origin.field
            ) or (origin.kind in {"file", "code", "doc", "config"} and not origin.path):
                item.origin = None
            elif origin.path or origin.field:
                text = _evidence_text(origin, source_root, profile)
                if (
                    not item.text
                    or item.text not in text
                    or (
                        origin.path
                        and origin.line is not None
                        and item.text.splitlines()[0]
                        not in text.splitlines()[origin.line - 1]
                    )
                ):
                    item.origin = None
                    text = ""
            else:
                item.origin = None
        verified = []
        if (
            "prompt" in item.enforced_by
            and item.text
            and item.text in contract.system_prompt
        ):
            verified.append("prompt")
        if (
            "code" in item.enforced_by
            and origin
            and origin.path.endswith(".py")
            and text
        ):
            try:
                nodes = ast.walk(ast.parse(text))
                if any(
                    isinstance(node, (ast.Assert, ast.Raise))
                    and (origin.line is None or node.lineno == origin.line)
                    and item.text in (ast.get_source_segment(text, node) or "")
                    for node in nodes
                ):
                    verified.append("code")
            except SyntaxError:
                pass
        goals = (catalogue or {}).get("sub_goals", [])
        goal = next(
            (
                goal
                for goal in goals
                if goal.get("name") == item.sub_goal
                and item.text
                and any(
                    item.text in goal.get(field, "")
                    for field in ("what", "check", "judged")
                )
            ),
            None,
        )
        if goal is None:
            item.sub_goal = ""
        elif "judge" in item.enforced_by and goal.get("judged"):
            verified.append("judge")
        item.enforced_by = verified

    for tool in contract.tools:
        if tool.origin is not None and (tool.origin.path or tool.origin.field):
            _evidence_text(tool.origin, source_root, profile)
        elif tool.origin is not None:
            tool.origin = None
        if tool.effect_kind is not None:
            evidence = tool.effect_evidence
            if evidence is None:
                raise ValueError(
                    f"tool[{tool.name}]: canonical effect requires evidence"
                )
            text = _evidence_text(evidence.origin, source_root, profile)
            if evidence.excerpt not in text:
                raise ValueError(
                    f"tool[{tool.name}]: effect evidence does not match its source"
                )
            if evidence.origin.path and evidence.origin.line is not None:
                line = text.splitlines()[evidence.origin.line - 1]
                if evidence.excerpt.splitlines()[0] not in line:
                    raise ValueError(
                        f"tool[{tool.name}]: effect evidence line does not match its source"
                    )
    for actor in contract.actors or []:
        evidence = _evidence_text(actor.origin, source_root, profile)
        if actor.name not in evidence or actor.role not in evidence:
            raise ValueError(
                f"actor[{actor.name}]: name or role is absent from its source"
            )
    for entry in contract.tool_entrypoints:
        if entry.origin is not None:
            if entry.origin.path or entry.origin.field:
                _evidence_text(entry.origin, source_root, profile)
            else:
                entry.origin = None
        if entry.source is not None:
            if source_root is None:
                entry.source = None
            else:
                text = _file_text(
                    source_root, Origin(kind="file", path=entry.source.path)
                )
                actual = "\n".join(
                    text.splitlines()[
                        entry.source.line_start - 1 : entry.source.line_end
                    ]
                )
                if (
                    entry.source.line_end > len(text.splitlines())
                    or entry.source.code != actual
                ):
                    raise ValueError(
                        f"tool[{entry.tool}]: source snippet does not match its file"
                    )
                entry.implementation_origin = "source"
        if entry.mode == "unreachable":
            amendment = Amendment(
                subject=entry.tool,
                kind="unreachable_tool",
                reason=entry.notes or "No runnable implementation seam was found.",
            )
            replaced = {
                f"{item.subject}: {item.reason}"
                for item in contract.amendment_details
                if item.kind == "unreachable_tool" and item.subject == entry.tool
            }
            contract.amendments = [
                sentence for sentence in contract.amendments if sentence not in replaced
            ]
            contract.amendment_details = [
                item
                for item in contract.amendment_details
                if not (item.kind == "unreachable_tool" and item.subject == entry.tool)
            ]
            contract.amendment_details.append(amendment)
            sentence = f"{amendment.subject}: {amendment.reason}"
            if sentence not in contract.amendments:
                contract.amendments.append(sentence)
    return contract
