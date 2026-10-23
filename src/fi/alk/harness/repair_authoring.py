"""Constrained model-assisted repair for generic authored artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .backends import FILE_TOOLS, SessionSpec, tool, tool_server
from .config import chosen_model
from .diagnostics import HarnessDiagnostic
from .repair_patch import (
    RepairPatchOperation,
    WorldIRRepairPatch,
    apply_world_ir_repair_patch,
)
from .session import Stage
from .source_data_invariants import world_from_ir
from .source_model import SourceModel
from .world_ir import WorldIR


class RepairPatchNotSubmittedError(RuntimeError):
    pass


class _PatchSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # An empty patch cannot repair a candidate. Accepting one previously consumed a full
    # authoring budget and launched another expensive clean-room validation with unchanged
    # semantics. Require an actual, typed mutation at the only write boundary.
    operations: tuple[RepairPatchOperation, ...] = Field(min_length=1)


async def request_world_ir_patch(
    source_root: Path,
    source: SourceModel,
    world: WorldIR,
    diagnostics: tuple[HarnessDiagnostic, ...],
    *,
    guidance: str = "",
) -> WorldIRRepairPatch:
    """Ask for one typed patch; validation happens inside the only mutating tool."""

    allowed = {item.code for item in diagnostics if item.repair_strategy}
    captured: list[WorldIRRepairPatch] = []

    @tool(
        "submit_world_ir_patch",
        "Submit the smallest evidence-backed patch to generated World IR. This tool cannot "
        "edit submitted source, delete scenarios, weaken checks, or change infrastructure.",
        _PatchSubmission.model_json_schema(),
    )
    async def submit(arguments: dict[str, object]) -> dict[str, object]:
        try:
            submission = _PatchSubmission.model_validate(arguments)
            patch = WorldIRRepairPatch.create(
                base_world_ir_hash=world.fingerprint,
                operations=submission.operations,
            )
            apply_world_ir_repair_patch(
                world, source, patch, allowed_reason_codes=allowed
            )
        except Exception as exc:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"patch rejected: {type(exc).__name__}: {exc}",
                    }
                ],
                "is_error": True,
            }
        captured[:] = [patch]
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"patch accepted with {len(patch.operations)} operations",
                }
            ]
        }

    backing = world_from_ir(world)
    backing.query("PRAGMA query_only = ON")

    def _rows(value: object) -> dict[str, object]:
        return {"content": [{"type": "text", "text": json.dumps(value, default=str)[:20000]}]}

    @tool(
        "inspect_world",
        "The generated world's tables: without a name, every table and its row count; with one, "
        "all of its rows.",
        {"type": "object", "properties": {"table": {"type": "string"}}},
    )
    async def inspect_world(arguments: dict[str, object]) -> dict[str, object]:
        name = str(arguments.get("table") or "").strip()
        if not name:
            return _rows({table.source_name: len(table.rows) for table in world.tables})
        return _rows(backing.query(f'SELECT * FROM "{name.replace(chr(34), "")}"'))

    @tool(
        "query_world",
        "Run one read-only SQL query against the generated world, for example the diagnostic's own "
        "query, to see exactly which rows are wrong.",
        {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
    )
    async def query_world(arguments: dict[str, object]) -> dict[str, object]:
        try:
            return _rows(backing.query(str(arguments.get("sql") or "")))
        except Exception as exc:  # noqa: BLE001 - a bad query is feedback, not a failed repair
            return {"content": [{"type": "text", "text": f"query failed: {exc}"}], "is_error": True}

    diagnostics_json = [
        item.model_dump(mode="json", exclude={"redacted_message"})
        | {"message": item.redacted_message}
        for item in diagnostics
    ]
    system_prompt = """You repair generated environment data for arbitrary agent runtimes. Your goal:
every supplied diagnostic passes afterwards. Explore as much as you need to get there: read the
submitted source with Read, Glob and Grep, list and read the generated world with inspect_world,
and run the failing queries yourself with query_world to see exactly which rows are wrong.
Submitted source and the canonical source model are authoritative. Never modify source, invent
an action/service, remove a scenario, weaken a check, or alter credentials/egress. Your only write
capability is submit_world_ir_patch. Every operation reason must exactly match one supplied
diagnostic code. Make the smallest patch that resolves the complete diagnostic set, call the
tool, then stop."""
    briefing = (
        "Repair this candidate.\n\nDIAGNOSTICS\n"
        + json.dumps(diagnostics_json, sort_keys=True, indent=2)[:16000]
        + "\n\nCANONICAL SOURCE MODEL\n"
        + source.model_dump_json(indent=2)[:24000]
        + "\n\nCURRENT GENERATED WORLD IR\n"
        + world.model_dump_json(indent=2)[:32000]
        + ("\n\nREQUIRED JOB INTENT\n" + guidance if guidance else "")
    )
    # Start with a deliberately narrow session. The complete typed source model, current World
    # IR, diagnostics, and their evidence references are already in ``briefing``; giving a model
    # repository exploration tools immediately can consume the entire turn budget reading a
    # large tree without ever reaching the one required write. If the compact pass cannot decide,
    # a fresh evidence-enabled session gets the original, larger budget. A fresh session matters:
    # Claude Agent SDK's max-turn bound is cumulative within a session, so merely prompting a
    # session that exhausted its first turn does not provide another repair opportunity.
    passes = (
        (
            FILE_TOOLS,
            24,
            "Look at the rows the diagnostics are about before patching; call "
            "submit_world_ir_patch.",
        ),
        (
            FILE_TOOLS,
            40,
            "The constrained repair pass did not submit an accepted patch. Inspect only the "
            "source evidence needed to resolve the diagnostics, then call "
            "submit_world_ir_patch.",
        ),
    )
    for builtins, max_turns, instruction in passes:
        stage = Stage(
            SessionSpec(
                system_prompt=system_prompt,
                servers={
                    "repair": tool_server(
                        "repair", tools=[submit, inspect_world, query_world]
                    )
                },
                builtins=builtins,
                cwd=str(source_root.resolve()),
                # Repair is part of the same authoring run and must use the explicitly
                # selected harness model. Leaving this unset falls back to the Claude SDK's
                # native model name, bypassing an AgentCC model route such as Vertex Gemini.
                model=chosen_model(),
                max_turns=max_turns,
                gated=True,
                thinking=True,
            ),
            name="repair-world-ir",
        )
        async with stage:
            await stage.say(f"{briefing}\n\n{instruction}")
            if not captured:
                await stage.say(
                    "No valid patch was submitted. Call submit_world_ir_patch now. If a prior "
                    "submission was rejected, correct its typed arguments using that rejection."
                )
        if captured:
            break
    if not captured:
        raise RepairPatchNotSubmittedError(
            "typed_world_ir_patch_not_submitted_after_constrained_and_evidence_passes"
        )
    return captured[0]


__all__ = ["request_world_ir_patch"]
