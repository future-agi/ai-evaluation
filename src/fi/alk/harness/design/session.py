"""The builder in design mode: one session, opened in two phases.

`open_builder` needs a contract and a saved world, and the scenario writers bake both in when they
open, so a design opens first on the world server and its own tools (interview, approval, world,
tools, rules) and, once the world is saved, on the builder's scenario tools over the design view
with its own tools still beside them. The phase follows `design.json`'s stage; the caller opens
the session again when `DesignTools.reopen` says the phase or the contract changed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..backends import ASK_TOOL, FILE_TOOLS, SessionSpec
from ..builder import (
    BUILDER,
    BUILDER_TURNS,
    DESIGNER,
    KNOWLEDGE_SERVER,
    knowledge_server,
    open_builder,
    skill_index,
)
from ..composition_edit import FIXED_ENVIRONMENT_TOOLS
from ..config import chosen_model, load_skill
from ..releases import WORLD_IMPLEMENTATION
from ..session import Stage
from ..world.snapshot import restore
from ..world.tools import WORLD_SERVER, world_tools
from .contract import design_contract
from .tools import DESIGN_SERVER, Approve, DesignTools
from .workspace import load_design

# The world server's tools a design never has: they bind a source, run its environment, or
# change the contract and policy behind the design tools' backs.
WORLD_LEFT_OUT = frozenset(
    {
        "adopt_state",
        "adopt_store",
        "adopt_tool",
        "write_store_ops",
        "run_env_command",
        "write_env_file",
        "fix_tool",
        "amend_contract",
        "add_rule",
        "drop_rule",
        "set_modality",
    }
)
_FILES = (
    "\n\nThe design's files are in {out}: design.json, tools.json, policy.json, handlers/, the "
    "world (manifest.json, world.sqlite), contract.json (derived from them), sub_goals.json, "
    "simulator_prompt.md and scenarios/. Change them only through the tools."
)


def open_for_designing(
    workspace: Path,
    *,
    approve: Approve,
    postgres_dsn: Callable[[], str],
    output: Path,
    ask: Callable[..., Any] | None = None,
    hosted_prompt: str = "",
) -> tuple[Stage, DesignTools]:
    design = load_design(workspace)
    contract = design_contract(workspace, design)
    options: dict[str, Any] = {
        "approve": approve,
        "postgres_dsn": postgres_dsn,
        "output": output,
    }
    if design.stage == "suite":
        tools = DesignTools(workspace, restore(workspace), contract, **options)
        stage = open_builder(
            contract,
            out=workspace,
            source_root=None,
            ask=ask,
            hosted_prompt=hosted_prompt,
            brief=f"## This design\n\n{tools.brief()}",
            sealed=False,
            more_servers={DESIGN_SERVER: tools.server()},
            files_note=_FILES.format(out=workspace),
            without=FIXED_ENVIRONMENT_TOOLS,
            skill=DESIGNER,
        )
        return stage, tools
    world_server, world = world_tools(
        contract, workspace, implementation=WORLD_IMPLEMENTATION
    )
    tools = DesignTools(workspace, world, contract, **options)
    spec = SessionSpec(
        system_prompt=(
            load_skill(DESIGNER)
            + "\n\n## What you know\n\nLoad any of these with `read_skill` before you do work "
            "it governs.\n\n"
            + "\n".join(f"- `{name}`: {about}" for name, about in skill_index().items())
            + f"\n\n## This design\n\n{tools.brief()}"
            + _FILES.format(out=workspace)
            + hosted_prompt
        ),
        servers={
            WORLD_SERVER: tools.guard_world(world_server, WORLD_LEFT_OUT),
            DESIGN_SERVER: tools.server(),
            KNOWLEDGE_SERVER: knowledge_server(contract, workspace),
        },
        builtins=(*FILE_TOOLS, ASK_TOOL) if ask is not None else FILE_TOOLS,
        cwd=str(workspace),
        max_turns=BUILDER_TURNS,
        model=chosen_model(),
        ask=ask,
    )
    return Stage(spec, name=BUILDER), tools


def unfinished(stage: str) -> str:
    """What the session is told when a turn ends and the design is not finished."""
    return (
        "[The design is not finished. "
        + {
            "interview": "Ask what you still need with AskUserQuestion, one question at a time, "
            "then propose_design.",
            "world": "Build the approved world, its tools and rules, then save_world.",
            "suite": "Write and prove the suite, run check_design, then finish_design.",
        }[stage]
        + " Or call stop_design with the reason.]"
    )
