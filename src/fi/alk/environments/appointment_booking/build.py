"""Build and seal the appointment environment release and its base suite.

    python -m fi.alk.environments.appointment_booking.build --out DIR

writes `DIR/environment` and `DIR/suite`, seals both through `fi.alk.harness.releases`, and prints
their digests. Sealing the suite proves every scenario against the environment, so a build that
finishes is a build whose scenarios all executed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from fi.alk.harness.catalogue import save_catalogue
from fi.alk.harness.folder import write_folder
from fi.alk.harness.releases import (
    POLICY,
    SERVERS,
    TOOLS,
    WORLD,
    WORLD_IMPLEMENTATION,
    ReleaseError,
    load_tools,
    seal_environment_release,
    seal_suite_release,
)
from fi.alk.harness.world.runtime import GeneratedWorld
from fi.alk.harness.world.snapshot import save

from . import suite

NAME = "appointment_booking"
HERE = Path(__file__).parent
_WORLD_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}

SIMULATOR_PROMPT = """You are a patient contacting Riverside Family Clinic by chat. The clinic has
primary care (Dr. Alicia Nguyen, Dr. Rohan Patel), pediatrics (Dr. Hannah Lee) and dermatology
(Dr. Sofia Morales) at its Main and Eastside clinics. Today is Monday 2 March 2026, 9am.

Who you are:
{{ persona }}

What you are here to do:
{{ instruction }}

Use only the facts your task gives you; if you are asked for something it does not say, say you
don't know.
"""


def _world_tool_specs(root: Path) -> list[dict]:
    """The world's own description of each tool, derived from the canonical definitions."""
    specs = []
    for tool in load_tools(root):
        properties = tool.parameters.get("properties") or {}
        specs.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    argument: {
                        "type": _WORLD_TYPES.get(str(schema.get("type")), "str"),
                        "values": schema.get("enum"),
                    }
                    for argument, schema in properties.items()
                },
            }
        )
    return specs


def build_environment(destination: Path) -> str:
    destination.mkdir(parents=True)
    shutil.copyfile(HERE / TOOLS, destination / TOOLS)
    shutil.copyfile(HERE / POLICY, destination / POLICY)
    # The read-only `sql` MCP server over the reference tables (providers, slots).
    shutil.copyfile(HERE / SERVERS, destination / SERVERS)
    world = GeneratedWorld(":memory:", kind="sqlite")
    world.name = NAME
    world.store.apply((HERE / "world.sql").read_text(encoding="utf-8"))
    world.handlers = {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted((HERE / "handlers").glob("*.py"))
    }
    world.tools = _world_tool_specs(destination)
    save(
        world,
        destination / WORLD,
        notes="Riverside Family Clinic: patients, providers, slots and appointments.",
        implementation=WORLD_IMPLEMENTATION,
    )
    world.close()
    release = seal_environment_release(
        destination, name=NAME, derived_from="template:healthcare_scheduling"
    )
    return release.digest


def build_suite(
    destination: Path, environment: Path, *, postgres_dsn: str | None = None
) -> str:
    destination.mkdir(parents=True)
    catalogue = suite.catalogue()
    for scenario in suite.scenarios():
        write_folder(scenario, catalogue, destination)
    save_catalogue(catalogue, destination)
    (destination / "simulator_prompt.md").write_text(SIMULATOR_PROMPT, encoding="utf-8")
    sealed = seal_suite_release(
        destination,
        name=f"{NAME}_base",
        environment_root=environment,
        policy_profile={"scored_agent_rules": suite.SCORED_AGENT_RULES},
        postgres_dsn=postgres_dsn,
    )
    return sealed.digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fi.alk.environments.appointment_booking.build"
    )
    parser.add_argument("--out", required=True, help="empty or missing directory")
    parser.add_argument(
        "--postgres-dsn",
        default=os.environ.get("ALK_RELEASE_POSTGRES_DSN"),
        help="role that may create databases; hosted runs need the suite proved on PostgreSQL",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} is not empty")
    try:
        environment = build_environment(out / "environment")
        suite_digest = build_suite(
            out / "suite", out / "environment", postgres_dsn=args.postgres_dsn
        )
    except ReleaseError as exc:
        print(json.dumps({"error": exc.code, "detail": exc.detail}), file=sys.stderr)
        return 1
    print(json.dumps({"environment": environment, "suite": suite_digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
