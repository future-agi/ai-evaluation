"""The tools that provision an environment, and the gate that decides it may be saved.

The difference from ``world/tools.py`` is the whole point of this path. There, the builder
writes a handler per tool and the world answers the agent's calls itself, so what gets tested
is a replica of the agent graded against a replica of its data. Here it writes none: it stands
up the engine the agent already uses, runs the agent's own migrations into it, and points the
agent at it. The agent's code, its client and its queries are untouched.

So there is deliberately no tool here that writes into the agent's repository. Not a guarded
one, not one that records what it changed. When a check fails there are two ways to make it
green -- fix the environment, or edit the agent until it stops failing -- and the second
produces a green suite about code nobody ships. That is not prevented by asking nicely in a
skill file. It is prevented by there being no verb for it.

The same three habits as the older surface, for the same reasons: execute immediately, say what
happened briefly, and never answer with nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..backends import tool, tool_server

from ..contract import AgentContract
from ..environment import (
    SubGoal,
    load_catalogue,
    save_catalogue,
    save_simulator_prompt,
    validate_simulator_prompt,
    validate_sub_goal,
)
from ..tools import schema
from .stores import Store, StoreError, resolve, supported
from .stores.prove import prove_checks_bite, prove_store

PROVISION_SERVER = "environment"

MANIFEST = "environment.json"

# What ``write_store_ops`` is actually given. Said here as well as in the skill because this is
# where the mistake surfaces: a store whose reset is wrong has a model reading *this* message.
from .stores.written import API as OPS_API  # noqa: E402


def _ok(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def _counts(state: dict[str, list[dict]]) -> str:
    return (
        ", ".join(f"{name}: {len(rows)}" for name, rows in sorted(state.items()))
        or "nothing"
    )


def provision_tools(
    contract: AgentContract, destination: Path, source: str | Path | None = None
) -> Any:
    """A server exposing the provisioning surface for one agent.

    ``source`` is the agent's repository. An in-memory store is reached by importing the
    agent's own loader, so the path its modules live under has to be known here; for a store
    behind a socket it is unused.
    """
    destination.mkdir(parents=True, exist_ok=True)
    catalogue = load_catalogue(destination)

    # Everything the stage builds up, held here until ``save_environment`` writes it out.
    standing: dict[str, Any] = {
        "store": None,  # the running Store
        "engine": "",
        "version": "",
        "seam": {},  # how the agent is pointed at it
        "migrations": "",
        "seed": "",
        "mutation": "",
        "proven": False,
    }

    def store() -> Store | None:
        return standing["store"]

    # -- what to stand up -------------------------------------------------------------

    @tool(
        "declare_engine",
        "Stand up the engine this agent already uses, and record how the agent will be "
        "pointed at it. Read the engine off the contract; never choose one for the agent. "
        "The container starts immediately, so a bad image comes back now rather than at "
        "save time.",
        schema(
            {
                "engine": str,
                "version": str,
                "dsn_env": str,
                "config_key": str,
                "host": str,
                "port": int,
                "database": str,
                "user": str,
                "loader_module": str,
                "loader_function": str,
            },
            ["engine"],
        ),
    )
    async def declare_engine(args: dict[str, Any]) -> dict[str, Any]:
        engine = str(args.get("engine") or "").strip()
        if engine not in supported():
            return _err(
                f"no store for {engine!r} yet. The harness can stand up "
                f"{', '.join(supported())}. Write one with write_store_ops first: it needs "
                f"the image, the port it listens on, and how to read and reset it.\n{OPS_API}"
            )
        if standing["store"] is not None:
            standing["store"].stop()

        # What the agent already expects, matched rather than changed. A hardcoded host is
        # redirected by a network alias; a hardcoded database name is simply the name we use.
        options: dict[str, Any] = {}
        if args.get("loader_module"):
            # An in-memory store is reached by importing the agent's own loader, so it takes
            # where to import from rather than anything to connect to.
            options["module"] = str(args["loader_module"])
            options["function"] = str(args.get("loader_function") or "load_data")
            if source:
                options["root"] = str(source)
        else:
            for key in ("database", "user"):
                if args.get(key):
                    options[key] = str(args[key])
            if args.get("version"):
                options["version"] = str(args["version"])

        try:
            running = resolve(engine, **options)
            running.start()
        except StoreError as exc:
            return _err(f"{engine} did not come up: {exc}")
        except TypeError as exc:
            return _err(
                f"{engine} does not take those: {exc}. A store behind a socket takes version, "
                "database and user; an in-memory one takes loader_module and loader_function."
            )

        standing.update(
            store=running,
            engine=engine,
            version=str(args.get("version") or ""),
            seam={
                key: args[key]
                for key in (
                    "dsn_env",
                    "config_key",
                    "host",
                    "port",
                    "database",
                    "user",
                    "loader_module",
                    "loader_function",
                )
                if args.get(key)
            },
            proven=False,
        )
        seam = standing["seam"]
        if args.get("loader_module"):
            return _ok(
                f"{engine} is up, holding what {args['loader_module']}."
                f"{options['function']} loaded: {_counts(running.state())}. Nothing connects "
                "to it — the agent's tools read this structure directly. Seed it if the "
                "contract carries data the loader does not."
            )
        pointed = (
            f"${seam['dsn_env']}"
            if seam.get("dsn_env")
            else (
                seam.get("config_key")
                or "NOTHING RECORDED — say how the agent reaches it"
            )
        )
        return _ok(
            f"{engine} is up at {running.dsn()}. The agent will be pointed at it with "
            f"{pointed}. Now run its own migrations."
        )

    @tool(
        "write_store_ops",
        "Teach the harness an engine it has never stood up: the image, the port, the boot "
        "environment, and how to read and reset what it holds. Only needed for an engine not "
        f"already known.\n\n{OPS_API}",
        schema(
            {
                "engine": str,
                "image": str,
                "container_port": int,
                "boot_env": dict,
                "dsn_template": str,
                "code": str,
            },
            ["engine", "image", "container_port", "code"],
        ),
    )
    async def write_store_ops(args: dict[str, Any]) -> dict[str, Any]:
        from .stores.written import register_written

        try:
            register_written(
                engine=str(args["engine"]),
                image=str(args["image"]),
                container_port=int(args["container_port"]),
                boot_env={
                    str(k): str(v) for k, v in (args.get("boot_env") or {}).items()
                },
                dsn_template=str(args.get("dsn_template") or ""),
                code=str(args["code"]),
            )
        except (StoreError, SyntaxError, ValueError) as exc:
            return _err(f"not registered: {exc}")
        return _ok(
            f"{args['engine']} registered. declare_engine can stand it up now; whether its "
            "reset is right is decided by prove_environment, not by either of us."
        )

    # -- what goes in it --------------------------------------------------------------

    @tool(
        "run_migrations",
        "Run the agent's OWN migrations into the store, so the schema is the agent's, spelled "
        "the way the agent spells it. Never write a schema yourself: one we invented is a "
        "guess, and every check written against it inherits the guess.",
        schema({"script": str}, ["script"]),
    )
    async def run_migrations(args: dict[str, Any]) -> dict[str, Any]:
        running = store()
        if running is None:
            return _err("nothing is standing yet. declare_engine first.")
        try:
            running.apply(str(args.get("script") or ""))
        except Exception as exc:  # noqa: BLE001 - the engine's complaint, reported as given
            return _err(f"the migration was refused: {exc}")
        standing["migrations"] = str(args.get("script") or "")
        state = running.state()
        if not state:
            return _err(
                "that ran but left no tables, so it was not the agent's schema. Find its "
                "migrations, its models, or the DDL it ships."
            )
        return _ok(f"schema is up. {_counts(state)}")

    @tool(
        "seed",
        "Put the agent's real starting data in, from the contract. Leave it in its natural "
        "starting state: empty carts, no in-flight work. Scenarios add what they need.",
        schema({"script": str}, ["script"]),
    )
    async def seed(args: dict[str, Any]) -> dict[str, Any]:
        running = store()
        if running is None:
            return _err("nothing is standing yet. declare_engine first.")
        try:
            running.apply(str(args.get("script") or ""))
        except Exception as exc:  # noqa: BLE001
            return _err(f"the seed was refused: {exc}")
        standing["seed"] += "\n" + str(args.get("script") or "")
        standing["proven"] = False
        return _ok(f"seeded. {_counts(running.state())}")

    @tool(
        "inspect_environment",
        "What is standing, and what it holds.",
        schema({}, []),
    )
    async def inspect_environment(_args: dict[str, Any]) -> dict[str, Any]:
        running = store()
        if running is None:
            # Said before anything is standing, because this is usually the first tool called
            # and "nothing yet" answers nothing. A stage that does not know inprocess already
            # exists goes looking for a server to put the agent's in-memory data in.
            return _ok(
                "nothing is standing yet. The harness can already stand up: "
                f"{', '.join(supported())}.\n"
                "'inprocess' is for an agent that holds its data in memory and whose tools read "
                "that structure directly — it starts no server and the agent's own loader fills "
                "it. Only write_store_ops for an engine genuinely absent from that list."
            )
        return _ok(
            f"{standing['engine']} {standing['version']} at {running.dsn()}\n"
            f"holds: {_counts(running.state())}\n"
            f"agent reaches it by: {json.dumps(standing['seam']) or 'nothing recorded'}\n"
            f"proven: {standing['proven']}"
        )

    # -- what it has to survive --------------------------------------------------------

    @tool(
        "add_sub_goal",
        "Add a named thing this agent can be checked on, shared by every scenario that needs "
        "it. Defined here, once, so results roll up: the same sub-goal failing in seven of "
        "twelve scenarios is one sentence.\n\n"
        "`check` is Python: define check(world, calls) returning a sentence when something is "
        "wrong, or None when it held. `world` is the environment afterwards, and "
        "world.state() gives every group and its rows; `calls` is every tool call made, each "
        "with .name and .arguments — so a check can insist a call happened with the right "
        "arguments, not merely that it happened.\n\n"
        "Do not write a check that asks whether a tool refused. Many agents report a refusal "
        "by returning an ordinary string, so nothing distinguishes it from success. Check what "
        "the world holds afterwards, and the arguments the agent actually used.\n\n"
        "`judged` is not a flag: it is the sentence saying what a model has to decide and why "
        "code cannot. Leave it empty for anything code can settle, which is most things.",
        schema(
            {"name": str, "what": str, "check": str, "judged": str},
            ["name", "what"],
        ),
    )
    async def add_sub_goal(args: dict[str, Any]) -> dict[str, Any]:
        one = SubGoal(
            name=str(args["name"]),
            what=str(args.get("what") or ""),
            check=str(args.get("check") or ""),
            judged=str(args.get("judged") or ""),
        )
        problems = validate_sub_goal(one)
        if problems:
            return _err(f"{one.name} not added:\n  - " + "\n  - ".join(problems))
        catalogue.sub_goals = [
            existing for existing in catalogue.sub_goals if existing.name != one.name
        ]
        catalogue.sub_goals.append(one)
        settled = [g.name for g in catalogue.sub_goals if g.deterministic()]
        standing["proven"] = False
        return _ok(
            f"{one.name} added. The catalogue has {len(catalogue.sub_goals)}, "
            f"{len(settled)} settled by code: {', '.join(sorted(settled)) or 'none'}"
        )

    @tool(
        "write_simulator_prompt",
        "The person on the other side of this conversation, with a {{ instruction }} slot each "
        "scenario fills. Only for a conversational agent.",
        schema({"prompt": str}, ["prompt"]),
    )
    async def write_simulator_prompt_tool(args: dict[str, Any]) -> dict[str, Any]:
        prompt = str(args.get("prompt") or "")
        problems = validate_simulator_prompt(prompt)
        if problems:
            return _err("not saved:\n  - " + "\n  - ".join(problems))
        path = save_simulator_prompt(prompt, destination)
        return _ok(f"Saved to {path}.")

    @tool(
        "prove_environment",
        "Put the environment through what it has to survive before anything is measured "
        "against it. `mutation` is any statement this engine accepts that changes something -- "
        "one insert is plenty -- and it is checked too, because a mutation that moves nothing "
        "would make a broken reset look perfect.",
        schema({"mutation": str}, ["mutation"]),
    )
    async def prove_environment(args: dict[str, Any]) -> dict[str, Any]:
        running = store()
        if running is None:
            return _err("nothing is standing yet. declare_engine first.")
        mutation = str(args.get("mutation") or "")

        report = prove_store(running, mutation)
        lines = [report.summary()]
        failed = [one.name for one in report.results if not one.passed]

        if not failed and catalogue.sub_goals:
            bites = prove_checks_bite(running, _checks(catalogue))
            lines.append(bites.summary())
            failed += [one.name for one in bites.results if not one.passed]

        standing["proven"] = not failed
        standing["mutation"] = mutation
        if failed:
            return _err("\n".join(lines))
        return _ok(
            "\n".join(lines) + "\nThe environment holds. save_environment will keep it."
        )

    @tool(
        "save_environment",
        "Freeze the environment and write it out. Refused unless it passes its own gate, "
        "which is re-run here rather than taken on trust.",
        schema({"notes": str}, []),
    )
    async def save_environment(args: dict[str, Any]) -> dict[str, Any]:
        running = store()
        if running is None:
            return _err("nothing is standing yet. declare_engine first.")
        # An in-memory store has no migration step: the agent's loader is what creates the
        # structure and fills it, and it already ran. Insisting on one here would be asking for
        # a schema this agent does not have.
        by_loader = bool(standing["seam"].get("loader_module"))
        if not standing["migrations"] and not by_loader:
            return _err(
                "Not saved. The agent's own migrations were never run, so this schema is not "
                "the agent's."
            )
        if not standing["seam"]:
            return _err(
                "Not saved. Nothing records how the agent reaches this store, so it cannot be "
                "pointed at it. If the agent genuinely has no configuration seam, that is a "
                "finding to report rather than something to work around."
            )
        if not catalogue.sub_goals:
            return _err(
                "Not saved. No sub-goals yet. They are defined here, once, and every scenario "
                "names the ones it needs — that is what makes results add up across the suite."
            )
        if not [one for one in catalogue.sub_goals if one.deterministic()]:
            return _err(
                "Not saved. Every sub-goal is judged by a model. Most of what this agent does "
                "leaves a trace in the store, and that should be settled by code."
            )

        # Re-run rather than trusting the flag: the builder does not get to declare its own
        # environment sound, for the same reason it does not get to declare a scenario passed.
        report = prove_store(running, standing["mutation"])
        failed = [one.name for one in report.results if not one.passed]
        if failed:
            return _err(
                f"Not saved, the environment does not hold up.\n{report.summary()}"
            )

        baseline = running.freeze()
        manifest = {
            "agent": contract.agent,
            "engine": standing["engine"],
            "version": standing["version"],
            "seam": standing["seam"],
            "migrations": standing["migrations"],
            "seed": standing["seed"],
            "mutation": standing["mutation"],
            "notes": str(args.get("notes") or ""),
            # The recipe, not the running container: run time stands the engine up again and
            # replays these, which is what makes the environment reproducible rather than a
            # thing that happened once on somebody's laptop.
            "baseline": {"rows": baseline.rows, "counters": baseline.counters},
        }
        path = destination / MANIFEST
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        save_catalogue(catalogue, destination)
        return _ok(
            f"Saved to {destination}. {standing['engine']} holding {_counts(baseline.rows)}, "
            f"{len(catalogue.sub_goals)} sub-goals."
        )

    server = tool_server(
        name=PROVISION_SERVER,
        version="0.1.0",
        tools=[
            declare_engine,
            write_store_ops,
            run_migrations,
            seed,
            inspect_environment,
            add_sub_goal,
            write_simulator_prompt_tool,
            prove_environment,
            save_environment,
        ],
    )
    return server, standing


def _checks(catalogue: Any) -> dict[str, Any]:
    """The catalogue's code checks, as things that can be run against a store.

    A check is written ``check(world, calls)``, and a store answers ``state()`` the same way a
    world does, so the same code runs against either. At build time nothing has been called
    yet, which is exactly the condition the bite gate wants: a check that still holds with no
    calls and no rows is not checking anything.
    """
    out: dict[str, Any] = {}
    for one in catalogue.sub_goals:
        if not one.deterministic():
            continue
        out[one.name] = _compiled(one)
    return out


def _compiled(one: Any) -> Any:
    def run(store: Store) -> str | None:
        namespace: dict[str, Any] = {}
        exec(compile(one.check, f"<check:{one.name}>", "exec"), namespace)  # nosec B102
        return namespace["check"](store, [])

    return run


TOOL_NAMES = (
    "declare_engine",
    "write_store_ops",
    "run_migrations",
    "seed",
    "inspect_environment",
    "add_sub_goal",
    "write_simulator_prompt",
    "prove_environment",
    "save_environment",
)
