"""A design's workspace, and the releases sealed from it.

The design view is the session's workspace, laid out flat the way the builder reads an
environment: the world snapshot (`manifest.json`, `world.sqlite`, `handlers/`, `world.py`), the
suite's files (`scenarios/`, `scenarios.json`, `sub_goals.json`, `simulator_prompt.md`), the
environment's own `tools.json`, `policy.json`, `servers.json` and `attached/`, and `design.json`.

Sealing hashes everything under a root, and the workspace also gathers the session's own files,
so the release view is materialized from the design view through an explicit allow-list.
`seal_design` seals that view into a root environment release and its base suite, proved on
PostgreSQL. A customized template is a new root, not an extension, because an extension can only
add; the template is recorded as a label.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..catalogue import load_catalogue
from ..folder import check_file_drift, read_all, write_index
from ..releases import (
    _SLUG_PATTERN,
    ATTACHED,
    POLICY,
    POLICY_TABLE,
    RELEASES_ENVIRONMENT,
    RELEASES_SUITE,
    SCENARIOS,
    SERVERS,
    SUB_GOALS,
    TOOLS,
    WORLD,
    WORLD_IMPLEMENTATION,
    EnvironmentReleaseV1,
    PolicyProfile,
    ReleaseError,
    ReleaseRef,
    SuiteReleaseV1,
    _policy_rows,
    load_environment_release,
    load_policy,
    load_suite_release,
    load_tools,
    seal_environment_release,
    seal_suite_release,
    world_tool_specs,
)
from ..world.snapshot import (
    DATABASE,
    HANDLERS,
    MANIFEST,
    WORLD_MODULE,
    read_manifest,
    restore,
    save,
    saved,
)
from .schema import unseeded_schema

DESIGN = "design.json"
DESIGN_SCHEMA = "futureagi.environment-design.v1"
SIMULATOR_PROMPT = "simulator_prompt.md"
_IGNORED = shutil.ignore_patterns(".DS_Store", "__pycache__", "*.pyc", "*.pyo")


class DesignTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: ReleaseRef
    suite: ReleaseRef


class Design(BaseModel):
    """`design.json`: what the design is called and what it started from."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["futureagi.environment-design.v1"] = DESIGN_SCHEMA
    name: str = Field(pattern=_SLUG_PATTERN)
    template: DesignTemplate | None = None
    policy_profile: PolicyProfile = Field(default_factory=PolicyProfile)
    # The template world's schema that PostgreSQL seeds do not enforce; listed, not refused.
    inherited_schema: list[str] = Field(default_factory=list)


def load_design(workspace: str | Path) -> Design:
    path = Path(workspace) / DESIGN
    try:
        return Design.model_validate_json(path.read_bytes())
    except FileNotFoundError as exc:
        raise ReleaseError("design_invalid", f"{DESIGN} is missing") from exc
    except ValidationError as exc:
        raise ReleaseError("design_invalid", str(exc)) from exc


def _write_design(workspace: Path, design: Design) -> None:
    (workspace / DESIGN).write_text(
        json.dumps(design.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def open_design(
    workspace: str | Path, *, name: str, template: tuple[Path, Path] | None = None
) -> Design:
    """The workspace's design: an empty one, or a root template's environment and suite unpacked
    into it. `template` is `(environment_root, suite_root)`. A workspace that already holds a
    design keeps it."""
    workspace = Path(workspace)
    if (workspace / DESIGN).is_file():
        return load_design(workspace)
    try:
        design = Design(name=name)
    except ValidationError as exc:
        raise ReleaseError("design_invalid", str(exc)) from exc
    workspace.mkdir(parents=True, exist_ok=True)
    if template is not None:
        design = _unpack(workspace, design, Path(template[0]), Path(template[1]))
    # Written last, so a failed unpack leaves no design and opening again lays it out afresh.
    _write_design(workspace, design)
    return design


def _unpack(
    workspace: Path, design: Design, environment_root: Path, suite_root: Path
) -> Design:
    environment = load_environment_release(environment_root)
    if environment.extends is not None:
        raise ReleaseError(
            "design_template_invalid",
            f"{environment.name} extends {environment.extends.name}; customize a root release",
        )
    suite = load_suite_release(suite_root)
    if suite.environment.digest != environment.digest:
        raise ReleaseError(
            "design_template_invalid",
            f"{suite.name} was not proved against {environment.name}",
        )
    for name in (TOOLS, POLICY, SERVERS):
        if (environment_root / name).is_file():
            shutil.copyfile(environment_root / name, workspace / name)
    _copy_tree(environment_root / ATTACHED, workspace / ATTACHED)
    # The world as sealed, default policy rows included: the suite's parameters stay in the
    # design's profile, because the environment seal requires the defaults.
    for path in (environment_root / WORLD).iterdir():
        if path.is_dir():
            _copy_tree(path, workspace / path.name)
        else:
            shutil.copyfile(path, workspace / path.name)
    shutil.rmtree(workspace / SCENARIOS, ignore_errors=True)
    for entry in suite.scenarios:
        _copy_tree(suite_root / entry.path, workspace / entry.path)
    for name in (SUB_GOALS, SIMULATOR_PROMPT):
        if (suite_root / name).is_file():
            shutil.copyfile(suite_root / name, workspace / name)
    write_index(read_all(workspace), workspace)
    return design.model_copy(
        update={
            "template": DesignTemplate(
                environment=ReleaseRef(
                    digest=environment.digest, name=environment.name
                ),
                suite=ReleaseRef(digest=suite.digest, name=suite.name),
            ),
            "policy_profile": suite.policy_profile,
            "inherited_schema": unseeded_schema(environment_root / WORLD / DATABASE),
        }
    )


def _copy_tree(source: Path, target: Path) -> None:
    # Symlinks stay symlinks, so sealing refuses them rather than copying what they point at.
    if source.is_dir():
        shutil.copytree(
            source, target, symlinks=True, ignore=_IGNORED, dirs_exist_ok=True
        )


def release_view(workspace: str | Path, out: str | Path) -> tuple[Path, Path]:
    """The design view's environment and suite roots, materialized under `out`."""
    workspace, out = Path(workspace), Path(out)
    environment, suite = out / RELEASES_ENVIRONMENT, out / RELEASES_SUITE
    world = environment / WORLD
    world.mkdir(parents=True)
    for name in (TOOLS, POLICY, SERVERS):
        if (workspace / name).is_file():
            shutil.copyfile(workspace / name, environment / name)
    _copy_tree(workspace / ATTACHED, environment / ATTACHED)
    for name in (MANIFEST, WORLD_MODULE):
        shutil.copyfile(workspace / name, world / name)
    _copy_tree(workspace / HANDLERS, world / HANDLERS)
    # Through SQLite's backup API, so a live session's database is copied consistently.
    origin = sqlite3.connect(f"file:{workspace / DATABASE}?mode=ro", uri=True)
    copy = sqlite3.connect(world / DATABASE)
    try:
        with copy:
            origin.backup(copy)
    finally:
        copy.close()
        origin.close()
    suite.mkdir()
    _copy_tree(workspace / SCENARIOS, suite / SCENARIOS)
    for name in (SUB_GOALS, SIMULATOR_PROMPT):
        if (workspace / name).is_file():
            shutil.copyfile(workspace / name, suite / name)
    return environment, suite


def _derive_policy_rows(workspace: Path) -> None:
    """The world holds `policy.json`'s defaults, as the environment seal requires; a world that
    already does is left byte for byte as it is."""
    defaults = {
        name: parameter.default
        for name, parameter in load_policy(workspace).parameters().items()
    }
    database = workspace / DATABASE
    if (_policy_rows(database) or {}) == defaults:
        return
    connection = sqlite3.connect(database)
    try:
        with connection:
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {POLICY_TABLE} "
                "(name TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(f"DELETE FROM {POLICY_TABLE}")
            connection.executemany(
                f"INSERT INTO {POLICY_TABLE} (name, value) VALUES (?, ?)",
                [(name, json.dumps(value)) for name, value in defaults.items()],
            )
    finally:
        connection.close()


def _resave_world(workspace: Path, design: Design) -> None:
    """The world saved again as an environment's: its handlers are the release's tools, and its
    notes, sequences and world checks are kept."""
    manifest = read_manifest(workspace)
    world = restore(workspace)
    try:
        world.name = design.name
        world.tools = world_tool_specs(workspace)
        world.handlers = {
            tool.name: path.read_text(encoding="utf-8")
            for tool in load_tools(workspace)
            if (path := workspace / HANDLERS / f"{tool.name}.py").is_file()
        }
        save(
            world,
            workspace,
            notes=str(manifest.get("notes") or ""),
            sequences=list(manifest.get("sequences") or []),
            world_checks=dict(manifest.get("world_checks") or {}),
            implementation=WORLD_IMPLEMENTATION,
        )
    finally:
        world.close()


def seal_design(
    workspace: str | Path, out: str | Path, *, postgres_dsn: str
) -> tuple[EnvironmentReleaseV1, SuiteReleaseV1]:
    """Seal the design into `out/environment` and `out/suite`, or raise `ReleaseError` and leave
    no `out`. Every scenario is proved on PostgreSQL, which hosted runs require."""
    workspace, out = Path(workspace), Path(out)
    design = load_design(workspace)
    if not saved(workspace):
        raise ReleaseError("design_world_missing", "the design has no saved world yet")
    _derive_policy_rows(workspace)
    _resave_world(workspace, design)
    if added := sorted(
        set(unseeded_schema(workspace / DATABASE)) - set(design.inherited_schema)
    ):
        raise ReleaseError(
            "design_world_unsafe",
            "hosted runs seed the world into PostgreSQL, which would drop or rewrite: "
            + "; ".join(added)
            + ". Enforce these rules in the handlers instead.",
        )
    shutil.rmtree(out, ignore_errors=True)
    environment_root, suite_root = release_view(workspace, out)
    try:
        environment = seal_environment_release(
            environment_root,
            name=design.name,
            derived_from=f"release:{design.template.environment.digest}"
            if design.template
            else None,
        )
        if drift := check_file_drift(suite_root, load_catalogue(suite_root)):
            raise ReleaseError("design_check_files_invalid", "; ".join(drift))
        suite = seal_suite_release(
            suite_root,
            name=f"{design.name}_base",
            environment_root=environment_root,
            policy_profile=design.policy_profile,
            metadata={"template_suite": design.template.suite.digest}
            if design.template
            else None,
            postgres_dsn=postgres_dsn,
        )
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return environment, suite
