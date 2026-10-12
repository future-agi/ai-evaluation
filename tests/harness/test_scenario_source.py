"""`scenario_source.py` -- the reader/compiler/wrapper that turns a bundle's own scenario documents
(the on-disk layout `folder.py` documents: `scenarios/<name>/scenario.json` + `setup.py` +
`ready.py` + `checks/<goal>.py`) into `hosted_scheduler.py`'s `Scenario`/`SubGoal` objects.

DUPLICATION DISCLOSURE: this file writes its own scenario-folder fixtures (`_write_scenario`) --
nothing else in the suite writes this layout, so there is nothing to import instead. The
lightweight `WorldPool`/`HostedScheduler` harness below (`_FakeWorld`, `_FakeWorldFactory`,
`_FakeCallRunner`, `_pool`) is adapted from `tests/harness/test_hosted_scheduler.py`'s own fakes of
the same names (not imported across test modules, per the brief -- copied and trimmed to what this
file needs; `_FakeWorld.read_only()` deliberately SHARES its `rows` dict with the writable world,
unlike that file's `InMemoryWorld.read_only()`, because the consumer-proof test below needs a
check to see what setup actually wrote). No `EnvironmentBundleV2` manifest-writing helper (the
`_build_bundle` pattern in `test_process_preflight.py`) is needed anywhere in this file: neither
`load_scenarios` nor `WorldPool`/`HostedScheduler` reads or validates one.

`asyncio.run` drives every `async def` seam here, matching every other file in this suite (no
pytest-asyncio dependency in this repo). Verified via `pytest tests/harness/test_scenario_source.py`.
"""

from __future__ import annotations

import asyncio
import json
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fi.alk.harness import scenario_source as ss
from fi.alk.harness.hosted_entrypoint import ScenarioPreallocationError
from fi.alk.harness.hosted_scheduler import (
    Call,
    CallOutcome,
    HostedScheduler,
    WorldPool,
)
from fi.alk.harness.process_runtime import EnvironmentRuntime, RuntimeState

# =================================================================================================
# Scenario-folder fixture writer -- `folder.py`'s documented layout, hand-written (never through
# `fi.alk.harness.folder`/`fi.alk.harness.scenario`, matching the module under test).
# =================================================================================================


@dataclass
class _FakeJob:
    """Stands in for `HarnessJob` wherever only `.run_id` is read (p13:
    `BundleScenarioSource.build` uses it as the platform-facing provision run name) -- avoids
    importing the real pydantic model into this module purely for one attribute."""

    run_id: str = "job-1"
    metadata: dict[str, Any] = field(default_factory=dict)


@pytest.fixture(autouse=True)
def _judged_sub_goals_decided_without_a_model(monkeypatch):
    """These tests are about wrapping scenarios, not about judging.

    A judged sub-goal now goes to a model, so without this the two consumer-proof tests would make
    a live call and assert on the verdict rather than on the wrapping.
    """
    from fi.alk.harness import hosted_scheduler

    async def _held(goal, world, calls, *, messages=()):
        return True, f"{goal.name}: stubbed for a scenario-source test"

    monkeypatch.setattr(hosted_scheduler, "_judge", _held)


def _write_scenario(
    scenarios_root: Path,
    name: str,
    *,
    scenario_key: str = "",
    scenario_id: str = "",
    sub_goals: list[str] | None = None,
    setup_code: str = "",
    ready_code: str = "",
    checks: dict[str, str] | None = None,
    raw_body: dict[str, Any] | None = None,
    write_body: bool = True,
) -> Path:
    """One scenario folder, written by hand -- deliberately not through pr63's `Scenario` model
    (which backfills an empty `scenario_key` via a validator, so a model instance can never
    produce the empty-key fixture the brief requires)."""
    folder = scenarios_root / name
    (folder / "checks").mkdir(parents=True, exist_ok=True)
    if write_body:
        body = (
            raw_body
            if raw_body is not None
            else {
                "name": name,
                "scenario_key": scenario_key,
                "scenario_id": scenario_id,
                "sub_goals": sub_goals or [],
            }
        )
        (folder / "scenario.json").write_text(json.dumps(body), encoding="utf-8")
    if setup_code:
        (folder / "setup.py").write_text(setup_code, encoding="utf-8")
    if ready_code:
        (folder / "ready.py").write_text(ready_code, encoding="utf-8")
    for goal_name, code in (checks or {}).items():
        (folder / "checks" / f"{goal_name}.py").write_text(code, encoding="utf-8")
    return folder


# =================================================================================================
# bundle_has_scenarios -- the LAYOUT DECISION's presence test.
# =================================================================================================


def test_bundle_has_scenarios_false_when_no_scenarios_directory(tmp_path: Path) -> None:
    assert ss.bundle_has_scenarios(tmp_path) is False


def test_bundle_has_scenarios_false_when_scenarios_directory_is_empty(
    tmp_path: Path,
) -> None:
    (tmp_path / ss.SCENARIOS_DIRNAME).mkdir()
    assert ss.bundle_has_scenarios(tmp_path) is False


def test_bundle_has_scenarios_false_when_subdirectory_has_no_scenario_json(
    tmp_path: Path,
) -> None:
    (tmp_path / ss.SCENARIOS_DIRNAME / "s1").mkdir(parents=True)
    assert ss.bundle_has_scenarios(tmp_path) is False


def test_bundle_has_scenarios_true_with_one_valid_folder(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", scenario_key="s1", sub_goals=[])
    assert ss.bundle_has_scenarios(tmp_path) is True


# =================================================================================================
# Reader -- work item 1.
# =================================================================================================


def test_load_scenarios_reads_the_documented_on_disk_layout(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root,
        "book_a_ride",
        scenario_key="book_a_ride",
        scenario_id="platform-42",
        sub_goals=["created_rider", "judged_tone"],
        setup_code="def setup(world):\n    world.put('riders', {'id': 1})\n",
        ready_code="def ready(world):\n    return None\n",
        checks={
            "created_rider": (
                "def check(world, calls):\n"
                "    del calls\n"
                "    return None if world.state('riders').get('riders') else 'missing'\n"
            )
        },
        # "judged_tone" deliberately has no checks/ file -- a JUDGED sub-goal.
    )
    scenarios = ss.load_scenarios(tmp_path)
    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario.scenario_key == "book_a_ride"
    assert scenario.scenario_id == "platform-42"
    assert [g.name for g in scenario.sub_goals] == ["created_rider", "judged_tone"]
    deterministic, judged = scenario.sub_goals
    assert deterministic.judged == ""
    assert judged.judged != ""  # mandatory, non-empty marker (CONTRACT QUESTIONS).


def test_load_scenarios_distinguishes_conversation_only_from_tool_action_solutions(
    tmp_path: Path,
) -> None:
    (tmp_path / "contract.json").write_text(
        json.dumps({"tools": [{"name": "book_ride"}]}), encoding="utf-8"
    )
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root,
        "refusal",
        scenario_key="refusal",
        raw_body={
            "scenario_key": "refusal",
            "sub_goals": ["safe_refusal"],
            "solution": [
                {"tool": "listen", "arguments": {}},
                {"tool": "respond", "arguments": {}},
            ],
        },
    )
    _write_scenario(
        root,
        "booking",
        scenario_key="booking",
        raw_body={
            "scenario_key": "booking",
            "sub_goals": ["booked"],
            "solution": [
                {"tool": "conversation_turn", "arguments": {}},
                {"tool": "book_ride", "arguments": {"destination": "airport"}},
            ],
        },
    )

    by_key = {
        scenario.scenario_key: scenario for scenario in ss.load_scenarios(tmp_path)
    }
    assert by_key["refusal"].requires_tool_evidence is False
    assert by_key["booking"].requires_tool_evidence is True


def test_load_scenarios_does_not_invent_tool_evidence_without_declared_tools(
    tmp_path: Path,
) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root,
        "conversation",
        scenario_key="conversation",
        raw_body={
            "scenario_key": "conversation",
            "sub_goals": ["helpful"],
            "solution": [
                {"tool": "listen", "arguments": {}},
                {"tool": "respond", "arguments": {}},
            ],
        },
    )

    scenario = ss.load_scenarios(tmp_path)[0]
    assert scenario.requires_tool_evidence is False


def test_load_scenarios_sorts_by_folder_name_like_folder_py(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "bravo", scenario_key="bravo")
    _write_scenario(root, "alpha", scenario_key="alpha")
    scenarios = ss.load_scenarios(tmp_path)
    assert [s.scenario_key for s in scenarios] == ["alpha", "bravo"]


def test_load_scenarios_missing_setup_or_ready_defaults_to_empty_text(
    tmp_path: Path,
) -> None:
    # `folder.py`'s `read_folder` treats a missing setup.py/ready.py as "" -- mirrored here.
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", scenario_key="s1")
    scenario = ss.load_scenarios(tmp_path)[0]
    assert scenario.setup(object()) is None
    assert scenario.ready(object()) is None


def test_load_scenarios_raises_typed_error_for_missing_scenario_json(
    tmp_path: Path,
) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", scenario_key="s1", write_body=False)
    with pytest.raises(ss.ScenarioDocumentInvalid):
        ss.load_scenarios(tmp_path)


def test_load_scenarios_raises_typed_error_for_invalid_json(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    folder = root / "s1"
    folder.mkdir(parents=True)
    (folder / "scenario.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ss.ScenarioDocumentInvalid):
        ss.load_scenarios(tmp_path)


def test_load_scenarios_raises_typed_error_for_non_object_json(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", raw_body=None, write_body=False)
    (root / "s1" / "scenario.json").write_text(
        json.dumps(["not", "an", "object"]), encoding="utf-8"
    )
    with pytest.raises(ss.ScenarioDocumentInvalid):
        ss.load_scenarios(tmp_path)


def test_load_scenarios_raises_typed_error_for_non_string_sub_goals(
    tmp_path: Path,
) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", raw_body={"scenario_key": "s1", "sub_goals": [1, 2]})
    with pytest.raises(ss.ScenarioDocumentInvalid):
        ss.load_scenarios(tmp_path)


def test_load_scenarios_raises_typed_error_when_scenarios_directory_absent(
    tmp_path: Path,
) -> None:
    with pytest.raises(ss.ScenarioDocumentInvalid):
        ss.load_scenarios(tmp_path)


def test_load_scenarios_never_silently_skips_a_bad_folder(tmp_path: Path) -> None:
    # folder.py's own `read_all` swallows a bad folder and continues -- this module must not: a
    # good scenario alongside a broken one still fails the whole load.
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "good", scenario_key="good")
    _write_scenario(root, "bad", write_body=False)
    with pytest.raises(ss.ScenarioDocumentInvalid):
        ss.load_scenarios(tmp_path)


# =================================================================================================
# Wrapper field mapping -- work item 3. `scenario_key`/`scenario_id` carried VERBATIM, including
# empty (the empty-key fixture: hand-written, since pr63's model backfills empty keys).
# =================================================================================================


def test_wrapper_carries_empty_scenario_key_and_id_verbatim(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", scenario_key="", scenario_id="", sub_goals=[])
    scenario = ss.load_scenarios(tmp_path)[0]
    assert scenario.scenario_key == ""
    assert scenario.scenario_id == ""


def test_wrapper_preserves_sub_goal_document_order(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", scenario_key="s1", sub_goals=["z", "a", "m"])
    scenario = ss.load_scenarios(tmp_path)[0]
    assert [g.name for g in scenario.sub_goals] == ["z", "a", "m"]


def test_judged_sub_goal_check_returns_none_and_judged_is_non_empty(
    tmp_path: Path,
) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", scenario_key="s1", sub_goals=["needs_judgment"])
    scenario = ss.load_scenarios(tmp_path)[0]
    goal = scenario.sub_goals[0]
    assert goal.judged != ""
    assert (
        goal.check(object(), []) is None
    )  # "held" by the shared return-value convention.


def test_deterministic_sub_goal_judged_is_empty(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root,
        "s1",
        scenario_key="s1",
        sub_goals=["holds"],
        checks={"holds": "def check(world, calls):\n    return None\n"},
    )
    scenario = ss.load_scenarios(tmp_path)[0]
    assert scenario.sub_goals[0].judged == ""


# =================================================================================================
# Compiler -- work item 2. Good/bad, compiled once at load.
# =================================================================================================


def test_compile_empty_source_is_a_no_op_success(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", scenario_key="s1", setup_code="", ready_code="")
    scenario = ss.load_scenarios(tmp_path)[0]
    assert scenario.setup(object()) is None
    assert scenario.ready(object()) is None


def test_setup_syntax_error_fails_at_load_as_scenario_document_invalid(
    tmp_path: Path,
) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root, "s1", scenario_key="s1", setup_code="def setup(world:\n    pass\n"
    )
    with pytest.raises(ss.ScenarioDocumentInvalid, match="would not compile"):
        ss.load_scenarios(tmp_path)


def test_check_syntax_error_fails_at_load_as_scenario_document_invalid(
    tmp_path: Path,
) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root,
        "s1",
        scenario_key="s1",
        sub_goals=["broken"],
        checks={"broken": "def check(world, calls\n    pass\n"},
    )
    with pytest.raises(ss.ScenarioDocumentInvalid, match="would not compile"):
        ss.load_scenarios(tmp_path)


def test_setup_missing_entry_point_fails_at_load(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", scenario_key="s1", setup_code="x = 1\n")
    with pytest.raises(ss.ScenarioDocumentInvalid, match="defines no setup"):
        ss.load_scenarios(tmp_path)


# =================================================================================================
# R1-2 (HIGH, p12-review-r1.md) -- an EXISTING but empty/whitespace-only checks/<goal>.py must not
# become a vacuously-passing deterministic goal. Absence of the file is what means "judged"; a
# present-but-empty file is malformed, matching `_compile_entry`'s `allow_empty=False` for `check`.
# =================================================================================================


def test_existing_but_empty_check_file_is_typed_document_invalid(
    tmp_path: Path,
) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root,
        "s1",
        scenario_key="s1",
        sub_goals=["never_checked"],
        checks={"never_checked": ""},
    )
    with pytest.raises(ss.ScenarioDocumentInvalid, match="defines no check"):
        ss.load_scenarios(tmp_path)


def test_existing_but_whitespace_only_check_file_is_typed_document_invalid(
    tmp_path: Path,
) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root,
        "s1",
        scenario_key="s1",
        sub_goals=["never_checked"],
        checks={"never_checked": "   \n\t\n"},
    )
    with pytest.raises(ss.ScenarioDocumentInvalid, match="defines no check"):
        ss.load_scenarios(tmp_path)


def test_setup_and_ready_still_allow_empty_source_after_the_r1_2_fix(
    tmp_path: Path,
) -> None:
    # Regression guard: R1-2's `allow_empty=False` is scoped to `check` only -- setup/ready must
    # still treat empty/whitespace source as the pre-existing no-op success (`folder.py` parity).
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", scenario_key="s1", setup_code="   \n", ready_code="")
    scenario = ss.load_scenarios(tmp_path)[0]
    assert scenario.setup(object()) is None
    assert scenario.ready(object()) is None


def test_mutation_vacuous_empty_check_pass_is_caught(tmp_path: Path) -> None:
    # Mutation table (R1-2): simulates deleting `allow_empty=False` from the `check` call site in
    # `_load_one`, i.e. reverting to the pre-fix behavior where an existing-but-empty check file
    # silently compiled to a no-op "held" callable -- a vacuous deterministic pass.
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root,
        "s1",
        scenario_key="s1",
        sub_goals=["never_checked"],
        checks={"never_checked": ""},
    )

    # Baseline: the real fix catches it.
    with pytest.raises(ss.ScenarioDocumentInvalid, match="defines no check"):
        ss.load_scenarios(tmp_path)

    def _always_allow_empty(
        source: str,
        *,
        label: str,
        entry: str,
        allow_empty: bool = True,
        defer_execution: bool = False,
    ):
        # The mutant: `allow_empty` is accepted but ignored -- `check` is treated exactly like
        # `setup`/`ready` again, as if the R1-2 fix's `allow_empty=False` call-site edit were
        # reverted. A standalone reimplementation of the pre-fix `_compile_entry` body, not a call
        # back into `ss._compile_entry` (which IS this mutant while the patch is active).
        del allow_empty
        if not source.strip():
            return lambda *args: None
        code = compile(source, f"<{label}>", "exec")
        namespace: dict[str, Any] = {}
        exec(code, namespace)  # noqa: S102
        function = namespace.get(entry)
        if not callable(function):
            raise ss.ScenarioDocumentInvalid(f"{label} defines no {entry}()")
        return function

    with mock.patch.object(ss, "_compile_entry", _always_allow_empty):
        scenarios = ss.load_scenarios(tmp_path)  # mutant: no longer raises
        goal = scenarios[0].sub_goals[0]
        assert goal.judged == ""  # still classified deterministic...
        assert (
            goal.check(object(), []) is None
        )  # ...and the mutant's vacuous "held" verdict

    # Restored: the guard is back.
    with pytest.raises(ss.ScenarioDocumentInvalid, match="defines no check"):
        ss.load_scenarios(tmp_path)


# =================================================================================================
# R1-3 (MEDIUM, p12-review-r1.md) -- `sub_goals[]` names are used verbatim to build
# `checks/<name>.py`; an absolute or traversal-shaped name must never resolve to a file outside
# `checks/` (bypassing the bundle's own integrity seal) or silently reclassify a deterministic goal
# as judged (a nonexistent path just fails `is_file()`).
# =================================================================================================


def test_absolute_path_subgoal_name_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text(
        "def check(world, calls):\n    return 'should never run'\n", encoding="utf-8"
    )
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root, "s1", scenario_key="s1", sub_goals=[str(outside.with_suffix(""))]
    )
    with pytest.raises(ss.ScenarioDocumentInvalid, match="not a plain filename"):
        ss.load_scenarios(tmp_path)


def test_relative_traversal_subgoal_name_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", scenario_key="s1", sub_goals=["../../../../etc/passwd"])
    with pytest.raises(ss.ScenarioDocumentInvalid, match="not a plain filename"):
        ss.load_scenarios(tmp_path)


def test_backslash_subgoal_name_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", scenario_key="s1", sub_goals=["a\\b"])
    with pytest.raises(ss.ScenarioDocumentInvalid, match="not a plain filename"):
        ss.load_scenarios(tmp_path)


def test_plain_subgoal_names_are_unaffected_by_the_r1_3_fix(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root,
        "s1",
        scenario_key="s1",
        sub_goals=["holds", "judged_one"],
        checks={"holds": "def check(world, calls):\n    return None\n"},
    )
    scenario = ss.load_scenarios(tmp_path)[0]
    assert [g.name for g in scenario.sub_goals] == ["holds", "judged_one"]


def test_mutation_subgoal_name_sanitization_reproduces_the_bundle_escape(
    tmp_path: Path,
) -> None:
    # Mutation table (R1-3): simulates deleting the `_validate_subgoal_name` call in `_load_one` --
    # an absolute-path sub_goal name would once again resolve `check_path` to a file entirely
    # outside `checks/` (and outside the sealed bundle), reading and compiling it.
    outside = tmp_path / "outside.py"
    outside.write_text(
        "def check(world, calls):\n    return 'ran from outside the bundle'\n",
        encoding="utf-8",
    )
    root = tmp_path / ss.SCENARIOS_DIRNAME
    escaping_name = str(outside.with_suffix(""))
    _write_scenario(root, "s1", scenario_key="s1", sub_goals=[escaping_name])

    with pytest.raises(ss.ScenarioDocumentInvalid, match="not a plain filename"):
        ss.load_scenarios(tmp_path)

    with mock.patch.object(
        ss, "_validate_subgoal_name", lambda name, *, folder_name: None
    ):
        scenarios = ss.load_scenarios(tmp_path)  # mutant: no longer raises
        goal = scenarios[0].sub_goals[0]
        assert (
            goal.check(object(), []) == "ran from outside the bundle"
        )  # read from OUTSIDE checks/

    with pytest.raises(ss.ScenarioDocumentInvalid, match="not a plain filename"):
        ss.load_scenarios(tmp_path)


# =================================================================================================
# R1-1 (CRITICAL, p12-review-r1.md) -- untyped/`BaseException` failures in the load path must never
# bypass the one `except ScenarioDocumentInvalid` clause in `hosted_entrypoint.py::run_job`: a
# scenario document must not be able to set the guest's own exit code (module-level `sys.exit()`)
# or vanish with zero terminal events (an unreadable file, a non-UTF-8 file). Reproduced here at
# the reader/compiler level, directly against `ss.load_scenarios` -- the e2e proof (through the
# REAL `run_job`, asserting the exact exit code and the terminal event) lives in
# `test_hosted_entrypoint.py` for the two triggers that survive `process_preflight.py`'s own
# byte-for-byte digest re-verification (module-level `sys.exit()`, non-UTF-8 bytes -- both hash
# fine as raw bytes). The unreadable-file (chmod 000) trigger is NOT reproduced through the full
# bundle/preflight pipeline: `process_preflight.py::_verify_digest` (a file outside this task's
# four-edit allowlist) re-opens every listed file with a bare `path.open("rb")`, unguarded, and
# would raise its own untyped `PermissionError` before `scenario_source.build()` is ever reached --
# an orthogonal, pre-existing gap in a module this task cannot touch. Reproduced here instead,
# directly at the boundary this task owns.
# =================================================================================================


def test_setup_module_level_sys_exit_zero_is_typed_document_invalid(
    tmp_path: Path,
) -> None:
    # The worst case in the finding: `sys.exit(0)` inside scenario code, unguarded, previously
    # meant the GUEST process itself exited 0 -- a "clean terminal that never happened" (§0.6).
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root, "s1", scenario_key="s1", setup_code="import sys\nsys.exit(0)\n"
    )
    with pytest.raises(ss.ScenarioDocumentInvalid, match="would not compile"):
        ss.load_scenarios(tmp_path)


def test_setup_module_level_sys_exit_three_is_typed_document_invalid(
    tmp_path: Path,
) -> None:
    # EXIT_FENCED == 3: previously the guest would exit 3, read by the platform as a fenced/
    # superseded attempt rather than a scenario content defect.
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root, "s1", scenario_key="s1", setup_code="import sys\nsys.exit(3)\n"
    )
    with pytest.raises(ss.ScenarioDocumentInvalid, match="would not compile"):
        ss.load_scenarios(tmp_path)


def test_check_module_level_bare_sys_exit_is_typed_document_invalid(
    tmp_path: Path,
) -> None:
    # Bare `sys.exit()` (no argument) is `SystemExit()`, not `SystemExit(int)` -- still a
    # `BaseException`, not an `Exception`; a `checks/<goal>.py` file is exactly where a generator
    # could emit this by omitting the `if __name__ == "__main__":` guard around `folder.py`'s own
    # `_RUNNABLE` tail (R1-4).
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root,
        "s1",
        scenario_key="s1",
        sub_goals=["broken"],
        checks={"broken": "import sys\nsys.exit()\n"},
    )
    with pytest.raises(ss.ScenarioDocumentInvalid, match="would not compile"):
        ss.load_scenarios(tmp_path)


def test_setup_unreadable_file_chmod_000_is_typed_document_invalid(
    tmp_path: Path,
) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    folder = _write_scenario(
        root, "s1", scenario_key="s1", setup_code="def setup(world):\n    pass\n"
    )
    setup_path = folder / "setup.py"
    setup_path.chmod(0o000)
    try:
        with pytest.raises(ss.ScenarioDocumentInvalid, match="cannot read"):
            ss.load_scenarios(tmp_path)
    finally:
        setup_path.chmod(0o644)  # restore so pytest's tmp_path cleanup can remove it


def test_setup_non_utf8_bytes_is_typed_document_invalid(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    folder = _write_scenario(root, "s1", scenario_key="s1")
    (folder / "setup.py").write_bytes(b"def setup(world):\n    return '\xff\xfe'\n")
    with pytest.raises(ss.ScenarioDocumentInvalid, match="cannot read"):
        ss.load_scenarios(tmp_path)


def test_scenario_json_non_utf8_bytes_is_typed_document_invalid(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    folder = root / "s1"
    folder.mkdir(parents=True)
    (folder / "scenario.json").write_bytes(b"\xff\xfe not valid utf-8 at all")
    with pytest.raises(ss.ScenarioDocumentInvalid, match="cannot read"):
        ss.load_scenarios(tmp_path)


def test_unreadable_scenarios_directory_is_typed_document_invalid_not_an_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    root.mkdir(parents=True)
    (root / "s1").mkdir()
    (root / "s1" / "scenario.json").write_text(
        json.dumps({"scenario_key": "s1"}), encoding="utf-8"
    )
    root.chmod(0o000)
    try:
        with pytest.raises(ss.ScenarioDocumentInvalid, match="cannot list"):
            ss.load_scenarios(tmp_path)
    finally:
        root.chmod(0o755)  # restore so pytest's tmp_path cleanup can remove it


def test_unreadable_scenarios_directory_makes_bundle_has_scenarios_false_not_an_escape(
    tmp_path: Path,
) -> None:
    # `bundle_has_scenarios` sits OUTSIDE `run_job`'s try/except entirely (R1-1) -- an unreadable
    # `scenarios/` must report False (falling back to the safe `NotWiredScenarioSource` default),
    # never raise, since there is nothing typed to catch it at that call site.
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", scenario_key="s1")
    root.chmod(0o000)
    try:
        assert ss.bundle_has_scenarios(tmp_path) is False
    finally:
        root.chmod(0o755)


# =================================================================================================
# Security-shaped (documentation, not enforcement) -- work item 5d. The precedent (`folder.py`'s
# `_run`) restricts nothing: a bare `{}` namespace with full default builtins still reaches `os`,
# `open`, `subprocess`. Pinned here rather than assumed. CONTRACT QUESTIONS: scenario code in the
# hosted guest is unsandboxed beyond the sandbox itself -- no sandbox is added by this module.
# =================================================================================================


def test_scenario_code_is_unsandboxed_importing_os_runs_successfully(
    tmp_path: Path,
) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root,
        "s1",
        scenario_key="s1",
        setup_code=(
            "import os\n"
            "def setup(world):\n"
            "    del world\n"
            "    return os.getcwd() and None\n"
        ),
    )
    scenario = ss.load_scenarios(tmp_path)[0]
    assert (
        scenario.setup(object()) is None
    )  # ran to completion -- `import os` was never blocked.


# =================================================================================================
# Consumer proof (work item 5b) -- real documents -> this adapter -> the REAL HostedScheduler, with
# a fake world/call runner. `WorldPool` needs no real `EnvironmentBundleV2`/provisioner fidelity
# for this (`bundle=object()`, matching test_hosted_scheduler.py's own `_pool` helper) -- only
# `provision()`/`reset()`/`healthy()`/`close()` are ever called on it.
# =================================================================================================


class _FakeProvisioner:
    name = "fake-process"

    def __init__(self, instances: int) -> None:
        self.instances = instances
        self.closed = False
        self._runtimes = {
            i: EnvironmentRuntime(
                runtime_id=f"digest:w{i}",
                world_index=i,
                bundle_digest="digest",
                state=RuntimeState.READY,
                endpoints={},
            )
            for i in range(instances)
        }

    async def provision(
        self, bundle, *, source, bundle_dir, work_directory, contract=None, instances=1
    ):
        del bundle, source, bundle_dir, work_directory, contract
        return [self._runtimes[i] for i in range(instances)]

    async def reset(self, runtime: EnvironmentRuntime, *, work_directory: Path) -> None:
        del runtime, work_directory

    async def healthy(
        self, runtime: EnvironmentRuntime, *, work_directory: Path
    ) -> bool:
        del runtime, work_directory
        return True

    async def close(self, *, work_directory: Path) -> None:
        del work_directory
        self.closed = True


class _FakeWorld:
    """Unlike `test_hosted_scheduler.py`'s `InMemoryWorld`, `read_only()` SHARES the `rows` dict
    rather than starting a fresh one -- this consumer-proof test needs a `check()` to see what
    `setup()` actually wrote, or the deterministic-check assertion below would pass vacuously."""

    def __init__(
        self,
        world_index: int,
        rng: Any,
        rows: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.world_index = world_index
        self.rng = rng
        self.rows: dict[str, list[dict[str, Any]]] = rows if rows is not None else {}

    def state(self, table: str | None = None) -> dict[str, list[dict[str, Any]]]:
        return (
            dict(self.rows)
            if table is None
            else {table: list(self.rows.get(table, []))}
        )

    def put(
        self, collection: str, record: dict[str, Any], *, key: str = ""
    ) -> dict[str, Any]:
        self.rows.setdefault(collection, []).append(record)
        return record

    def change(
        self, collection: str, key: str, changes: dict[str, Any], *, by: str = ""
    ) -> int:
        del key, changes, by
        return 0

    def drop(self, collection: str, key: str = "", *, by: str = "") -> int:
        del key, by
        return 0

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Call:
        raise NotImplementedError

    def query(self, sql: str, params: Any = ()) -> list[dict[str, Any]]:
        del sql, params
        return []

    def read_only(self) -> "_FakeWorld":
        return _FakeWorld(self.world_index, self.rng, rows=self.rows)


class _FakeWorldFactory:
    async def create(
        self, runtime: EnvironmentRuntime, *, rng: random.Random
    ) -> _FakeWorld:
        return _FakeWorld(runtime.world_index, rng)


class _FakeCallRunner:
    def __init__(self, outcomes: dict[str, CallOutcome | Exception]) -> None:
        self._outcomes = outcomes

    async def run(self, scenario: Any, runtime: EnvironmentRuntime) -> CallOutcome:
        del runtime
        outcome = self._outcomes[scenario.scenario_key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class _FakeOutbound:
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    receipts: list[Any] = field(default_factory=list)

    async def scenario_started(
        self, *, scenario_key: str, world_index: int, scenario_attempt: int
    ) -> None:
        self.events.append(("scenario_started", {"scenario_key": scenario_key}))

    async def scenario_retried(
        self, *, scenario_key: str, from_world: int, to_world: int, cause: str = ""
    ) -> None:
        self.events.append(("scenario_retried", {"cause": cause}))

    async def world_unhealthy(self, *, world_index: int, cause: str) -> None:
        self.events.append(("world_unhealthy", {"cause": cause}))

    async def log(self, *, level: str, message: str) -> None:
        self.events.append(("log", {"message": message}))

    async def receipt(self, receipt: Any) -> None:
        self.receipts.append(receipt)


def _call_outcome() -> CallOutcome:
    return CallOutcome(
        calls=(
            Call(
                name="tool",
                arguments={},
                result="ok",
                ok=True,
                error="",
                refused=False,
                at=0.0,
            ),
        ),
        turns=1,
        started_at="2026-08-25T00:00:00.000Z",
        ended_at="2026-08-25T00:00:01.000Z",
        duration_ms=1000,
    )


def _build_fixture_bundle(tmp_path: Path) -> Path:
    """Two scenarios: one whose deterministic check genuinely holds against what `setup` wrote,
    one whose deterministic check genuinely does not -- proving verdicts are evaluated for real,
    not just that a key made it through (the brief's own "key-only assertion cannot detect a
    wrapper bug" warning). Each also carries one judged sub-goal."""
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root,
        "passing",
        scenario_key="passing",
        scenario_id="",
        sub_goals=["created_rider", "needs_judgment"],
        setup_code="def setup(world):\n    world.put('riders', {'id': 1})\n",
        ready_code="def ready(world):\n    return None\n",
        checks={
            "created_rider": (
                "def check(world, calls):\n"
                "    del calls\n"
                "    return None if world.state('riders').get('riders') else 'missing rider'\n"
            )
        },
    )
    _write_scenario(
        root,
        "failing",
        scenario_key="failing",
        scenario_id="",
        sub_goals=["created_rider"],
        setup_code="def setup(world):\n    return None\n",  # never creates the rider
        ready_code="def ready(world):\n    return None\n",
        checks={
            "created_rider": (
                "def check(world, calls):\n"
                "    del calls\n"
                "    return None if world.state('riders').get('riders') else 'missing rider'\n"
            )
        },
    )
    return tmp_path


def test_consumer_proof_real_scheduler_evaluates_wrapped_scenarios() -> None:
    async def scenario() -> None:
        tmp_path = Path(tempfile.mkdtemp(prefix="p12-consumer-"))
        bundle_dir = _build_fixture_bundle(tmp_path)
        scenarios = ss.load_scenarios(bundle_dir)
        assert [s.scenario_key for s in scenarios] == [
            "failing",
            "passing",
        ]  # sorted by folder name

        outbound = _FakeOutbound()
        provisioner = _FakeProvisioner(1)
        pool = WorldPool(
            provisioner,
            bundle=object(),
            source=Path("/work/source"),
            bundle_dir=bundle_dir,
            work_directory=tmp_path,
            instances=1,
            outbound=outbound,
        )
        await pool.start()
        call_runner = _FakeCallRunner(
            {"passing": _call_outcome(), "failing": _call_outcome()}
        )
        scheduler = HostedScheduler(
            pool=pool,
            world_factory=_FakeWorldFactory(),
            call_runner=call_runner,
            outbound=outbound,
            job_seed=1,
        )
        result = await asyncio.wait_for(scheduler.run(scenarios), timeout=10.0)
        await pool.close()

        assert result.aborted is None
        receipts_by_key = {r.scenario_key: r for r in result.receipts}
        assert set(receipts_by_key) == {"passing", "failing"}

        passing = receipts_by_key["passing"]
        assert passing.status == "passed"
        goals_by_name = {g.name: g for g in passing.sub_goals}
        assert (
            goals_by_name["created_rider"].held is True
        )  # a real deterministic check, evaluated
        assert goals_by_name["created_rider"].judged is False
        assert (
            goals_by_name["needs_judgment"].held is True
        )  # placeholder "held" convention
        assert (
            goals_by_name["needs_judgment"].judged is True
        )  # marks it as not really settled yet

        failing = receipts_by_key["failing"]
        assert (
            failing.status == "failed"
        )  # the SAME check, genuinely evaluated, genuinely fails
        assert failing.sub_goals[0].held is False
        assert failing.sub_goals[0].reason == "missing rider"

    asyncio.run(scenario())


def test_runtime_error_inside_setup_reaches_setup_crashed_through_the_real_scheduler() -> (
    None
):
    # Divergence (a) from folder.py's `_run`: a syntax error is caught at LOAD time. A RUNTIME
    # error inside a setup() that compiles fine is a different story -- it must reach
    # `hosted_scheduler.py`'s own classification (`_run_phase`'s `_PhaseCrashed` ->
    # `setup_crashed`), proven here through the real scheduler, not a fake standing in for it.
    async def scenario() -> None:
        tmp_path = Path(tempfile.mkdtemp(prefix="p12-setup-crash-"))
        root = tmp_path / ss.SCENARIOS_DIRNAME
        _write_scenario(
            root,
            "boom",
            scenario_key="boom",
            scenario_id="",
            sub_goals=[],
            setup_code="def setup(world):\n    raise ValueError('setup blew up')\n",
        )
        scenarios = ss.load_scenarios(tmp_path)

        outbound = _FakeOutbound()
        provisioner = _FakeProvisioner(1)
        pool = WorldPool(
            provisioner,
            bundle=object(),
            source=Path("/work/source"),
            bundle_dir=tmp_path,
            work_directory=tmp_path,
            instances=1,
            outbound=outbound,
        )
        await pool.start()
        scheduler = HostedScheduler(
            pool=pool,
            world_factory=_FakeWorldFactory(),
            call_runner=_FakeCallRunner({}),
            outbound=outbound,
            job_seed=1,
        )
        result = await asyncio.wait_for(scheduler.run(scenarios), timeout=10.0)
        await pool.close()

        assert len(result.receipts) == 1
        receipt = result.receipts[0]
        assert receipt.status == "errored"
        assert receipt.failure is not None
        assert receipt.failure.code == "setup_crashed"
        assert "setup blew up" in receipt.failure.message

    asyncio.run(scenario())


# =================================================================================================
# R1-4 (MEDIUM, p12-review-r1.md) -- nothing else in this suite feeds the adapter an actual
# `folder.py::write_folder` product; every other fixture here is hand-written. This pins the reader
# against the REAL producer once: a future change to `write_folder`'s layout (e.g. setting
# `namespace["__name__"]` inside `_compile_entry` to make tracebacks readable) would turn every real
# check file's `_RUNNABLE` tail (which every one of them carries) into a module-level `SystemExit`
# at load -- exactly R1-1's worst case -- and an all-hand-written suite would never see it.
# =================================================================================================


def test_real_write_folder_round_trip_matches_the_adapters_reading(
    tmp_path: Path,
) -> None:
    from fi.alk.harness import folder as fmod
    from fi.alk.harness.catalogue import Catalogue, SubGoal
    from fi.alk.harness.scenario import Scenario

    catalogue = Catalogue(
        sub_goals=[
            SubGoal(
                name="created_rider",
                what="rider row exists",
                check=(
                    "def check(world, calls):\n"
                    "    del calls\n"
                    "    return None if world.state('riders').get('riders') else 'missing rider'\n"
                ),
            ),
            SubGoal(
                name="polite_tone",
                what="agent was polite",
                judged="was the refusal explained?",
            ),
        ]
    )
    scenario_model = Scenario(
        name="book_a_ride",
        instruction="book a ride",
        sub_goals=["created_rider", "polite_tone"],
        setup_code="def setup(world):\n    world.put('riders', {'id': 1})\n",
        ready_code="def ready(world):\n    return None\n",
    )
    fmod.write_folder(scenario_model, catalogue, tmp_path)

    # The real producer's own on-disk layout must satisfy `bundle_has_scenarios`'s presence test
    # unchanged.
    assert ss.bundle_has_scenarios(tmp_path) is True

    scenarios = ss.load_scenarios(tmp_path)
    assert len(scenarios) == 1
    scenario = scenarios[0]
    goals_by_name = {g.name: g for g in scenario.sub_goals}
    assert set(goals_by_name) == {"created_rider", "polite_tone"}
    assert goals_by_name["created_rider"].judged == ""  # has a real checks/ file
    assert (
        goals_by_name["polite_tone"].judged != ""
    )  # no checks/ file -- judged, per deterministic()

    world = _FakeWorld(0, random.Random(0))
    assert scenario.setup(world) is None
    assert world.rows == {"riders": [{"id": 1}]}
    assert scenario.ready(world) is None
    # The real producer appends `_RUNNABLE` (ending `if __name__ == "__main__": ... raise
    # SystemExit(...)`) to every checks/ file it writes -- this proves `exec(code, {})` resolves
    # `__name__` to `'builtins'` (never `'__main__'`), so that tail stays inert, against the REAL
    # producer's own bytes rather than a hand-written stand-in that never carries the tail at all.
    assert (
        goals_by_name["created_rider"].check(world, []) is None
    )  # held, genuinely evaluated
    assert (
        goals_by_name["polite_tone"].check(world, []) is None
    )  # judged placeholder convention


# =================================================================================================
# R1-5 (MEDIUM, p12-review-r1.md) -- divergence (a) (compile once, at load) moves every scenario
# file's compile+exec OUTSIDE every phase budget the scheduler enforces. A wall-clock budget around
# the load converts a hang into a typed terminal instead of an unbounded one (a worker thread
# cannot actually be canceled, so the background thread is accepted as leaked -- see
# `_LOAD_TIMEOUT_SECONDS`'s docstring).
# =================================================================================================


def test_hosted_load_never_executes_module_level_scenario_code() -> None:
    async def scenario() -> None:
        tmp_path = Path(tempfile.mkdtemp(prefix="p12-load-timeout-"))
        root = tmp_path / ss.SCENARIOS_DIRNAME
        _write_scenario(
            root,
            "s1",
            scenario_key="s1",
            # Module-level, not inside setup() -- runs during `_compile_entry`'s `exec`, i.e.
            # during the load itself, which is exactly what a real budget must bound.
            setup_code="raise RuntimeError('module executed in loader')\ndef setup(world):\n    pass\n",
        )
        source = ss.BundleScenarioSource()

        async def passthrough(client, scenarios, **kwargs):
            return scenarios

        with mock.patch.object(ss, "register_with_platform", passthrough):
            loaded = await source.build(
                _FakeJob(run_id="job-1"),
                object(),
                object(),
                pool=object(),
                world_factory=object(),
                bundle_dir=tmp_path,
            )
        assert [item.scenario_key for item in loaded] == ["s1"]

    asyncio.run(scenario())


def test_load_without_a_hang_is_unaffected_by_the_budget(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / ss.SCENARIOS_DIRNAME
        _write_scenario(root, "s1", scenario_key="s1", sub_goals=[])
        source = ss.BundleScenarioSource()

        # p13: `build()` now calls `register_with_platform` after load -- this test is about the
        # R1-5 timeout BUDGET specifically, not registration, so registration is stubbed to a
        # passthrough (registration's own behavior is covered separately, below).
        async def _passthrough(scenarios_client, scenarios, *, run_name, **rest):
            del scenarios_client, run_name, rest
            return scenarios

        with mock.patch.object(ss, "register_with_platform", _passthrough):
            scenarios = await source.build(
                _FakeJob(run_id="job-1"),
                object(),
                object(),
                pool=object(),
                world_factory=object(),
                bundle_dir=tmp_path,
            )
        assert [s.scenario_key for s in scenarios] == ["s1"]

    asyncio.run(scenario())


# =================================================================================================
# Mutations (work item 6) -- proved via in-memory monkeypatching (`unittest.mock.patch.object`)
# rather than editing bytes on any tracked file on disk: this module (`scenario_source.py`) is a
# brand-new, untracked file at the start of this work, so there is no tracked-file shasum to record
# for these -- the git-recovery safety net the brief's mutation section is guarding against
# (touching a TRACKED file's bytes with no git-based way back) does not apply to a file this task
# itself created. Each mutation patches one function/attribute, runs the assertion that should now
# fail (or the behavior that should now differ), and restores in a `finally` -- `mock.patch.object`
# does this atomically and cannot leave the module in a mutated state even if the assertion raises.
# =================================================================================================


def test_mutation_skip_compile_check_is_killed(tmp_path: Path) -> None:
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(
        root, "s1", scenario_key="s1", setup_code="def setup(world:\n    pass\n"
    )

    # Baseline: the real compiler catches the syntax error.
    with pytest.raises(ss.ScenarioDocumentInvalid):
        ss.load_scenarios(tmp_path)

    # Mutant: a "compiler" that never raises on a bad compile, returning a no-op instead --
    # simulates deleting the try/except around `compile()`/`exec()` in `_compile_entry`.
    def _never_fails(
        source: str, *, label: str, entry: str, defer_execution: bool = False
    ):
        del source, label, entry
        return lambda *args: None

    with mock.patch.object(ss, "_compile_entry", _never_fails):
        scenarios = ss.load_scenarios(tmp_path)  # mutant: no longer raises
        assert (
            scenarios[0].setup(object()) is None
        )  # confirms the mutant path actually ran

    # Restored: the guard is back.
    with pytest.raises(ss.ScenarioDocumentInvalid):
        ss.load_scenarios(tmp_path)


def test_mutation_judged_flag_dropped_is_killed_by_5b_verdict_assertions() -> None:
    # Mutant: `_load_one` always sets `judged=""`, as if the "no checks/<name>.py -> judged"
    # branch were deleted.
    #
    # R1-6/R1-8 fold-in (p12-review-r1.md LOW findings): the original version of this test named
    # the 5b consumer-proof test in its own identifier but never actually ran it -- it asserted on
    # `_load_one`'s output directly instead, which is true but does not exercise the killing
    # assertion it claims to. This version runs the REAL
    # `test_consumer_proof_real_scheduler_evaluates_wrapped_scenarios` under the patch and checks
    # THAT it raises. No `async def`/`asyncio.run` wrapper needed either (R1-8): that test already
    # drives its own `asyncio.run` internally, and nothing here awaits anything.
    original_load_one = ss._load_one

    def _dropped_judged(folder: Path, **_read_as_the_real_one_is):
        compiled = original_load_one(folder)
        broken_goals = tuple(
            ss._CompiledSubGoal(name=g.name, judged="", check=g.check)
            for g in compiled.sub_goals
        )
        return ss._CompiledScenario(
            scenario_key=compiled.scenario_key,
            scenario_id=compiled.scenario_id,
            sub_goals=broken_goals,
            setup=compiled.setup,
            ready=compiled.ready,
        )

    with mock.patch.object(ss, "_load_one", _dropped_judged):
        with pytest.raises(AssertionError):
            test_consumer_proof_real_scheduler_evaluates_wrapped_scenarios()

    # Restored: the real 5b test passes again.
    test_consumer_proof_real_scheduler_evaluates_wrapped_scenarios()


def test_mutation_empty_key_reader_synthesizing_a_key_is_caught(tmp_path: Path) -> None:
    # Mutant: the reader backfills an empty `scenario_key` with a placeholder instead of carrying
    # it verbatim, as pr63's own model validator would. The empty-key fixture's own guarantee
    # (verbatim empty carry) must catch this.
    root = tmp_path / ss.SCENARIOS_DIRNAME
    _write_scenario(root, "s1", scenario_key="", scenario_id="", sub_goals=[])

    real = ss.load_scenarios(tmp_path)[0]
    assert real.scenario_key == ""  # baseline: verbatim carry holds

    original_load_one = ss._load_one

    def _synthesizes_key(folder: Path, **_read_as_the_real_one_is):
        compiled = original_load_one(folder)
        key = compiled.scenario_key or "synthesized-key"
        return ss._CompiledScenario(
            scenario_key=key,
            scenario_id=compiled.scenario_id,
            sub_goals=compiled.sub_goals,
            setup=compiled.setup,
            ready=compiled.ready,
        )

    with mock.patch.object(ss, "_load_one", _synthesizes_key):
        mutant = ss.load_scenarios(tmp_path)[0]
        assert mutant.scenario_key != ""  # the mutant's defect: no longer verbatim

    restored = ss.load_scenarios(tmp_path)[0]
    assert restored.scenario_key == ""  # confirms the patch was fully undone


# =================================================================================================
# p13 -- scenario pre-allocation (`register_with_platform`), wired against the platform's actual
# route (a single `POST .../scenarios/`, discriminated by a body-level `operation` field, keyed
# provision response, full-set `begin`) rather than the documented two-path/position-ordered
# shape -- see `ScenariosClient`'s and `register_with_platform`'s own docstrings for the file:line
# evidence, and reports/p13-worker-r2.md CONTRACT NOTES for where the two disagree.
# =================================================================================================


def _scenario(key: str, *, scenario_id: str = "") -> ss._CompiledScenario:
    return ss._CompiledScenario(
        scenario_key=key,
        scenario_id=scenario_id,
        sub_goals=(),
        setup=lambda w: None,
        ready=lambda w: None,
    )


@dataclass
class _FakeScenariosClient:
    """Stands in for `hosted_entrypoint.ScenariosClient` -- records every payload it is called
    with (so a test can assert on exactly what `register_with_platform` sends) and returns a
    canned `{"result": {...}}`-unwrapped body per call, matching what the real
    `ScenariosClient._post` already hands back (the envelope itself is that class's concern, not
    this module's -- see `test_scenarios_client_provision_unwraps_the_result_envelope` in
    `test_hosted_entrypoint.py`)."""

    provision_response: dict[str, Any]
    begin_response: dict[str, Any] = field(
        default_factory=lambda: {"test_execution_id": "exec-1", "scenarios": []}
    )
    provision_error: Exception | None = None
    begin_error: Exception | None = None
    provision_calls: list[dict[str, Any]] = field(default_factory=list)
    begin_calls: list[dict[str, Any]] = field(default_factory=list)

    def provision(
        self, payload: dict[str, Any], *, deadline: float | None = None
    ) -> dict[str, Any]:
        del deadline
        self.provision_calls.append(payload)
        if self.provision_error is not None:
            raise self.provision_error
        return self.provision_response

    def begin(
        self, payload: dict[str, Any], *, deadline: float | None = None
    ) -> dict[str, Any]:
        del deadline
        self.begin_calls.append(payload)
        if self.begin_error is not None:
            raise self.begin_error
        return self.begin_response


# -------------------------------------------------------------------------------------------------
# `BundleScenarioSource.build` wiring -- registration runs after load, before the scheduler; a
# LOCALLY-detectable defect (empty scenario_key) must never reach the network first.
# -------------------------------------------------------------------------------------------------


def test_build_rejects_empty_scenario_key_before_calling_register_with_platform(
    tmp_path: Path,
) -> None:
    # An empty scenario_key is carried verbatim off the document (this module's own LAYOUT
    # DECISION -- never synthesized), but `hosted_entrypoint.py`'s own `_validate_scenarios` would
    # reject it downstream as a deterministic, `environment`-domain content defect. Checked here,
    # before `register_with_platform` is ever called, so that existing classification wins over a
    # round trip that would only rediscover the same defect as a `platform_sync` failure instead.
    async def scenario() -> None:
        root = tmp_path / ss.SCENARIOS_DIRNAME
        _write_scenario(root, "s1", scenario_key="", sub_goals=[])
        source = ss.BundleScenarioSource()

        register_calls: list[Any] = []

        async def _spy(scenarios_client, scenarios, *, run_name):
            register_calls.append((scenarios_client, scenarios, run_name))
            return scenarios

        with mock.patch.object(ss, "register_with_platform", _spy):
            with pytest.raises(ss.ScenarioDocumentInvalid, match="scenario_key"):
                await source.build(
                    _FakeJob(),
                    object(),
                    object(),
                    pool=object(),
                    world_factory=object(),
                    bundle_dir=tmp_path,
                )
        assert register_calls == []  # never reached the network

    asyncio.run(scenario())


# -------------------------------------------------------------------------------------------------
# Payload shape -- quote-driven against Azain's serializers (file:line in each docstring/comment).
# -------------------------------------------------------------------------------------------------


def test_provision_payload_only_sends_fields_azains_serializer_declares() -> None:
    # HarnessProvisionPersonaSerializer (futureagi/simulate/serializers/hosted_harness.py:168-174):
    # scenario_key is the only REQUIRED field; name/role/situation/outcome/persona are all
    # `required=False`. `_CompiledScenario` carries none of the optional ones (this module's own
    # LAYOUT DECISION -- see the module docstring), so the payload must send bare scenario_key
    # only, never a synthesized value for a field this reader does not have.
    scenarios = (_scenario("book_a_ride"), _scenario("cancel_ride"))
    payload = ss._provision_payload("run-1", scenarios)
    assert payload == {
        "operation": "provision",
        "name": "run-1",
        "personas": [{"scenario_key": "book_a_ride"}, {"scenario_key": "cancel_ride"}],
    }
    for persona in payload["personas"]:
        assert set(persona) == {"scenario_key"}




# -------------------------------------------------------------------------------------------------
# Keyed-response parsing (`_scenario_ids_by_key`) -- dict lookup by scenario_key, never a
# positional zip; every malformed/mismatched shape raises rather than returning a partial mapping.
# -------------------------------------------------------------------------------------------------


def test_scenario_ids_by_key_matches_regardless_of_response_order() -> None:
    # The platform's response order is not guaranteed to match submission order -- this is the
    # load-bearing case a positional zip would get wrong (mutation-killed below).
    submitted = (_scenario("a"), _scenario("b"))
    reordered_response = [
        {"scenario_key": "b", "scenario_id": "platform-b"},
        {"scenario_key": "a", "scenario_id": "platform-a"},
    ]
    assert ss._scenario_ids_by_key(submitted, reordered_response) == {
        "a": "platform-a",
        "b": "platform-b",
    }


def test_scenario_ids_by_key_raises_typed_error_for_non_list_response() -> None:
    with pytest.raises(ScenarioPreallocationError) as exc_info:
        ss._scenario_ids_by_key((_scenario("a"),), {"not": "a list"})
    assert exc_info.value.error.code == "scenarios_provision_response_invalid"


def test_scenario_ids_by_key_raises_typed_error_for_a_non_object_entry() -> None:
    with pytest.raises(ScenarioPreallocationError) as exc_info:
        ss._scenario_ids_by_key((_scenario("a"),), ["not an object"])
    assert exc_info.value.error.code == "scenarios_provision_response_invalid"


def test_scenario_ids_by_key_raises_typed_error_for_unknown_key() -> None:
    submitted = (_scenario("a"),)
    raw = [
        {"scenario_key": "a", "scenario_id": "platform-a"},
        {"scenario_key": "never-submitted", "scenario_id": "platform-x"},
    ]
    with pytest.raises(ScenarioPreallocationError) as exc_info:
        ss._scenario_ids_by_key(submitted, raw)
    assert exc_info.value.error.code == "scenario_registration_unknown_key"


def test_scenario_ids_by_key_raises_typed_error_for_missing_key() -> None:
    submitted = (_scenario("a"), _scenario("b"))
    raw = [{"scenario_key": "a", "scenario_id": "platform-a"}]  # "b" never comes back
    with pytest.raises(ScenarioPreallocationError) as exc_info:
        ss._scenario_ids_by_key(submitted, raw)
    assert exc_info.value.error.code == "scenario_registration_missing"


def test_scenario_ids_by_key_raises_typed_error_for_duplicate_key() -> None:
    submitted = (_scenario("a"),)
    raw = [
        {"scenario_key": "a", "scenario_id": "platform-a"},
        {"scenario_key": "a", "scenario_id": "platform-a-again"},
    ]
    with pytest.raises(ScenarioPreallocationError) as exc_info:
        ss._scenario_ids_by_key(submitted, raw)
    assert exc_info.value.error.code == "scenario_registration_duplicate_key"


def test_scenario_ids_by_key_raises_typed_error_for_empty_scenario_id() -> None:
    with pytest.raises(ScenarioPreallocationError) as exc_info:
        ss._scenario_ids_by_key(
            (_scenario("a"),), [{"scenario_key": "a", "scenario_id": ""}]
        )
    assert exc_info.value.error.code == "scenarios_provision_response_invalid"


# -------------------------------------------------------------------------------------------------
# Authoring provisions identities but never begins execution.
# -------------------------------------------------------------------------------------------------


def test_register_with_platform_assigns_platform_ids_without_beginning() -> None:
    async def scenario() -> None:
        submitted = (_scenario("a"), _scenario("b"))
        client = _FakeScenariosClient(
            provision_response={
                "run_test_id": "run-test-1",
                # Deliberately out of order relative to `submitted` -- proves the match is by key.
                "scenarios": [
                    {"scenario_key": "b", "scenario_id": "platform-b"},
                    {"scenario_key": "a", "scenario_id": "platform-a"},
                ],
            },
        )
        result = await ss.register_with_platform(client, submitted, run_name="run-1")

        assert [s.scenario_key for s in result] == [
            "a",
            "b",
        ]  # submission order preserved
        assert [s.scenario_id for s in result] == [
            "platform-a",
            "platform-b",
        ]  # matched by key
        assert (
            result[0].setup is submitted[0].setup
        )  # untouched fields carried through verbatim
        assert result[0].ready is submitted[0].ready
        assert result[0].sub_goals is submitted[0].sub_goals

        assert client.provision_calls == [
            {
                "operation": "provision",
                "name": "run-1",
                "personas": [{"scenario_key": "a"}, {"scenario_key": "b"}],
            }
        ]
        assert client.begin_calls == []

    asyncio.run(scenario())


def test_execution_manifest_selects_and_expands_trials_without_reauthoring() -> None:
    authored = (_scenario("a"), _scenario("b"))
    selected = ss._scenarios_for_execution_manifest(
        authored,
        [
            {
                "execution_key": "a-trial-1",
                "scenario_key": "a",
                "scenario_id": "platform-a",
                "trial_index": 1,
            },
            {
                "execution_key": "a-trial-2",
                "scenario_key": "a",
                "scenario_id": "platform-a",
                "trial_index": 2,
            },
        ],
    )

    assert [scenario.scenario_key for scenario in selected] == [
        "a-trial-1",
        "a-trial-2",
    ]
    assert [scenario.scenario_id for scenario in selected] == [
        "platform-a",
        "platform-a",
    ]
    assert selected[0].setup is authored[0].setup
    assert selected[1].ready is authored[0].ready


def test_register_with_platform_guard_failure_never_calls_begin() -> None:
    # "never partial assignment": a guard failure during provision-response parsing must stop
    # BEFORE begin is ever called -- no scenario in this batch gets sealed for execution against a
    # registration the client-side guard has already rejected.
    async def scenario() -> None:
        submitted = (_scenario("a"), _scenario("b"))
        client = _FakeScenariosClient(
            provision_response={
                "run_test_id": "run-test-1",
                "scenarios": [
                    {"scenario_key": "a", "scenario_id": "platform-a"}
                ],  # "b" missing
            },
        )
        with pytest.raises(ScenarioPreallocationError) as exc_info:
            await ss.register_with_platform(client, submitted, run_name="run-1")
        assert exc_info.value.error.code == "scenario_registration_missing"
        assert client.begin_calls == []

    asyncio.run(scenario())


def test_register_with_platform_missing_run_test_id_is_a_typed_failure_before_begin() -> (
    None
):
    async def scenario() -> None:
        submitted = (_scenario("a"),)
        client = _FakeScenariosClient(
            provision_response={
                "scenarios": [{"scenario_key": "a", "scenario_id": "platform-a"}],
            },
        )
        with pytest.raises(ScenarioPreallocationError) as exc_info:
            await ss.register_with_platform(client, submitted, run_name="run-1")
        assert exc_info.value.error.code == "scenarios_provision_response_invalid"
        assert client.begin_calls == []

    asyncio.run(scenario())


# -------------------------------------------------------------------------------------------------
# Mutations (p13 work item 5).
# -------------------------------------------------------------------------------------------------


def test_mutation_positional_zip_matching_is_killed() -> None:
    # Mutant: `_scenario_ids_by_key` replaced with a positional zip (the documented shape --
    # not what the platform actually returns). A reordered response must silently mismatch ids
    # under the mutant; the real (key-matching) implementation must not.
    submitted = (_scenario("a"), _scenario("b"))
    reordered_response = [
        {"scenario_key": "b", "scenario_id": "platform-b"},
        {"scenario_key": "a", "scenario_id": "platform-a"},
    ]

    real = ss._scenario_ids_by_key(submitted, reordered_response)
    assert real == {
        "a": "platform-a",
        "b": "platform-b",
    }  # baseline: correct regardless of order

    def _positional_zip_mutant(submitted, raw_scenarios):
        return {
            scenario.scenario_key: entry["scenario_id"]
            for scenario, entry in zip(submitted, raw_scenarios, strict=True)
        }

    with mock.patch.object(ss, "_scenario_ids_by_key", _positional_zip_mutant):
        mutant = ss._scenario_ids_by_key(submitted, reordered_response)
        assert mutant == {
            "a": "platform-b",
            "b": "platform-a",
        }  # mutant's defect: swapped ids
        assert mutant != real

    restored = ss._scenario_ids_by_key(submitted, reordered_response)
    assert restored == real  # confirms the patch was fully undone


def test_mutation_missing_scenario_guard_removed_is_killed() -> None:
    # Mutant: the "every submitted key must appear in the response" check deleted from
    # `_scenario_ids_by_key` -- as if a job with fewer provisioned scenarios than requested were
    # silently accepted instead of failing the whole registration.
    submitted = (_scenario("a"), _scenario("b"))
    incomplete_response = [
        {"scenario_key": "a", "scenario_id": "platform-a"}
    ]  # "b" never comes back

    with pytest.raises(ScenarioPreallocationError) as exc_info:
        ss._scenario_ids_by_key(
            submitted, incomplete_response
        )  # baseline: the real guard catches it
    assert exc_info.value.error.code == "scenario_registration_missing"

    def _no_missing_guard_mutant(submitted, raw_scenarios):
        del submitted
        return {entry["scenario_key"]: entry["scenario_id"] for entry in raw_scenarios}

    with mock.patch.object(ss, "_scenario_ids_by_key", _no_missing_guard_mutant):
        mutant = ss._scenario_ids_by_key(
            submitted, incomplete_response
        )  # mutant: no longer raises
        assert (
            "b" not in mutant
        )  # confirms the mutant's defect: an incomplete mapping got through

    with pytest.raises(ScenarioPreallocationError) as exc_info:
        ss._scenario_ids_by_key(submitted, incomplete_response)  # restored
    assert exc_info.value.error.code == "scenario_registration_missing"


def test_mutation_id_assignment_skipped_is_killed() -> None:
    # Mutant: `register_with_platform`'s final identity-assignment step deleted.
    # Provision still succeeds, but execution would receive scenarios without
    # the platform identity required by receipts.
    async def scenario() -> None:
        submitted = (_scenario("a"),)
        client = _FakeScenariosClient(
            provision_response={
                "run_test_id": "run-test-1",
                "scenarios": [{"scenario_key": "a", "scenario_id": "platform-a"}],
            },
        )

        real = await ss.register_with_platform(client, submitted, run_name="run-1")
        assert real[0].scenario_id == "platform-a"  # baseline: the real fix assigns it

        async def _skip_assignment_mutant(scenarios_client, scenarios, *, run_name):
            await asyncio.to_thread(
                scenarios_client.provision, ss._provision_payload(run_name, scenarios)
            )
            return scenarios  # mutant's defect: returned VERBATIM, ids never merged in

        with mock.patch.object(ss, "register_with_platform", _skip_assignment_mutant):
            mutant = await ss.register_with_platform(
                client, submitted, run_name="run-1"
            )
            assert mutant[0].scenario_id == ""  # mutant's defect: id never assigned

        restored = await ss.register_with_platform(client, submitted, run_name="run-1")
        assert (
            restored[0].scenario_id == "platform-a"
        )  # confirms the patch was fully undone

    asyncio.run(scenario())


# -------------------------------------------------------------------------------------------------
# Harness-chosen platform evals. The guest sends names only; the platform owns the mapping.
# -------------------------------------------------------------------------------------------------


def test_provision_payload_omits_chosen_evals_and_prompt_when_there_are_none() -> None:
    # Omitted rather than empty, so a platform that predates these fields is unaffected.
    payload = ss._provision_payload("run-1", (_scenario("book_a_ride"),))
    assert "chosen_evals" not in payload and "agent_prompt" not in payload


def test_provision_payload_carries_the_chosen_names_and_the_agent_prompt() -> None:
    payload = ss._provision_payload(
        "run-1",
        (_scenario("book_a_ride"),),
        ["customer_agent_loop_detection", "dead_air_detection"],
        "You are a ride booking agent.",
    )
    assert payload["chosen_evals"] == [
        "customer_agent_loop_detection",
        "dead_air_detection",
    ]
    assert payload["agent_prompt"] == "You are a ride booking agent."


def test_chosen_evals_and_prompt_are_read_off_the_bundle_contract(tmp_path) -> None:
    (tmp_path / "contract.json").write_text(
        json.dumps(
            {
                "agent": "ride",
                "system_prompt_excerpt": "  You book rides.  ",
                "chosen_evals": ["customer_agent_conversation_quality", " ", ""],
            }
        ),
        encoding="utf-8",
    )
    names, prompt, modality = ss._chosen_evals_and_prompt(tmp_path)
    assert names == ["customer_agent_conversation_quality"]
    assert prompt == "You book rides."


def test_a_bundle_without_a_readable_contract_costs_the_run_nothing(tmp_path) -> None:
    assert ss._chosen_evals_and_prompt(tmp_path) == ([], "", "")
    (tmp_path / "contract.json").write_text("{not json", encoding="utf-8")
    assert ss._chosen_evals_and_prompt(tmp_path) == ([], "", "")


def test_the_provision_payload_carries_the_modality_so_evals_bind_to_the_right_input(
    tmp_path,
) -> None:
    """Provisioning defaults to text, and a voice run that stays quiet has its evals judge a
    transcript.

    For a spoken call the conversation is the recording.
    """
    (tmp_path / "contract.json").write_text(
        json.dumps(
            {"modality": "voice", "chosen_evals": ["customer_agent_context_retention"]}
        ),
        encoding="utf-8",
    )
    names, _prompt, modality = ss._chosen_evals_and_prompt(tmp_path)
    assert modality == "voice"

    payload = ss._provision_payload("run", [], names, "", modality)
    assert payload["modality"] == "voice"
    assert payload["chosen_evals"] == ["customer_agent_context_retention"]

    # Omitted rather than empty, so an older platform is unaffected.
    assert "modality" not in ss._provision_payload("run", [], [], "", "")


def test_a_judged_sub_goal_is_exactly_one_with_no_check_file(tmp_path: Path) -> None:
    """The scheduler routes on `judged` alone, so judged has to mean "no code settles this".

    If a coded sub-goal ever arrived judged-truthy, its check would be skipped in favour of a model
    call. Nothing asserted the equivalence until this test.
    """
    _write_scenario(
        tmp_path / ss.SCENARIOS_DIRNAME,
        "mixed",
        scenario_key="mixed",
        sub_goals=["coded", "judged_only"],
        checks={"coded": "def check(world, calls):\n    return None\n"},
    )
    scenario = ss.load_scenarios(tmp_path)[0]
    by_name = {goal.name: goal for goal in scenario.sub_goals}

    assert by_name["coded"].judged == "", "a sub-goal with a check file must not be judged"
    assert by_name["coded"].check is not ss._judged_placeholder_check
    assert by_name["judged_only"].judged, "no check file means a model has to decide it"
    assert by_name["judged_only"].check is ss._judged_placeholder_check


def test_restoring_claims_never_makes_a_coded_sub_goal_judged(tmp_path: Path) -> None:
    """`_with_claims` guards this deliberately; removing the guard would skip real checks."""
    _write_scenario(
        tmp_path / ss.SCENARIOS_DIRNAME,
        "coded",
        scenario_key="coded",
        sub_goals=["coded"],
        checks={"coded": "def check(world, calls):\n    return None\n"},
    )
    scenario = ss.load_scenarios(tmp_path)[0]
    # A catalogue that claims the coded sub-goal is judged, which is the shape validate_sub_goal
    # accepts today: a working check AND a judged claim on the same sub-goal.
    restored = ss._with_claims(
        scenario,
        {"coded": {"what": "the row is written", "judged": "a model must weigh tone"}},
    )
    goal = restored.sub_goals[0]
    assert goal.judged == "", "a coded sub-goal must stay coded whatever the catalogue claims"
    assert goal.what == "the row is written", "the description is still restored"


def test_the_index_carries_each_scenario_in_full(tmp_path: Path) -> None:
    from fi.alk.harness.folder import write_index
    from fi.alk.harness.scenario import Persona, Scenario, Step

    scenario = Scenario(
        name="quote_for_a_young_adult",
        use_case="quote a plan",
        branch="applicant is 26",
        tests="quotes the youth plan",
        instruction="ask for a quote",
        sub_goals=["plan_quoted"],
        persona=Persona(name="Ada", age_group="25-32"),
        fixture={"age": 26, "email": "ada@example.com"},
        solution=[Step(tool="find_applicant"), Step(tool="quote_plan")],
        max_turns=12,
    )
    write_index([scenario], tmp_path)

    row = json.loads((tmp_path / "scenarios.json").read_text(encoding="utf-8"))[0]
    assert row["persona"]["name"] == "Ada"
    assert row["branch"] == "applicant is 26"
    assert row["fixture"]["age"] == 26
    assert [step["tool"] for step in row["solution"]] == ["find_applicant", "quote_plan"]
    assert row["max_turns"] == 12
    assert row["folder"] == "scenarios/quote_for_a_young_adult"
    assert row["steps"] == 2
    assert "setup_code" not in row
    assert "ready_code" not in row


def test_keyword_problems_catches_what_a_hundred_scenario_suite_did(tmp_path: Path) -> None:
    from fi.alk.harness.scenario import Persona, Scenario, keyword_problems

    def suite(keywords_for) -> list[Scenario]:
        return [
            Scenario(
                name=f"s{i}",
                keywords=keywords_for(i),
                persona=Persona(name=f"P{i}"),
            )
            for i in range(20)
        ]

    said = " ".join(keyword_problems(suite(lambda i: ["weather", f"x{i}"])))
    assert "more than 40%" in said and "weather" in said

    said = " ".join(keyword_problems(suite(lambda i: [f"only{i}", f"also{i}", f"third{i}"])))
    assert "fewer than 3 scenarios" in said

    said = " ".join(keyword_problems(suite(lambda i: [f"k{n}" for n in range(8)] if i == 0 else ["k0", "k1", "k2"])))
    assert "more than 5 keywords" in said

    vocabulary = [
        "call_termination",
        "disambiguation",
        "weather_lookup",
        "interruption",
        "multi_intent",
        "code_switching",
    ]
    assert keyword_problems(
        suite(lambda i: [vocabulary[i % 6], vocabulary[(i + 2) % 6]])
    ) == []


def test_coverage_report_answers_how_much_of_the_space_was_tested(tmp_path: Path) -> None:
    from fi.alk.harness.scenario import Scenario, coverage_report

    def suite(coords) -> list[Scenario]:
        return [
            Scenario(name=f"s{i}", use_case="book a ride", coverage=c)
            for i, c in enumerate(coords)
        ]

    even = coverage_report(
        suite([{"task": t, "overlay": o} for t in ("book", "cancel") for o in ("none", "interrupt")])
    )
    assert even["scenarios"] == 4
    assert even["placed"] == 4
    assert even["axes"]["task"]["levels"] == 2
    assert even["axes"]["task"]["spread"] == 1.0
    assert even["pairs"]["overlay x task"]["covered"] == 4
    assert even["pairs"]["overlay x task"]["share"] == 1.0

    lopsided = coverage_report(
        suite([{"task": "book"}] * 9 + [{"task": "cancel"}])
    )
    assert lopsided["axes"]["task"]["levels"] == 2
    assert lopsided["axes"]["task"]["spread"] < 0.5

    mixed = coverage_report(suite([{"task": "book"}, {}, {}]))
    assert mixed["scenarios"] == 3 and mixed["placed"] == 1
    assert "use_case" not in mixed["axes"]
    assert mixed["use_cases"] == {"book a ride": 3}

    assert coverage_report([])["scenarios"] == 0


def test_coverage_report_tells_a_gap_from_a_cell_that_was_never_legal(tmp_path: Path) -> None:
    from fi.alk.harness.scenario import Persona, Scenario, coverage_report

    suite = [
        Scenario(
            name=f"s{i}",
            persona=Persona(name=f"P{i}"),
            coverage={"task": task, "counterparty": who},
        )
        for i, (task, who) in enumerate(
            [("book", "first_time"), ("book", "regular"), ("cancel", "first_time"), ("cancel", "regular")]
        )
    ]

    blind = coverage_report(suite)
    assert blind["pairs"]["counterparty x task"]["share"] == 1.0

    told = coverage_report(
        suite,
        {
            "axes": {
                "task": ["book", "cancel", "reschedule"],
                "counterparty": ["first_time", "regular", "minor"],
            },
            "masked": [["task=book", "counterparty=minor"]],
        },
    )
    pair = told["pairs"]["counterparty x task"]
    assert (pair["covered"], pair["possible"], pair["masked"]) == (4, 8, 1)
    assert told["axes"]["task"]["unused"] == ["reschedule"]
    assert told["axes"]["counterparty"]["unused"] == ["minor"]


def test_check_problems_names_the_check_that_only_proves_the_tool_was_reached(tmp_path: Path) -> None:
    from fi.alk.harness.folder import check_problems

    def write(scenario: str, name: str, body: str) -> None:
        folder = tmp_path / "scenarios" / scenario / "checks"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.py").write_text(
            f"def check(world, calls):\n{body}\n    return None\n", encoding="utf-8"
        )

    write("twice", "lookup_weather_executed",
          "    lookups = [c for c in calls if c.name == 'lookup_weather' and c.ok]\n"
          "    if not any('location' in c.arguments for c in lookups):\n"
          "        return 'no location'")
    write("twice", "weather_lookup_succeeded",
          "    lookups = [c for c in calls if c.name == 'lookup_weather' and c.ok]\n"
          "    if not any(len(c.arguments.get('location', '')) > 0 for c in lookups):\n"
          "        return 'empty location'")
    write("plumbing", "call_ended_on_request",
          "    if not [c for c in calls if c.name == 'end_call']:\n"
          "        return 'not ended'")
    write("double_quoted", "subgoal_book_ride_confirmed",
          '    book = next((c for c in calls if c.name == "book_ride" and c.ok), None)\n'
          '    if book.arguments.get("caller_explicitly_confirmed") is not True:\n'
          '        return "must be explicitly confirmed"')

    said = " ".join(check_problems(tmp_path))
    assert "twice: lookup_weather_executed, weather_lookup_succeeded" in said
    assert "1 checks assert only that a tool was called" in said
    assert "double_quoted" not in said


def test_fixture_credentials_are_only_flagged_when_they_name_a_record() -> None:
    from fi.alk.harness.scenario_tools import _handed_to_the_caller

    ungrounded = {
        "origin": "seed",
        "identity": {"rider_id": "rdr_dana", "phone": "+14155550101"},
        "credentials": {"otp_code": "265512"},
    }
    assert sorted(_handed_to_the_caller(ungrounded)) == [
        ("otp_code", "265512"),
        ("phone", "+14155550101"),
    ]

    assert _handed_to_the_caller({"origin": "seed", "location": "New York"}) == []
    assert _handed_to_the_caller({"pickup_time": "18:30", "passengers": 3}) == []
    assert _handed_to_the_caller({"payment": {"card_number": "4242424242424242"}}) == [
        ("card_number", "4242424242424242")
    ]


def test_unpinned_callers_calibrates_against_the_suite_it_is_given() -> None:
    from fi.alk.harness.scenario import Persona, Scenario, unpinned_callers

    def one(name: str, fixture: dict) -> Scenario:
        return Scenario(name=name, persona=Persona(name=name), fixture=fixture)

    weatherish = [one(f"w{i}", {"location": "London"}) for i in range(4)]
    assert unpinned_callers(weatherish) == []

    alike = [one(f"r{i}", {"phone": f"+1415555010{i}"}) for i in range(4)]
    assert unpinned_callers(alike) == []

    mixed = [one(f"r{i}", {"phone": f"+1415555010{i}"}) for i in range(3)]
    mixed.append(one("book_ride_guest", {"caller_name": "Carlos"}))
    said = " ".join(unpinned_callers(mixed))
    assert "book_ride_guest" in said
    assert "pin one that matches no row" in said
    assert "r0" not in said


def test_a_broken_suite_remark_costs_the_remark_and_not_the_save(tmp_path: Path) -> None:
    from fi.alk.harness.folder import check_problems

    checks = tmp_path / "scenarios" / "unreadable" / "checks"
    checks.mkdir(parents=True)
    (checks / "broken.py").write_bytes(b"def check(world, calls):\n    return '\xff\xfe not utf-8'\n")

    with pytest.raises(UnicodeDecodeError):
        check_problems(tmp_path)

    noted: list[str] = ["something already noted"]
    for remark in (lambda: check_problems(tmp_path),):
        try:
            noted = noted + remark()
        except Exception:
            pass
    assert noted == ["something already noted"]


def test_uncovered_cells_names_what_the_plan_allows_and_nothing_reached() -> None:
    from fi.alk.harness.scenario import Scenario, uncovered_cells

    design = {"axes": {"task": ["book", "cancel"], "who": ["new", "vip"]}}
    written = [Scenario(name="a", coverage={"task": "book", "who": "new"})]

    assert uncovered_cells(written, design) == [
        "task=book x who=vip",
        "task=cancel x who=new",
        "task=cancel x who=vip",
    ]
    assert len(uncovered_cells([], design)) == 4
    assert uncovered_cells(written, {"axes": {"task": ["book"]}}) == []


def test_a_masked_pair_is_not_an_empty_cell() -> None:
    from fi.alk.harness.scenario import Scenario, uncovered_cells

    design = {
        "axes": {"task": ["book", "cancel"], "who": ["new", "vip"]},
        "masked": [["task=cancel", "who=vip"]],
    }
    written = [Scenario(name="a", coverage={"task": "book", "who": "new"})]

    assert uncovered_cells(written, design) == [
        "task=book x who=vip",
        "task=cancel x who=new",
    ]


def test_uncovered_cells_stops_at_the_limit() -> None:
    from fi.alk.harness.scenario import uncovered_cells

    design = {"axes": {"a": [str(n) for n in range(10)], "b": [str(n) for n in range(10)]}}

    assert len(uncovered_cells([], design)) == 24
    assert len(uncovered_cells([], design, limit=5)) == 5


def test_a_writer_is_offered_only_the_tools_it_writes_with(monkeypatch) -> None:
    from pathlib import Path as _Path

    from fi.alk.harness import scenarios as stage
    from fi.alk.harness.backends import ToolServer, ToolSpec
    from fi.alk.harness.contract import AgentContract

    monkeypatch.setattr(stage, "world_summary", lambda _: "a world")
    monkeypatch.setattr(stage, "load_skill", lambda *a, **k: "a skill")
    monkeypatch.setattr(stage, "discovered_skills", lambda **k: "")

    async def handler(args):
        return {"content": "ok"}

    from fi.alk.harness.scenario_tools import TOOL_NAMES

    server = ToolServer(
        name="scenarios",
        version="1",
        tools=[
            ToolSpec(name=name, description=name, input_schema={}, handler=handler)
            for name in TOOL_NAMES
        ],
    )
    worker = stage.writer_worker(
        AgentContract(agent="a"), _Path("/tmp"), server, budget=40
    )[stage.WRITER]

    offered = {spec.name for spec in worker.servers[stage.SCENARIO_SERVER].tools}
    assert offered == set(stage.WRITER_TOOLS)
    assert not offered & {
        "inspect_scenario",
        "suite_progress",
        "save_scenarios",
        "aim_for",
        "drop_scenario",
        "amend_contract",
    }


def test_the_submit_reply_says_what_the_names_are_for() -> None:
    from fi.alk.harness import scenario_tools

    source = (
        __import__("pathlib").Path(scenario_tools.__file__).read_text(encoding="utf-8")
    )
    assert "inspect_scenario names them all" not in source
    assert "do not read them" in source


def test_only_a_value_the_agent_looks_up_is_a_missing_credential() -> None:
    from pathlib import Path as _P
    import tempfile

    from fi.alk.harness.scenario import Scenario, Step
    from fi.alk.harness.scenario_tools import grounding_problems

    guest = Scenario(
        name="guest",
        fixture={"credentials": {"phone": "+14155550220"}},
        solution=[Step(tool="book_ride", arguments={"first_name": "Dana"})],
    )
    looked_up = Scenario(
        name="known",
        fixture={"credentials": {"phone": "+14155550220"}},
        solution=[Step(tool="find_rider", arguments={"phone": "+14155550220"})],
    )
    with tempfile.TemporaryDirectory() as tmp:
        assert grounding_problems([guest], _P(tmp)) == []
        assert grounding_problems([looked_up], _P(tmp)) == []


def test_a_suite_is_always_handed_out(tmp_path, monkeypatch) -> None:
    from fi.alk.harness import scenarios as stage
    from fi.alk.harness.contract import AgentContract

    from fi.alk.harness.backends import ToolServer, ToolSpec
    from fi.alk.harness.scenario_tools import TOOL_NAMES

    async def handler(args):
        return {"content": "ok"}

    monkeypatch.setattr(stage, "world_summary", lambda _: "a world")
    monkeypatch.setattr(stage, "load_skill", lambda *a, **k: "a skill")
    monkeypatch.setattr(stage, "discovered_skills", lambda **k: "")
    monkeypatch.setattr(
        stage,
        "scenario_tools",
        lambda *a, **k: (
            ToolServer(
                name="scenarios",
                version="1",
                tools=[
                    ToolSpec(name=n, description=n, input_schema={}, handler=handler)
                    for n in TOOL_NAMES
                ],
            ),
            [],
        ),
    )
    contract = AgentContract(agent="a")

    big, _ = stage.open_stage(contract, out=tmp_path / "big", wanted=50)
    small, _ = stage.open_stage(contract, out=tmp_path / "small", wanted=8)
    offered = lambda spec: {
        t.name for t in spec.servers[stage.SCENARIO_SERVER].tools
    }

    assert not (set(stage.WRITES_A_SCENARIO) & offered(big.spec)), (
        "a loop that can write, writes"
    )
    assert big.spec.workers, "the loop must still be able to hand work out"
    assert "aim_for" in offered(big.spec) and "suite_progress" in offered(big.spec)
    assert "save_scenarios" in offered(big.spec)
    assert not (set(stage.WRITES_A_SCENARIO) & offered(small.spec))
    assert small.spec.workers, "a small suite is handed out too"


def test_suite_progress_names_overlays_that_assert_nothing() -> None:
    from fi.alk.harness.scenario import Scenario, redteam_problems

    plain = [
        Scenario(name=f"plain{n}", coverage={"task": "book"}, sub_goals=["booked"])
        for n in range(8)
    ]
    lazy = Scenario(
        name="lazy",
        coverage={"task": "book", "overlay": "prompt_injection"},
        sub_goals=["booked"],
    )
    real = Scenario(
        name="real",
        coverage={"task": "book", "overlay": "prompt_injection"},
        sub_goals=["booked", "injection_refused"],
    )
    said = " ".join(redteam_problems(plain + [lazy, real]))
    assert "lazy" in said
    assert "real:" not in said


def test_a_resumed_suite_is_told_to_read_before_it_changes_anything() -> None:
    from fi.alk.harness.contract import AgentContract
    from fi.alk.harness.scenarios import opening

    contract = AgentContract(agent="a")
    big = opening(contract, 50, existing=50)
    small = opening(contract, 8, existing=8)

    assert "inspect_scenario" in big and "inspect_scenario" in small
    assert "do not write scenarios yourself" not in big


def test_a_writer_is_told_which_named_tools_are_not_its_job() -> None:
    import re
    from unittest import mock
    from pathlib import Path as _P

    from fi.alk.harness import scenarios as stage
    from fi.alk.harness.backends import ToolServer, ToolSpec
    from fi.alk.harness.config import SKILLS_ROOT
    from fi.alk.harness.contract import AgentContract
    from fi.alk.harness.scenario_tools import TOOL_NAMES

    async def handler(args):
        return {"content": "ok"}

    server = ToolServer(
        name="scenarios",
        version="1",
        tools=[
            ToolSpec(name=n, description=n, input_schema={}, handler=handler)
            for n in TOOL_NAMES
        ],
    )
    with mock.patch.object(stage, "world_summary", lambda _: "w"), mock.patch.object(
        stage, "load_skill", lambda *a, **k: "s"
    ), mock.patch.object(stage, "discovered_skills", lambda **k: ""):
        said = stage.writer_worker(
            AgentContract(agent="a"), _P("/tmp"), server, budget=400
        )[stage.WRITER].instructions

    method = re.sub(
        r"```.*?```",
        "",
        (SKILLS_ROOT / "write-scenarios" / "SKILL.md").read_text(encoding="utf-8"),
        flags=re.S,
    )
    named = set(re.findall(r"`([a-z_][a-z0-9_]*)\(?`", method)) & set(TOOL_NAMES)
    for missing in named - set(stage.WRITER_TOOLS):
        assert f"`{missing}`" in said, f"the writer is never told it lacks {missing}"


def test_the_loop_learns_what_is_left_without_reading_a_scenario_body() -> None:
    import asyncio
    import json
    from pathlib import Path as _P

    from fi.alk.harness.contract import AgentContract
    from fi.alk.harness.scenario_tools import scenario_tools

    root = _P(__file__).parent.parent / "fixtures"
    if not (root / "contract.json").exists():
        return  # the shape is pinned above; the sizing check needs a real suite
    contract = AgentContract.model_validate(json.loads((root / "contract.json").read_text()))
    server, _ = scenario_tools(contract, root, root, wanted=5)
    tool = next(t for t in server.tools if t.name == "suite_progress")

    async def said(args):
        out = await tool.handler(args)
        return "".join(p.get("text", "") for p in out["content"])

    plain, named = asyncio.run(said({})), asyncio.run(said({"names": True}))
    assert len(named) >= len(plain)


def test_a_world_is_refused_when_it_invents_tables_or_columns(tmp_path) -> None:
    from fi.alk.harness.world.tools import _tables_the_source_lacks

    (tmp_path / "db").mkdir()
    (tmp_path / "db" / "schema.sql").write_text(
        'DROP TABLE IF EXISTS users, products CASCADE;\n'
        "CREATE TABLE users (rider_id TEXT PRIMARY KEY);\n"
        'CREATE TABLE IF NOT EXISTS "products" (id TEXT PRIMARY KEY);\n',
        encoding="utf-8",
    )
    adopted = {
        "users": [{"rider_id": "u1"}],
        "products": [{"id": "p1"}],
        "sqlite_sequence": [{"anything": 1}],
    }
    invented_table = {"users": [{"rider_id": "u1"}], "fare_products": [{"id": "f1"}]}
    invented_column = {"users": [{"rider_id": "u1", "created_at": "2026-01-01"}]}

    assert _tables_the_source_lacks(adopted, str(tmp_path), None) == []
    assert _tables_the_source_lacks(invented_table, str(tmp_path), None) == [
        "fare_products (the whole table)"
    ]
    assert _tables_the_source_lacks(invented_column, str(tmp_path), None) == [
        "users.{created_at}"
    ]
    (tmp_path / "db" / "schema.sql").write_text(
        "CREATE TABLE users (\n"
        "  rider_id TEXT PRIMARY KEY,\n"
        "  phone TEXT NOT NULL,  -- matched against caller_ani\n"
        "  status TEXT NOT NULL DEFAULT 'active'\n"
        ");\n",
        encoding="utf-8",
    )
    assert _tables_the_source_lacks(
        {"users": [{"rider_id": "u1", "phone": "+1"}]}, str(tmp_path), None
    ) == []
    assert _tables_the_source_lacks(
        {"users": [{"rider_id": "u1"}]}, str(tmp_path), None
    ) == ["users rows leave out phone, which the schema requires"]
    assert _tables_the_source_lacks(invented_table, "", None) == []


def test_a_credential_the_world_lacks_is_refused_at_submit() -> None:
    from types import SimpleNamespace

    from fi.alk.harness.scenario import Scenario, Step
    from fi.alk.harness.scenario_tools import _credentials_the_world_lacks

    world = SimpleNamespace(state=lambda: {"places": [{"place_id": "plc_real_01"}]})

    looked_up_and_missing = Scenario(
        name="a",
        fixture={"pickup_place_id": "plc_dana_home_01"},
        solution=[Step(tool="set_pickup", arguments={"place_id": "plc_dana_home_01"})],
    )
    looked_up_and_present = Scenario(
        name="b",
        fixture={"pickup_place_id": "plc_real_01"},
        solution=[Step(tool="set_pickup", arguments={"place_id": "plc_real_01"})],
    )
    never_looked_up = Scenario(
        name="c",
        fixture={"caller_phone": "+14155550199"},
        solution=[Step(tool="book_ride", arguments={"product": "uberx"})],
    )

    assert _credentials_the_world_lacks(looked_up_and_missing, world)
    assert _credentials_the_world_lacks(looked_up_and_present, world) == []
    assert _credentials_the_world_lacks(never_looked_up, world) == []


def test_the_seed_renders_a_boolean_the_agent_schema_declares(tmp_path) -> None:
    from fi.alk.harness.bundle_author_v2 import _schema_column_types

    (tmp_path / "schema.sql").write_text(
        "CREATE TABLE payment_methods (\n"
        "  id TEXT PRIMARY KEY,\n"
        "  is_default BOOLEAN NOT NULL,  -- stored as 0/1 in sqlite\n"
        "  last4 TEXT NOT NULL\n"
        ");\n",
        encoding="utf-8",
    )
    found = _schema_column_types([tmp_path / "schema.sql"])
    assert found[("payment_methods", "is_default")] == "boolean"
    assert ("payment_methods", "last4") not in found


def test_a_primary_key_counts_as_required(tmp_path) -> None:
    from fi.alk.harness.world.tools import _tables_the_source_lacks

    (tmp_path / "db").mkdir()
    (tmp_path / "db" / "schema.sql").write_text(
        "CREATE TABLE users (\n"
        "  rider_id TEXT PRIMARY KEY,\n"
        "  nickname TEXT\n"
        ");\n",
        encoding="utf-8",
    )
    assert _tables_the_source_lacks(
        {"users": [{"rider_id": "u1"}]}, str(tmp_path), None
    ) == []
    assert _tables_the_source_lacks(
        {"users": [{"nickname": "Dana"}]}, str(tmp_path), None
    ) == ["users rows leave out rider_id, which the schema requires"]


def test_an_array_column_is_typed_from_the_adopted_schema(tmp_path) -> None:
    from fi.alk.harness.bundle_author_v2 import _schema_column_types

    (tmp_path / "schema.sql").write_text(
        "CREATE TABLE places (\n"
        "  place_id TEXT PRIMARY KEY,\n"
        "  aliases TEXT[] NOT NULL DEFAULT '{}',\n"
        "  is_open BOOLEAN NOT NULL\n"
        ");\n",
        encoding="utf-8",
    )
    found = _schema_column_types([tmp_path / "schema.sql"])
    assert found[("places", "aliases")] == "text[]"
    assert found[("places", "is_open")] == "boolean"


def test_a_dangling_foreign_key_is_refused(tmp_path) -> None:
    from fi.alk.harness.world.tools import _tables_the_source_lacks

    (tmp_path / "db").mkdir()
    (tmp_path / "db" / "schema.sql").write_text(
        "CREATE TABLE users (rider_id TEXT PRIMARY KEY);\n"
        "CREATE TABLE wallets (\n"
        "  id TEXT PRIMARY KEY,\n"
        "  rider_id TEXT NOT NULL REFERENCES users(rider_id)\n"
        ");\n",
        encoding="utf-8",
    )
    sound = {
        "users": [{"rider_id": "u1"}],
        "wallets": [{"id": "w1", "rider_id": "u1"}],
    }
    dangling = {
        "users": [{"rider_id": "u1"}],
        "wallets": [{"id": "w1", "rider_id": "u9"}],
    }
    empty_target = {"users": [], "wallets": [{"id": "w1", "rider_id": "u1"}]}

    assert _tables_the_source_lacks(sound, str(tmp_path), None) == []
    assert _tables_the_source_lacks(dangling, str(tmp_path), None) == [
        "wallets.rider_id points at users.rider_id rows that do not exist: u9"
    ]
    assert _tables_the_source_lacks(empty_target, str(tmp_path), None) == [
        "wallets.rider_id points at users.rider_id rows that do not exist: u1"
    ]


def test_a_caller_the_record_does_not_know_is_reported(monkeypatch) -> None:
    from pathlib import Path as _P

    from fi.alk.harness import scenario_tools
    from fi.alk.harness.scenario import Persona, Scenario

    rows = {
        "users": [{"rider_id": "rdr_eli", "first_name": "Eli", "phone": "+14155550108"}],
        "saved_places": [{"rider_id": "rdr_eli", "name": "Home"}],
    }
    monkeypatch.setattr(
        scenario_tools, "restore", lambda _root: type("W", (), {"state": lambda _s: rows})()
    )

    def caller(name: str) -> Scenario:
        return Scenario(
            name=f"call_{name.lower()}",
            fixture={"rider_id": "rdr_eli", "phone": "+14155550108"},
            persona=Persona(name=name),
        )

    said = scenario_tools.persona_off_the_record([caller("Liam")], _P("/tmp"))
    assert said and "is named Eli" in said[0]
    assert scenario_tools.persona_off_the_record([caller("Eli")], _P("/tmp")) == []
    assert scenario_tools.persona_off_the_record([caller("Eli Navarro")], _P("/tmp")) == []
    guest = Scenario(
        name="guest", fixture={"phone": "+14165550199"}, persona=Persona(name="Chloe")
    )
    assert scenario_tools.persona_off_the_record([guest], _P("/tmp")) == []



def test_one_spelling_per_level_however_the_plan_wrote_it() -> None:
    from fi.alk.harness.scenario import Scenario
    from fi.alk.harness.scenario_tools import _off_the_grid

    slashes = Scenario(name="a", coverage={"Overlay": "privacy/PII", "task": "book_ride"})
    assert slashes.coverage == {"overlay": "privacy_pii", "task": "book_ride"}

    grid = {"overlay": ["none", "privacy/PII"], "task": ["book_ride"]}
    assert _off_the_grid({"overlay": "privacy_pii", "task": "book_ride"}, grid) == ""
    assert "not a level the plan deals" in _off_the_grid(
        {"overlay": "invented", "task": "book_ride"}, grid
    )
    assert "says nothing about" in _off_the_grid({"task": "book_ride"}, grid)


def test_an_empty_collection_names_its_fields(tmp_path) -> None:
    import sqlite3

    from fi.alk.harness import scenario_tools

    class _World:
        def __init__(self, path):
            self.connection = sqlite3.connect(path)
            self.connection.execute("CREATE TABLE bookings (booking_ref TEXT, status TEXT)")
            self.connection.execute("CREATE TABLE users (rider_id TEXT)")
            self.connection.execute("INSERT INTO users VALUES ('rdr_dana')")

        def state(self):
            return {"bookings": [], "users": [{"rider_id": "rdr_dana"}]}

        def close(self):
            self.connection.close()

    world = _World(tmp_path / "w.sqlite")
    assert scenario_tools._fields_of(world, "bookings") == ["booking_ref", "status"]
    assert scenario_tools._fields_of(world, "users") == ["rider_id"]
    assert scenario_tools._fields_of(object(), "bookings") == []
    world.close()


def test_a_broken_proof_says_so_instead_of_blaming_setup() -> None:
    from fi.alk.harness.prove import Proof

    judged_only = Proof()
    judged_only.broken = ["none of this sub-goal's checks is in code"]
    said = judged_only.why()
    assert "broken, not failing" in said
    assert "setup.py" not in said

    not_ready = Proof()
    not_ready.why_not_ready = "no booking row exists for bk_1"
    assert "the world is not ready" in not_ready.why()
    assert "no booking row exists for bk_1" in not_ready.why()

    both = Proof()
    both.broken = ["ready.py raised TypeError"]
    both.why_not_ready = "ready.py raised TypeError"
    assert "the world is not ready" in both.why()


def test_the_grid_has_to_be_the_frameworks_axes() -> None:
    from fi.alk.harness.scenario_tools import CANONICAL_AXES, _grid_off_the_framework

    whole = {axis: ["one"] for axis in CANONICAL_AXES}
    whole["task"] = ["cancel-ride"]
    assert _grid_off_the_framework(whole) == ""
    assert _grid_off_the_framework({}) == ""

    drifted = {**whole, "payment_state": ["card_expired"]}
    del drifted["disposition"]
    said = _grid_off_the_framework(drifted)
    assert "payment_state (a level of disposition)" in said
    assert "Missing: disposition" in said

    short = {axis: ["one"] for axis in CANONICAL_AXES if axis != "interaction"}
    assert "Missing: interaction" in _grid_off_the_framework(short)


def test_an_instruction_naming_a_record_the_world_lacks_is_refused() -> None:
    from types import SimpleNamespace

    from fi.alk.harness.scenario import Scenario
    from fi.alk.harness.scenario_tools import _identifiers_the_instruction_invents

    world = SimpleNamespace(
        state=lambda: {"places": [{"place_id": "plc_market_st"}, {"place_id": "plc_sfo"}]}
    )

    invented = Scenario(
        name="a",
        instruction="You ask for a ride from plc_blr_airport to plc_market_st.",
    )
    said = _identifiers_the_instruction_invents(invented, world)
    assert said and "plc_blr_airport" in said[0]
    assert "plc_market_st" not in said[0]

    real = Scenario(name="b", instruction="From plc_market_st to plc_sfo, please.")
    assert _identifiers_the_instruction_invents(real, world) == []

    other = Scenario(name="c", instruction="Charge it to acct_99 like last time.")
    assert _identifiers_the_instruction_invents(other, world) == []

    bare = SimpleNamespace(state=lambda: {"notes": [{"text": "hello"}]})
    assert _identifiers_the_instruction_invents(invented, bare) == []

def test_coverage_counts_a_planned_level_however_its_separators_are_spelled():
    from collections import Counter

    from fi.alk.harness.scenario import _against_plan

    report = _against_plan(
        "task",
        Counter({"explain_billing_policy": 2}),
        {"task": ["explain-billing_policy", "cancel-order"]},
    )
    assert report["unused"] == ["cancel-order"]
    assert report["share"] == 0.5


def test_a_level_past_a_third_of_the_suite_is_refused_while_somewhere_thinner_exists() -> None:
    from fi.alk.harness.scenario import Scenario
    from fi.alk.harness.scenario_tools import _over_its_share

    grid = {"task": ["book_ride", "cancel_ride", "check_status"]}
    booked = [
        Scenario(name=f"b{one}", coverage={"task": "book_ride"}) for one in range(7)
    ]

    said = _over_its_share({"task": "book_ride"}, grid, booked, 20)
    assert "whole share" in said and "cancel_ride" in said
    assert _over_its_share({"task": "cancel_ride"}, grid, booked, 20) == ""
    assert _over_its_share({"task": "book_ride"}, grid, booked, 8) == ""
    everywhere = booked + [
        Scenario(name=f"c{one}", coverage={"task": "cancel_ride"}) for one in range(7)
    ] + [Scenario(name=f"s{one}", coverage={"task": "check_status"}) for one in range(7)]
    assert _over_its_share({"task": "book_ride"}, grid, everywhere, 20) == ""


def test_interface_and_overlay_levels_are_not_held_to_a_third() -> None:
    from fi.alk.harness.scenario import Scenario
    from fi.alk.harness.scenario_tools import _over_its_share

    grid = {
        "interface": ["noisy_line", "quiet_line", "accented"],
        "overlay": ["none", "prompt_injection", "social_engineering"],
    }
    kept = [
        Scenario(name=f"n{one}", coverage={"interface": "noisy_line", "overlay": "none"})
        for one in range(15)
    ]
    assert _over_its_share({"interface": "noisy_line", "overlay": "none"}, grid, kept, 20) == ""


def test_a_slice_that_would_fall_below_the_red_team_floor_refuses_another_plain_scenario() -> None:
    from fi.alk.harness.scenario import Scenario
    from fi.alk.harness.scenario_tools import _red_team_short

    grid = {"overlay": ["none", "prompt_injection", "social_engineering"]}
    plain = [Scenario(name=f"p{n}", coverage={"overlay": "none"}) for n in range(16)]
    ask_plain = {"name": "x", "coverage": {"overlay": "none"}}

    assert "prompt_injection" in _red_team_short(ask_plain, grid, plain, 20)
    attack = {"name": "x", "coverage": {"overlay": "prompt_injection"}}
    assert _red_team_short(attack, grid, plain, 20) == ""
    assert _red_team_short(ask_plain, grid, plain[:15], 20) == ""
    assert _red_team_short(ask_plain, {"overlay": ["none"]}, plain, 20) == ""


def test_a_quiet_line_is_held_to_a_sixth_of_a_slice() -> None:
    from fi.alk.harness.scenario import Scenario
    from fi.alk.harness.scenario_tools import _over_its_share

    grid = {"interface": ["quiet_line", "noisy_line"]}
    quiet = [Scenario(name=f"q{n}", coverage={"interface": "quiet_line"}) for n in range(3)]

    assert "noisy_line" in _over_its_share({"interface": "quiet_line"}, grid, quiet, 18)
    assert _over_its_share({"interface": "quiet_line"}, grid, quiet[:2], 18) == ""
    assert _over_its_share({"interface": "noisy_line"}, grid, quiet, 18) == ""
    # Resubmitting one already held replaces it, so the count stays where it is.
    assert _over_its_share({"interface": "quiet_line"}, grid, quiet, 18, "q0") == ""


def test_a_family_name_is_not_reached_for_a_third_time() -> None:
    from fi.alk.harness.scenario import Persona, Scenario
    from fi.alk.harness.scenario_tools import _already_in_the_suite

    kept = [Scenario(name="a", persona=Persona(name="Marcus Vance"))]
    ask = {"name": "c", "persona": {"name": "Heather Vance"}}

    assert _already_in_the_suite(ask, kept, {"philip vance"})
    assert _already_in_the_suite(ask, kept, set()) == ""
    other = {"name": "c", "persona": {"name": "Heather Lam"}}
    assert _already_in_the_suite(other, kept, {"philip vance"}) == ""


def test_names_a_sibling_writer_journalled_count_as_taken(tmp_path) -> None:
    from fi.alk.harness.scenario import Persona, Scenario
    from fi.alk.harness.scenario_tools import _first_names_on_disk, journal_scenario

    journal_scenario(Scenario(name="a", persona=Persona(name="Marcus Vance")), tmp_path)
    journal_scenario(Scenario(name="b", persona=Persona(name="Philip Vance")), tmp_path)

    assert _first_names_on_disk(tmp_path) == {"marcus vance", "philip vance"}
    assert _first_names_on_disk(tmp_path, "b") == {"marcus vance"}


def test_callers_are_told_apart_by_full_name_not_first_name() -> None:
    from fi.alk.harness.scenario import Persona, Scenario
    from fi.alk.harness.scenario_tools import _already_in_the_suite

    kept = [Scenario(name="a", persona=Persona(name="Maria Lopez"))]

    assert _already_in_the_suite({"name": "b", "persona": {"name": "Maria Chen"}}, kept, set()) == ""
    assert "Maria Lopez" in _already_in_the_suite(
        {"name": "b", "persona": {"name": "  maria   LOPEZ "}}, kept, set()
    )
    assert _already_in_the_suite({"name": "b", "persona": {"name": "Maria Lopez"}}, [], {"maria lopez"})


def test_the_red_team_floor_leaves_a_tiny_slice_alone() -> None:
    from fi.alk.harness.scenario import Scenario
    from fi.alk.harness.scenario_tools import _red_team_short

    grid = {"overlay": ["none", "prompt_injection"]}
    plain = [Scenario(name=f"p{n}", coverage={"overlay": "none"}) for n in range(2)]

    assert _red_team_short({"name": "x", "coverage": {"overlay": "none"}}, grid, plain, 3) == ""
    assert _red_team_short({"name": "x"}, grid, plain, 20) == ""


def test_the_run_specific_policy_reaches_writers_and_planner_before_their_checklists(tmp_path: Path) -> None:
    from fi.alk.harness import scenarios as stage
    from fi.alk.harness.backends import ToolServer, ToolSpec
    from fi.alk.harness.contract import AgentContract
    from fi.alk.harness.scenario_tools import TOOL_NAMES

    async def handler(args):
        return {"content": "ok"}

    server = ToolServer(
        name="scenarios",
        version="1",
        tools=[ToolSpec(name=n, description=n, input_schema={}, handler=handler) for n in TOOL_NAMES],
    )
    policy = "POLICY-MARKER"
    with mock.patch.object(stage, "world_summary", lambda _: "w"), mock.patch.object(
        stage, "load_skill", lambda *a, **k: "s"
    ), mock.patch.object(stage, "discovered_skills", lambda **k: ""):
        writer = stage.writer_worker(
            AgentContract(agent="a"), tmp_path, server, budget=40, authoring_guidance=policy
        )[stage.WRITER].instructions
        planner, _ = stage.open_stage(
            AgentContract(agent="a"), out=tmp_path / "suite", wanted=20, authoring_guidance=policy
        )
    said = planner._spec.system_prompt

    assert writer.index(policy) < writer.index("a scenario coherent in itself")
    assert said.index(policy) < said.index("Before you brief anyone")

