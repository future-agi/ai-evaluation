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


@pytest.fixture(autouse=True)
def _judged_sub_goals_decided_without_a_model(monkeypatch):
    """These tests are about wrapping scenarios, not about judging.

    A judged sub-goal now goes to a model, so without this the two consumer-proof tests would make
    a live call and assert on the verdict rather than on the wrapping.
    """
    from fi.alk.harness import hosted_scheduler

    async def _held(goal, world, calls):
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
        source: str, *, label: str, entry: str, allow_empty: bool = True
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


def test_mutation_revert_r1_1_containment_reproduces_the_untyped_escapes(
    tmp_path: Path,
) -> None:
    # Revert-verify-restore: a SCRATCH copy of this module's pre-R1-1-fix content (never a tracked
    # file -- this module did not exist as a tracked file before this task either, see the mutation
    # section's own DUPLICATION DISCLOSURE below) is imported under a private name and driven
    # through the exact same fixtures the tests above use. It reproduces every one of the four
    # untyped escapes the fix closes; the real, fixed `ss` module does not.
    import importlib.util
    import sys as _sys

    prefix_path = Path(
        "/private/tmp/claude-501/-Users-khushalsonawat-Desktop-future-agi/"
        "12a30b1b-5fe7-4808-ae3f-103ab50c6ebc/scratchpad/p12fix1/scenario_source_prefix.py"
    )
    if not prefix_path.is_file():
        pytest.skip("pre-fix scratch copy not present in this environment")
    module_name = "_p12_scenario_source_prefix"
    spec = importlib.util.spec_from_file_location(module_name, prefix_path)
    assert spec is not None and spec.loader is not None
    prefix = importlib.util.module_from_spec(spec)
    # dataclasses' `from __future__ import annotations` string-annotation resolution looks the
    # module up in `sys.modules` by name -- registered (and cleaned up after) purely for that,
    # never left behind for anything else to import.
    _sys.modules[module_name] = prefix
    try:
        spec.loader.exec_module(prefix)

        # (a) module-level sys.exit(0) -- pre-fix: raw SystemExit escapes `load_scenarios` itself.
        root = tmp_path / "a" / ss.SCENARIOS_DIRNAME
        _write_scenario(
            root, "s1", scenario_key="s1", setup_code="import sys\nsys.exit(0)\n"
        )
        with pytest.raises(SystemExit):
            prefix.load_scenarios(tmp_path / "a")
        with pytest.raises(ss.ScenarioDocumentInvalid):
            ss.load_scenarios(tmp_path / "a")

        # (b) non-UTF-8 setup.py -- pre-fix: raw UnicodeDecodeError escapes.
        root_b = tmp_path / "b" / ss.SCENARIOS_DIRNAME
        folder_b = _write_scenario(root_b, "s1", scenario_key="s1")
        (folder_b / "setup.py").write_bytes(
            b"def setup(world):\n    return '\xff\xfe'\n"
        )
        with pytest.raises(UnicodeDecodeError):
            prefix.load_scenarios(tmp_path / "b")
        with pytest.raises(ss.ScenarioDocumentInvalid):
            ss.load_scenarios(tmp_path / "b")

        # (c) unreadable setup.py (chmod 000) -- pre-fix: raw PermissionError escapes.
        root_c = tmp_path / "c" / ss.SCENARIOS_DIRNAME
        folder_c = _write_scenario(
            root_c, "s1", scenario_key="s1", setup_code="def setup(world):\n    pass\n"
        )
        setup_path = folder_c / "setup.py"
        setup_path.chmod(0o000)
        try:
            with pytest.raises(PermissionError):
                prefix.load_scenarios(tmp_path / "c")
            with pytest.raises(ss.ScenarioDocumentInvalid):
                ss.load_scenarios(tmp_path / "c")
        finally:
            setup_path.chmod(0o644)
    finally:
        del _sys.modules[module_name]


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
        self, *, scenario_key: str, from_world: int, to_world: int
    ) -> None:
        self.events.append(("scenario_retried", {}))

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


def test_load_timeout_converts_a_hanging_module_level_scenario_into_a_typed_failure() -> (
    None
):
    async def scenario() -> None:
        tmp_path = Path(tempfile.mkdtemp(prefix="p12-load-timeout-"))
        root = tmp_path / ss.SCENARIOS_DIRNAME
        _write_scenario(
            root,
            "s1",
            scenario_key="s1",
            # Module-level, not inside setup() -- runs during `_compile_entry`'s `exec`, i.e.
            # during the load itself, which is exactly what a real budget must bound.
            setup_code="import time\ntime.sleep(1.5)\ndef setup(world):\n    pass\n",
        )
        source = ss.BundleScenarioSource()
        with mock.patch.object(ss, "_LOAD_TIMEOUT_SECONDS", 0.1):
            with pytest.raises(ss.ScenarioDocumentInvalid, match="exceeded"):
                await source.build(
                    object(),
                    object(),
                    object(),
                    pool=object(),
                    world_factory=object(),
                    bundle_dir=tmp_path,
                )

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
    def _never_fails(source: str, *, label: str, entry: str):
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


def test_begin_payload_carries_operation_run_test_id_and_the_full_key_set() -> None:
    # HarnessScenarioBeginSerializer (futureagi/simulate/serializers/hosted_harness.py:193-198):
    # `scenario_keys` is REQUIRED and `allow_empty=False` -- the full sealed set every time, never
    # a subset (`begin_scenarios`, services/hosted_harness.py:323-329, 409s on anything less).
    scenarios = (_scenario("a"), _scenario("b"))
    payload = ss._begin_payload("run-test-1", scenarios)
    assert payload == {
        "operation": "begin",
        "run_test_id": "run-test-1",
        "scenario_keys": ["a", "b"],
    }


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
# `register_with_platform` -- full sequence: provision -> match (guards) -> begin -> assign.
# -------------------------------------------------------------------------------------------------


def test_register_with_platform_assigns_platform_ids_and_begins_the_full_set() -> None:
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
        assert client.begin_calls == [
            {
                "operation": "begin",
                "run_test_id": "run-test-1",
                "scenario_keys": ["a", "b"],
            },
        ]

    asyncio.run(scenario())


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
    # Mutant: `register_with_platform`'s final `replace(scenario, scenario_id=...)` step deleted --
    # provision/begin both still run (a response-shape bug would not be caught by this mutant), but
    # the scenarios handed back to the scheduler never actually carry the platform's id.
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
            provision_result = await asyncio.to_thread(
                scenarios_client.provision, ss._provision_payload(run_name, scenarios)
            )
            await asyncio.to_thread(
                scenarios_client.begin,
                ss._begin_payload(provision_result["run_test_id"], scenarios),
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
    """Provisioning defaults to text, and a voice run that stays quiet has its evals judge a transcript.

    Measured on run 0734ab2e: four eval configs were created with `conversation -> transcript` on a
    voice run, because the guest never sent `modality` and `_resolve_scenario_modality` fell back to
    text. For a spoken call the conversation is the recording.
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
