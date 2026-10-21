import asyncio
import json
from pathlib import Path

from fi.alk.harness import nudge
from fi.alk.harness.backends import SessionSpec, tool, tool_server
from fi.alk.harness.catalogue import Catalogue, SubGoal, load_catalogue, save_catalogue
from fi.alk.harness.config import gate_hooks
from fi.alk.harness.contract import AgentContract, ToolSpec
from fi.alk.harness.proofs import load_proofs, record_unprovable
from fi.alk.harness.scenario_tools import (
    accept_scenario,
    load_scenarios,
    scenario_tools,
)
from fi.alk.harness.source_data_invariants import ARTIFACT


def _cart(root: Path) -> Path:
    from fi.alk.harness.world import GeneratedWorld
    from fi.alk.harness.world.snapshot import save

    class Cart(GeneratedWorld):
        name = "cart"
        tools = [{"name": "add"}, {"name": "lst"}]
        handlers = {
            "add": (
                "def handle(args, db):\n"
                "    m = db.one('SELECT * FROM menu WHERE id=?', [args['item_id']])\n"
                "    if not m: raise ToolError('no item')\n"
                "    db.execute('INSERT INTO cart (item_id) VALUES (?)', [args['item_id']])\n"
                "    return {'ok': 1}\n"
            ),
            "lst": "def handle(args, db):\n    return db.query('SELECT * FROM cart')\n",
        }

    world = Cart(":memory:")
    world.connection.executescript(
        "CREATE TABLE menu(id TEXT PRIMARY KEY); CREATE TABLE cart(item_id TEXT);"
        "INSERT INTO menu VALUES ('big_mac');"
    )
    world.connection.commit()
    save(world, root, notes="test", sequences=[])
    contract = AgentContract(
        agent="cart",
        real_use_cases=["add an item"],
        tools=[
            ToolSpec(name="add", args=["item_id"], arg_values={"item_id": ["big_mac"]}),
            ToolSpec(name="lst"),
        ],
    )
    (root / "contract.json").write_text(contract.model_dump_json(), encoding="utf-8")
    save_catalogue(
        Catalogue(
            sub_goals=[
                SubGoal(
                    name="item-added",
                    what="the item reached the cart",
                    check=(
                        "def check(world, calls):\n"
                        "    rows = world.state()['cart']\n"
                        "    return None if len(rows) == 1 else 'not one row'\n"
                    ),
                ),
                SubGoal(
                    name="right-item",
                    what="the call carried the item asked for",
                    check=(
                        "def check(world, calls):\n"
                        "    made = [c for c in calls if c.name == 'add' and c.ok]\n"
                        "    if not made: return 'add was never called'\n"
                        "    got = made[0].arguments.get('item_id')\n"
                        "    return None if got == 'big_mac' else 'wrong item'\n"
                    ),
                ),
            ]
        ),
        root,
    )
    return root


def _scenario(name: str = "adds-a-big-mac", **overrides) -> dict:
    payload = {
        "name": name,
        "use_case": "order an item",
        "instruction": "Order one Big Mac.",
        "tests": "whether the agent reaches the menu before adding the item it was asked for",
        "fixture": {"origin": "seed", "item_id": "big_mac"},
        "solution": [
            {"tool": "lst", "arguments": {}},
            {"tool": "add", "arguments": {"item_id": "big_mac"}},
        ],
        "sub_goals": ["item-added", "right-item"],
    }
    payload.update(overrides)
    return payload


def _suite(root: Path, *names: str) -> Path:
    _cart(root)
    kept: list = []
    for name in names:
        said = accept_scenario(
            _scenario(name), world_root=root, catalogue=load_catalogue(root), kept=kept
        )
        assert not said.get("is_error"), said
    return root


def _break_scenario(root: Path, name: str) -> None:
    body = root / "scenarios" / name / "scenario.json"
    payload = json.loads(body.read_text(encoding="utf-8"))
    payload["solution"] = [{"tool": "lst", "arguments": {}}]
    body.write_text(json.dumps(payload), encoding="utf-8")


def _call(server, name: str, args: dict) -> dict:
    handler = next(one.handler for one in server.tools if one.name == name)
    return asyncio.run(handler(args))


def _session(root: Path, stage: str):
    """A scenario server and a nudge server sharing one catalogue and one kept list."""
    contract = AgentContract.model_validate_json(
        (root / "contract.json").read_text(encoding="utf-8")
    )
    catalogue = load_catalogue(root)
    scenarios, kept = scenario_tools(
        contract, root, root, wanted=10, catalogue=catalogue
    )
    server = nudge.nudge_server(
        root, {"nudge_id": "n1", "stage": stage}, catalogue=catalogue, kept=kept
    )
    return scenarios, server, catalogue, kept


# 1. retire_sub_goal (both stages)


def test_retiring_a_sub_goal_in_the_environment_removes_only_the_catalogue_entry(
    tmp_path,
):
    root = _suite(tmp_path, "adds-a-big-mac")

    said = nudge.retire_sub_goal(
        root, "right-item", "the tool lost it", stage="environment"
    )

    assert "retired" in said and not said.startswith("Not")
    assert load_catalogue(root).names() == {"item-added"}
    assert load_scenarios(root)[0].sub_goals == ["item-added", "right-item"]


def test_retiring_in_the_suite_refuses_while_the_catalogue_holds_it(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")

    said = nudge.retire_sub_goal(root, "right-item", "gone", stage="suite")

    assert said.startswith("Not retired")
    assert load_scenarios(root)[0].sub_goals == ["item-added", "right-item"]


def test_retiring_in_the_suite_drops_the_name_from_scenarios(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    nudge.retire_sub_goal(root, "right-item", "gone", stage="environment")

    said = nudge.retire_sub_goal(root, "right-item", "gone", stage="suite")

    assert "dropped from 1 scenario" in said
    assert load_scenarios(root)[0].sub_goals == ["item-added"]
    assert not (
        root / "scenarios" / "adds-a-big-mac" / "checks" / "right-item.py"
    ).exists()


def test_a_sub_goal_added_after_an_environment_retire_does_not_bring_it_back(
    tmp_path,
):
    from fi.alk.harness.world.tools import world_tools

    root = _suite(tmp_path, "adds-a-big-mac")
    scenarios, server, catalogue, _ = _session(root, "environment")
    contract = AgentContract.model_validate_json(
        (root / "contract.json").read_text(encoding="utf-8")
    )
    world, _ = world_tools(contract, root, catalogue=catalogue)

    retired = _call(server, "retire_sub_goal", {"name": "right-item", "why": "gone"})
    added = _call(
        scenarios,
        "add_sub_goal",
        {
            "name": "one-row",
            "what": "the cart holds at most one row",
            "check": (
                "def check(world, calls):\n"
                "    rows = world.state()['cart']\n"
                "    return None if len(rows) <= 1 else 'more than one row'\n"
            ),
        },
    )

    assert not retired.get("is_error") and not added.get("is_error"), added
    assert load_catalogue(root).names() == {"item-added", "one-row"}

    again = _call(
        world,
        "add_sub_goal",
        {
            "name": "cart-exists",
            "what": "the cart table is still there",
            "check": (
                "def check(world, calls):\n"
                "    return None if 'cart' in world.state() else 'no cart table'\n"
            ),
        },
    )

    assert not again.get("is_error"), again
    assert load_catalogue(root).names() == {"item-added", "one-row", "cart-exists"}


def test_a_drop_or_submit_after_a_suite_retire_does_not_bring_it_back(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac", "adds-one-big-mac")
    nudge.retire_sub_goal(root, "right-item", "gone", stage="environment")
    scenarios, server, catalogue, kept = _session(root, "suite")

    retired = _call(server, "retire_sub_goal", {"name": "right-item", "why": "gone"})
    dropped = _call(scenarios, "drop_scenario", {"name": "adds-one-big-mac"})

    assert not retired.get("is_error") and not dropped.get("is_error"), dropped
    assert [one.sub_goals for one in load_scenarios(root)] == [["item-added"]]

    said = accept_scenario(
        _scenario("adds-two-big-macs", sub_goals=["item-added"]),
        world_root=root,
        catalogue=catalogue,
        kept=kept,
    )

    assert not said.get("is_error"), said
    assert {one.name: one.sub_goals for one in load_scenarios(root)} == {
        "adds-a-big-mac": ["item-added"],
        "adds-two-big-macs": ["item-added"],
    }


def test_retiring_needs_a_reason(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    assert nudge.retire_sub_goal(
        root, "right-item", " ", stage="environment"
    ).startswith("Not retired")
    assert load_catalogue(root).names() == {"item-added", "right-item"}


# 2. report_update writes the result; the last call wins


def test_report_update_writes_the_result_and_the_last_call_wins(tmp_path):
    root = _cart(tmp_path)
    server = nudge.nudge_server(root, {"nudge_id": "n1", "stage": "environment"})
    (root / "handlers" / "add.py").write_text("def handle(:\n", encoding="utf-8")

    first = _call(server, "report_update", {"changed": True, "notes": "added a tool"})
    written = json.loads((root / nudge.NUDGE_RESULT).read_text(encoding="utf-8"))

    assert first.get("is_error") is True
    assert written["nudge_id"] == "n1" and written["stage"] == "environment"
    assert written["changed"] is True and written["notes"] == "added a tool"
    assert written["checks"]["passed"] is False
    assert [one["item"] for one in written["checks"]["problems"]] == ["handlers/add.py"]
    assert set(written["checks"]) == {"passed", "problems", "warnings", "not_checked"}
    assert written["left_out"] == []

    (root / "handlers" / "add.py").write_text("def handle(args, db):\n    return 1\n")
    second = _call(server, "report_update", {"changed": False, "notes": "fixed"})
    written = json.loads((root / nudge.NUDGE_RESULT).read_text(encoding="utf-8"))

    assert not second.get("is_error")
    assert written["checks"] == {
        "passed": True,
        "problems": [],
        "warnings": [],
        "not_checked": [],
    }
    assert written["changed"] is False and written["notes"] == "fixed"


# 3. environment_gate


def test_a_healthy_environment_passes_the_static_gate(tmp_path):
    assert nudge.environment_gate(_cart(tmp_path)) == ([], [])


def test_an_invalid_contract_is_reported(tmp_path):
    root = _cart(tmp_path)
    (root / "contract.json").write_text('{"agent": "cart"', encoding="utf-8")
    problems, _ = nudge.environment_gate(root)
    assert [(one["item"], one["kind"]) for one in problems] == [
        ("contract.json", "contract")
    ]


def test_a_mistyped_contract_is_reported_without_its_values(tmp_path):
    root = _cart(tmp_path)
    (root / "contract.json").write_text(
        json.dumps({"agent": "cart", "tools": "sk-live-not-a-list"}), encoding="utf-8"
    )
    problems, _ = nudge.environment_gate(root)
    assert [(one["item"], one["kind"]) for one in problems] == [
        ("contract.json", "contract")
    ]
    assert "tools" in problems[0]["message"]
    assert "sk-live-not-a-list" not in problems[0]["message"]


def test_a_contract_that_fails_validation_is_reported(tmp_path):
    root = _cart(tmp_path)
    (root / "contract.json").write_text(
        AgentContract(agent="cart").model_dump_json(), encoding="utf-8"
    )
    problems, _ = nudge.environment_gate(root)
    assert problems and {one["kind"] for one in problems} == {"contract"}


def test_an_unparsable_world_file_is_reported(tmp_path):
    root = _cart(tmp_path)
    (root / "state.json").write_text("{nope", encoding="utf-8")
    problems, not_checked = nudge.environment_gate(root)
    assert [(one["item"], one["kind"]) for one in problems] == [("state.json", "world")]
    assert not_checked == []


def test_a_handler_with_a_syntax_error_is_reported_and_nothing_is_written(tmp_path):
    root = _cart(tmp_path)
    (root / "handlers" / "lst.py").write_text(
        "def handle(args db):\n", encoding="utf-8"
    )
    problems, _ = nudge.environment_gate(root)
    assert [(one["item"], one["kind"]) for one in problems] == [
        ("handlers/lst.py", "handler")
    ]
    assert not (root / "handlers" / "__pycache__").exists()


def test_a_sub_goal_check_that_does_not_compile_is_reported(tmp_path):
    root = _cart(tmp_path)
    catalogue = load_catalogue(root)
    catalogue.sub_goals[0].check = "def check(world, calls)\n    return None\n"
    save_catalogue(catalogue, root)
    problems, _ = nudge.environment_gate(root)
    assert ("item-added", "sub_goal") in [
        (one["item"], one["kind"]) for one in problems
    ]


def test_a_dangling_scenario_sub_goal_is_ignored(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    nudge.retire_sub_goal(root, "right-item", "gone", stage="environment")
    assert nudge.environment_gate(root) == ([], [])


def test_a_world_not_in_sqlite_is_not_checked(tmp_path):
    root = _cart(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["store"] = "postgres"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert nudge.environment_gate(root) == ([], ["world"])


def test_a_corrupt_sqlite_world_is_reported(tmp_path):
    root = _cart(tmp_path)
    (root / "world.sqlite").write_bytes(b"not a database at all" * 100)
    problems, _ = nudge.environment_gate(root)
    assert [(one["item"], one["kind"]) for one in problems] == [
        ("world.sqlite", "world")
    ]


def _add_tool(root: Path, name: str) -> None:
    path = root / "contract.json"
    contract = AgentContract.model_validate_json(path.read_text(encoding="utf-8"))
    contract.tools.append(ToolSpec(name=name))
    path.write_text(contract.model_dump_json(), encoding="utf-8")


def _set_manifest(root: Path, **fields) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    (root / "manifest.json").write_text(
        json.dumps({**manifest, **fields}), encoding="utf-8"
    )


def test_a_tool_added_without_a_saved_handler_is_reported(tmp_path):
    root = _cart(tmp_path)
    _add_tool(root, "send_otp")
    problems, _ = nudge.environment_gate(root)
    assert [(one["item"], one["kind"]) for one in problems] == [("send_otp", "world")]


def test_a_tool_added_with_a_saved_handler_or_runtime_tool_passes(tmp_path):
    root = _cart(tmp_path)
    _add_tool(root, "send_otp")
    _add_tool(root, "verify_otp")
    _set_manifest(root, tools=["add", "lst", "send_otp"], runtime_tools=["verify_otp"])
    assert nudge.environment_gate(root) == ([], [])


def test_a_tool_added_to_an_empty_world_passes(tmp_path):
    root = _cart(tmp_path)
    _add_tool(root, "send_otp")
    _set_manifest(root, tools=[], tables={})
    assert nudge.environment_gate(root) == ([], [])


def test_a_handler_gap_already_in_the_contract_is_reported(tmp_path):
    root = _cart(tmp_path)
    _add_tool(root, "send_otp")
    server = nudge.nudge_server(root, {"nudge_id": "n1", "stage": "environment"})

    said = _call(server, "report_update", {"changed": False, "notes": "fits"})

    assert said.get("is_error")
    written = json.loads((root / nudge.NUDGE_RESULT).read_text(encoding="utf-8"))
    assert [(one["item"], one["kind"]) for one in written["checks"]["problems"]] == [
        ("send_otp", "world")
    ]


def test_an_uncovered_tool_is_reported_without_a_turn(tmp_path):
    root = _cart(tmp_path)
    _add_tool(root, "send_otp")
    problems, _ = nudge.environment_gate(root)
    assert [(one["item"], one["kind"]) for one in problems] == [("send_otp", "world")]


def _add_tool_sub_goal(root: Path, name: str, tool_name: str) -> None:
    catalogue = load_catalogue(root)
    catalogue.sub_goals.append(
        SubGoal(
            name=name,
            what=f"{tool_name} was used",
            check=(
                "def check(world, calls):\n"
                f"    made = [c for c in calls if c.name == '{tool_name}' and c.ok]\n"
                f"    if not made: return '{tool_name} was never called'\n"
                "    got = made[0].arguments.get('label')\n"
                "    return None if got == 'home' else 'wrong place'\n"
            ),
        )
    )
    save_catalogue(catalogue, root)


def test_a_tool_added_in_the_turn_that_no_sub_goal_checks_is_reported(tmp_path):
    root = _cart(tmp_path)
    server = nudge.nudge_server(root, {"nudge_id": "n1", "stage": "environment"})
    _add_tool(root, "saved_places")
    _set_manifest(root, tools=["add", "lst", "saved_places"])

    said = _call(server, "report_update", {"changed": True, "notes": "fits"})

    assert said.get("is_error")
    written = json.loads((root / nudge.NUDGE_RESULT).read_text(encoding="utf-8"))
    assert [(one["item"], one["kind"]) for one in written["checks"]["problems"]] == [
        ("saved_places", "tool")
    ]


def test_a_tool_added_in_the_turn_with_a_sub_goal_naming_it_passes(tmp_path):
    root = _cart(tmp_path)
    server = nudge.nudge_server(root, {"nudge_id": "n1", "stage": "environment"})
    _add_tool(root, "saved_places")
    _set_manifest(root, tools=["add", "lst", "saved_places"])
    # A longer name that contains it is not a mention.
    _add_tool_sub_goal(root, "other", "saved_places_v2")
    assert nudge.environment_gate(root, {"add", "lst"})[0] == [
        nudge._problem(
            "saved_places", "tool", "added in this rebuild; no sub-goal checks it"
        )
    ]
    _add_tool_sub_goal(root, "used-saved-places", "saved_places")

    said = _call(server, "report_update", {"changed": True, "notes": "fits"})

    assert not said.get("is_error"), said


def test_a_renamed_tool_with_its_sub_goals_updated_or_a_removed_tool_passes(tmp_path):
    root = _cart(tmp_path)
    server = nudge.nudge_server(root, {"nudge_id": "n1", "stage": "environment"})
    path = root / "contract.json"
    contract = AgentContract.model_validate_json(path.read_text(encoding="utf-8"))
    contract.tools = [
        ToolSpec(name="add_item", args=["item_id"], arg_values={"item_id": ["big_mac"]})
    ]
    path.write_text(contract.model_dump_json(), encoding="utf-8")
    _set_manifest(root, tools=["add_item"])
    catalogue = load_catalogue(root)
    for one in catalogue.sub_goals:
        one.check = one.check.replace("'add'", "'add_item'")
    save_catalogue(catalogue, root)

    said = _call(server, "report_update", {"changed": True, "notes": "fits"})

    assert not said.get("is_error"), said


def test_an_untested_tool_already_in_the_contract_is_not_reported(tmp_path):
    root = _cart(tmp_path)
    _add_tool(root, "saved_places")
    _set_manifest(root, tools=["add", "lst", "saved_places"])
    server = nudge.nudge_server(root, {"nudge_id": "n1", "stage": "environment"})

    said = _call(server, "report_update", {"changed": False, "notes": "fits"})

    assert not said.get("is_error"), said


def test_unsaved_world_changes_fail_the_report(tmp_path):
    root = _cart(tmp_path)
    result = nudge.report_update(
        root,
        nudge={"nudge_id": "n1", "stage": "environment"},
        changed=True,
        notes="",
        world={"unsaved": True},
    )
    assert [(one["item"], one["kind"]) for one in result["problems"]] == [
        ("world", "world")
    ]
    assert nudge.report_update(
        root,
        nudge={"nudge_id": "n1", "stage": "environment"},
        changed=True,
        notes="",
        world={"unsaved": False},
    )["passed"]


def test_track_world_holds_unsaved_until_a_save_succeeds():
    outcomes = {"save_world": {"is_error": True}}

    def _named(name: str):
        @tool(name, name, {})
        async def handler(_args: dict) -> dict:
            return outcomes.get(name, {"content": []})

        return handler

    held: dict = {}
    server = nudge.track_world(
        tool_server(
            name="world",
            tools=[_named("seed"), _named("save_world"), _named("check_health")],
        ),
        held,
    )

    _call(server, "check_health", {})
    assert held == {}
    _call(server, "seed", {})
    assert held == {"unsaved": True}
    _call(server, "save_world", {})
    assert held == {"unsaved": True}
    outcomes.pop("save_world")
    _call(server, "save_world", {})
    assert held == {"unsaved": False}


def test_a_world_check_edit_cites_the_agents_source_and_is_reported(tmp_path):
    root = _cart(tmp_path / "work")
    source = tmp_path / "source"
    (source / "db").mkdir(parents=True)
    (source / "db" / "seed.sql").write_text("INSERT INTO menu VALUES ('fries');\n")
    added: list[dict] = []

    @tool("add_world_check", "Add one check.", {})
    async def add_world_check(args: dict) -> dict:
        added.append(args)
        return {"content": []}

    held: dict = {}
    server = nudge.track_world(
        tool_server(name="world", tools=[add_world_check]), held, source
    )
    check = {"name": "menu", "code": "def check(world): return None"}

    uncited = _call(server, "add_world_check", check)
    misquoted = _call(
        server,
        "add_world_check",
        {
            **check,
            "source_path": "db/seed.sql",
            "source_quote": "VALUES ('onion rings')",
        },
    )
    assert uncited["is_error"] and misquoted["is_error"]
    assert added == [] and held == {}

    cited = {
        **check,
        "source_path": "db/seed.sql",
        "source_quote": "INSERT INTO menu VALUES ('fries');",
    }
    assert not _call(server, "add_world_check", cited).get("is_error")
    assert added == [cited]
    assert held == {"unsaved": True, "checks": {"menu": "db/seed.sql"}}
    assert {"source_path", "source_quote"} <= set(
        server.tools[0].input_schema["required"]
    )

    held["unsaved"] = False
    result = nudge.report_update(
        root,
        nudge={"nudge_id": "n1", "stage": "environment"},
        changed=True,
        notes="",
        world=held,
    )
    assert result["passed"]
    assert "world checks changed: menu (cites db/seed.sql)" in result["warnings"]


def test_the_world_reproduces_in_flight_seed_rows_the_facts_do_not_list():
    from fi.alk.harness.config import SKILLS_ROOT

    def said(*parts: str) -> str:
        return " ".join(
            SKILLS_ROOT.joinpath(*parts).read_text(encoding="utf-8").split()
        )

    update = said("builder", "topics", "agent-update.md")
    assert "including in-flight ones such as an active booking" in update
    assert "even when the facts do not list that row as changed" in update
    assert "A world check that keeps such a table empty pins the old seed" in update
    for skill in ("build-environment", "provision-environment"):
        assert (
            "no in-flight work, unless the agent's own seed has them; then reproduce "
            "those rows as they are" in said(skill, "SKILL.md")
        ), skill


# 4. suite_gate


def test_the_suite_gate_proves_into_proofs_json(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    (root / "proofs.json").unlink()

    assert nudge.suite_gate(root, set()) == ([], [], [])
    assert load_proofs(root)["adds-a-big-mac"]["holds"] is True


def test_a_failing_user_draft_scenario_is_not_checked(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    _break_scenario(root, "adds-a-big-mac")

    problems, not_checked, _ = nudge.suite_gate(root, {"adds-a-big-mac"})

    assert problems == [] and not_checked == ["adds-a-big-mac"]
    assert load_proofs(root)["adds-a-big-mac"]["holds"] is False


def test_a_failing_generated_scenario_is_a_problem(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    _break_scenario(root, "adds-a-big-mac")

    problems, not_checked, _ = nudge.suite_gate(root, set())

    assert [(one["item"], one["kind"]) for one in problems] == [
        ("adds-a-big-mac", "scenario")
    ]
    assert problems[0]["message"] and not_checked == []


def test_a_scenario_naming_a_retired_sub_goal_is_a_problem(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    nudge.retire_sub_goal(root, "right-item", "gone", stage="environment")

    problems, not_checked, _ = nudge.suite_gate(root, set())

    assert [(one["item"], one["kind"]) for one in problems] == [
        ("adds-a-big-mac", "scenario")
    ]
    assert "right-item" in problems[0]["message"] and not_checked == []
    assert load_proofs(root)["adds-a-big-mac"]["holds"] is False
    assert nudge.suite_gate(root, {"adds-a-big-mac"})[:2] == ([], ["adds-a-big-mac"])


def test_a_scenario_with_no_sub_goals_left_is_a_problem(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    body = root / "scenarios" / "adds-a-big-mac" / "scenario.json"
    payload = json.loads(body.read_text(encoding="utf-8"))
    payload["sub_goals"], payload["solution"] = [], []
    body.write_text(json.dumps(payload), encoding="utf-8")

    problems, _, _ = nudge.suite_gate(root, set())

    assert [(one["item"], one["kind"]) for one in problems] == [
        ("adds-a-big-mac", "scenario")
    ]


def test_a_solution_calling_a_tool_the_contract_lacks_is_a_problem(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    (root / "contract.json").write_text(
        AgentContract(agent="cart", tools=[ToolSpec(name="lst")]).model_dump_json(),
        encoding="utf-8",
    )

    problems, not_checked, _ = nudge.suite_gate(root, set())

    assert [(one["item"], one["kind"]) for one in problems] == [
        ("adds-a-big-mac", "scenario")
    ]
    assert "contract lacks: add" in problems[0]["message"] and not_checked == []
    assert nudge.suite_gate(root, {"adds-a-big-mac"})[:2] == ([], ["adds-a-big-mac"])


def test_a_gate_that_cannot_run_leaves_no_earlier_result(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    server = nudge.nudge_server(root, {"nudge_id": "n1", "stage": "suite"})
    _call(server, "report_update", {"changed": True, "notes": "fits"})
    (root / "sub_goals.json").write_text("{nope", encoding="utf-8")

    try:
        _call(server, "report_update", {"changed": True, "notes": "again"})
    except ValueError:
        pass

    assert not (root / nudge.NUDGE_RESULT).exists()


def test_user_draft_keys_from_the_nudge_reach_the_suite_gate(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    _break_scenario(root, "adds-a-big-mac")
    server = nudge.nudge_server(
        root,
        {"nudge_id": "n1", "stage": "suite", "user_draft_keys": ["adds-a-big-mac"]},
    )

    _call(server, "report_update", {"changed": True, "notes": "kept the draft"})
    written = json.loads((root / nudge.NUDGE_RESULT).read_text(encoding="utf-8"))

    assert written["checks"]["not_checked"] == ["adds-a-big-mac"]
    assert written["checks"]["passed"] is True


def test_a_dropped_scenario_is_left_out(tmp_path):
    from fi.alk.harness.scenario_tools import write_scenarios

    root = _suite(tmp_path, "adds-a-big-mac", "adds-one-big-mac")
    record_unprovable(root, "dropped-last-week", "dropped")
    server = nudge.nudge_server(root, {"nudge_id": "n1", "stage": "suite"})
    kept = [one for one in load_scenarios(root) if one.name != "adds-one-big-mac"]
    write_scenarios(kept, root)

    _call(server, "report_update", {"changed": True, "notes": "dropped one"})
    written = json.loads((root / nudge.NUDGE_RESULT).read_text(encoding="utf-8"))

    assert [one["key"] for one in written["left_out"]] == ["adds-one-big-mac"]
    assert written["checks"]["passed"] is True
    problems, not_checked, left_out = nudge.suite_gate(root, set())
    assert [one["key"] for one in left_out] == [
        "adds-one-big-mac",
        "dropped-last-week",
    ]


def _add_sub_goal(root: Path, name: str) -> None:
    catalogue = load_catalogue(root)
    catalogue.sub_goals.append(
        SubGoal(
            name=name,
            what="the cart was listed",
            check=(
                "def check(world, calls):\n"
                "    listed = any(c.name == 'lst' and c.ok for c in calls)\n"
                "    return None if listed else 'lst was never called'\n"
            ),
        )
    )
    save_catalogue(catalogue, root)


def test_a_sub_goal_the_environment_added_must_be_named_by_a_scenario(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    _add_sub_goal(root, "cart-listed")
    server = nudge.nudge_server(
        root,
        {
            "nudge_id": "n1",
            "stage": "suite",
            "new_sub_goals": ["cart-listed", "not-in-catalogue"],
        },
    )

    said = _call(server, "report_update", {"changed": True, "notes": "none added"})

    assert said.get("is_error")
    written = json.loads((root / nudge.NUDGE_RESULT).read_text(encoding="utf-8"))
    assert [(one["item"], one["kind"]) for one in written["checks"]["problems"]] == [
        ("cart-listed", "sub_goal")
    ]
    assert nudge.suite_gate(root, set())[0] == []

    body = root / "scenarios" / "adds-a-big-mac" / "scenario.json"
    payload = json.loads(body.read_text(encoding="utf-8"))
    payload["sub_goals"].append("cart-listed")
    body.write_text(json.dumps(payload), encoding="utf-8")

    said = _call(server, "report_update", {"changed": True, "notes": "named it"})
    assert not said.get("is_error"), said


# 5. source_review_stale


def test_a_stale_source_review_is_a_warning_and_nothing_is_re_authored(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    agent = source / "agent.py"
    quote = "def refund_order(order_id: str) -> dict:"
    agent.write_text(quote + "\n    return {}\n", encoding="utf-8")
    root = _cart(tmp_path / "ws")
    review = {
        "checks": [
            {
                "name": "refunds-need-an-order",
                "violations_sql": "SELECT 1 WHERE 0",
                "evidence": [{"path": "agent.py", "quote": quote, "sha256": "x"}],
                "scenarios": [],
            }
        ]
    }
    (root / ARTIFACT).write_text(json.dumps(review), encoding="utf-8")

    assert nudge.source_review_stale(source, root) == []

    agent.write_text("def refund(order_id):\n    return {}\n", encoding="utf-8")
    warnings = nudge.source_review_stale(source, root)

    assert len(warnings) == 1 and "refunds-need-an-order" in warnings[0]
    assert quote not in warnings[0]
    assert json.loads((root / ARTIFACT).read_text(encoding="utf-8")) == review

    result = nudge.report_update(
        root,
        nudge={"nudge_id": "n1", "stage": "environment"},
        changed=True,
        notes="",
        source=source,
    )
    assert result["passed"] is True and result["warnings"] == warnings


def test_a_missing_source_is_not_reported_as_a_stale_review(tmp_path):
    root = _cart(tmp_path / "ws")
    review = {
        "checks": [
            {
                "name": "refunds-need-an-order",
                "evidence": [
                    {"path": "agent.py", "quote": "def refund", "sha256": "x"}
                ],
            }
        ]
    }
    (root / ARTIFACT).write_text(json.dumps(review), encoding="utf-8")

    warnings = nudge.source_review_stale(tmp_path / "not-mounted", root)

    assert warnings == ["source not available; source review not re-checked"]


# 6. filtered_spec + gate_hooks


def _spec() -> SessionSpec:
    async def handler(_args):
        return {"content": []}

    def server(name: str, *tools: str):
        return tool_server(
            name=name, tools=[tool(one, one, {})(handler) for one in tools]
        )

    return SessionSpec(
        system_prompt="",
        servers={
            "contract": server("contract", "amend_contract", "fix_tool"),
            "scenarios": server("scenarios", "submit_scenario", "drop_scenario"),
            "run": server("run", "run_scenarios"),
        },
        builtins=("Read", "Glob", "Grep", "Delegate"),
    )


def _decision(granted, name: str) -> str:
    refuse = gate_hooks(granted)["PreToolUse"][0].hooks[0]
    said = asyncio.run(refuse({"tool_name": name}, None, None))
    return said.get("hookSpecificOutput", {}).get("permissionDecision", "allow")


def test_the_environment_stage_cannot_submit_scenarios():
    spec = _spec()
    granted = nudge.filtered_spec(spec, nudge.ENVIRONMENT_TOOLS).granted()

    assert _decision(granted, "mcp__contract__amend_contract") == "allow"
    assert _decision(granted, "mcp__scenarios__submit_scenario") == "deny"
    assert _decision(granted, "mcp__run__run_scenarios") == "deny"
    assert _decision(granted, "Delegate") == "deny"
    assert _decision(granted, "Read") == "allow"
    assert len(spec.servers["scenarios"].tools) == 2


def test_the_suite_stage_cannot_amend_the_contract():
    granted = nudge.filtered_spec(_spec(), nudge.SUITE_TOOLS).granted()

    assert _decision(granted, "mcp__scenarios__submit_scenario") == "allow"
    assert _decision(granted, "mcp__contract__amend_contract") == "deny"
    assert _decision(granted, "mcp__contract__fix_tool") == "deny"


# 9. renames refused in a suite nudge turn


def test_a_rename_in_a_suite_nudge_turn_is_refused(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    submitted: list[dict] = []

    async def submit(args):
        submitted.append(args)
        return {"content": [{"type": "text", "text": "kept"}]}

    server = nudge.guard_renames(
        tool_server(name="scenarios", tools=[tool("submit_scenario", "", {})(submit)]),
        root,
    )

    renamed = _call(
        server,
        "submit_scenario",
        _scenario("orders-a-big-mac", scenario_key="adds-a-big-mac"),
    )
    rekeyed = _call(server, "submit_scenario", _scenario(scenario_key="something-else"))
    repaired = _call(server, "submit_scenario", _scenario())
    added = _call(server, "submit_scenario", _scenario("adds-two-big-macs"))

    assert renamed["is_error"] and "rename" in renamed["content"][0]["text"]
    assert rekeyed["is_error"]
    assert not repaired.get("is_error") and not added.get("is_error")
    assert [one["name"] for one in submitted] == ["adds-a-big-mac", "adds-two-big-macs"]


def test_the_retire_tool_refuses_as_an_error_and_answers_otherwise(tmp_path):
    root = _suite(tmp_path, "adds-a-big-mac")
    server = nudge.nudge_server(root, {"nudge_id": "n1", "stage": "suite"})

    refused = _call(server, "retire_sub_goal", {"name": "item-added", "why": " "})
    answered = _call(server, "retire_sub_goal", {"name": "nope", "why": "gone"})

    assert refused == {
        "content": [
            {"type": "text", "text": "Not retired: name the sub-goal and say why."}
        ],
        "is_error": True,
    }
    assert answered == {
        "content": [
            {"type": "text", "text": "No scenario names nope; nothing to retire."}
        ]
    }


def test_usage_delta_counts_only_what_the_turn_spent():
    before = {
        "total_usd": 1.0,
        "unpriced_turns": 1,
        "stages": [
            {
                "stage": "builder",
                "turns": 2,
                "models": ["m1"],
                "tokens_in": 10,
                "tokens_out": 5,
                "tokens_cached": 0,
            },
            {
                "stage": "writer",
                "turns": 1,
                "models": ["m0"],
                "tokens_in": 7,
                "tokens_out": 1,
                "tokens_cached": 0,
            },
        ],
    }
    after = {
        "total_usd": 1.25,
        "unpriced_turns": 2,
        "stages": [
            {
                "stage": "builder",
                "turns": 3,
                "models": ["m1", "m2"],
                "tokens_in": 30,
                "tokens_out": 8,
                "tokens_cached": 4,
            },
            {
                "stage": "writer",
                "turns": 1,
                "models": ["m0"],
                "tokens_in": 7,
                "tokens_out": 1,
                "tokens_cached": 0,
            },
        ],
    }
    assert nudge.usage_delta(before, after) == {
        "tokens_in": 20,
        "tokens_out": 3,
        "tokens_cached": 4,
        "usd": 0.25,
        "unpriced_turns": 1,
        "models": ["m1", "m2"],
    }
