"""The world's cast: sub-actors, how scenarios bind them, and that they are sealed with the world."""

from __future__ import annotations

import asyncio
import json

import pytest

from fi.alk.harness.cast import (
    Binding,
    Cast,
    FaultProfile,
    SubActor,
    Behaviour,
    Trigger,
    binding_problems,
    bound_faults,
    bound_voices,
    load_cast,
    save_cast,
    sub_actor_problems,
)
from fi.alk.harness.contract import AgentContract, ToolSpec


def _voice() -> SubActor:
    return SubActor(
        id="subactor_bystander",
        name="Bystander",
        role="someone near the caller",
        kind="behavioral",
        channel="audio_mix",
        behaviour=Behaviour(prompt="wants the caller to hurry and asks for an extra item"),
    )


def _service() -> SubActor:
    return SubActor(
        id="subactor_service",
        name="Item service",
        role="third-party service behind add",
        kind="system",
        channel="tool_response",
        depends_on=["add"],
        faults=[FaultProfile(tool="add", kind="timeout")],
    )


def _cart_world():
    from fi.alk.harness.world import GeneratedWorld

    class W(GeneratedWorld):
        name = "cart"
        tools = [{"name": "add"}, {"name": "lst"}]
        handlers = {
            "add": (
                "def handle(args, db):\n"
                "    m = db.one('SELECT * FROM menu WHERE id=?', [args['item_id']])\n"
                "    if not m: raise ToolError('no item %r' % args['item_id'])\n"
                "    db.execute('INSERT INTO cart (item_id) VALUES (?)', [args['item_id']])\n"
                "    return {'ok': 1}\n"
            ),
            "lst": "def handle(args, db):\n    return db.query('SELECT * FROM cart')\n",
        }

    world = W(":memory:")
    world.connection.executescript(
        "CREATE TABLE menu(id TEXT PRIMARY KEY); CREATE TABLE cart(item_id TEXT);"
    )
    world.connection.execute("INSERT INTO menu VALUES ('big_mac')")
    world.connection.commit()
    contract = AgentContract(
        agent="cart",
        modality="voice",
        real_use_cases=["add an item"],
        tools=[
            ToolSpec(name="add", args=["item_id"], arg_values={"item_id": ["big_mac"]}),
            ToolSpec(name="lst"),
        ],
    )
    return world, contract


def _environment(tmp_path):
    from fi.alk.harness.catalogue import Catalogue, SubGoal, save_catalogue
    from fi.alk.harness.world.snapshot import save

    world, contract = _cart_world()
    save(world, tmp_path, notes="test", sequences=[])
    catalogue = Catalogue(
        sub_goals=[
            SubGoal(
                name="item-added",
                what="the item reached the cart",
                check=(
                    "def check(world, calls):\n"
                    "    rows = world.state()['cart']\n"
                    "    if len(rows) != 1: return '%d rows, expected 1' % len(rows)\n"
                    "    return None\n"
                ),
            ),
            SubGoal(
                name="right-item",
                what="the call carried the item that was asked for",
                check=(
                    "def check(world, calls):\n"
                    "    made = [c for c in calls if c.name == 'add' and c.ok]\n"
                    "    if not made: return 'add was never called'\n"
                    "    got = made[0].arguments.get('item_id')\n"
                    "    return None if got == 'big_mac' else 'added %r' % got\n"
                ),
            ),
        ]
    )
    save_catalogue(catalogue, tmp_path)
    return tmp_path, contract, catalogue


def _scenario(**overrides):
    payload = {
        "name": "adds-a-big-mac",
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


def test_the_cast_round_trips_and_a_missing_file_is_an_empty_cast(tmp_path):
    assert load_cast(tmp_path) == Cast()
    cast = Cast(sub_actors=[_voice(), _service()])
    save_cast(tmp_path, cast)
    assert load_cast(tmp_path) == cast


def test_a_binding_can_be_written_as_a_bare_id():
    assert Binding.model_validate("subactor_service").id == "subactor_service"


@pytest.mark.parametrize(
    ("change", "said"),
    [
        ({"channel": "pager"}, "is not one of"),
        ({"depends_on": ["refund"]}, "does not have: refund"),
        ({"faults": [FaultProfile(tool="add", kind="explode")]}, "'explode'"),
        ({"faults": [FaultProfile(tool="add", kind="latency")]}, "needs latency_ms"),
        ({"faults": [FaultProfile(tool="add", kind="content")]}, "needs the content"),
        ({"faults": [FaultProfile(tool="add", kind="error", on_call=0)]}, "counts from 1"),
        ({"faults": []}, "needs at least one fault"),
    ],
)
def test_a_system_sub_actor_is_refused_for_each_way_it_cannot_run(change, said):
    actor = _service().model_copy(update=change)
    problems = sub_actor_problems(actor, modality="voice", tools={"add", "lst"}, faults_reach=True)
    assert any(said in one for one in problems), problems


def test_a_voice_sub_actor_needs_a_voice_agent_and_a_behaviour():
    assert not sub_actor_problems(_voice(), modality="voice", tools=set(), faults_reach=False)
    assert sub_actor_problems(_voice(), modality="chat", tools=set(), faults_reach=False)
    silent = _voice().model_copy(update={"behaviour": None})
    assert sub_actor_problems(silent, modality="voice", tools=set(), faults_reach=False)


def test_a_fault_on_tools_nothing_in_the_run_stands_between_is_refused():
    problems = sub_actor_problems(_service(), modality="voice", tools={"add"}, faults_reach=False)
    assert any("could never be applied" in one for one in problems)


def test_bindings_are_refused_when_they_cannot_run():
    cast = Cast(sub_actors=[_voice(), _service()])

    def problems(*bindings, modality="voice", max_turns=10):
        return binding_problems(list(bindings), cast, modality=modality, max_turns=max_turns)

    assert not problems(Binding(id="subactor_bystander", trigger=Trigger(kind="mid_journey", turn=2)))
    assert not problems(Binding(id="subactor_service", trigger=Trigger(kind="on_tool", tool="add", on_call=2)))
    assert problems(Binding(id="nobody"))
    assert problems(Binding(id="subactor_bystander", trigger=Trigger(kind="probability")))
    assert problems(Binding(id="subactor_bystander", trigger=Trigger(kind="mid_journey", turn=11)))
    assert problems(Binding(id="subactor_service", trigger=Trigger(kind="on_tool", tool="lst")))
    assert problems(Binding(id="subactor_bystander"), modality="chat")
    assert problems(Binding(id="subactor_service"), Binding(id="subactor_service"))


def test_a_call_carries_the_faults_and_voices_its_scenario_binds():
    cast = Cast(sub_actors=[_voice(), _service()])
    bindings = [
        Binding(id="subactor_service", trigger=Trigger(kind="on_tool", tool="add", on_call=3)),
        Binding(id="subactor_bystander", trigger=Trigger(kind="mid_journey", turn=2)),
    ]
    assert bound_faults(bindings, cast) == [
        {"tool": "add", "kind": "timeout", "on_call": 3, "latency_ms": 0, "content": "", "sub_actor": "subactor_service"}
    ]
    voices = bound_voices(bindings, cast)
    assert [one["id"] for one in voices] == ["subactor_bystander"]
    assert voices[0]["trigger"]["turn"] == 2
    assert bound_faults([], cast) == [] and bound_voices([], cast) == []


def _cast_tools(tmp_path, monkeypatch, *, enabled=True):
    from fi.alk.harness.world.tools import world_tools

    if enabled:
        monkeypatch.setenv("HARNESS_SUB_ACTORS", "1")
    else:
        monkeypatch.delenv("HARNESS_SUB_ACTORS", raising=False)
    root, contract, _catalogue = _environment(tmp_path)
    server, _world = world_tools(contract, root)
    return root, {one.name: one for one in server.tools}


def test_the_cast_tools_are_offered_only_with_the_switch_on(tmp_path, monkeypatch):
    _root, tools = _cast_tools(tmp_path, monkeypatch, enabled=False)
    assert "add_sub_actor" not in tools and "drop_sub_actor" not in tools


def test_add_sub_actor_writes_the_cast_and_refuses_what_cannot_run(tmp_path, monkeypatch):
    root, tools = _cast_tools(tmp_path, monkeypatch)
    add = tools["add_sub_actor"].handler

    said = asyncio.run(add({**_voice().model_dump(), "primary_role": "someone ordering food"}))
    assert not said.get("is_error"), said
    said = asyncio.run(add(_service().model_dump()))
    assert not said.get("is_error"), said
    cast = load_cast(root)
    assert [one.id for one in cast.sub_actors] == ["subactor_bystander", "subactor_service"]
    assert cast.primary.role == "someone ordering food"

    refused = asyncio.run(add({**_service().model_dump(), "id": "subactor_other", "depends_on": ["refund"]}))
    assert refused["is_error"] and "refund" in refused["content"][0]["text"]


def test_a_bound_sub_actor_cannot_be_dropped(tmp_path, monkeypatch):
    root, tools = _cast_tools(tmp_path, monkeypatch)
    asyncio.run(tools["add_sub_actor"].handler(_service().model_dump()))
    folder = root / "scenarios" / "uses-the-service"
    folder.mkdir(parents=True)
    (folder / "scenario.json").write_text(json.dumps({"sub_actors": [{"id": "subactor_service"}]}))

    refused = asyncio.run(tools["drop_sub_actor"].handler({"id": "subactor_service"}))
    assert refused["is_error"] and "uses-the-service" in refused["content"][0]["text"]

    (folder / "scenario.json").write_text(json.dumps({}))
    dropped = asyncio.run(tools["drop_sub_actor"].handler({"id": "subactor_service"}))
    assert not dropped.get("is_error") and load_cast(root).sub_actors == []


def test_a_scenario_binds_only_what_the_cast_holds(tmp_path, monkeypatch):
    from fi.alk.harness.scenario_tools import accept_scenario

    monkeypatch.setenv("HARNESS_SUB_ACTORS", "1")
    root, _contract, catalogue = _environment(tmp_path)
    save_cast(root, Cast(sub_actors=[_voice(), _service()]))

    refused = accept_scenario(
        _scenario(sub_actors=[{"id": "nobody"}]), world_root=root, catalogue=catalogue, kept=[]
    )
    assert refused["is_error"] and "nobody is not in the cast" in refused["content"][0]["text"]

    kept = []
    said = accept_scenario(
        _scenario(sub_actors=[{"id": "subactor_bystander", "trigger": {"kind": "mid_journey", "turn": 2}}]),
        world_root=root,
        catalogue=catalogue,
        kept=kept,
    )
    assert not said.get("is_error"), said
    written = json.loads((root / "scenarios" / "adds-a-big-mac" / "scenario.json").read_text())
    assert written["sub_actors"] == [{"id": "subactor_bystander", "trigger": {"kind": "mid_journey", "turn": 2}}]


def test_with_the_switch_off_a_binding_is_refused_and_nothing_changes(tmp_path, monkeypatch):
    from fi.alk.harness.scenario_tools import accept_scenario

    monkeypatch.delenv("HARNESS_SUB_ACTORS", raising=False)
    root, _contract, catalogue = _environment(tmp_path)
    refused = accept_scenario(
        _scenario(sub_actors=[{"id": "subactor_bystander"}]), world_root=root, catalogue=catalogue, kept=[]
    )
    assert refused["is_error"] and "no sub-actors" in refused["content"][0]["text"]

    said = accept_scenario(_scenario(), world_root=root, catalogue=catalogue, kept=[])
    assert not said.get("is_error"), said
    written = json.loads((root / "scenarios" / "adds-a-big-mac" / "scenario.json").read_text())
    assert "sub_actors" not in written


def test_the_bundle_seals_the_cast_beside_the_catalogue(tmp_path):
    from fi.alk.harness.bundle_author_v2 import _copy_sub_goal_catalogue

    authoring, staging = tmp_path / "authoring", tmp_path / "staging"
    authoring.mkdir()
    staging.mkdir()
    (authoring / "sub_goals.json").write_text('{"sub_goals": []}')
    assert _copy_sub_goal_catalogue(authoring, staging) == ["sub_goals.json"]
    assert not (staging / "cast.json").exists()

    save_cast(authoring, Cast(sub_actors=[_service()]))
    assert _copy_sub_goal_catalogue(authoring, staging) == ["sub_goals.json", "cast.json"]
    assert load_cast(staging).sub_actors[0].id == "subactor_service"


def test_the_builder_keeps_the_cast_from_chat_and_asks_before_changing_it(tmp_path, monkeypatch):
    from fi.alk.harness.builder import suite_server
    from fi.alk.harness.hosted_chat_entrypoint import _NEEDS_CONFIRMATION

    root, contract, _catalogue = _environment(tmp_path)
    monkeypatch.delenv("HARNESS_SUB_ACTORS", raising=False)
    assert "add_sub_actor" not in {one.name for one in suite_server(root, [], contract).tools}

    monkeypatch.setenv("HARNESS_SUB_ACTORS", "1")
    tools = {one.name: one for one in suite_server(root, [], contract).tools}
    assert {"add_sub_actor", "drop_sub_actor"} <= set(tools)
    assert {"add_sub_actor", "drop_sub_actor"} <= _NEEDS_CONFIRMATION
    said = asyncio.run(tools["add_sub_actor"].handler(_service().model_dump()))
    assert not said.get("is_error"), said
    assert load_cast(root).sub_actors[0].id == "subactor_service"


def test_every_trigger_kind_is_checked_before_a_run():
    cast = Cast(sub_actors=[_voice(), _service()])

    def problems(binding):
        return binding_problems([binding], cast, modality="voice", max_turns=10, tools={"add", "lst", "transfer"})

    assert not problems(Binding(id="subactor_bystander", trigger=Trigger(kind="on_keyword", words=["manager"])))
    assert problems(Binding(id="subactor_bystander", trigger=Trigger(kind="on_keyword")))
    assert not problems(Binding(id="subactor_bystander", trigger=Trigger(kind="on_escalation", tool="transfer")))
    assert problems(Binding(id="subactor_bystander", trigger=Trigger(kind="on_escalation", tool="refund")))
    assert not problems(Binding(id="subactor_bystander", trigger=Trigger(kind="probability", p=0.4)))
    assert problems(Binding(id="subactor_bystander", trigger=Trigger(kind="probability", p=1.0)))
    assert problems(Binding(id="subactor_service", trigger=Trigger(kind="on_keyword", words=["x"])))


def test_a_channel_is_refused_where_no_run_can_carry_it():
    speaker = _voice().model_copy(update={"channel": "thread_message"})
    assert not sub_actor_problems(speaker, modality="chat", tools=set(), faults_reach=False)
    assert sub_actor_problems(speaker, modality="voice", tools=set(), faults_reach=False)
    page = _voice().model_copy(update={"channel": "page_content"})
    assert any("no run" in one for one in sub_actor_problems(page, modality="browser", tools=set(), faults_reach=False))
    arrives = SubActor(id="s", name="Inbox", kind="behavioral", channel="inbound_message", behaviour=Behaviour(prompt="x"))
    assert sub_actor_problems(arrives, modality="chat", tools=set(), faults_reach=False)
    arrives = arrives.model_copy(update={"inbound": {"collection": "messages", "record": {"body": "hi"}}})
    assert not sub_actor_problems(arrives, modality="chat", tools=set(), faults_reach=False)


def test_a_sub_actor_that_may_not_enter_needs_a_sub_goal_about_it(tmp_path, monkeypatch):
    from fi.alk.harness.catalogue import SubGoal, save_catalogue
    from fi.alk.harness.scenario_tools import accept_scenario

    monkeypatch.setenv("HARNESS_SUB_ACTORS", "1")
    root, _contract, catalogue = _environment(tmp_path)
    save_cast(root, Cast(sub_actors=[_voice()]))
    bound = [{"id": "subactor_bystander", "trigger": {"kind": "probability", "p": 0.5}}]
    refused = accept_scenario(_scenario(sub_actors=bound), world_root=root, catalogue=catalogue, kept=[])
    assert refused["is_error"] and "may not enter" in refused["content"][0]["text"]

    catalogue.sub_goals.append(
        SubGoal(name="ignored-bystander", what="w", judged="Applies when: x\nPass when: y\nFail when: z\nIf it does not arise: Pass", sub_actor="subactor_bystander")
    )
    save_catalogue(catalogue, root)
    said = accept_scenario(
        _scenario(sub_actors=bound, sub_goals=["item-added", "right-item", "ignored-bystander"]),
        world_root=root,
        catalogue=catalogue,
        kept=[],
    )
    assert "may not enter" not in str(said)


def test_the_answer_key_is_proved_with_the_bound_service_failing(tmp_path, monkeypatch):
    from fi.alk.harness.scenario_tools import accept_scenario

    monkeypatch.setenv("HARNESS_SUB_ACTORS", "1")
    root, _contract, catalogue = _environment(tmp_path)
    failing = _service().model_copy(update={"faults": [FaultProfile(tool="add", kind="error", on_call=1)]})
    save_cast(root, Cast(sub_actors=[failing]))
    bound = [{"id": "subactor_service", "trigger": {"kind": "always"}}]

    gives_up = accept_scenario(_scenario(sub_actors=bound), world_root=root, catalogue=catalogue, kept=[])
    assert gives_up["is_error"]

    retries = _scenario(
        sub_actors=bound,
        solution=[
            {"tool": "lst", "arguments": {}},
            {"tool": "add", "arguments": {"item_id": "big_mac"}},
            {"tool": "add", "arguments": {"item_id": "big_mac"}},
        ],
    )
    said = accept_scenario(retries, world_root=root, catalogue=catalogue, kept=[])
    assert not said.get("is_error"), said
