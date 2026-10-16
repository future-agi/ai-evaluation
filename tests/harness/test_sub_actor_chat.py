"""Sub-actors in a chat: a second participant the agent and the person both see, or a record arriving."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fi.alk.harness.cast import Behaviour, Binding, Cast, SubActor, Trigger, save_cast
from fi.alk.harness.run import conversation
from fi.alk.harness.run.conversation import Exchange, Transcript, _ChatSubActors
from fi.alk.harness.scenario import Scenario


class _Stage:
    def __init__(self, spec, name=""):
        self.spec = spec

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def say(self, _text):
        return SimpleNamespace(text="Can you add one more for me?")


class _World:
    def __init__(self):
        self.calls = []
        self.added = []

    def add(self, collection, record):
        self.added.append((collection, record))


class _Target:
    def __init__(self):
        self.world = _World()
        self.heard = []

    def interject(self, name, text):
        self.heard.append((name, text))


def _setup(tmp_path, monkeypatch, actor, trigger):
    monkeypatch.setenv("HARNESS_SUB_ACTORS", "1")
    monkeypatch.setattr(conversation, "Stage", _Stage)
    save_cast(tmp_path, Cast(sub_actors=[actor]))
    scenario = Scenario(name="s", instruction="Order one item.", sub_actors=[Binding(id=actor.id, trigger=trigger)])
    return _ChatSubActors(scenario, tmp_path, model=None, seed="run-1")


def test_a_second_participant_writes_after_the_agent_turn_its_trigger_names(tmp_path, monkeypatch):
    actor = SubActor(
        id="subactor_friend",
        name="Friend",
        kind="behavioral",
        channel="thread_message",
        behaviour=Behaviour(prompt="wants an extra item", max_lines=1),
    )
    others = _setup(tmp_path, monkeypatch, actor, Trigger(kind="mid_journey", turn=2))
    target = _Target()

    async def scenario():
        first = await others.after_agent(1, "What would you like?", target)
        second = await others.after_agent(2, "One item, got it.", target)
        third = await others.after_agent(3, "Anything else?", target)
        return first, second, third

    first, second, third = asyncio.run(scenario())
    assert first == [] and third == []
    assert second == [Exchange("sub_actor", "Can you add one more for me?", name="Friend", sub_actor="subactor_friend")]
    assert target.heard == [("Friend", "Can you add one more for me?")]


def test_a_record_arrives_in_the_world_when_its_trigger_fires(tmp_path, monkeypatch):
    actor = SubActor(
        id="subactor_inbox",
        name="Inbox",
        kind="behavioral",
        channel="inbound_message",
        behaviour=Behaviour(prompt="a message arrives"),
        inbound={"collection": "messages", "record": {"body": "Your order changed"}},
    )
    others = _setup(tmp_path, monkeypatch, actor, Trigger(kind="on_keyword", words=["confirm"]))
    target = _Target()

    async def scenario():
        return await others.after_agent(1, "Let me look.", target), await others.after_agent(2, "Please confirm.", target)

    before, after = asyncio.run(scenario())
    assert before == []
    assert target.world.added == [("messages", {"body": "Your order changed"})]
    assert after[0].speaker == "sub_actor" and after[0].sub_actor == "subactor_inbox"


def test_a_sub_actor_that_cannot_speak_is_skipped_and_the_call_goes_on(tmp_path, monkeypatch):
    actor = SubActor(id="subactor_friend", name="Friend", kind="behavioral", channel="thread_message", behaviour=Behaviour(prompt="x"))
    others = _setup(tmp_path, monkeypatch, actor, Trigger(kind="mid_journey", turn=1))

    async def refused(self, _text):
        raise RuntimeError("model refused")

    monkeypatch.setattr(_Stage, "say", refused)
    target = _Target()
    assert asyncio.run(others.after_agent(1, "Hello.", target)) == [] and target.heard == []


def test_with_the_switch_off_nobody_else_enters(tmp_path, monkeypatch):
    actor = SubActor(id="subactor_friend", name="Friend", kind="behavioral", channel="thread_message", behaviour=Behaviour(prompt="x"))
    others = _setup(tmp_path, monkeypatch, actor, Trigger())
    monkeypatch.delenv("HARNESS_SUB_ACTORS")
    scenario = Scenario(name="s", instruction="x", sub_actors=[Binding(id="subactor_friend")])
    assert _ChatSubActors(scenario, tmp_path, model=None, seed="run-1").entering == []
    assert others.entering


def test_a_sub_actor_turn_keeps_its_name_in_both_message_forms():
    transcript = Transcript(
        exchanges=[
            Exchange("customer", "hi"),
            Exchange("sub_actor", "add one", name="Friend", sub_actor="subactor_friend"),
            Exchange("agent", "hello"),
        ]
    )
    assert transcript.canonical_messages()[1] == {
        "role": "sub_actor",
        "name": "Friend",
        "sub_actor": "subactor_friend",
        "content": "add one",
    }
    assert [one["role"] for one in transcript.canonical_messages()] == ["user", "sub_actor", "assistant"]
    assert b'"role": "sub_actor"' in transcript.artifact()


def test_the_hosted_chat_target_hears_a_second_participant_by_name():
    from fi.alk.harness.chat_call_runner import _HostedChatTarget

    target = _HostedChatTarget.__new__(_HostedChatTarget)
    target._messages = []
    target.interject("Second person", "add one")
    assert target._messages == [{"role": "user", "name": "Second_person", "content": "add one"}]
