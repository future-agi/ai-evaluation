"""Grading with sub-actors: their turns are attributed, and a sub-goal about one that never entered cannot fail."""

from __future__ import annotations

import json
from fi.alk.harness.catalogue import Catalogue, SubGoal, load_catalogue, save_catalogue
from fi.alk.harness.cast import absent
from fi.alk.harness.judge import _others_present, _transcript




def test_a_sub_actor_turn_is_named_and_marked_as_not_the_customer():
    messages = [
        {"role": "assistant", "content": "How can I help?"},
        {"role": "sub_actor", "name": "Bystander", "sub_actor": "subactor_bystander", "content": "Add one more."},
        {"role": "user", "content": "Just the one, thanks."},
    ]
    said = _transcript(messages)
    assert "Bystander (not the customer): Add one more." in said
    assert "Agent: How can I help?" in said and "Customer: Just the one" in said
    assert "Bystander" in _others_present(messages)
    assert _others_present([{"role": "user", "content": "hi"}]) == ""


def test_a_sub_goal_about_a_sub_actor_that_never_entered_holds_with_the_reason():
    said = absent("subactor_bystander", [{"role": "user", "content": "hi"}])
    assert said.startswith("Not applicable") and "subactor_bystander" in said

    heard = [{"role": "sub_actor", "sub_actor": "subactor_bystander", "content": "Add one."}]
    assert absent("subactor_bystander", heard) == ""
    assert absent("", []) == ""


def test_the_catalogue_names_a_sub_actor_only_when_one_is_named(tmp_path):
    save_catalogue(
        Catalogue(
            sub_goals=[
                SubGoal(name="plain", what="w", judged="j"),
                SubGoal(name="about", what="w", judged="j", sub_actor="subactor_bystander"),
            ]
        ),
        tmp_path,
    )
    written = json.loads((tmp_path / "sub_goals.json").read_text())["sub_goals"]
    assert "sub_actor" not in written[0] and written[1]["sub_actor"] == "subactor_bystander"
    assert load_catalogue(tmp_path).named("about").sub_actor == "subactor_bystander"


def test_the_hosted_sub_goal_carries_its_sub_actor_from_the_catalogue(tmp_path):
    from fi.alk.harness.scenario_source import _CompiledScenario, _CompiledSubGoal, _load_catalogue_claims, _with_claims

    (tmp_path / "sub_goals.json").write_text(
        json.dumps({"sub_goals": [{"name": "about", "what": "w", "judged": "j", "sub_actor": "subactor_bystander"}]})
    )
    scenario = _CompiledScenario(
        scenario_key="k",
        scenario_id="",
        sub_goals=(_CompiledSubGoal(name="about", judged="judged", check=lambda *_a: None),),
        setup=lambda *_a: None,
        ready=lambda *_a: None,
    )
    restored = _with_claims(scenario, _load_catalogue_claims(tmp_path))
    assert restored.sub_goals[0].sub_actor == "subactor_bystander"


def test_local_grading_lets_a_sub_goal_about_an_absent_sub_actor_stand_aside(tmp_path):
    from fi.alk.harness.run.grade import grade_sub_goals
    from fi.alk.harness.scenario import Scenario

    catalogue = Catalogue(
        sub_goals=[
            SubGoal(name="about", what="w", check="def check(world, calls):\n    return 'never reached'\n", sub_actor="subactor_bystander")
        ]
    )
    scenario = Scenario(name="s", instruction="x", sub_goals=["about"])
    outcome = grade_sub_goals(None, scenario, catalogue, [], [])[0]
    assert outcome.held and outcome.said.startswith("Not applicable")

    heard = [{"role": "sub_actor", "sub_actor": "subactor_bystander", "content": "Add one."}]
    assert not grade_sub_goals(None, scenario, catalogue, [], heard)[0].held


def test_a_sub_actor_turn_keeps_its_name_in_the_spoken_transcript():
    from fi.alk.harness.run.conversation import Exchange, Transcript

    transcript = Transcript(exchanges=[Exchange("sub_actor", "add one", name="Friend", sub_actor="s")])
    assert transcript.spoken() == "Friend (not the customer): add one"
