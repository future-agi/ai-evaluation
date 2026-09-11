from types import SimpleNamespace

import pytest

from fi.alk.harness import prove as module
from fi.alk.harness.catalogue import Catalogue, SubGoal
from fi.alk.harness.world.runtime import Call


@pytest.mark.parametrize(
    "condition,call,accepted",
    [
        ("c.arguments.get('method')", Call("select", {"method": "card"}), False),
        (
            "c.ok and c.arguments.get('method')",
            Call("select", {"method": "card"}),
            True,
        ),
        (
            "c.refused and c.arguments.get('method')",
            Call("select", {"method": "card"}, ok=False, refused=True),
            True,
        ),
        (
            "not c.ok and c.arguments.get('method')",
            Call("select", {"method": "card"}, ok=False, refused=True),
            False,
        ),
    ],
)
def test_proof_requires_outcome_not_just_attempt(
    monkeypatch, tmp_path, condition, call, accepted
):
    world = SimpleNamespace(close=lambda: None)
    outcome = SimpleNamespace(ok=True, said="", broken=False)
    monkeypatch.setattr(module, "prepared", lambda *a: (world, outcome, outcome))
    monkeypatch.setattr(
        module,
        "_run",
        lambda *a, with_solution: (world, [call] if with_solution else [], [], []),
    )
    catalogue = Catalogue(
        sub_goals=[
            SubGoal(
                name="selected",
                what="Select payment",
                check=(
                    "def check(world, calls):\n"
                    f'    return None if any({condition} for c in calls) else "not achieved"\n'
                ),
            )
        ]
    )
    proof = module.prove(SimpleNamespace(sub_goals=["selected"]), catalogue, tmp_path)
    assert proof.holds is accepted
    assert bool(proof.failed_attempts) is not accepted
    if not accepted:
        assert "every tool attempt crashes" in proof.why()
    assert call.error == ""  # Mutation never edits the original evidence.
