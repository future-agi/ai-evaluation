"""Unit 4 (BBG U4 / ARCH §2e) — extension registries + extension_admission."""
from __future__ import annotations

import pytest

from fi.alk import extensions as E
from fi.alk.extensions import ExtensionError
from fi.simulate.simulation import contract as C


@pytest.fixture(autouse=True)
def _clean():
    E._reset_extensions()
    yield
    E._reset_extensions()


def test_registration_round_trip_each_point():
    rec = E.register_objective({"name": "acme.cost", "evidence_class_capability": ["local_gate"]})
    assert rec["point"] == "loss"
    assert "acme.cost" in E.registered("loss")
    assert E.resolve("loss", "acme.cost")["name"] == "acme.cost"


def test_collision_rejected():
    E.register_generator({"name": "acme.gen"})
    with pytest.raises(ExtensionError, match="collision"):
        E.register_generator({"name": "acme.gen"})


def test_optimizer_without_budgets_rejected():
    with pytest.raises(ExtensionError, match="declared_budgets"):
        E.register_optimizer({"name": "acme.opt"})


def test_optimizer_with_budgets_ok():
    rec = E.register_optimizer({"name": "acme.opt", "declared_budgets": {"eval_budget": 8},
                                "evidence_class_capability": ["local_gate"]})
    assert rec["name"] == "acme.opt"


def test_admission_refusal_when_gated_empty_evidence():
    rec = E.register_objective({"name": "acme.loss", "evidence_class_capability": []})
    out = E.extension_admission(rec, {"gated": True})
    assert out["admitted"] is False
    assert out["type"] == "extension_evidence_inadmissible"


def test_admission_passthrough_non_gated():
    rec = E.register_objective({"name": "acme.loss"})
    out = E.extension_admission(rec, {"gated": False})
    assert out["admitted"] is True


def test_admission_gated_green():
    rec = E.register_objective({"name": "acme.loss", "evidence_class_capability": ["captured_fixture"]})
    rec["conformance_green"] = True
    out = E.extension_admission(rec, {"gated": True})
    assert out["admitted"] is True


def test_world_kind_registration_pushes_into_contract():
    E.register_environment({
        "name": "acme.simworld",
        "kind_token": "acme.simworld",
        "spec_validator": "acme.validate",
        "rung_ladder": {1: "local_gate"},
        "evidence_class_capability": ["local_gate"],
    })
    assert "acme.simworld" in C.resolved_world_kinds()
    # resolution now sees the custom kind
    ws = C.WorldSpec(kind="acme.simworld")
    assert ws.kind == "acme.simworld"


def test_world_kind_missing_validator_rejected():
    with pytest.raises(ExtensionError, match="spec_validator"):
        E.register_environment({"name": "acme.bad", "kind_token": "acme.bad"})


def test_canon_tuples_unmutated_after_registration():
    before_kinds = C.SIMULATION_WORLD_KINDS
    before_roles = C.SIMULATION_CAST_ROLES
    E.register_environment({
        "name": "acme.k", "kind_token": "acme.k", "spec_validator": "v",
        "rung_ladder": {1: "local_gate"},
    })
    E.register_role({"name": "acme.role", "role": "acme.role"})
    assert C.SIMULATION_WORLD_KINDS is before_kinds
    assert C.SIMULATION_CAST_ROLES is before_roles
    assert C.SIMULATION_WORLD_KINDS == ("conversation", "tool_api", "browser",
                                        "computer_use", "code_exec", "voice_telephony")
