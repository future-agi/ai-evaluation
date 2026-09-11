"""Phase 9D — optimizer profile matrix modality target-kind expansion.

Machinery tier: no extras, no env flags, no network, no real keys. Covers the
five 9D properties:

  1. the new modality cells appear in the produced exact-set (and the gate's
     exact-set contract: produced refs == "/".join(c) for the V1 mirror);
  2. the modality-coverage clause flips when a LANDED modality token has zero
     declared cells (tested at the filter level — the same
     ``cell_ref.split("/")[1] == token`` logic the gate uses);
  3. the runtime ``apply_plans`` cover the modality cells (not just whole_agent)
     — both the runtime producer and the gate filter generalize together;
  4. determinism — building + running the matrix twice yields byte-identical
     modality-cell winner/score/selected_patch_paths;
  5. each modality cell carries ``evidence_class == "local_gate"``,
     ``setting.engine == "local_text"``, ``native_proof_closed`` and a winner.

The modality tokens are voice_agent / image_agent / cua_agent (9C landed, so all
three are live in ONE pass). The matrix grows 33 -> 40 cells.
"""

from __future__ import annotations

import os

import pytest

from fi.alk import optimize, trinity

_MODALITY_VOICE_CELL = "livekit/voice_agent/society"
_MODALITY_IMAGE_CELL = "llamaindex/image_agent/society"
_MODALITY_CUA_CELL = "langgraph/cua_agent/society"
_MODALITY_CELLS = (
    "livekit/voice_agent/society",
    "livekit/voice_agent/evolution_elo",
    "livekit/voice_agent/tpe",
    "llamaindex/image_agent/society",
    "llamaindex/image_agent/evolution_elo",
    "langgraph/cua_agent/society",
    "langgraph/cua_agent/regression_replay",
)


@pytest.fixture(autouse=True)
def _release_local_env():
    """The matrix runs credential-free under the release-local env (no real
    key, no network)."""
    name = "AGENT_LEARNING_SDK_OPTIMIZER_PROFILE_MATRIX_KEY"
    previous = os.environ.get(name)
    os.environ[name] = f"agent-learning-release-local-{name.lower()}"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


# --- 1. new cells appear in the exact-set -----------------------------------


def test_modality_cells_appear_in_produced_exact_set() -> None:
    manifests = optimize.build_optimizer_profile_matrix_manifests()
    assert len(manifests) == len(optimize.OPTIMIZER_PROFILE_MATRIX_CELLS) == 40
    for cell_ref in _MODALITY_CELLS:
        assert cell_ref in manifests, cell_ref
    # The gate's exact-set contract: produced refs == the V1 mirror's refs.
    expected = {"/".join(cell) for cell in trinity.V1_OPTIMIZER_PROFILE_MATRIX_CELLS}
    assert set(manifests) == expected
    assert sorted(manifests) == sorted(expected)


def test_modality_tokens_in_vocab_and_per_axis_coverage_lockstep() -> None:
    for token in ("voice_agent", "image_agent", "cua_agent"):
        assert token in optimize.OPTIMIZER_PROFILE_MATRIX_TARGET_KINDS
        assert token in trinity.V1_OPTIMIZER_PROFILE_MATRIX_TARGET_KINDS
    assert list(optimize.OPTIMIZER_PROFILE_MATRIX_TARGET_KINDS) == (
        trinity.V1_OPTIMIZER_PROFILE_MATRIX_TARGET_KINDS
    )
    assert trinity.V1_OPTIMIZER_PROFILE_MATRIX_CELLS == list(
        optimize.OPTIMIZER_PROFILE_MATRIX_CELLS
    )


# --- 2. the coverage clause flips when a landed modality has zero cells ------


def _modality_cell_refs(declared_cell_refs, token):
    """The same filter the gate's modality-coverage clause uses."""
    return [
        cell_ref
        for cell_ref in declared_cell_refs
        if cell_ref.split("/")[1] == token
    ]


def test_modality_coverage_clause_passes_on_the_real_declared_set() -> None:
    declared = ["/".join(cell) for cell in trinity.V1_OPTIMIZER_PROFILE_MATRIX_CELLS]
    for token in trinity.V1_OPTIMIZER_PROFILE_MATRIX_MODALITY_TARGET_KINDS:
        if token in trinity.V1_OPTIMIZER_PROFILE_MATRIX_TARGET_KINDS:
            assert _modality_cell_refs(declared, token), token


def test_modality_coverage_clause_flips_when_a_landed_token_has_zero_cells() -> None:
    # A modality token present in the vocabulary with ZERO declared cells is a
    # coverage failure — drop every voice cell and assert the filter is empty.
    declared = [
        "/".join(cell)
        for cell in trinity.V1_OPTIMIZER_PROFILE_MATRIX_CELLS
        if cell[1] != "voice_agent"
    ]
    assert "voice_agent" in trinity.V1_OPTIMIZER_PROFILE_MATRIX_TARGET_KINDS
    assert _modality_cell_refs(declared, "voice_agent") == []
    # the real set still has voice cells (the positive control)
    real = ["/".join(cell) for cell in trinity.V1_OPTIMIZER_PROFILE_MATRIX_CELLS]
    assert _modality_cell_refs(real, "voice_agent")


# --- 3. apply-plan coverage includes the modality cells ----------------------


def test_apply_plans_cover_the_modality_cells() -> None:
    manifests = optimize.build_optimizer_profile_matrix_manifests()
    result = optimize.run_optimizer_profile_matrix(manifests)
    plan_refs = {plan["cell_ref"] for plan in result["apply_plans"]}
    # whole_agent cells still export plans
    assert "livekit/whole_agent/society" in plan_refs
    # the modality cells now export plans too (PRD-9D §4.7 / A2)
    assert _MODALITY_VOICE_CELL in plan_refs
    assert _MODALITY_IMAGE_CELL in plan_refs
    assert _MODALITY_CUA_CELL in plan_refs
    # apply-plan-exporting kinds = {whole_agent, voice_agent, image_agent, cua_agent}
    exporting_kinds = {ref.split("/")[1] for ref in plan_refs}
    assert exporting_kinds == {
        "whole_agent",
        "voice_agent",
        "image_agent",
        "cua_agent",
    }


# --- 4. determinism ----------------------------------------------------------


def _modality_trajectories():
    manifests = optimize.build_optimizer_profile_matrix_manifests()
    result = optimize.run_optimizer_profile_matrix(manifests)
    return {
        cell["cell_ref"]: (
            cell["winner"],
            cell["score"],
            tuple(cell.get("selected_patch_paths") or []),
        )
        for cell in result["cells"]
        if cell["cell_ref"] in _MODALITY_CELLS
    }


def test_modality_cells_are_deterministic_across_runs() -> None:
    first = _modality_trajectories()
    second = _modality_trajectories()
    assert set(first) == set(_MODALITY_CELLS)
    assert first == second


# --- 5. per-cell local_gate evidence -----------------------------------------


def test_every_modality_cell_emits_local_gate_evidence() -> None:
    manifests = optimize.build_optimizer_profile_matrix_manifests()
    result = optimize.run_optimizer_profile_matrix(manifests)
    by_ref = {cell["cell_ref"]: cell for cell in result["cells"]}
    for cell_ref in _MODALITY_CELLS:
        cell = by_ref[cell_ref]
        assert cell["evidence_class"] == "local_gate", cell_ref
        assert cell["setting"]["engine"] == "local_text", cell_ref
        assert cell["native_proof_closed"] is True, cell_ref
        assert cell["status"] == "passed", cell_ref
        assert cell["winner"], cell_ref
        # within the per-cell eval-budget cap (ARCH §6)
        assert int(cell["evaluations_used"]) <= int(cell["eval_budget"]) <= 24


def test_modality_cells_route_as_whole_agent_manifests_with_typed_world_kind() -> None:
    manifests = optimize.build_optimizer_profile_matrix_manifests()
    expected_world = {
        "livekit/voice_agent/society": "voice_telephony",
        "llamaindex/image_agent/society": "image",
        "langgraph/cua_agent/society": "browser",
    }
    for cell_ref, world_kind in expected_world.items():
        manifest = manifests[cell_ref]
        # whole_agent IN MECHANISM: it carries the whole-agent contract block and
        # produces an agent-learning.optimization.v1 manifest (runnable).
        assert manifest["version"] == "agent-learning.optimization.v1"
        assert manifest.get("whole_agent")
        assert manifest["scenario"]["world"]["kind"] == world_kind
        cell = manifest["metadata"]["optimizer_profile_matrix_cell"]
        assert cell["setting"]["engine"] == "local_text"
