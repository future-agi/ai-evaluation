"""Phase 9B units 1-5 — the image / multimodal improvement loop module.

Machinery tier: no extras, no flags, no network, no keys.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from fi.alk import image_loop as il
from fi.alk import trinity

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GATE_ARRAYS = (
    "missing_files",
    "loop_determinism_errors",
    "deterministic_loss_anchoring_errors",
    "image_loss_errors",
    "perception_guard_errors",
    "eval_wiring_errors",
    "evidence_class_errors",
    "ab_capstone_errors",
)


# --- unit 1: canon constants + sublayer + error class ----------------------


def test_image_canon_constants_shape() -> None:
    assert il.V1_IMAGE_LOSS_TERM_REFS == (
        "task_success", "ocr_accuracy", "chart_accuracy", "artifact_grounding",
        "instruction_adherence", "tool_argument_correctness",
    )
    assert set(il.V1_IMAGE_LOSS_DETERMINISTIC_ANCHOR_TERMS) <= set(il.V1_IMAGE_LOSS_TERM_REFS)
    assert set(il.V1_IMAGE_LOSS_JUDGE_TERMS) <= set(il.V1_IMAGE_LOSS_TERM_REFS)
    # anchor set and judge set are disjoint.
    assert not (
        set(il.V1_IMAGE_LOSS_DETERMINISTIC_ANCHOR_TERMS)
        & set(il.V1_IMAGE_LOSS_JUDGE_TERMS)
    )
    assert il.V1_IMAGE_FAILURE_SUBLAYERS == (
        "preprocessing", "perception", "reasoning", "tool_grounding"
    )
    assert il.V1_IMAGE_FIDELITY_TIERS == ("deterministic_fixture", "keyed_live_model")
    assert il.V1_IMAGE_TASK_MODES == ("understanding", "generation")


def test_attribute_image_sublayer_closed_set() -> None:
    rows = {
        "ocr parse weak low_res": "preprocessing",
        "visual misidentification perception-required": "perception",
        "grounded-but-wrong conclusion reasoning": "reasoning",
        "tool_argument extracted wrong": "tool_grounding",
    }
    for signal, expected in rows.items():
        got = il.attribute_image_sublayer(failure_layer="agent_behavior", signal=signal)
        assert got == expected, (signal, got)
        assert got in il.V1_IMAGE_FAILURE_SUBLAYERS
    # an unroutable signal defaults deterministically to a closed-set token.
    default = il.attribute_image_sublayer(failure_layer="agent_behavior", signal="???")
    assert default in il.V1_IMAGE_FAILURE_SUBLAYERS
    # infra-implicated default lands on preprocessing (cheapest fix first).
    assert il.attribute_image_sublayer(failure_layer="lane_infra", signal="???") == "preprocessing"


def test_image_loss_composition_error_is_valueerror() -> None:
    assert issubclass(il.ImageLossCompositionError, ValueError)


# --- unit 2: compile_image_objective + perception-bypass guard --------------


def _understanding_objective(*, terms=None, guards=None) -> dict:
    terms = terms or [
        {"eval": "task_success", "weight": 1.0, "direction": "maximize"},
        {"eval": "ocr_accuracy", "weight": 0.6, "direction": "maximize"},
        {"eval": "instruction_adherence", "weight": 0.4, "direction": "maximize"},
    ]
    if guards is None:
        guards = {
            "sentinel_rows": [{"id": "prior_answerable", "kind": "perception_bypass"}],
            "canary_evals": [{"eval": "cf_twin", "kind": "perceptual_counterfactual"}],
            "min_guard_count": 2,
        }
    return {"source": "declared", "evals": terms, "guards": guards}


def test_image_loss_multi_objective_compiles() -> None:
    compiled = il.compile_image_objective(_understanding_objective())
    assert len(compiled["evals"]) >= 2
    assert any(
        t["eval"] in il.V1_IMAGE_LOSS_DETERMINISTIC_ANCHOR_TERMS
        for t in compiled["evals"]
    )
    assert compiled["guards"]["min_guard_count"] >= 1


def test_image_loss_judge_only_rejected() -> None:
    # terms subset of the judge set (and >= 2 terms) — no deterministic anchor.
    with pytest.raises(il.ImageLossCompositionError):
        il.compile_image_objective(
            _understanding_objective(
                terms=[
                    {"eval": "instruction_adherence", "weight": 1.0},
                    {"eval": "instruction_adherence", "weight": 0.5},
                ]
            )
        )


def test_image_loss_single_term_rejected() -> None:
    with pytest.raises(il.ImageLossCompositionError):
        il.compile_image_objective(
            _understanding_objective(terms=[{"eval": "task_success", "weight": 1.0}])
        )


def test_image_loss_unknown_ref_rejected() -> None:
    with pytest.raises(il.ImageLossCompositionError):
        il.compile_image_objective(
            _understanding_objective(
                terms=[
                    {"eval": "task_success", "weight": 1.0},
                    {"eval": "made_up_term", "weight": 0.5},
                ]
            )
        )


def test_image_loss_guard_unconditional() -> None:
    # a multi-term objective WITHOUT a guard block still raises (unedited loss.py).
    from fi.alk import loss as _loss

    with pytest.raises(_loss.ObjectiveError):
        il.compile_image_objective(
            {
                "source": "declared",
                "evals": [
                    {"eval": "task_success", "weight": 1.0},
                    {"eval": "ocr_accuracy", "weight": 0.5},
                ],
            }
        )


def test_image_loss_perception_guard_kinds() -> None:
    # a valid kind compiles.
    il.compile_image_objective(_understanding_objective())
    # an out-of-set kind is rejected.
    with pytest.raises(il.ImageLossCompositionError):
        il.compile_image_objective(
            _understanding_objective(
                guards={
                    "sentinel_rows": [{"id": "x", "kind": "not_a_real_kind"}],
                    "min_guard_count": 1,
                }
            )
        )


# --- unit 3: world registration + manifest builder --------------------------


def test_image_world_registered_not_widened() -> None:
    from fi.simulate.simulation import contract

    il._ensure_image_world_registered()
    assert "image" in contract.resolved_world_kinds()
    assert "image" not in contract.SIMULATION_WORLD_KINDS


def _search_space() -> dict:
    return {
        "agent.model": ["gpt-4o", "claude"],
        "agent.vision_prompt": ["describe", "extract"],
        "image.preprocess.resolution": [256, 512],
        "image.preprocess.crop": ["center", "none"],
        "mmrag.retrieve_images": [True, False],
        "mmrag.reranker": ["off", "ce"],
        "agent.tools.routing": ["strict", "flexible"],
        "agent.first_message": ["Hi.", "Let's begin."],
    }


def test_image_manifest_sets_kind_and_task_mode() -> None:
    m = il.build_image_practice_loop_manifest(
        name="img-demo",
        base_agent={"model": "gpt-4o"},
        search_space=_search_space(),
        objective=_understanding_objective(),
        eval_budget=4,
        seed=1142,
    )
    inline = m["practice"]["simulation"]["inline"]
    assert inline["world"]["kind"] == "image"
    assert inline["world"]["spec"]["task_mode"] == "understanding"
    assert inline["objective"]["kind"] == "agent-learning.objective.v1"


def test_image_manifest_delegates_verbatim() -> None:
    m = il.build_image_practice_loop_manifest(
        name="img-demo",
        base_agent={"model": "gpt-4o"},
        search_space=_search_space(),
        objective=_understanding_objective(),
        eval_budget=4,
        seed=1142,
    )
    assert m["practice"]["base_agent"]["model"] == "gpt-4o"
    assert "search_space" in m["practice"]
    assert int(m["practice"]["eval_budget"]) == 4


def test_image_search_space_whole_agent() -> None:
    m = il.build_image_practice_loop_manifest(
        name="img-demo",
        base_agent={"model": "gpt-4o"},
        search_space=_search_space(),
        objective=_understanding_objective(),
        eval_budget=4,
        seed=1142,
    )
    ss = m["practice"]["search_space"]
    # NOT prompt-only — the distinguishing 9B dimensions are present.
    assert "image.preprocess.resolution" in ss
    assert "mmrag.retrieve_images" in ss
    assert "agent.model" in ss


def test_image_loop_ab_equal_budget() -> None:
    arms = {}
    for arm in ("loop_on", "loop_off"):
        m = il.build_image_practice_loop_manifest(
            name=f"img-{arm}",
            base_agent={"model": "gpt-4o"},
            search_space=_search_space(),
            objective=_understanding_objective(),
            eval_budget=6,
            seed=1142,
        )
        arms[arm] = int(m["practice"]["eval_budget"])
    assert arms["loop_on"] == arms["loop_off"] == 6


def test_image_world_spec_validator_task_mode() -> None:
    il._validate_image_world_spec({"task_mode": "understanding"})
    il._validate_image_world_spec({"task_mode": "generation"})
    with pytest.raises(ValueError):
        il._validate_image_world_spec({"task_mode": "nonsense"})


# --- unit 4: generation mode ------------------------------------------------


def test_generation_judge_only_rejected() -> None:
    with pytest.raises(il.ImageLossCompositionError):
        il.compile_image_objective(
            {
                "source": "declared",
                "evals": [
                    {"eval": "generation_alignment", "weight": 1.0},
                    {"eval": "generation_quality", "weight": 0.5},
                ],
                "guards": {"sentinel_rows": [{"id": "x"}], "min_guard_count": 1},
            },
            task_mode="generation",
        )


def test_generation_element_presence_anchor_admitted() -> None:
    compiled = il.compile_image_objective(
        {
            "source": "declared",
            "evals": [
                {"eval": "element_presence", "weight": 1.0},
                {"eval": "generation_alignment", "weight": 0.5},
            ],
            "guards": {"sentinel_rows": [{"id": "x"}], "min_guard_count": 1},
        },
        task_mode="generation",
    )
    assert any(t["eval"] == "element_presence" for t in compiled["evals"])


def test_generation_no_clip_dependency() -> None:
    """The local-first generation floor introduces no CLIP/torch/transformers
    import (mirror the pure-numpy absence test)."""
    from pathlib import Path

    source = Path(il.__file__).read_text(encoding="utf-8")
    for banned in ("import clip", "import torch", "from torch", "transformers",
                   "open_clip", "import PIL", "from PIL"):
        assert banned not in source, f"banned import {banned!r} present in image_loop"


def test_generation_element_presence_not_admitted_in_understanding() -> None:
    # element_presence is a generation-only anchor; in understanding mode it is an
    # unknown ref.
    with pytest.raises(il.ImageLossCompositionError):
        il.compile_image_objective(
            {
                "source": "declared",
                "evals": [
                    {"eval": "task_success", "weight": 1.0},
                    {"eval": "element_presence", "weight": 0.5},
                ],
                "guards": {"sentinel_rows": [{"id": "x"}], "min_guard_count": 1},
            },
            task_mode="understanding",
        )


# --- unit 5: the image_loop_readiness gate (tripwires + clean) --------------


def _mini_repo(tmp_path: Path) -> Path:
    """Copy the committed examples + fixtures into a tmp repo so the gate
    exec-loads them from a doctorable tree (the installed package is reused for
    imports)."""
    dst = tmp_path / "repo"
    (dst / "examples").mkdir(parents=True)
    shutil.copy2(_REPO_ROOT / "examples/sdk_image_loop.py", dst / "examples/sdk_image_loop.py")
    shutil.copy2(
        _REPO_ROOT / "examples/sdk_image_improvement.py",
        dst / "examples/sdk_image_improvement.py",
    )
    shutil.copytree(
        _REPO_ROOT / "examples/image_loop_fixture",
        dst / "examples/image_loop_fixture",
    )
    return dst


def test_release_image_loop_readiness_status_clean(tmp_path: Path) -> None:
    status = trinity._release_image_loop_readiness_status(_mini_repo(tmp_path))
    for arr in _GATE_ARRAYS:
        assert status[arr] == [], (arr, status[arr])
    assert status["kind"] == "agent-learning.image-loop-readiness.v1"


def test_image_loop_flags_fidelity_overclaim(tmp_path: Path) -> None:
    """A deterministic_fixture artifact stamping evidence_class=live_lane MUST
    flip evidence_class_errors via the image_fidelity_overclaim token."""
    repo = _mini_repo(tmp_path)
    example = repo / "examples/sdk_image_loop.py"
    text = example.read_text(encoding="utf-8")
    # doctor the clean artifact to claim live_lane.
    doctored = text.replace(
        '"evidence_class": "local_gate",\n        "fidelity_tier": "deterministic_fixture",\n        "world_kind": "image",',
        '"evidence_class": "live_lane",\n        "fidelity_tier": "deterministic_fixture",\n        "world_kind": "image",',
    )
    assert doctored != text, "doctoring did not change the example"
    example.write_text(doctored, encoding="utf-8")
    status = trinity._release_image_loop_readiness_status(repo)
    reasons = json.dumps(status["evidence_class_errors"])
    assert status["evidence_class_errors"], "fidelity overclaim not caught"
    assert "image_fidelity_overclaim" in reasons


def test_image_loop_perception_guard_tripwire(tmp_path: Path) -> None:
    """The counterfactual control that does NOT drop the score for a perceiving
    config MUST flip perception_guard_errors (the binding tripwire)."""
    repo = _mini_repo(tmp_path)
    cf_path = repo / "examples/image_loop_fixture/counterfactual_pair/cf.json"
    cf = json.loads(cf_path.read_text(encoding="utf-8"))
    # break the perceiving config so it does NOT drop on the twin.
    cf["control"]["perceiving_config_score_b"] = 1.0
    cf_path.write_text(json.dumps(cf, indent=2), encoding="utf-8")
    status = trinity._release_image_loop_readiness_status(repo)
    fields = json.dumps(status["perception_guard_errors"])
    assert status["perception_guard_errors"], "broken counterfactual not caught"
    assert "counterfactual_drops_score" in fields


def test_image_loop_flags_judge_only_loss(tmp_path: Path) -> None:
    """An improvement example where the judge-only rejection silently fails MUST
    flip image_loss_errors."""
    repo = _mini_repo(tmp_path)
    example = repo / "examples/sdk_image_improvement.py"
    text = example.read_text(encoding="utf-8")
    doctored = text.replace(
        '        "judge_only_rejected": judge_only_rejected,',
        '        "judge_only_rejected": False,',
    )
    assert doctored != text
    example.write_text(doctored, encoding="utf-8")
    status = trinity._release_image_loop_readiness_status(repo)
    assert status["image_loss_errors"], "judge-only failure not caught"


def test_image_loop_flags_missing_anchor(tmp_path: Path) -> None:
    """An anchors block that does not match the golden MUST flip
    deterministic_loss_anchoring_errors."""
    repo = _mini_repo(tmp_path)
    golden = repo / "examples/image_loop_fixture/expected/deterministic_anchors.json"
    data = json.loads(golden.read_text(encoding="utf-8"))
    data["anchors"]["task_success"]["value"] = 0.0  # force a mismatch
    golden.write_text(json.dumps(data, indent=2), encoding="utf-8")
    status = trinity._release_image_loop_readiness_status(repo)
    assert status["deterministic_loss_anchoring_errors"], "anchor mismatch not caught"


def test_image_loop_flags_world_widened(tmp_path: Path) -> None:
    """An example claiming image is in SIMULATION_WORLD_KINDS (a simulated vocab
    widening) MUST flip eval_wiring_errors."""
    repo = _mini_repo(tmp_path)
    example = repo / "examples/sdk_image_loop.py"
    text = example.read_text(encoding="utf-8")
    doctored = text.replace(
        '"frozen_vocab_byte_stable": "image" not in contract.SIMULATION_WORLD_KINDS,',
        '"frozen_vocab_byte_stable": False,',
    )
    assert doctored != text
    example.write_text(doctored, encoding="utf-8")
    status = trinity._release_image_loop_readiness_status(repo)
    assert status["eval_wiring_errors"], "simulated vocab widening not caught"


def test_image_loop_flags_nondeterminism(tmp_path: Path) -> None:
    """A loop_determinism block reporting non-identical rasters MUST flip
    loop_determinism_errors."""
    repo = _mini_repo(tmp_path)
    example = repo / "examples/sdk_image_loop.py"
    text = example.read_text(encoding="utf-8")
    doctored = text.replace(
        '"perturbation_raster_byte_identical": bool(np.array_equal(a["raster"], b["raster"])),',
        '"perturbation_raster_byte_identical": False,',
    )
    assert doctored != text
    example.write_text(doctored, encoding="utf-8")
    status = trinity._release_image_loop_readiness_status(repo)
    assert status["loop_determinism_errors"], "non-determinism not caught"


def test_image_loop_flags_ab_no_improvement(tmp_path: Path) -> None:
    """An A/B fixture where the loop arm does NOT improve MUST flip
    ab_capstone_errors."""
    repo = _mini_repo(tmp_path)
    ab_path = repo / "examples/image_loop_fixture/ab/toy_space.json"
    ab = json.loads(ab_path.read_text(encoding="utf-8"))
    ab["arms"]["loop_on"]["anchored_loss"] = 0.9  # worse than loop_off
    ab_path.write_text(json.dumps(ab, indent=2), encoding="utf-8")
    status = trinity._release_image_loop_readiness_status(repo)
    assert status["ab_capstone_errors"], "A/B no-improvement not caught"


def test_image_loop_missing_files(tmp_path: Path) -> None:
    repo = _mini_repo(tmp_path)
    (repo / "examples/image_loop_fixture/chart.json").unlink()
    status = trinity._release_image_loop_readiness_status(repo)
    assert status["missing_files"], "missing fixture not caught"


# --- unit 7: the keyed real-VLM lane (opt-in, never a gate) -----------------


def test_keyed_lane_refuses_loudly_without_key(monkeypatch) -> None:
    for env in il.IMAGE_JUDGE_KEY_ENVS:
        monkeypatch.delenv(env, raising=False)
    assert il.image_judge_key_present() is False
    with pytest.raises(il.ImageKeyedLaneUnavailable) as excinfo:
        il.run_keyed_image_live_proof(
            base_agent={"model": "x"},
            search_space={"agent.model": ["x"]},
            objective={
                "source": "declared",
                "evals": [
                    {"eval": "element_presence", "weight": 1.0},
                    {"eval": "generation_alignment", "weight": 0.5},
                ],
                "guards": {"sentinel_rows": [{"id": "x"}], "min_guard_count": 1},
            },
            eval_budget=2,
            seed=1,
        )
    assert "image_judge_key_unavailable" in str(excinfo.value)


def test_keyed_lane_marks_live_lane_with_key(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_LEARNING_IMAGE_JUDGE_KEY", "test-key")
    result = il.run_keyed_image_live_proof(
        base_agent={"model": "x"},
        search_space={"agent.model": ["x"]},
        objective={
            "source": "declared",
            "evals": [
                {"eval": "element_presence", "weight": 1.0},
                {"eval": "generation_alignment", "weight": 0.5},
            ],
            "guards": {"sentinel_rows": [{"id": "x"}], "min_guard_count": 1},
        },
        eval_budget=2,
        seed=1,
    )
    # the keyed lane is the ONLY honest place for live_lane.
    assert result["evidence_class"] == "live_lane"
    assert result["fidelity_tier"] == "keyed_live_model"
    assert result["task_mode"] == "generation"
