"""Phase 9B units 1-4 — the image / multimodal improvement loop (the 13D
Practice Loop on ``world.kind = image``).

ARCH-9B §2.1/§2.2/§2.3/§2.4 / decisions 9B-D1..9B-D6, 9B-A1/A2/A3/A7/A8.

This module invents NO optimizer, NO artifact kind, NO loss machinery, NO world.
It is the IMAGE analogue of ``voice_loop.py`` — a thin composition layer over
verbatim engines:

  * the multi-objective image loss compiles via ``loss.compile_objective`` (the
    Goodhart guard at ``loss.py:106-116`` is reused VERBATIM — "There is no
    override."); the 9B-A2 composition rule (>= 2 terms, >= 1 deterministic
    ground-truth anchor — a judge-only loss is INVALID) is a thin validator on
    top, raising ``image_loss_guard_missing`` (``ImageLossCompositionError``);
  * the whole multimodal-agent config is the search space, assembled by
    ``optimize.build_practice_loop_manifest`` (the same ``base_agent`` +
    ``search_space`` whole-agent contract) with ``world.kind=image`` +
    ``task_mode`` (understanding | generation) on ``WorldSpec.spec``;
  * the image sub-attribution is an additive tag stamped alongside the base
    ``FAILURE_LAYERS`` tag (the existing ``practice/_diagnose.py`` machinery is
    consumed, not rewritten);
  * ``world.kind=image`` enters the world-kind space through the R4 registry hook
    (``extensions.register_extension``) — never by widening the frozen
    ``SIMULATION_WORLD_KINDS`` tuple. ``image`` is "typed -> executable".

The canon constants below are this module's home; ``trinity.py`` carries literal
mirrors that the milestone test cross-pins (the GUNA_AXES cross-pin pattern —
trinity never imports this module so the gate runs even if this is broken). The
pure-numpy perturbation operators live in the companion ``image_perturb.py``
(9B-A1b — substrate, not loop).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

# --- canon (ARCH-9B §2.1 image-loss term refs + §2.3 sub-attribution) -------
# The deterministic-anchored UNDERSTANDING-mode menu (the 6-tuple analogue of
# V1_VOICE_LOSS_TERM_REFS, the 9-tuple).
V1_IMAGE_LOSS_TERM_REFS = (
    "task_success",
    "ocr_accuracy",
    "chart_accuracy",
    "artifact_grounding",
    "instruction_adherence",
    "tool_argument_correctness",
)
# The mandatory ground-truth quality anchors — an image loss MUST carry >= 1 of
# these (9B-A2 / 9B-D3). The analogue of V1_VOICE_LOSS_NON_TIMING_QUALITY_TERMS.
# ``element_presence`` joins this admissible set under task_mode=generation.
V1_IMAGE_LOSS_DETERMINISTIC_ANCHOR_TERMS = (
    "task_success",
    "ocr_accuracy",
    "chart_accuracy",
    "artifact_grounding",
)
# The bounded/guarded judge contributors (the analogue of V1_VOICE_LOSS_TIMING_
# TERMS, the hackable-alone set). A judge-only loss (terms subset of this set) is
# structurally rejected (9B-D3). generation_alignment / generation_quality join
# this set under task_mode=generation.
V1_IMAGE_LOSS_JUDGE_TERMS = ("instruction_adherence",)

# Generation-mode terms (ARCH-9B §2.4 / 9B-A7/A8), admitted ONLY under
# task_mode=generation.
V1_IMAGE_GENERATION_ANCHOR_TERMS = ("element_presence",)  # deterministic floor (9B-A8)
V1_IMAGE_GENERATION_JUDGE_TERMS = ("generation_alignment", "generation_quality")

# The four-token image sub-attribution closed set (9B §2.3), stamped alongside
# the base FAILURE_LAYERS tag.
V1_IMAGE_FAILURE_SUBLAYERS = ("preprocessing", "perception", "reasoning", "tool_grounding")

# A MARKER field on artifact metadata — NOT a new evidence class (R5/A18; the
# frozen EVIDENCE_CLASSES 4-tuple is unchanged). The analogue of
# V1_VOICE_FIDELITY_TIERS. (ARCH-9B §2.6)
V1_IMAGE_FIDELITY_TIERS = ("deterministic_fixture", "keyed_live_model")

# The typed ``kind`` discriminators a perception-bypass guard row may carry,
# beyond the base sentinel/canary rows the loss guard already allows (ARCH-9B
# §2.2).
V1_IMAGE_PERCEPTION_GUARD_KINDS = ("perception_bypass", "perceptual_counterfactual")

# The task_mode switch on WorldSpec.spec (ARCH-9B §2.3 / 9B-D2). ONE world kind,
# two loss profiles.
V1_IMAGE_TASK_MODES = ("understanding", "generation")

# The registered world-kind token + the namespaced extension name (R4 hook).
IMAGE_WORLD_KIND = "image"
IMAGE_EXTENSION_NAME = "agentlearning.image"

# The R4 rung -> evidence-class ladder (ARCH-9B §2.6). The deterministic core is
# local_gate/captured_fixture; live_lane is added ONLY on the keyed lane record
# (unit 7), never the day-one deterministic record.
_IMAGE_RUNG_LADDER = {
    "rung1": ["local_gate"],
    "perturbed": ["live_stressed", "captured_fixture"],
    "keyed_vlm": ["live_lane"],
}


class ImageLossCompositionError(ValueError):
    """Raised when an image objective violates the 9B-A2 composition rule (the
    ``image_loss_guard_missing`` finding — an image specialization of
    ``objective_guards_missing``). A ``ValueError`` subclass so callers can
    ``except ValueError`` exactly as for ``VoiceLossCompositionError``."""


def _term_refs(objective: Mapping[str, Any]) -> list[str]:
    """The objective's eval refs (read from ``evals`` — the loss.py schema)."""
    return [
        str(term.get("eval"))
        for term in (objective.get("evals") or [])
        if isinstance(term, Mapping) and term.get("eval")
    ]


def _admissible_anchor_terms(task_mode: str) -> tuple[str, ...]:
    if task_mode == "generation":
        return V1_IMAGE_LOSS_DETERMINISTIC_ANCHOR_TERMS + V1_IMAGE_GENERATION_ANCHOR_TERMS
    return V1_IMAGE_LOSS_DETERMINISTIC_ANCHOR_TERMS


def _admissible_term_refs(task_mode: str) -> tuple[str, ...]:
    if task_mode == "generation":
        return (
            V1_IMAGE_LOSS_TERM_REFS
            + V1_IMAGE_GENERATION_ANCHOR_TERMS
            + V1_IMAGE_GENERATION_JUDGE_TERMS
        )
    return V1_IMAGE_LOSS_TERM_REFS


def compile_image_objective(
    payload: Mapping[str, Any], *, task_mode: str = "understanding"
) -> dict:
    """Compile a multi-objective image loss with a perception-bypass Goodhart
    guard (ARCH-9B §2.2 / 9B-A2 / 9B-D3). The image analogue of
    ``compile_voice_objective`` (voice_loop.py:70). Enforces, ON TOP of the
    verbatim ``loss.compile_objective`` Goodhart guard:

      (a) >= 2 terms (a single-term image objective is reward-hackable);
      (b) >= 1 deterministic ground-truth anchor — a judge-only loss is INVALID
          (9B-D3). ``task_mode`` selects the admissible anchor set:
          understanding -> V1_IMAGE_LOSS_DETERMINISTIC_ANCHOR_TERMS; generation
          adds ``element_presence`` (the deterministic floor, 9B-A8);
      (c) unknown-ref rejection (every term must be a member of the mode's menu);
      (d) when sentinel/canary rows carry a perception ``kind`` discriminator it
          must be in V1_IMAGE_PERCEPTION_GUARD_KINDS (the closed set).

    Then delegates to ``loss.compile_objective`` VERBATIM — which unconditionally
    enforces the populated guard block (sentinel_rows / canary_evals,
    min_guard_count >= 1, "There is no override.")."""

    from . import loss as _loss  # downward facade import (legal; voice_loop.py idiom)

    if task_mode not in V1_IMAGE_TASK_MODES:
        raise ImageLossCompositionError(
            f"image_loss_guard_missing: task_mode {task_mode!r} not in "
            f"{V1_IMAGE_TASK_MODES}"
        )

    refs = _term_refs(payload)

    # rule (a): >= 2 terms.
    if len(refs) < 2:
        raise ImageLossCompositionError(
            "image_loss_guard_missing: an image objective is reward-hackable as a "
            "single term; it MUST be multi-objective (>= 2 terms). "
            f"got {refs}"
        )

    # rule (b): >= 1 deterministic ground-truth anchor (judge-only REJECTED).
    anchors = _admissible_anchor_terms(task_mode)
    if not any(ref in anchors for ref in refs):
        raise ImageLossCompositionError(
            "image_loss_guard_missing: an image loss MUST carry >= 1 deterministic "
            f"ground-truth anchor {anchors}; a judge-only loss is INVALID by "
            f"contract (9B-D3). got {refs}"
        )

    # rule (c): unknown-ref rejection.
    allowed = _admissible_term_refs(task_mode)
    for ref in refs:
        if ref not in allowed:
            raise ImageLossCompositionError(
                f"image_loss_guard_missing: unknown image loss term {ref!r}; "
                f"expected members of {allowed} (task_mode={task_mode})"
            )

    # rule (d): the perception-bypass guard rows ride the existing
    # sentinel_rows/canary_evals with a typed ``kind`` discriminator (no new
    # ObjectiveSpec field, ARCH-9B §2.2). When present it must be in the closed
    # set (plus any untyped/base rows the loss guard already allows).
    guards = payload.get("guards") or {}
    for bucket in ("sentinel_rows", "canary_evals"):
        for row in guards.get(bucket) or []:
            if isinstance(row, Mapping):
                kind = row.get("kind")
                if kind is not None and kind not in V1_IMAGE_PERCEPTION_GUARD_KINDS:
                    raise ImageLossCompositionError(
                        f"image_loss_guard_missing: guard row kind {kind!r} not in "
                        f"{V1_IMAGE_PERCEPTION_GUARD_KINDS}"
                    )

    # the verbatim Goodhart guard (loss.py:106-116) — "There is no override."
    return _loss.compile_objective(payload)


def attribute_image_sublayer(
    *,
    failure_layer: str,
    deficit: Mapping[str, Any] | None = None,
    signal: str | None = None,
) -> str:
    """Map a weak image cell to a ``V1_IMAGE_FAILURE_SUBLAYERS`` token, stamped
    ALONGSIDE the base ``FAILURE_LAYERS`` tag (a weak cell carries both, e.g.
    ``{failure_layer:"agent_behavior", image_sublayer:"perception"}``). The base
    attribution rides the existing ``practice/_diagnose.py`` machinery; this is
    the thin sublayer helper (the image analogue of ``attribute_voice_sublayer``,
    voice_loop.py:109).

    Routing (ARCH-9B §2.3 table, grounded in DISCO's parse->reason split
    2603.23511 + AgentVista visual-misidentification=perception 2602.23166):
      * OCR/parse weak; low-res/compression hurts -> ``preprocessing``
        (Fix-Before-Search: try a resolution/crop change before blaming the LLM);
      * visual misidentification; perception-required cell weak -> ``perception``
        (the dominant AgentVista failure);
      * grounded-but-wrong-conclusion -> ``reasoning`` (parse ok, reasoning fails);
      * tool-argument extracted wrong from the image -> ``tool_grounding``."""

    sig = str(signal or (deficit or {}).get("signal") or "").lower()
    if any(
        k in sig
        for k in ("ocr", "parse", "low_res", "low-res", "resolution", "compression",
                  "compress", "blur", "preprocess")
    ):
        return "preprocessing"
    if any(
        k in sig
        for k in ("tool_argument", "tool-argument", "tool_grounding", "tool argument",
                  "argument", "extracted")
    ):
        return "tool_grounding"
    if any(
        k in sig
        for k in ("misidentif", "perception", "visual", "occlusion", "occluded",
                  "perceive", "see ")
    ):
        return "perception"
    if any(
        k in sig
        for k in ("reason", "conclusion", "grounded-but-wrong", "wrong_conclusion",
                  "inference")
    ):
        return "reasoning"
    # default: infra-implicated cells land on preprocessing (the cheapest fix
    # before blaming the model); otherwise the reasoning/policy layer.
    if failure_layer in ("lane_infra", "framework_runtime", "provider"):
        return "preprocessing"
    return "reasoning"


def _ensure_image_world_registered() -> None:
    """Register ``world.kind=image`` via the R4 hook (ARCH-9B §2.1 / 9B-D2).
    Idempotent. Pushes DOWN into ``contract.register_world_kind`` so
    ``resolved_world_kinds()`` contains ``image`` WITHOUT touching the frozen
    ``SIMULATION_WORLD_KINDS`` tuple (contract.py:55)."""

    from fi.simulate.simulation import contract as _contract

    # idempotent: register_extension raises on a name collision, and the world
    # kind only needs to land once per process.
    if IMAGE_WORLD_KIND in _contract.resolved_world_kinds():
        return

    from . import extensions as _ext

    _ext.register_extension(
        "environment",
        {
            "name": IMAGE_EXTENSION_NAME,          # vendor.name shape (_validate_record)
            "kind_token": IMAGE_WORLD_KIND,        # the registered world.kind token
            "spec_validator": _validate_image_world_spec,  # R4 mandate
            "rung_ladder": _IMAGE_RUNG_LADDER,             # R4 mandate
            # the deterministic core; live_lane is added ONLY on the keyed lane
            # record (unit 7), never here.
            "evidence_class_capability": ["local_gate", "captured_fixture"],
            # gated_contexts_runnable stays False until rung1_fixture_green
            # (extensions.py admission); 9B never silently claims executable.
        },
    )


def _validate_image_world_spec(spec: Mapping[str, Any]) -> None:
    """The R4 ``spec_validator`` for the image world: validate the ``task_mode``
    switch (understanding | generation) on ``WorldSpec.spec``. Raises ValueError
    on an unknown mode (the closed-set guard)."""

    task_mode = str((spec or {}).get("task_mode", "understanding"))
    if task_mode not in V1_IMAGE_TASK_MODES:
        raise ValueError(
            f"image world.spec.task_mode {task_mode!r} not in {V1_IMAGE_TASK_MODES}"
        )


def build_image_practice_loop_manifest(
    *,
    name: str,
    base_agent: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any]],
    objective: Mapping[str, Any],
    eval_budget: int,
    seed: int,
    task_mode: str = "understanding",
    scenario_inline: Optional[Mapping[str, Any]] = None,
    max_rounds: int = 8,
) -> dict[str, Any]:
    """Assemble the image improvement-loop manifest: the 13D Practice Loop on
    ``world.kind=image`` + ``task_mode`` with the multi-objective guarded image
    loss + the whole multimodal-agent search space (9B-D5). Delegates to
    ``optimize.build_practice_loop_manifest`` so its validators hold VERBATIM
    (9B-A3). The objective is compiled by ``compile_image_objective`` (the 9B-A2
    rule) before it rides the simulation.

    Byte-parallel to ``build_voice_practice_loop_manifest`` except: (a) the
    ``_ensure_image_world_registered()`` call (voice's kind is built-in, image's
    is registered through the R4 hook); (b) ``world["kind"]="image"`` instead of
    ``"voice_telephony"``; (c) the ``spec["task_mode"]`` write (voice has no mode
    switch)."""

    from . import optimize as _optimize  # downward facade import (legal)

    if task_mode not in V1_IMAGE_TASK_MODES:
        raise ImageLossCompositionError(
            f"image_loss_guard_missing: task_mode {task_mode!r} not in "
            f"{V1_IMAGE_TASK_MODES}"
        )

    _ensure_image_world_registered()                       # step 1 (§2.1)
    compiled = compile_image_objective(objective, task_mode=task_mode)  # step 2 (unit 2)
    inline = dict(scenario_inline or {})
    inline.setdefault("version", "agent-learning.simulation.v1")
    inline["objective"] = compiled
    world = dict(inline.get("world") or {})
    world["kind"] = IMAGE_WORLD_KIND                        # step 3 — the registered kind
    spec = dict(world.get("spec") or {})
    spec["task_mode"] = task_mode                           # the task_mode switch on WorldSpec.spec
    world["spec"] = spec
    inline["world"] = world

    return _optimize.build_practice_loop_manifest(          # step 4 — VERBATIM delegate
        name=name,
        simulation={"version": inline["version"], "inline": inline},
        base_agent=base_agent,
        search_space=search_space,
        eval_budget=eval_budget,
        seed=seed,
        max_rounds=max_rounds,
    )


# === Unit 7 — the keyed real-VLM lane (opt-in, NEVER a gate prerequisite) ===
# ARCH-9B §2.4 / §2.6 / 9B-D1/D6. The judge-anchored terms, the full generation
# profile, and the one real-multimodal-agent live-proof are owner-keyed, opt-in,
# never a release gate. The deterministic core stays local_gate/captured_fixture;
# the keyed lane is the ONLY honest place for live_lane (a real keyed model ran).

KEYED_IMAGE_EXTENSION_NAME = "agentlearning.image.keyed"
_KEYED_IMAGE_RUNG_LADDER = {
    "rung1": ["local_gate"],
    "perturbed": ["live_stressed", "captured_fixture"],
    "keyed_vlm": ["live_lane"],
}

# the env keys that gate the keyed lane (checked, never required by any gate).
IMAGE_JUDGE_KEY_ENVS = ("AGENT_LEARNING_IMAGE_JUDGE_KEY", "OPENAI_API_KEY")


class ImageKeyedLaneUnavailable(RuntimeError):
    """Raised by the keyed lane when no judge/VLM key is present — the loud
    refusal (the ``image_judge_key_unavailable`` finding). The deterministic core
    NEVER raises this; only the opt-in keyed path does."""


def image_judge_key_present() -> bool:
    """True iff a judge/VLM key is configured for the keyed lane."""
    import os

    return any(os.environ.get(env) for env in IMAGE_JUDGE_KEY_ENVS)


def register_keyed_image_lane() -> None:
    """Register the SEPARATE keyed-lane extension record that adds ``live_lane``
    to ``evidence_class_capability`` (ARCH-9B §2.6, unit 7). Idempotent. This is
    the ONLY record that may carry ``live_lane`` — the deterministic-core record
    (``_ensure_image_world_registered``) stays ``("local_gate","captured_fixture")``.
    NEVER called by the gate; opt-in only."""
    from . import extensions as _ext

    if _ext.resolve("environment", KEYED_IMAGE_EXTENSION_NAME) is not None:
        return
    _ext.register_extension(
        "environment",
        {
            "name": KEYED_IMAGE_EXTENSION_NAME,
            # the keyed lane reuses the SAME world.kind token only when the base
            # record is absent; here it declares the keyed capability without a
            # second kind_token (the world kind is already registered).
            "evidence_class_capability": ["local_gate", "captured_fixture", "live_lane"],
        },
    )


def run_keyed_image_live_proof(
    *,
    base_agent: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any]],
    objective: Mapping[str, Any],
    eval_budget: int,
    seed: int,
    task_mode: str = "generation",
    name: str = "image-keyed-live-proof",
) -> dict[str, Any]:
    """The one owner-keyed live-proof entry (WORKFLOW Step 5 real-keys ground
    rule). Refuses LOUDLY without a key (``ImageKeyedLaneUnavailable`` ->
    ``image_judge_key_unavailable``) — never a fake number, never a release
    prerequisite. With a key, it builds the generation-profile manifest and marks
    the run ``live_lane`` / ``fidelity_tier=keyed_live_model``.

    The keyed run itself (calling the judge/VLM) is left to the caller's runtime;
    this returns the keyed manifest + the honest evidence-class stamp so an
    owner can execute it once with real keys."""
    if not image_judge_key_present():
        raise ImageKeyedLaneUnavailable(
            "image_judge_key_unavailable: the keyed real-VLM lane requires a "
            f"judge/VLM key (one of {IMAGE_JUDGE_KEY_ENVS}); withheld -- never a "
            "fake number, never a release prerequisite"
        )
    register_keyed_image_lane()
    manifest = build_image_practice_loop_manifest(
        name=name,
        base_agent=base_agent,
        search_space=search_space,
        objective=objective,
        eval_budget=eval_budget,
        seed=seed,
        task_mode=task_mode,
    )
    return {
        "manifest": manifest,
        "evidence_class": "live_lane",          # the ONLY honest live_lane
        "fidelity_tier": "keyed_live_model",
        "task_mode": task_mode,
    }
