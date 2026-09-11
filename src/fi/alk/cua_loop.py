"""Phase 9C units 1-4 — the CUA / browser / computer-use improvement loop (the
13D Practice Loop on ``world.kind = browser`` / ``computer_use`` with a
``cua_surface = browser | desktop`` sub-kind switch).

ARCH-9C §2.1/§2.2/§2.3/§2.4 / decisions 9C-D1..9C-D6, 9C-A1/A1b/A1c/A2/A3/A7/A7b/A8.

This module invents NO optimizer, NO artifact kind, NO loss machinery, NO world,
NO perturbation module. It is the CUA analogue of ``image_loop.py`` /
``voice_loop.py`` — a thin composition layer over verbatim engines:

  * the multi-objective CUA loss compiles via ``loss.compile_objective`` (the
    Goodhart guard at ``loss.py:106-116`` is reused VERBATIM — "There is no
    override."); the 9C-A2 composition rule (>= 2 terms, >= 1 deterministic
    post-state ground-truth anchor — a judge-only loss is INVALID) is a thin
    validator on top, raising ``cua_loss_guard_missing`` (``CuaLossCompositionError``);
  * the whole CUA-agent config is the search space, assembled by
    ``optimize.build_practice_loop_manifest`` (the same ``base_agent`` +
    ``search_space`` whole-agent contract) with ``world.kind=browser`` /
    ``computer_use`` + ``cua_surface`` (browser | desktop) on ``WorldSpec.spec``;
  * the CUA sub-attribution is an additive tag stamped alongside the base
    ``FAILURE_LAYERS`` tag (the existing ``practice/_diagnose.py`` machinery is
    consumed, not rewritten);
  * ``browser`` / ``computer_use`` enter EXECUTABLE-LOOP status through the R4
    registry hook (``extensions.register_extension``) — never by widening the
    frozen ``SIMULATION_WORLD_KINDS`` tuple. They are already FROZEN typed-only
    members (the §0/9C-A1b nuance vs 9B's ``image``): so the registration gates on
    the EXECUTABLE-LOOP RECORD presence in ``_EXTRA_WORLD_KINDS``, NOT on the
    verbatim image idempotence guard (which would short-circuit immediately).

The CUA perturbation operators (selector-drift / layout-shift / stale-screenshot
/ injected-DOM) are ALREADY in the kit's ``BrowserEnvironment`` mutation pack
(9C-A1c / 9C-D4) — there is NO ``cua_perturb.py`` and NO ``apply_cua_perturbations``
function (the sharpest contrast with 9B's ``image_perturb.py``);
``V1_CUA_PERTURBATION_OPERATORS`` is a NAMING MIRROR ONLY.

The canon constants below are this module's home; ``trinity.py`` carries literal
mirrors that the milestone test cross-pins (the GUNA_AXES cross-pin pattern —
trinity never imports this module so the gate runs even if this is broken).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

# --- canon (ARCH-9C §2.1 CUA-loss term refs + §2.3 sub-attribution) ----------
# The browser-surface loss menu (the 9-tuple analogue of V1_IMAGE_LOSS_TERM_REFS,
# the 6-tuple). ``grounding_step_accuracy`` is admitted only under
# cua_surface=desktop — see V1_CUA_DESKTOP_ANCHOR_TERMS / unit 4.
V1_CUA_LOSS_TERM_REFS = (
    "task_success",
    "state_match",
    "grounding_mutation_resilience",
    "action_correctness",
    "step_efficiency",
    "safety_adherence",
    "tool_evidence",
    "trace_coverage",
    "completion_judge",
)

# The MANDATORY deterministic post-state anchors (analogue of
# V1_IMAGE_LOSS_DETERMINISTIC_ANCHOR_TERMS). Both alias the SAME state_quality
# anchor (optimize.py:17565-17568 — url/final_state/order_id match); both admitted
# so authors can name the outcome either way. A browser-surface loss MUST carry
# >= 1 of these (unit 2). For desktop the admissible anchor narrows to
# V1_CUA_DESKTOP_ANCHOR_TERMS (unit 4).
V1_CUA_LOSS_DETERMINISTIC_ANCHOR_TERMS = ("task_success", "state_match")

# The narrower deterministic anchor admitted under cua_surface=desktop
# (ARCH §2.4 / 9C-A7b) — grounding/step accuracy, NOT full task success.
V1_CUA_DESKTOP_ANCHOR_TERMS = ("grounding_step_accuracy",)

# The bounded/guarded judge contributors (analogue of V1_IMAGE_LOSS_JUDGE_TERMS);
# keyed (unit 4). A judge-only loss (terms subset of this set) is structurally
# rejected (unit 2 / 9C-D3).
V1_CUA_LOSS_JUDGE_TERMS = ("completion_judge",)

# The dual-objective safety axis (ST-WebAgentBench CuP / HarmonyGuard); the
# unsafe-completion canary rides it (unit 2). CuP-proxy:
# prompt_injection_touched==0 AND blocked_action==0 (the kit's action_quality
# safety half, optimize.py:17552-17554).
V1_CUA_LOSS_MANDATORY_SAFETY_TERMS = ("safety_adherence",)

# The four-token CUA sub-attribution closed set (9C §2.3), stamped alongside the
# base FAILURE_LAYERS tag (analogue of V1_IMAGE_FAILURE_SUBLAYERS).
V1_CUA_FAILURE_SUBLAYERS = ("perception", "grounding", "action_policy", "reasoning_memory")

# The cua_surface switch on WorldSpec.spec (analogue of V1_IMAGE_TASK_MODES). ONE
# world loop, two surface profiles (9C-D2): browser -> world.kind=browser (full
# post-state); desktop -> world.kind=computer_use (grounding/step, unit 4).
V1_CUA_SURFACES = ("browser", "desktop")

# A MARKER field on artifact metadata — NOT a new evidence class (R5/A18; the
# frozen EVIDENCE_CLASSES 4-tuple live/_contract.py:18 is unchanged). The analogue
# of V1_IMAGE_FIDELITY_TIERS. (ARCH §2.6)
V1_CUA_FIDELITY_TIERS = ("deterministic_fixture", "keyed_live_model")

# The typed ``kind`` discriminators a completion guard row may carry, beyond the
# base sentinel/canary rows the loss guard already allows (ARCH §2.2; the analogue
# of V1_IMAGE_PERCEPTION_GUARD_KINDS).
V1_CUA_COMPLETION_GUARD_KINDS = ("fake_completion", "unsafe_completion")

# NAMING MIRROR ONLY (9C-A1c / 9C-D4). References the kit's EXISTING mutation-pack
# operators (normalize_browser_mutation_pack, environment.py:5146;
# _browser_mutation_perturbations, :28727; the prompt-injection surfaces,
# :29350/:2903). There is NO cua_perturb.py and NO apply_cua_perturbations
# function — the sharpest contrast with 9B's image_perturb.py.
V1_CUA_PERTURBATION_OPERATORS = ("selector_drift", "layout_shift", "stale_screenshot", "injected_dom")

# The registered world-kind tokens + the namespaced extension names (R4 hook). The
# CUA loop registers EXECUTABLE-LOOP status for two already-frozen kinds.
CUA_BROWSER_WORLD_KIND = "browser"
CUA_DESKTOP_WORLD_KIND = "computer_use"
CUA_BROWSER_EXTENSION_NAME = "agentlearning.browser_cua"
CUA_DESKTOP_EXTENSION_NAME = "agentlearning.computer_use_cua"

# The R4 rung -> evidence-class ladder (ARCH §2.6). The deterministic core is
# local_gate/captured_fixture; live_lane is added ONLY on the keyed lane record
# (unit 7), never the day-one deterministic record.
_CUA_RUNG_LADDER = {
    "rung1": ["local_gate"],
    "perturbed": ["live_stressed", "captured_fixture"],
    "keyed_browser_vm": ["live_lane"],
}


class CuaLossCompositionError(ValueError):
    """Raised when a CUA objective violates the 9C-A2 composition rule (the
    ``cua_loss_guard_missing`` finding — a CUA specialization of
    ``objective_guards_missing``). A ``ValueError`` subclass so callers can
    ``except ValueError`` exactly as for ``ImageLossCompositionError`` /
    ``VoiceLossCompositionError``."""


def _term_refs(objective: Mapping[str, Any]) -> list[str]:
    """The objective's eval refs (read from ``evals`` — the loss.py schema; also
    tolerant of a ``terms`` alias)."""
    rows = objective.get("evals") or objective.get("terms") or []
    return [
        str(term.get("eval"))
        for term in rows
        if isinstance(term, Mapping) and term.get("eval")
    ]


def _admissible_anchor_terms(cua_surface: str) -> tuple[str, ...]:
    """The surface-admissible deterministic anchor set (the analogue of the image
    ``_admissible_anchor_terms(task_mode)``): browser -> the full post-state
    anchors; desktop -> the narrower grounding/step anchor (unit 4)."""
    if cua_surface == "desktop":
        return V1_CUA_DESKTOP_ANCHOR_TERMS
    return V1_CUA_LOSS_DETERMINISTIC_ANCHOR_TERMS


def _admissible_term_refs(cua_surface: str) -> tuple[str, ...]:
    """The surface-admissible loss-term menu: browser -> V1_CUA_LOSS_TERM_REFS;
    desktop -> the narrower grounding/step anchor + the deterministic-composition
    + safety + judge terms (unit 4). Desktop drops the browser-only post-state
    anchors (task_success / state_match) since the credential-free desktop rung is
    grounding/step ONLY, not full task success."""
    if cua_surface == "desktop":
        return (
            V1_CUA_DESKTOP_ANCHOR_TERMS
            + (
                "grounding_mutation_resilience",
                "action_correctness",
                "step_efficiency",
                "safety_adherence",
                "tool_evidence",
                "trace_coverage",
            )
            + V1_CUA_LOSS_JUDGE_TERMS
        )
    return V1_CUA_LOSS_TERM_REFS


def attribute_cua_sublayer(
    *,
    failure_layer: str,
    deficit: Mapping[str, Any] | None = None,
    signal: str | None = None,
) -> str:
    """Map a weak CUA cell to a ``V1_CUA_FAILURE_SUBLAYERS`` token, stamped
    ALONGSIDE the base ``FAILURE_LAYERS`` tag (a weak cell carries both, e.g.
    ``{failure_layer:"agent_behavior", cua_sublayer:"grounding"}``). The base
    attribution rides the existing ``practice/_diagnose.py`` machinery; this is the
    thin sublayer helper (the CUA analogue of ``attribute_image_sublayer``,
    image_loop.py:208).

    Routing (ARCH-9C §2.3 table, grounded in the observe->ground->act decomposition
    + the kit's layers ["browser","cua","security","evaluator"] optimize.py:17267 +
    the step-level stuck/milestone split 2604.27151):
      * stale screenshot, didn't refresh; missed an observed change -> ``perception``
        (observation-channel failure);
      * selector drifted, mis-clicked; coordinate off -> ``grounding`` (the
        observe->ground seam, the dominant mutation-resilience failure);
      * looped on the same step / 2.5-2.8x too many steps; touched injected banner
        -> ``action_policy`` (action/escalation policy + safety);
      * right perception, wrong plan; bad memory of prior steps ->
        ``reasoning_memory`` (plan/memory failure — ACuRL / Reflexion)."""

    sig = str(signal or (deficit or {}).get("signal") or "").lower()
    # Precedence-ordered (the ARCH §2.3 routing table). reasoning_memory is
    # checked FIRST among the higher-cognition cues so a "right perception, wrong
    # plan; bad memory of prior steps" cell routes to reasoning_memory even though
    # it mentions perception (the word is a red herring; the ARCH perception row is
    # "stale screenshot / missed an observed change", which carries none of these
    # cues).
    if any(
        k in sig
        for k in ("wrong plan", "wrong_plan", "bad memory", "memory", "reflect",
                  "reasoning", "prior step", "prior_step")
    ):
        return "reasoning_memory"
    if any(
        k in sig
        for k in ("stale screenshot", "stale_screenshot", "didn't refresh",
                  "did not refresh", "missed an observed", "missed_change",
                  "observed change", "screenshot")
    ):
        return "perception"
    if any(
        k in sig
        for k in ("selector drift", "selector_drift", "drifted selector",
                  "selector drifted", "mis-click", "misclick", "mis click",
                  "coordinate off", "coordinate_off", "ground")
    ):
        return "grounding"
    if any(
        k in sig
        for k in ("loop", "too many steps", "step_efficiency", "redundant",
                  "injected banner", "injected_banner", "injection", "blocked",
                  "escalation", "action_policy", "action policy", "unsafe")
    ):
        return "action_policy"
    # default: infra-implicated cells land on perception (the cheapest observation
    # fix before blaming the policy); otherwise the reasoning/memory layer.
    if failure_layer in ("lane_infra", "framework_runtime", "provider"):
        return "perception"
    return "reasoning_memory"


def compile_cua_objective(
    payload: Mapping[str, Any], *, cua_surface: str = "browser"
) -> dict:
    """Compile a multi-objective CUA loss with a fake/unsafe-completion Goodhart
    guard (ARCH-9C §2.2 / 9C-A2 / 9C-D3). The CUA analogue of
    ``compile_image_objective`` (image_loop.py:132). Enforces, ON TOP of the
    verbatim ``loss.compile_objective`` Goodhart guard:

      rule 1: closed-set ``cua_surface`` (browser | desktop);
      rule 2: >= 2 terms (a single-term CUA objective is reward-hackable);
      rule 3: >= 1 surface-admissible deterministic post-state anchor — a
              judge-only loss is INVALID (9C-D3). ``cua_surface`` selects the
              admissible anchor set: browser -> V1_CUA_LOSS_DETERMINISTIC_ANCHOR_TERMS;
              desktop -> V1_CUA_DESKTOP_ANCHOR_TERMS (the narrower grounding/step
              anchor, unit 4);
      rule 4: unknown-ref rejection (every term must be a member of the surface
              menu);
      rule 5: when sentinel/canary rows carry a completion ``kind`` discriminator
              it must be in V1_CUA_COMPLETION_GUARD_KINDS (the closed set).

    Then delegates to ``loss.compile_objective`` VERBATIM — which unconditionally
    enforces the populated guard block (sentinel_rows / canary_evals,
    min_guard_count >= 1, "There is no override.")."""

    from . import loss as _loss  # downward facade import (legal; image_loop.py idiom)

    # rule 1: closed-set cua_surface.
    if cua_surface not in V1_CUA_SURFACES:
        raise CuaLossCompositionError(
            f"cua_loss_guard_missing: cua_surface {cua_surface!r} not in "
            f"{V1_CUA_SURFACES}"
        )

    refs = _term_refs(payload)

    # rule 2: >= 2 terms.
    if len(refs) < 2:
        raise CuaLossCompositionError(
            "cua_loss_guard_missing: a CUA objective is reward-hackable as a "
            "single term; it MUST be multi-objective (>= 2 terms). "
            f"got {refs}"
        )

    # rule 3: >= 1 surface-admissible deterministic post-state anchor (judge-only
    # REJECTED — 9C-D3).
    anchors = _admissible_anchor_terms(cua_surface)
    if not any(ref in anchors for ref in refs):
        raise CuaLossCompositionError(
            "cua_loss_guard_missing: a CUA loss MUST carry >= 1 deterministic "
            f"post-state anchor {anchors}; a judge-only loss is INVALID by "
            f"contract (9C-D3). got {refs} (cua_surface={cua_surface})"
        )

    # rule 4: unknown-ref rejection (surface-filtered).
    allowed = _admissible_term_refs(cua_surface)
    for ref in refs:
        if ref not in allowed:
            raise CuaLossCompositionError(
                f"cua_loss_guard_missing: unknown CUA loss term {ref!r}; "
                f"expected members of {allowed} (cua_surface={cua_surface})"
            )

    # rule 5: the fake/unsafe-completion guard rows ride the existing
    # sentinel_rows/canary_evals with a typed ``kind`` discriminator (no new
    # ObjectiveSpec field, ARCH-9C §2.2). When present it must be in the closed set
    # (plus any untyped/base rows the loss guard already allows).
    guards = payload.get("guards") or {}
    for bucket in ("sentinel_rows", "canary_evals"):
        for row in guards.get(bucket) or []:
            if isinstance(row, Mapping):
                kind = row.get("kind")
                if kind is not None and kind not in V1_CUA_COMPLETION_GUARD_KINDS:
                    raise CuaLossCompositionError(
                        f"cua_loss_guard_missing: guard row kind {kind!r} not in "
                        f"{V1_CUA_COMPLETION_GUARD_KINDS}"
                    )

    # the verbatim Goodhart guard (loss.py:106-116) — "There is no override."
    return _loss.compile_objective(payload)


def _validate_cua_world_spec(spec: Mapping[str, Any]) -> None:
    """The R4 ``spec_validator`` for the CUA world: validate the ``cua_surface``
    switch (browser | desktop) on ``WorldSpec.spec`` (the analogue of
    ``_validate_image_world_spec``, image_loop.py:293). Raises ValueError on an
    unknown surface (the closed-set guard)."""

    cua_surface = str((spec or {}).get("cua_surface", "browser"))
    if cua_surface not in V1_CUA_SURFACES:
        raise ValueError(
            f"cua world.spec.cua_surface {cua_surface!r} not in {V1_CUA_SURFACES}"
        )


def _ensure_cua_world_registered(cua_surface: str = "browser") -> None:
    """Flip ``browser`` / ``computer_use`` from typed-only to EXECUTABLE-LOOP
    status via the R4 hook (ARCH-9C §2.1 / §2.3 / 9C-D2 / 9C-A1b). Idempotent BY
    VENDOR.NAME.

    THE 9C-A1b RULE (binding): this does NOT use the verbatim image idempotence
    guard (``if kind in resolved_world_kinds(): return``, image_loop.py:272),
    because ``browser`` / ``computer_use`` are ALREADY in ``resolved_world_kinds()``
    as frozen built-ins (contract.py:55) — that guard would short-circuit
    IMMEDIATELY and never record the R4 executable-loop evidence (spec_validator +
    rung_ladder + rung1_fixture_green). Instead it gates on the EXECUTABLE-LOOP
    RECORD presence in ``_EXTRA_WORLD_KINDS`` (the additive R4 record keyed by
    vendor.name). ``register_world_kind`` pushes the CUA record into
    ``_EXTRA_WORLD_KINDS``; built-ins shadow extensions at resolution
    (contract.py:88-91), so ``WorldSpec(kind="browser")`` keeps validating against
    the built-in entry and the frozen ``SIMULATION_WORLD_KINDS`` tuple stays
    byte-stable. ``browser`` / ``computer_use`` stay in
    ``TYPED_ONLY_WORLD_KINDS_V1`` — executable-loop status is carried by the
    registry record + the ``cua_loop_readiness`` gate, NOT by the frozen
    executable tuple."""

    from fi.simulate.simulation import contract as _contract

    if cua_surface not in V1_CUA_SURFACES:
        raise CuaLossCompositionError(
            f"cua_loss_guard_missing: cua_surface {cua_surface!r} not in "
            f"{V1_CUA_SURFACES}"
        )

    if cua_surface == "browser":
        kind_token = CUA_BROWSER_WORLD_KIND
        vendor_name = CUA_BROWSER_EXTENSION_NAME
    else:
        kind_token = CUA_DESKTOP_WORLD_KIND
        vendor_name = CUA_DESKTOP_EXTENSION_NAME

    from . import extensions as _ext

    # 9C-A1b: gate on the EXECUTABLE-LOOP RECORD, not on bare admissibility. The
    # kind is already admissible (built-in); the executable-loop marker is the
    # _EXTRA_WORLD_KINDS record carrying the CUA kind_token + the vendor.name.
    # register_extension keys _EXTRA_WORLD_KINDS by the kind_token (it calls
    # contract.register_world_kind(token, stored)), so the contract record is read
    # by kind_token; the vendor.name lives inside the record's ``name`` field
    # (the idempotence key per 9C-A1b — one executable-loop record per vendor).
    existing = _contract._EXTRA_WORLD_KINDS.get(kind_token)  # additive record, never the built-in
    if existing and existing.get("kind_token") == kind_token and existing.get("name") == vendor_name:
        return  # executable-loop record already present (idempotent by vendor.name)

    # The persistent extension registry (extensions._REGISTRY) outlives the
    # contract's _EXTRA_WORLD_KINDS within a process. register_extension RAISES on
    # a name collision, so if the extension is ALREADY in the registry (e.g. the
    # contract record was cleared but the registry was not), re-push the existing
    # stored record into the contract directly rather than re-registering. This
    # keeps the gate idempotent by vendor.name AND restores the executable-loop
    # evidence in _EXTRA_WORLD_KINDS.
    stored = _ext.resolve("environment", vendor_name)
    if stored is not None and stored.get("kind_token") == kind_token:
        _contract.register_world_kind(kind_token, stored)
        return

    _ext.register_extension(
        "environment",
        {
            "name": vendor_name,                       # vendor.name shape (_validate_record)
            "kind_token": kind_token,                  # "browser" / "computer_use" token
            "spec_validator": _validate_cua_world_spec,  # R4 mandate
            "rung_ladder": _CUA_RUNG_LADDER,             # R4 mandate
            # the deterministic core; live_lane is added ONLY on the keyed lane
            # record (unit 7), never here. gated_contexts_runnable stays False
            # until rung1_fixture_green (extensions.py admission); 9C never
            # silently claims executable.
            "evidence_class_capability": ["local_gate", "captured_fixture"],
        },
    )


def build_cua_practice_loop_manifest(
    *,
    name: str,
    base_agent: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any]],
    objective: Mapping[str, Any],
    eval_budget: int,
    seed: int,
    cua_surface: str = "browser",
    scenario_inline: Optional[Mapping[str, Any]] = None,
    max_rounds: int = 8,
) -> dict[str, Any]:
    """Assemble the CUA improvement-loop manifest: the 13D Practice Loop on
    ``world.kind=browser`` / ``computer_use`` + ``cua_surface`` with the
    multi-objective guarded CUA loss + the whole CUA-agent search space (9C-D5).
    Delegates to ``optimize.build_practice_loop_manifest`` so its validators hold
    VERBATIM (9C-A3). The objective is compiled by ``compile_cua_objective`` (the
    9C-A2 rule) before it rides the simulation.

    Byte-parallel to ``build_image_practice_loop_manifest`` except: (a) the
    ``_ensure_cua_world_registered(cua_surface)`` call uses the 9C-A1b
    executable-loop-record gate (NOT the verbatim image idempotence guard); (b)
    ``world["kind"]`` is ``"browser"`` / ``"computer_use"`` driven by
    ``cua_surface`` (image's is always ``"image"``); (c) the ``spec["cua_surface"]``
    write (instead of ``spec["task_mode"]``)."""

    from . import optimize as _optimize  # downward facade import (legal)

    if cua_surface not in V1_CUA_SURFACES:
        raise CuaLossCompositionError(
            f"cua_loss_guard_missing: cua_surface {cua_surface!r} not in "
            f"{V1_CUA_SURFACES}"
        )

    _ensure_cua_world_registered(cua_surface)                  # step 1 (§3.1, 9C-A1b)
    compiled = compile_cua_objective(objective, cua_surface=cua_surface)  # step 2 (unit 2)
    inline = dict(scenario_inline or {})
    inline.setdefault("version", "agent-learning.simulation.v1")
    inline["objective"] = compiled
    world = dict(inline.get("world") or {})
    world["kind"] = (
        CUA_BROWSER_WORLD_KIND if cua_surface == "browser" else CUA_DESKTOP_WORLD_KIND
    )                                                          # step 3 — the registered kind
    spec = dict(world.get("spec") or {})
    spec["cua_surface"] = cua_surface                          # the cua_surface switch on WorldSpec.spec
    world["spec"] = spec
    inline["world"] = world

    return _optimize.build_practice_loop_manifest(             # step 4 — VERBATIM delegate
        name=name,
        simulation={"version": inline["version"], "inline": inline},
        base_agent=base_agent,
        search_space=search_space,
        eval_budget=eval_budget,
        seed=seed,
        max_rounds=max_rounds,
    )


# === Unit 7 — the keyed real-browser/VM lane (opt-in, NEVER a gate prerequisite) ===
# ARCH-9C §2.4 / §2.6 / 9C-D1/D6/A8. The keyed completion_judge term, the desktop
# full-post-state rungs, and the one real-browser/CUA-agent live-proof are
# owner-keyed/infra-provisioned, opt-in, never a release gate. The deterministic
# core stays local_gate/captured_fixture; the keyed lane is the ONLY honest place
# for live_lane (a real keyed browser/VM/judge ran).

KEYED_CUA_EXTENSION_NAME = "agentlearning.cua.keyed"

# the env keys that gate the keyed completion_judge lane (checked, never required
# by any gate). The GRADE-shaped judge term (9C-A8) calls a judge model.
CUA_JUDGE_KEY_ENVS = ("AGENT_LEARNING_CUA_JUDGE_KEY", "OPENAI_API_KEY")


class CuaKeyedLaneUnavailable(RuntimeError):
    """Raised by the keyed lane when no judge key / VM infra is present — the loud
    refusal (the ``cua_judge_key_unavailable`` / ``cua_desktop_infra_unavailable``
    finding). The deterministic core NEVER raises this; only the opt-in keyed path
    does."""


def cua_judge_key_present() -> bool:
    """True iff a judge key is configured for the keyed completion_judge lane."""
    import os

    return any(os.environ.get(env) for env in CUA_JUDGE_KEY_ENVS)


def register_keyed_cua_lane() -> None:
    """Register the SEPARATE keyed-lane extension record that adds ``live_lane`` to
    ``evidence_class_capability`` (ARCH-9C §2.6, unit 7). Idempotent. This is the
    ONLY record that may carry ``live_lane`` — the deterministic-core records
    (``_ensure_cua_world_registered``) stay ``("local_gate","captured_fixture")``.
    NEVER called by the gate; opt-in only."""
    from . import extensions as _ext

    if _ext.resolve("environment", KEYED_CUA_EXTENSION_NAME) is not None:
        return
    _ext.register_extension(
        "environment",
        {
            "name": KEYED_CUA_EXTENSION_NAME,
            # the keyed lane declares the keyed capability without a second
            # kind_token (the world kinds are already registered).
            "evidence_class_capability": ["local_gate", "captured_fixture", "live_lane"],
        },
    )


def run_keyed_cua_live_proof(
    *,
    base_agent: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any]],
    objective: Mapping[str, Any],
    eval_budget: int,
    seed: int,
    cua_surface: str = "browser",
    name: str = "cua-keyed-live-proof",
) -> dict[str, Any]:
    """The one owner-keyed live-proof entry (WORKFLOW Step 5 real-keys ground
    rule). Refuses LOUDLY without a key (``CuaKeyedLaneUnavailable`` ->
    ``cua_judge_key_unavailable``) — never a fake number, never a release
    prerequisite. With a key, it builds the manifest and marks the run
    ``live_lane`` / ``fidelity_tier=keyed_live_model``.

    The keyed run itself (calling the GRADE judge / a real browser / a VM) is left
    to the caller's runtime; this returns the keyed manifest + the honest
    evidence-class stamp so an owner can execute it once with real keys/infra."""
    if not cua_judge_key_present():
        raise CuaKeyedLaneUnavailable(
            "cua_judge_key_unavailable: the keyed real-browser/VM lane requires a "
            f"judge key (one of {CUA_JUDGE_KEY_ENVS}); withheld -- never a fake "
            "number, never a release prerequisite"
        )
    register_keyed_cua_lane()
    manifest = build_cua_practice_loop_manifest(
        name=name,
        base_agent=base_agent,
        search_space=search_space,
        objective=objective,
        eval_budget=eval_budget,
        seed=seed,
        cua_surface=cua_surface,
    )
    return {
        "manifest": manifest,
        "evidence_class": "live_lane",          # the ONLY honest live_lane
        "fidelity_tier": "keyed_live_model",
        "cua_surface": cua_surface,
    }
