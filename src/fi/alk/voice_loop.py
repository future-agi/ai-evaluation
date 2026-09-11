"""Phase 9A unit 4 — the voice improvement loop (the 13D Practice Loop on
``world.kind = voice_telephony``).

ARCH §2.3 / decisions 9A-D5 (the 13D loop on voice_telephony; NO new optimizer),
9A-A4 (multi-objective-mandatory voice loss; single-timing-term rejection),
9A-A14 (``V1_VOICE_FAILURE_SUBLAYERS`` voice sub-attribution).

This module invents NO optimizer, NO artifact kind, NO loss machinery. It is a
thin composition layer over verbatim engines:

  * the multi-objective voice loss compiles via ``loss.compile_objective`` (the
    Goodhart guard at ``loss.py:106-116`` is reused VERBATIM — "There is no
    override."); the 9A-A4 composition rule (≥2 terms, ≥1 non-timing quality
    term) is a thin validator on top, raising ``voice_loss_guard_missing``;
  * the whole voice-agent config is the search space, assembled by
    ``optimize.build_practice_loop_manifest`` (the same ``base_agent`` +
    ``search_space`` whole-agent contract) with ``world.kind=voice_telephony``;
  * the voice sub-attribution is an additive tag stamped alongside the base
    ``FAILURE_LAYERS`` tag (the existing ``practice/_diagnose.py`` machinery is
    consumed, not rewritten).

The canon constants below are this module's home; ``trinity.py`` carries literal
mirrors that the milestone test cross-pins (the GUNA_AXES cross-pin pattern —
trinity never imports this module so the gate runs even if this is broken).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

# --- canon (ARCH §4 voice-loss term refs + §2.3 sub-attribution) ------------
# the gate-pinned eval refs (ARCH §4 names are binding in code/gate; the UI-UX
# display names — tool_arg_correctness/selectivity_jir/asr_wer_delta/
# perturbation_robustness_delta — are presentation aliases only).
V1_VOICE_LOSS_TERM_REFS = (
    "task_success",
    "tool_argument_correctness",
    "barge_in_latency",
    "ttfb",
    "wer_delta",
    "recovery",
    "selectivity",
    "codec_survival",
    "perturbation_robustness",
)
# the non-timing quality anchors — a voice loss MUST carry >= 1 of these (9A-A4).
V1_VOICE_LOSS_NON_TIMING_QUALITY_TERMS = ("task_success", "tool_argument_correctness")
# the timing-only terms (a single-timing-term objective is reward-hackable —
# ASPIRin R§2.3 — and is structurally rejected).
V1_VOICE_LOSS_TIMING_TERMS = ("barge_in_latency", "ttfb", "recovery")
# the four-token voice sub-attribution closed set (9A-A14), stamped alongside
# the base FAILURE_LAYERS tag.
V1_VOICE_FAILURE_SUBLAYERS = ("acoustic_codec", "asr_mishear", "llm", "tts_endpointing")


class VoiceLossCompositionError(ValueError):
    """Raised when a voice objective violates the 9A-A4 composition rule
    (the ``voice_loss_guard_missing`` finding — a voice specialization of
    ``objective_guards_missing``)."""


def _term_refs(objective: Mapping[str, Any]) -> list[str]:
    return [
        str(term.get("eval"))
        for term in (objective.get("evals") or [])
        if isinstance(term, Mapping) and term.get("eval")
    ]


def compile_voice_objective(payload: Mapping[str, Any]) -> dict:
    """Compile a multi-objective voice loss (9A-A4). Enforces, ON TOP of the
    verbatim ``loss.compile_objective`` Goodhart guard:

      (a) >= 2 terms,
      (b) >= 1 non-timing quality term
          (``task_success`` / ``tool_argument_correctness``),
      (c) a populated guard block (delegated to ``compile_objective``).

    A single-timing-term voice objective is structurally rejected
    (``VoiceLossCompositionError`` / the ``voice_loss_guard_missing`` finding).
    The underlying guard is the UNEDITED ``loss.py`` enforcement."""

    from . import loss as _loss  # downward facade import (legal)

    refs = _term_refs(payload)
    if len(refs) < 2:
        raise VoiceLossCompositionError(
            "voice_loss_guard_missing: a voice objective is reward-hackable as a "
            "single term (ASPIRin); it MUST be multi-objective (>= 2 terms). "
            f"got {refs}"
        )
    if not any(ref in V1_VOICE_LOSS_NON_TIMING_QUALITY_TERMS for ref in refs):
        raise VoiceLossCompositionError(
            "voice_loss_guard_missing: a voice objective MUST carry >= 1 "
            "non-timing quality term "
            f"({V1_VOICE_LOSS_NON_TIMING_QUALITY_TERMS}); a timing-only loss is "
            f"reward-hackable and is structurally rejected. got {refs}"
        )
    for ref in refs:
        if ref not in V1_VOICE_LOSS_TERM_REFS:
            raise VoiceLossCompositionError(
                f"voice_loss_guard_missing: unknown voice loss term {ref!r}; "
                f"expected members of {V1_VOICE_LOSS_TERM_REFS}"
            )
    # the verbatim Goodhart guard (loss.py:106-116) — "There is no override."
    return _loss.compile_objective(payload)


def attribute_voice_sublayer(
    *,
    failure_layer: str,
    deficit: Mapping[str, Any] | None = None,
    signal: str | None = None,
) -> str:
    """Map a weak voice cell to a ``V1_VOICE_FAILURE_SUBLAYERS`` token, stamped
    ALONGSIDE the base ``FAILURE_LAYERS`` tag (a weak cell carries both). The
    base attribution rides the existing ``practice/_diagnose.py`` machinery; this
    is the thin sublayer helper.

    Mapping (ARCH §2.3): a weak ``selectivity`` / endpointing signal →
    ``tts_endpointing`` (not ``llm``); a mis-heard value under clean audio →
    ``asr_mishear``; a claim that died through the codec → ``acoustic_codec``;
    otherwise the reasoning/policy layer → ``llm``."""

    sig = str(signal or (deficit or {}).get("signal") or "").lower()
    if any(k in sig for k in ("selectivity", "endpoint", "barge", "vad", "interrupt", "recovery", "ttfb")):
        return "tts_endpointing"
    if any(k in sig for k in ("codec", "survival", "packet", "band_energy", "perturbation")):
        return "acoustic_codec"
    if any(k in sig for k in ("wer", "mishear", "asr", "tool_argument", "transcription")):
        return "asr_mishear"
    # default to the reasoning/policy layer unless infra clearly implicated.
    if failure_layer in ("lane_infra", "framework_runtime", "provider"):
        return "acoustic_codec"
    return "llm"


def build_voice_practice_loop_manifest(
    *,
    name: str,
    base_agent: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any]],
    objective: Mapping[str, Any],
    eval_budget: int,
    seed: int,
    scenario_inline: Optional[Mapping[str, Any]] = None,
    max_rounds: int = 8,
) -> dict[str, Any]:
    """Assemble the voice improvement-loop manifest: the 13D Practice Loop on
    ``world.kind=voice_telephony`` with the multi-objective voice loss + the
    whole voice-agent search space (9A-D5). Delegates to
    ``optimize.build_practice_loop_manifest`` so its validators hold VERBATIM.
    The objective is compiled by ``compile_voice_objective`` (the 9A-A4 rule)
    before it rides the simulation."""

    from . import optimize as _optimize  # downward facade import (legal)

    compiled = compile_voice_objective(objective)
    inline = dict(scenario_inline or {})
    inline.setdefault("version", "agent-learning.simulation.v1")
    inline["objective"] = compiled
    world = dict(inline.get("world") or {})
    world["kind"] = "voice_telephony"
    inline["world"] = world

    return _optimize.build_practice_loop_manifest(
        name=name,
        simulation={"version": inline["version"], "inline": inline},
        base_agent=base_agent,
        search_space=search_space,
        eval_budget=eval_budget,
        seed=seed,
        max_rounds=max_rounds,
    )
