"""Vendor import parsers — Vapi / Retell (Phase 7, unit 7; ARCH Decision 8).

Import-only: we read their formats, we never call their platforms (R§3.6).
Lossless: the full source text is retained at ``provenance.raw`` and
``render_vendor_text`` reproduces it byte-exact (the gate parity check).
Anything not in the strict-subset mapping tables stays verbatim in
``identity.style_notes``. Imports earn no evidence-class shortcut —
calibration is the equalizer (evidence_class=hand_written).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from fi.simulate.simulation.models import (
    BehaviorPolicy,
    Persona,
    PersonaIdentity,
    PersonaProvenance,
    ScenarioGoal,
)

PERSONA_VENDOR_IMPORT_FORMATS = ("vapi", "retell")

_VAPI_SECTION_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")
_RETELL_SECTIONS = ("identity", "goal", "personality")
_RETELL_SECTION_RE = re.compile(
    r"^(?:#+\s*)?(?P<name>identity|goal|personality)\s*:?\s*$",
    re.IGNORECASE,
)

# Fixed keyword table (exhaustive; ARCH Decision 8 / BUILD §7). The verbosity
# dial is post-v1.x (ARCH Decision 4) — brief/talkative stay verbatim notes.
_IMPATIENT_ESCALATION_PRESET = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_INTERRUPT_PROPENSITY = 0.6


def _parse_sections(
    text: str, *, fmt: str
) -> Tuple[Dict[str, List[str]], List[str]]:
    """-> (ordered section name -> lines, free lines outside any section)."""
    sections: Dict[str, List[str]] = {}
    free_lines: List[str] = []
    current: Optional[str] = None
    for line in text.splitlines():
        if fmt == "vapi":
            match = _VAPI_SECTION_RE.match(line.strip())
            if match:
                current = match.group("name").strip().lower()
                sections.setdefault(current, [])
                continue
        else:
            match = _RETELL_SECTION_RE.match(line.strip())
            if match:
                current = match.group("name").strip().lower()
                sections.setdefault(current, [])
                continue
        if current is None:
            free_lines.append(line)
        else:
            sections[current].append(line)
    return sections, free_lines


def _clean(lines: List[str]) -> List[str]:
    return [line.strip() for line in lines if line.strip()]


def _apply_keyword_table(
    lines: List[str],
    style_notes: List[str],
) -> Optional[BehaviorPolicy]:
    """Fixed keyword table ONLY — unmatched lines go to style_notes verbatim."""
    escalation: Optional[List[float]] = None
    interruption: Optional[float] = None
    for line in lines:
        lowered = line.lower()
        matched = False
        if "impatien" in lowered:  # impatient / impatience / shows impatience
            escalation = list(_IMPATIENT_ESCALATION_PRESET)
            matched = True
        if "interrupt" in lowered:
            interruption = _INTERRUPT_PROPENSITY
            matched = True
        if not matched:
            style_notes.append(line)
        else:
            # the trajectory spec is made executable AND the prose is kept
            style_notes.append(line)
    if escalation is None and interruption is None:
        return None
    kwargs = {}
    if escalation is not None:
        kwargs["escalation_schedule"] = escalation
    if interruption is not None:
        kwargs["interruption_propensity"] = interruption
    return BehaviorPolicy(**kwargs)


def import_vendor_persona(
    text: str,
    *,
    format: str,
) -> Tuple[Persona, Optional[ScenarioGoal]]:
    """Parse a Vapi/Retell persona file into (Persona, ScenarioGoal stub).

    Goals belong to the Scenario, not the Persona (2601.15290 split); the
    legacy ``outcome`` field gets the first goal line for back-compat."""
    if format not in PERSONA_VENDOR_IMPORT_FORMATS:
        raise ValueError(
            f"unsupported vendor format {format!r}; "
            f"expected one of {list(PERSONA_VENDOR_IMPORT_FORMATS)}"
        )
    sections, free_lines = _parse_sections(text, fmt=format)

    identity_lines = _clean(sections.get("identity", []))
    personality_lines = _clean(sections.get("personality", []))
    goal_lines = _clean(sections.get("goals", []) or sections.get("goal", []))
    style_lines = _clean(sections.get("interaction style", []))

    name: Optional[str] = None
    summary_lines: List[str] = []
    for line in identity_lines:
        if line.startswith("Name:"):
            name = line.split("Name:", 1)[1].strip()
        else:
            summary_lines.append(line)
    if not sections:
        # no sections (free text): whole text -> identity.summary; nothing inferred
        summary_lines = _clean(free_lines)

    style_notes: List[str] = []
    policy = _apply_keyword_table(personality_lines + style_lines, style_notes)

    goal: Optional[ScenarioGoal] = None
    if goal_lines:
        goal = ScenarioGoal(states=list(goal_lines), success_state=goal_lines[0])

    summary = " ".join(summary_lines).strip() or None
    identity = PersonaIdentity(name=name, summary=summary, style_notes=style_notes)
    provenance = PersonaProvenance(
        evidence_class="hand_written",
        source_format=format,
        raw=text,
    )
    persona = Persona(
        persona=({"name": name} if name else {}),
        situation=summary or "Imported vendor persona session.",
        outcome=goal_lines[0] if goal_lines else "The conversation completes naturally.",
        identity=identity,
        behavior_policy=policy,
        provenance=provenance,
    )
    return persona, goal


def render_vendor_text(persona: Persona) -> str:
    """Reproduce the imported source byte-exact (the gate parity check)."""
    if persona.provenance is None or persona.provenance.raw is None:
        raise ValueError(
            "persona carries no provenance.raw — only vendor-imported "
            "personas can be rendered back to vendor text"
        )
    return persona.provenance.raw


__all__ = [
    "PERSONA_VENDOR_IMPORT_FORMATS",
    "import_vendor_persona",
    "render_vendor_text",
]
