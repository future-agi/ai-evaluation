"""Legacy embedded-dict persona auto-upgrade (Phase 7, unit 1.3)."""

from __future__ import annotations

from typing import Any, Mapping

from fi.simulate.simulation.models import (
    Persona,
    PersonaIdentity,
    PersonaProvenance,
)


def upgrade_legacy_persona(row: Mapping[str, Any]) -> Persona:
    """Free-dict persona row -> typed Persona, provenance=legacy (PRD §4.1).

    Lossless: the original dict stays in ``.persona`` untouched. Only EXACT
    key matches lift into identity (name/role/language); everything else
    remains free-form. No temperament/policy is invented — a legacy persona
    is untyped (``is_typed == False``), runs fine, and simply cannot produce
    fidelity evidence (lowest class; cannot back release claims)."""
    if isinstance(row, Persona):
        persona = row
    else:
        persona = Persona(**dict(row))
    if persona.provenance is not None:
        # already studio-managed (provenance deliberately set) — never
        # re-touched, so content addressing stays stable across load.
        return persona
    embedded = dict(persona.persona)
    identity = persona.identity or PersonaIdentity(
        name=embedded.get("name"),
        role=embedded.get("role"),
        language=embedded.get("language"),
    )
    return persona.model_copy(update={
        "identity": identity,
        "provenance": PersonaProvenance(evidence_class="legacy"),
    })


__all__ = ["upgrade_legacy_persona"]
