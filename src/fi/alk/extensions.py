"""Unit 4 (BBG U4 / ARCH §2e) — the four 13D-4 registries + extension_admission.

Explicit in-process registration (AD-J: no entry-points, no import-time
discovery — local-first, gate-scannable). One uniform record shape across four
points; ONE choke point (``extension_admission``) enforces 13D-D5 (extension ≠
exemption) in gated contexts. The facade pushes world-kind/role registrations
DOWN into ``fi.simulate.simulation.contract`` setters (Appendix C-1); the engine
never imports up.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .live._contract import EVIDENCE_CLASSES, RELEASE_ADMISSIBLE_EVIDENCE_CLASSES

EXTENSION_POINTS = ("environment", "loss", "optimizer", "generator")

# point -> name -> record
_REGISTRY: Dict[str, Dict[str, dict]] = {point: {} for point in EXTENSION_POINTS}


class ExtensionError(ValueError):
    """Raised when a registration record is malformed."""


def _validate_record(point: str, record: Mapping[str, Any]) -> dict:
    if point not in EXTENSION_POINTS:
        raise ExtensionError(f"point {point!r} not in {EXTENSION_POINTS}")
    name = record.get("name")
    if not name or "." not in str(name):
        raise ExtensionError(
            "extension record requires a namespaced name 'vendor.name'"
        )
    if str(name) in _REGISTRY[point]:
        raise ExtensionError(f"extension name collision: {name!r} already registered for {point}")
    caps = record.get("evidence_class_capability") or []
    for cap in caps:
        if cap not in EVIDENCE_CLASSES:
            raise ExtensionError(
                f"evidence_class_capability {cap!r} not in {EVIDENCE_CLASSES}"
            )
    if point == "optimizer" and not record.get("declared_budgets"):
        raise ExtensionError(
            "optimizer extensions REQUIRE declared_budgets (refused otherwise)"
        )
    stored = dict(record)
    stored["point"] = point
    stored.setdefault("provides", None)
    stored.setdefault("conformance_manifest", None)
    stored.setdefault("evidence_class_capability", list(caps))
    stored.setdefault("version", "0.0.0")
    stored["gated_contexts_runnable"] = False  # until a green conformance run lands
    return stored


def register_extension(point: str, record: Mapping[str, Any]) -> dict:
    stored = _validate_record(point, record)
    # world-kind/role registrations push down into the engine contract (C-1).
    if point == "environment":
        token = stored.get("kind_token")
        if token:
            if not stored.get("spec_validator") or not stored.get("rung_ladder"):
                raise ExtensionError(
                    "a custom world.kind extension MUST declare spec_validator + rung_ladder (R4)"
                )
            from fi.simulate.simulation import contract as _contract
            _contract.register_world_kind(str(token), stored)
    _REGISTRY[point][str(stored["name"])] = stored
    return stored


def register_environment(record: Mapping[str, Any]) -> dict:
    """Register a descriptive environment **metadata record** (studio extension).

    Canon correspondence (assessment §8 Gap B): the runtime sibling
    ``fi.simulate.registry.register_environment`` registers a **runnable** plugin
    factory in ``environment_registry``. This one records metadata (and, for a
    ``world.kind`` extension carrying a ``kind_token``, writes the contract's
    extension side-table via ``contract.register_world_kind`` — the frozen canon
    constants never mutate). A record is not a factory — the two are deliberately
    unwired.
    """
    return register_extension("environment", record)


def register_objective(record: Mapping[str, Any]) -> dict:
    return register_extension("loss", record)


def register_optimizer(record: Mapping[str, Any]) -> dict:
    return register_extension("optimizer", record)


def register_generator(record: Mapping[str, Any]) -> dict:
    return register_extension("generator", record)


def register_role(record: Mapping[str, Any]) -> dict:
    """Register a namespaced cast role (pushes into the engine contract)."""
    stored = _validate_record("environment", {**record, "name": record["name"]})
    token = stored.get("kind_token") or stored.get("role")
    from fi.simulate.simulation import contract as _contract
    if token:
        _contract.register_cast_role(str(token), stored)
    _REGISTRY["environment"][str(stored["name"])] = stored
    return stored


def resolve(point: str, token: str) -> Optional[dict]:
    """Built-ins first — callers check the canon BEFORE consulting this."""
    return _REGISTRY.get(point, {}).get(str(token))


def registered(point: str) -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY.get(point, {})))


def extension_admission(record: Mapping[str, Any], context: Mapping[str, Any]) -> dict:
    """THE one choke point. In gated contexts (release-check/promotion/training)
    a registered extension that cannot produce admissible evidence does not run.
    Returns ``{admitted: bool}`` or the structured refusal finding."""
    gated = bool(context.get("gated"))
    if not gated:
        return {"admitted": True, "reason": "non_gated_passthrough"}

    point = record.get("point")
    caps = set(record.get("evidence_class_capability") or [])
    admissible_caps = caps & set(RELEASE_ADMISSIBLE_EVIDENCE_CLASSES)
    conformance_green = bool(record.get("conformance_green") or record.get("gated_contexts_runnable"))

    def refuse(reason: str) -> dict:
        return {
            "admitted": False,
            "type": "extension_evidence_inadmissible",
            "level": "error",
            "name": record.get("name"),
            "point": point,
            "reason": reason,
            "remediation": (
                "run the extension's conformance_manifest green with evidence in "
                f"{RELEASE_ADMISSIBLE_EVIDENCE_CLASSES} before any gated use"
            ),
        }

    # (i) conformance manifest ran green with release-admissible evidence
    if not conformance_green or not admissible_caps:
        return refuse("no green conformance run with release-admissible evidence")
    # (iii) optimizer extensions without declared_budgets never run
    if point == "optimizer" and not record.get("declared_budgets"):
        return refuse("optimizer extension has no declared_budgets")
    # (iv) environment/world-kind extensions claiming an executable kind need a
    #      green rung-1 fixture run recorded.
    if point == "environment" and record.get("kind_token"):
        if not record.get("rung1_fixture_green"):
            return refuse("world-kind extension lacks a green rung-1 fixture run")
    return {"admitted": True, "reason": "gated_admitted"}


def _reset_extensions() -> None:  # test-only
    for point in EXTENSION_POINTS:
        _REGISTRY[point].clear()
    from fi.simulate.simulation import contract as _contract
    _contract._reset_contract_extensions()
