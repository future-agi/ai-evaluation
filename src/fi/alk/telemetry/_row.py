"""Canonical ledger-row construction + serialization (Phase 8, ARCH §2a).

Imports: stdlib only plus the two reused live/ seams. The content address must
be byte-identical on any machine, so serialization replicates the
``_schema.py:_json_sha256`` recipe exactly (``sort_keys=True``,
``separators=(",", ":")``, ``default=str``) — one canonicalization discipline,
no second divergable serializer (ARCH Decision 2). Redaction runs BEFORE the
row is content-addressed or written: the address is computed over redacted
bytes, so a re-run that re-redacts produces the same address.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping, Sequence

from ..live._contract import AGENT_LEARNING_RUN_KIND, EVIDENCE_CLASSES, VERDICTS
from ..live._transcript import redact_env_values  # the ONE redaction seam
from ._contract import (
    LEDGER_ROW_SCHEMA,
    NON_CANONICAL_FIELDS,
    PHASES,
    SEMCONV_VERSION_ENV,
)

# Fixed precision for floats in the addressed core — the kit's existing
# rounding rule (live/_transcript.py:105 rounds to 6 places).
_FLOAT_PRECISION = 6


def canonical_row_bytes(row: Mapping[str, Any]) -> bytes:
    """The exact bytes ``run_id`` is the SHA-256 of (the addressed core).

    Excludes ``created_at``/``run_id``/``chain`` — and ONLY those three
    (ARCH §2a): wall-clock and the chain digest are envelope fields that must
    never enter the content address.
    """

    preimage = {k: v for k, v in row.items() if k not in NON_CANONICAL_FIELDS}
    return json.dumps(
        preimage, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")  # == _schema.py:_json_sha256 recipe, byte-identical


def canonical_row_address(row: Mapping[str, Any]) -> str:
    """``run_id = SHA-256(canonical addressed core)`` (P8-D3)."""

    return hashlib.sha256(canonical_row_bytes(row)).hexdigest()


def _redact_value(value: Any, required_env: Sequence[str]) -> Any:
    """Walk a row value: redact env VALUES out of every string leaf and round
    floats to fixed precision so the addressed core has no platform-variant
    repr (ARCH §2a determinism rules)."""

    if isinstance(value, str):
        return redact_env_values(value, required_env)
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, _FLOAT_PRECISION)
    if isinstance(value, Mapping):
        return {
            _redact_value(key, required_env): _redact_value(item, required_env)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(item, required_env) for item in value]
    return value


def build_ledger_row(
    payload: Mapping[str, Any], *, required_env: Sequence[str] = ()
) -> dict[str, Any]:
    """Project an ``agent-learning.run.v1`` payload into a small ledger row of
    metadata + content-addressed asset REFERENCES, never copies (PRD §4.1).

    Redaction-before-serialize is the load-bearing ordering: ``_redact_value``
    runs on the last step before ``canonical_row_address`` and before any disk
    write — the same seam+placement as ``live/_transcript.py:111``.
    """

    summary = payload.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    evidence_class = payload.get("evidence_class")
    if evidence_class not in EVIDENCE_CLASSES:
        evidence_class = "local_gate"  # absence => local_gate (BUILD §1.3)
    capture = payload.get("capture")
    capture = capture if isinstance(capture, Mapping) else {}
    row: dict[str, Any] = {
        "schema": LEDGER_ROW_SCHEMA,
        "kind": AGENT_LEARNING_RUN_KIND,  # always the canonical run kind
        "phase": _infer_phase(payload),
        "evidence_class": evidence_class,
        "verdict": _project_verdict(payload, summary),
        "scores": _project_scores(summary),
        "gate_outcomes": _project_gate_outcomes(payload),
        "semconv_version": os.environ.get(SEMCONV_VERSION_ENV) or "unset",
        "manifest_address": _manifest_address(payload),
        # ASSET REFERENCES — content addresses, never copies (R§3.3):
        "asset_refs": _asset_refs(payload),
        "trace_ids": _trace_ids(payload),
        "content_bearing": _content_bearing(payload, capture),
        "redaction": _redaction_contract(capture),
    }
    # Redact env VALUES out of every string field BEFORE the row is
    # content-addressed or written (R§1 2507.06350; PRD §4.1):
    row = _redact_value(row, tuple(required_env))
    row["run_id"] = canonical_row_address(row)  # address AFTER redaction
    return row  # created_at/chain are added by the ledger append (envelope)


def content_admissible(run_payload: Mapping[str, Any]) -> bool:
    """The content-sync admission predicate (PRD §4.2): the same
    ``capture.redaction`` non-empty mapping + ``capture.reviewed is True``
    shape the ``live_lane_boundary`` gate demands on captured fixtures."""

    capture = run_payload.get("capture")
    capture = capture if isinstance(capture, Mapping) else {}
    redaction = capture.get("redaction")
    has_map = isinstance(redaction, Mapping) and bool(redaction)
    return has_map and capture.get("reviewed") is True


def declared_required_env(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect declared env names from the run payload (names only — the
    redaction seam replaces their VALUES with ``[redacted:NAME]``)."""

    names: list[str] = []
    for source in (
        payload.get("required_env"),
        _mapping(payload.get("live_lane")).get("required_env"),
        _mapping(payload.get("lane")).get("required_env"),
        _mapping(_mapping(payload.get("capture")).get("redaction")),
    ):
        if isinstance(source, Mapping):
            names.extend(str(name) for name in source)
        elif isinstance(source, (list, tuple)):
            names.extend(str(name) for name in source)
    seen: dict[str, None] = {}
    for name in names:
        if name:
            seen.setdefault(name, None)
    return tuple(seen)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _infer_phase(payload: Mapping[str, Any]) -> str:
    explicit = payload.get("phase")
    if isinstance(explicit, str) and explicit in PHASES:
        return explicit
    if isinstance(payload.get("live_lane"), Mapping) or isinstance(
        payload.get("lane"), (str, Mapping)
    ):
        return "live"
    if payload.get("optimization") is not None:
        return "optimize"
    if payload.get("redteam") is not None or payload.get("attacks") is not None:
        return "redteam"
    if payload.get("suite") is not None or payload.get("result_kinds") is not None:
        return "suite"
    if payload.get("evaluations") is not None or payload.get("evals") is not None:
        return "evals"
    return "simulate"


def _project_verdict(
    payload: Mapping[str, Any], summary: Mapping[str, Any]
) -> str | None:
    """Echo the run's own verdict — never recompute or reinterpret it
    (ARCH §1.4: the ledger records the verdict it is handed)."""

    for candidate in (payload.get("verdict"), summary.get("verdict")):
        if isinstance(candidate, str) and candidate in VERDICTS:
            return candidate
    status = payload.get("status")
    if status == "passed":
        return "pass"
    if status == "failed":
        return "fail"
    return None


def _project_scores(summary: Mapping[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            scores[str(key)] = round(float(value), _FLOAT_PRECISION)
    return scores


def _project_gate_outcomes(payload: Mapping[str, Any]) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    declared = payload.get("gate_outcomes")
    if isinstance(declared, Mapping):
        for key, value in declared.items():
            outcomes[str(key)] = bool(value)
        return outcomes
    checks = payload.get("checks")
    if isinstance(checks, (list, tuple)):
        for check in checks:
            if isinstance(check, Mapping) and check.get("id") is not None:
                outcomes[str(check["id"])] = bool(
                    check.get("passed", check.get("status") == "passed")
                )
    return outcomes


def _manifest_address(payload: Mapping[str, Any]) -> str | None:
    manifest = payload.get("manifest")
    if isinstance(manifest, Mapping) and manifest:
        data = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()
    address = payload.get("manifest_address")
    return str(address) if isinstance(address, str) and address else None


def _asset_refs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    declared = payload.get("asset_refs")
    if isinstance(declared, (list, tuple)):
        for item in declared:
            if not isinstance(item, Mapping):
                continue
            address = item.get("content_address") or item.get("content_hash")
            if not address:
                continue
            ref: dict[str, Any] = {
                "kind": str(item.get("kind") or "asset"),
                "content_address": str(address),
            }
            if item.get("account_object_id"):
                ref["account_object_id"] = str(item["account_object_id"])
            refs.append(ref)
    for plural, singular in (("personas", "persona"), ("scenarios", "scenario")):
        for item in payload.get(plural) or []:
            if not isinstance(item, Mapping):
                continue
            address = (
                item.get("content_address")
                or item.get("content_hash")
                or item.get("version")
            )
            if not address:
                continue
            ref = {"kind": singular, "content_address": str(address)}
            if item.get("account_object_id"):
                ref["account_object_id"] = str(item["account_object_id"])
            refs.append(ref)
    return refs


def _trace_ids(payload: Mapping[str, Any]) -> list[str]:
    declared = payload.get("trace_ids")
    if isinstance(declared, (list, tuple)):
        return [str(item) for item in declared if item]
    return []


def _content_bearing(
    payload: Mapping[str, Any], capture: Mapping[str, Any]
) -> bool:
    """True iff the row references captured content — transcripts/prompts/
    tool I/O (ARCH §2a); the sync content gate keys off it."""

    if capture:
        return True
    if payload.get("transcripts"):
        return True
    declared = payload.get("asset_refs")
    if isinstance(declared, (list, tuple)):
        for item in declared:
            if isinstance(item, Mapping) and item.get("kind") == "transcript":
                return True
    return False


def _redaction_contract(capture: Mapping[str, Any]) -> dict[str, Any] | None:
    """The capture+redaction mapping for content-bearing rows: env NAMES +
    strategy — names always, values never. ``None`` on metadata-only rows."""

    redaction = capture.get("redaction")
    if isinstance(redaction, Mapping) and redaction:
        return {str(name): str(strategy) for name, strategy in redaction.items()}
    return None
