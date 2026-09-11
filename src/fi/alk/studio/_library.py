"""Content-addressed persona/scenario library (Phase 7, ARCH §2d).

One library root (default ``.agent-learning/library/``), no DB, no service:

    personas/<slug>/<sha256-hex>.json     # library content; filename IS the hash
    scenarios/<slug>/<sha256-hex>.json
    calibrations/<persona-hash>.json      # agent-learning.persona-calibration.v1
    coverage/<report-timestamp>.json      # raw coverage data (index blocks)
    quarantine/                           # refused pulls — never loadable
    index.json                            # agent-learning.persona-library.v1

``save_persona`` refuses to overwrite a hash-named file with different bytes
(content addressing makes tampering loud); ``load_persona`` re-hashes and
rejects mismatches. Runtime fidelity floors live in the index as data, seeded
from the engine floor table (trinity constants pin the same values at gate
time).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from fi.simulate.simulation.fidelity import PERSONA_FIDELITY_FLOORS
from fi.simulate.simulation.models import Persona, Scenario

from ._upgrade import upgrade_legacy_persona

DEFAULT_LIBRARY_ROOT = ".agent-learning/library"
PERSONA_LIBRARY_KIND = "agent-learning.persona-library.v1"
PERSONA_CALIBRATION_KIND = "agent-learning.persona-calibration.v1"

_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return slug or "unnamed"


def library_root(library: Union[str, Path, None] = None) -> Path:
    return Path(library) if library is not None else Path(DEFAULT_LIBRARY_ROOT)


def ensure_library(library: Union[str, Path, None] = None) -> Path:
    root = library_root(library)
    for sub in ("personas", "scenarios", "calibrations", "coverage", "quarantine"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    if not index_path.exists():
        _write_index(root, {
            "kind": PERSONA_LIBRARY_KIND,
            "personas": [],
            "scenarios": [],
            # Runtime per-class floors are library-index DATA seeded from the
            # engine constants (ARCH §2c); legacy is omitted on purpose.
            "floors": {k: dict(v) for k, v in PERSONA_FIDELITY_FLOORS.items()},
            "bias_lint": None,
            "pull_receipts": [],
            "representativeness_claim": "none",
        })
    return root


def load_index(library: Union[str, Path, None] = None) -> Dict[str, Any]:
    root = ensure_library(library)
    with open(root / "index.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_index(root: Path, index: Mapping[str, Any]) -> None:
    payload = json.dumps(index, indent=2, sort_keys=True, default=str)
    (root / "index.json").write_text(payload + "\n", encoding="utf-8")


def save_index(library: Union[str, Path, None], index: Mapping[str, Any]) -> None:
    _write_index(ensure_library(library), index)


def _hash_hex(model: Union[Persona, Scenario]) -> str:
    return model.content_hash().split(":", 1)[1]


def _dump_model(model: Union[Persona, Scenario]) -> str:
    payload = model.model_dump(exclude_none=True)
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def _refuse_quarantined(path: Path) -> None:
    if "quarantine" in path.parts:
        raise ValueError(
            f"refusing to load quarantined content: {path} "
            "(quarantined artifacts are never loadable)"
        )


def _write_content_addressed(directory: Path, hex_digest: str, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{hex_digest}.json"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise ValueError(
                f"content-address collision: {path} already exists with "
                "different bytes; refusing to overwrite (tampering is loud)"
            )
        return path
    path.write_text(content, encoding="utf-8")
    return path


def _persona_slug(persona: Persona, slug: Optional[str]) -> str:
    if slug:
        return _slugify(slug)
    name = None
    if persona.identity is not None and persona.identity.name:
        name = persona.identity.name
    if not name:
        name = persona.persona.get("name") if isinstance(persona.persona, dict) else None
    return _slugify(name or "persona")


def _upsert_entry(entries: List[Dict[str, Any]], entry: Dict[str, Any]) -> None:
    for index, existing in enumerate(entries):
        if existing.get("content_digest") == entry["content_digest"]:
            entries[index] = {**existing, **entry}
            return
    entries.append(entry)


def save_persona(
    persona: Persona,
    *,
    library: Union[str, Path, None] = None,
    slug: Optional[str] = None,
    admit: bool = False,
    lint_result: Optional[Mapping[str, Any]] = None,
    calibration_ref: Optional[str] = None,
) -> Dict[str, Any]:
    """Write a persona as content-addressed library content.

    ``admit=True`` is the library-admission gate: it requires a calibrated
    persona AND a current green set-level bias lint (ARCH §2f) — and refuses
    loudly otherwise. Demographics-bearing personas can never be admitted
    without the lint stamp (P7-D4)."""
    root = ensure_library(library)
    if admit:
        provenance = persona.provenance
        if provenance is None or not provenance.calibrated:
            raise ValueError(
                "admit refused: persona is not calibrated "
                "(run calibrate_persona first; uncalibrated personas still "
                "run at the lowest evidence class)"
            )
        if lint_result is None or lint_result.get("status") != "passed":
            raise ValueError(
                "admit refused: no current green set-level bias lint for the "
                "receiving library (run bias_lint and pass the result)"
            )
    hex_digest = _hash_hex(persona)
    slug_value = _persona_slug(persona, slug)
    path = _write_content_addressed(root / "personas" / slug_value, hex_digest, _dump_model(persona))
    evidence_class = (
        persona.provenance.evidence_class if persona.provenance is not None else "legacy"
    )
    stage = "admitted" if admit else (
        "interrogated" if persona.provenance is not None and persona.provenance.calibrated
        else "sampled"
    )
    entry: Dict[str, Any] = {
        "ref": str(path.relative_to(root)),
        "slug": slug_value,
        "content_digest": f"sha256:{hex_digest}",
        "evidence_class": evidence_class,
        "calibration_stage": stage,
        "calibration_ref": calibration_ref or (
            persona.provenance.calibration_ref if persona.provenance is not None else None
        ),
        "bias_lint_stamp": (
            {
                "status": lint_result.get("status"),
                "locales_linted": list(lint_result.get("locales_linted", [])),
                "stamped_at": _now(),
            }
            if lint_result is not None else None
        ),
        "locale_stamps": list(lint_result.get("locales_linted", [])) if lint_result else [],
    }
    index = load_index(root)
    _upsert_entry(index.setdefault("personas", []), entry)
    if lint_result is not None:
        index["bias_lint"] = {
            "status": lint_result.get("status"),
            "locales_linted": list(lint_result.get("locales_linted", [])),
            "stamped_at": _now(),
        }
    _write_index(root, index)
    return {
        "path": str(path),
        "ref": entry["ref"],
        "content_digest": entry["content_digest"],
        "admitted": admit,
        "evidence_class": evidence_class,
        "calibration_stage": stage,
    }


def load_persona(
    ref: Union[str, Path],
    *,
    library: Union[str, Path, None] = None,
) -> Persona:
    """Load + re-hash library content; reject mismatches and quarantine.

    Bare legacy rows auto-upgrade through the shim (provenance=legacy)."""
    path = Path(ref)
    if not path.exists():
        path = library_root(library) / ref
    _refuse_quarantined(path.resolve())
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    persona = Persona(**data)
    if _HEX_RE.match(path.stem):
        actual = _hash_hex(persona)
        if actual != path.stem:
            raise ValueError(
                f"content hash mismatch for {path}: expected sha256:{path.stem}, "
                f"found sha256:{actual} — artifact was tampered with or "
                "hand-edited; re-pull or fork it as a NEW persona"
            )
    return upgrade_legacy_persona(persona)


def save_scenario(
    scenario: Scenario,
    *,
    library: Union[str, Path, None] = None,
    slug: Optional[str] = None,
) -> Dict[str, Any]:
    root = ensure_library(library)
    hex_digest = _hash_hex(scenario)
    slug_value = _slugify(slug or scenario.name)
    path = _write_content_addressed(root / "scenarios" / slug_value, hex_digest, _dump_model(scenario))
    entry = {
        "ref": str(path.relative_to(root)),
        "slug": slug_value,
        "content_digest": f"sha256:{hex_digest}",
        "kind": scenario.kind,
        "version": scenario.version,
        "parent_version": scenario.parent_version,
    }
    index = load_index(root)
    _upsert_entry(index.setdefault("scenarios", []), entry)
    _write_index(root, index)
    return {"path": str(path), "ref": entry["ref"], "content_digest": entry["content_digest"]}


def load_scenario(
    ref: Union[str, Path],
    *,
    library: Union[str, Path, None] = None,
) -> Scenario:
    path = Path(ref)
    if not path.exists():
        path = library_root(library) / ref
    _refuse_quarantined(path.resolve())
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    scenario = Scenario(**data)
    if _HEX_RE.match(path.stem):
        actual = _hash_hex(scenario)
        if actual != path.stem:
            raise ValueError(
                f"content hash mismatch for {path}: expected sha256:{path.stem}, "
                f"found sha256:{actual} — artifact was tampered with"
            )
    return scenario


def save_calibration(
    artifact: Mapping[str, Any],
    persona_hash_hex: str,
    *,
    library: Union[str, Path, None] = None,
) -> Path:
    root = ensure_library(library)
    path = root / "calibrations" / f"{persona_hash_hex}.json"
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def quarantine_payload(
    name: str,
    payload: Any,
    findings: List[Dict[str, Any]],
    *,
    library: Union[str, Path, None] = None,
) -> Path:
    """Write a refused payload under ``quarantine/`` (never loadable)."""
    root = ensure_library(library)
    path = root / "quarantine" / f"{_slugify(name)}.scan.json"
    path.write_text(
        json.dumps(
            {
                "disposition": "quarantined",
                "payload": payload,
                "findings": findings,
                "quarantined_at": _now(),
            },
            indent=2, sort_keys=True, default=str,
        ) + "\n",
        encoding="utf-8",
    )
    return path


def record_pull_receipt(
    receipt: Mapping[str, Any],
    *,
    library: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Pull receipts are provenance entries in the library index — never a
    standalone artifact kind (ARCH §2g)."""
    root = ensure_library(library)
    index = load_index(root)
    entry = {**dict(receipt), "recorded_at": _now()}
    index.setdefault("pull_receipts", []).append(entry)
    _write_index(root, index)
    return entry


def list_library(library: Union[str, Path, None] = None) -> Dict[str, Any]:
    index = load_index(library)
    return {
        "kind": PERSONA_LIBRARY_KIND,
        "personas": list(index.get("personas", [])),
        "scenarios": list(index.get("scenarios", [])),
        "bias_lint": index.get("bias_lint"),
        "floors": index.get("floors", {}),
        "pull_receipts": list(index.get("pull_receipts", [])),
    }


__all__ = [
    "DEFAULT_LIBRARY_ROOT",
    "PERSONA_CALIBRATION_KIND",
    "PERSONA_LIBRARY_KIND",
    "ensure_library",
    "library_root",
    "list_library",
    "load_index",
    "load_persona",
    "load_scenario",
    "quarantine_payload",
    "record_pull_receipt",
    "save_calibration",
    "save_index",
    "save_persona",
    "save_scenario",
]
