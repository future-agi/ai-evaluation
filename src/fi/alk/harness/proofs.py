"""Each scenario's latest proof, saved beside the suite so the platform can record it."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .prove import Proof

PROOFS = "proofs.json"


def load_proofs(destination: Path) -> dict[str, dict[str, Any]]:
    path = Path(destination) / PROOFS
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _save(destination: Path, proofs: dict[str, dict[str, Any]]) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=destination, prefix=".proofs-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(proofs, out, indent=2, sort_keys=True, ensure_ascii=False)
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination / PROOFS)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def record_proof(destination: Path, scenario_key: str, proof: Proof) -> None:
    """Save the three gates for one scenario, replacing its earlier proof."""
    proofs = load_proofs(destination)
    proofs[scenario_key] = {
        "holds": proof.holds,
        "ready": proof.ready,
        "solvable": proof.solvable,
        "not_vacuous": not proof.vacuous,
        "why": "" if proof.holds else proof.why(),
        "proved_at": _now(),
    }
    _save(destination, proofs)


def record_unprovable(destination: Path, scenario_key: str, why: str) -> None:
    """Save that a scenario could not be proved at all."""
    proofs = load_proofs(destination)
    proofs[scenario_key] = {
        "holds": False,
        "ready": False,
        "solvable": False,
        "not_vacuous": False,
        "why": why,
        "proved_at": _now(),
    }
    _save(destination, proofs)
