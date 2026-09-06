"""Which recorded mailbox greeting, if any, a voicemail scenario should be heard through.

A greeting is the one part of a mailbox that can carry a name, and the name belongs to the
scenario's persona, so a recorded personal greeting would be wrong for every scenario but one. The
clips this reads are chosen for saying "it's me" rather than a name, which is what makes a recording
usable at all: it fits whoever the persona happens to be.

The catalogue is a local file and is deliberately not committed, so everything here degrades to
nothing when it is absent: no catalogue means the mailbox speaks its greeting the way it did before
and the tone is generated.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CATALOG = Path(__file__).with_name("data") / "voicemail" / "catalog.json"
# A run may point somewhere else, which is how a deployment serves these from object storage
# instead of from the tree.
CATALOG_ENV = "ALK_VOICEMAIL_CATALOG"


def _entries() -> list[dict[str, Any]]:
    path = Path(os.environ.get(CATALOG_ENV, "").strip() or CATALOG)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no catalogue is the ordinary case, never an error
        return []
    return [one for one in body if isinstance(one, dict)] if isinstance(body, list) else []


def _resolved(entry: dict[str, Any]) -> str:
    """Where the audio actually is: a URL if the catalogue serves one, else a file beside us."""
    url = str(entry.get("url") or "").strip()
    if url:
        return url
    raw = str(entry.get("path") or "").strip()
    if not raw:
        return ""
    here = Path(raw)
    if here.is_file():
        return str(here)
    # Catalogue paths are written relative to the repository root, so fall back to the file's own
    # name beside this module, which is where it lives once the package is installed.
    beside = CATALOG.parent / Path(str(entry.get("file_name") or here.name))
    return str(beside) if beside.is_file() else ""


def clip_for(style: str, language: str = "") -> dict[str, Any] | None:
    """A clip whose style and language match, or None where the catalogue has nothing to offer.

    Language matters as much as style: a scenario whose mailbox greets in Hindi cannot be served an
    English recording, and no clip is the right answer there because the session speaks the greeting
    itself in the language the scenario asked for.

    Deterministic rather than random: the first matching entry wins, so two runs of the same suite
    are heard through the same mailbox and a difference between them is never the audio.
    """
    wanted = str(style or "").strip().lower()
    if not wanted:
        return None
    spoken = (str(language or "").strip().lower() or "en").split("-")[0]
    for entry in _entries():
        if str(entry.get("style") or "").strip().lower() != wanted:
            continue
        if str(entry.get("language") or "en").strip().lower().split("-")[0] != spoken:
            continue
        source = _resolved(entry)
        if not source:
            continue
        return {
            "source": source,
            # A clip that ends with its own tone must not be given a second one.
            "has_tone": bool(entry.get("has_tone")),
            "id": str(entry.get("id") or ""),
            # The greeting must reach the transcript, or the call reads as the agent talking to nobody.
            "transcript": str(entry.get("transcript") or "").strip(),
        }
    return None
