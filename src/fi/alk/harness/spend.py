"""What the harness itself spent, written where a deleted sandbox cannot take it with it.

Every model call the harness makes arrives at one place, `Stage`'s handling of `StageDone`, so the
ledger is fed from there rather than from each stage's own code: a writer added later is counted
without anybody remembering to count it. Parallel scenario writers and the suite review each open
their own session, which is why per-call-site accounting would have missed most of a large run's
spend.

The file is rewritten after every turn so the newest total is always on disk. A run that dies
mid-turn loses that turn only, and the platform reads the file while the sandbox is alive.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

JOURNAL_ALIAS = "ALK_SPEND_JOURNAL"

_stages: dict[str, dict[str, Any]] = {}
_path: Path | None = None


def journal_to(path: str | os.PathLike[str] | None) -> None:
    """Where to keep the ledger. Nothing is written until this is set or the alias names a path."""
    global _path
    _path = Path(path) if path else None
    if _path is not None:
        _flush()


def _destination() -> Path | None:
    if _path is not None:
        return _path
    named = os.environ.get(JOURNAL_ALIAS, "").strip()
    return Path(named) if named else None


def record(
    stage: str,
    usd: float | None,
    turns: int = 0,
    models: set[str] | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> None:
    """Add one session's reported spend. A backend that cannot price a call reports None."""
    name = (stage or "stage").strip() or "stage"
    entry = _stages.setdefault(
        name,
        {
            "usd": 0.0,
            "turns": 0,
            "models": [],
            "priced": 0,
            "unpriced": 0,
            "tokens_in": 0,
            "tokens_out": 0,
        },
    )
    if usd is None:
        entry["unpriced"] += 1
    else:
        entry["usd"] = round(entry["usd"] + float(usd), 6)
        entry["priced"] += 1
    entry["turns"] += int(turns or 0)
    entry["tokens_in"] += int(tokens_in or 0)
    entry["tokens_out"] += int(tokens_out or 0)
    for model in sorted(models or set()):
        if model not in entry["models"]:
            entry["models"].append(model)
    _flush()


def total_usd() -> float:
    return round(sum(float(entry["usd"]) for entry in _stages.values()), 6)


def unpriced_turns() -> int:
    """Turns whose backend reported no price. Nonzero means the total is a floor, not the answer."""
    return sum(int(entry["unpriced"]) for entry in _stages.values())


def snapshot() -> dict[str, Any]:
    return {
        "schema": "futureagi.harness-spend.v1",
        "total_usd": total_usd(),
        "unpriced_turns": unpriced_turns(),
        "stages": [
            {
                "stage": name,
                **{
                    key: entry[key]
                    for key in (
                        "usd",
                        "turns",
                        "models",
                        "priced",
                        "unpriced",
                        "tokens_in",
                        "tokens_out",
                    )
                },
            }
            for name, entry in sorted(_stages.items())
        ],
    }


def _flush() -> None:
    destination = _destination()
    if destination is None:
        return
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Written whole, then moved: a poll that reads mid-write must never see half a total.
        handle = tempfile.NamedTemporaryFile(
            "w", dir=destination.parent, prefix=".spend-", suffix=".json", delete=False
        )
        with handle as writing:
            json.dump(snapshot(), writing, indent=2, sort_keys=True)
        os.replace(handle.name, destination)
    except OSError:
        # Accounting must never be the reason a stage fails.
        pass
