"""Guard: no competitor names in the shippable kit (review BH-15 follow-up).

The kit must carry no competitor product names in source, tests, examples, or
docs (a public, shippable library). Prior scrubs missed prose comparisons like
"HUD's premise" and a "TB-style" comment; this test makes the class fail fast so
it cannot recur. Research/design that legitimately references prior art lives in
the separate internal-docs repo, not here.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Scanned trees (the shippable surfaces).
_SCAN_DIRS = ("src", "tests", "examples", "docs")
_SCAN_SUFFIXES = (".py", ".md", ".json", ".txt", ".toml", ".yaml", ".yml")

# Banned competitor terms. ``HUD`` is word-boundary + case-sensitive (avoid
# matching "should"/"include"); the benchmark names are matched loosely.
_BANNED = [
    re.compile(r"\bHUD\b"),
    re.compile(r"terminal[-\s]?bench", re.IGNORECASE),
    re.compile(r"\bTB-style\b"),
    re.compile(r"swe[-\s]?bench", re.IGNORECASE),
    re.compile(r"\bvivaria\b", re.IGNORECASE),
    re.compile(r"\binspect_ai\b"),
]

# This guard file necessarily contains the patterns; exclude it.
_SELF = Path(__file__).resolve()


def _iter_files():
    for d in _SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix not in _SCAN_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            if path.resolve() == _SELF:
                continue
            yield path


def test_no_competitor_names_in_shipped_tree() -> None:
    hits: list[str] = []
    for path in _iter_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            for pat in _BANNED:
                if pat.search(line):
                    rel = path.relative_to(ROOT)
                    hits.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not hits, "competitor names found in shippable kit:\n" + "\n".join(hits)
