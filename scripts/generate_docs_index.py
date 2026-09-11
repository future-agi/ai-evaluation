#!/usr/bin/env python3
"""Regenerate docs/llms.txt from page frontmatter.

The docs_executability release gate regenerates the same content in memory and
byte-compares it against the committed file — run this script after any docs
page change; never hand-edit docs/llms.txt.

Usage: uv run python scripts/generate_docs_index.py [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fi.alk import trinity  # noqa: E402


def build_page_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in trinity._docs_page_paths(root):
        text = path.read_text(encoding="utf-8")
        metadata = trinity._parse_docs_frontmatter(text)
        if metadata is None:
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "title": trinity._docs_page_title(text),
                "track": metadata.get("track"),
                "backing": [str(item) for item in metadata.get("backing") or []],
                "artifact_kinds": [
                    str(item) for item in metadata.get("artifact_kinds") or []
                ],
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the committed index differs from the regeneration.",
    )
    args = parser.parse_args(argv)

    rendered = trinity._render_docs_machine_index(build_page_records(ROOT))
    index_path = ROOT / trinity.V1_DOCS_MACHINE_INDEX_FILE
    if args.check:
        committed = (
            index_path.read_text(encoding="utf-8")
            if index_path.is_file()
            else None
        )
        if committed != rendered:
            print(f"STALE: {index_path} differs from regeneration", file=sys.stderr)
            return 1
        print("index up to date")
        return 0
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(rendered, encoding="utf-8")
    print(f"wrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
