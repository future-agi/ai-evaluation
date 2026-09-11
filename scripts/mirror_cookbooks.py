#!/usr/bin/env python3
"""One-way, byte-deterministic mirror: kit docs -> code/core/cookbooks.

The kit repo is the source of truth (decision P2-D1). The mirror carries the
docs tree (minus brand assets), the machine index, and every backing object
referenced by page frontmatter, plus a generated README pointing back at the
kit. MIRROR_MANIFEST.json holds content hashes only — no timestamps — so
--check is a pure hash comparison.

Usage:
  python scripts/mirror_cookbooks.py --dest ../cookbooks/agent-learning-kit [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fi.alk import trinity  # noqa: E402

MANIFEST_NAME = "MIRROR_MANIFEST.json"
MIRROR_README = """# Agent Learning Kit — cookbooks mirror

Generated one-way from the `agent-learning-kit` repository (`docs/` +
referenced `examples/`). The kit repo is canonical; edit there, then rerun
`python scripts/mirror_cookbooks.py`. Pages are admitted by the kit's
`docs_executability` release gate before they can appear here.
"""


def collect_mirror_set(root: Path) -> list[Path]:
    """Docs pages + llms.txt + every backing object referenced by frontmatter."""

    files: set[Path] = set()
    for page_path in trinity._docs_page_paths(root):
        files.add(page_path)
        metadata = trinity._parse_docs_frontmatter(
            page_path.read_text(encoding="utf-8")
        )
        if not metadata:
            continue
        for backing in metadata.get("backing") or []:
            backing_path = root / str(backing)
            if backing_path.is_file():
                files.add(backing_path)
    index_path = root / trinity.V1_DOCS_MACHINE_INDEX_FILE
    if index_path.is_file():
        files.add(index_path)
    return sorted(files)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_mirror(root: Path, dest: Path, files: list[Path]) -> dict[str, str]:
    """Copy files preserving relative layout; return {relative_path: sha256}."""

    hashes: dict[str, str] = {}
    for source in files:
        relative = source.relative_to(root)
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = _sha256(source)
    readme_path = dest / "README.md"
    readme_path.write_text(MIRROR_README, encoding="utf-8")
    hashes["README.md"] = _sha256(readme_path)
    manifest_path = dest / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps({"files": dict(sorted(hashes.items()))}, indent=2) + "\n",
        encoding="utf-8",
    )
    return hashes


def check_mirror(root: Path, dest: Path, files: list[Path]) -> list[str]:
    """Return drift findings (missing/extra/hash mismatch)."""

    findings: list[str] = []
    manifest_path = dest / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]
    recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
    expected: dict[str, str] = {}
    for source in files:
        expected[str(source.relative_to(root))] = _sha256(source)
    readme_path = dest / "README.md"
    if readme_path.is_file():
        expected["README.md"] = _sha256(readme_path)
    for relative, digest in expected.items():
        if relative not in recorded:
            findings.append(f"missing from manifest: {relative}")
        elif recorded[relative] != digest:
            findings.append(f"hash mismatch: {relative}")
        target = dest / relative
        if not target.is_file():
            findings.append(f"missing from mirror: {relative}")
        elif target.is_file() and _sha256(target) != digest:
            findings.append(f"mirror content drift: {relative}")
    for relative in recorded:
        if relative not in expected:
            findings.append(f"stale in manifest: {relative}")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    files = collect_mirror_set(ROOT)
    if not files:
        print("nothing to mirror (no docs pages found)", file=sys.stderr)
        return 1
    if args.check:
        findings = check_mirror(ROOT, args.dest, files)
        if findings:
            for finding in findings:
                print(f"DRIFT: {finding}", file=sys.stderr)
            return 1
        print(f"mirror in sync ({len(files)} files)")
        return 0
    hashes = write_mirror(ROOT, args.dest, files)
    print(f"mirrored {len(hashes)} files to {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
