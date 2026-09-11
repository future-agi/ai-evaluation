"""Mirror script coverage (Phase 2B): collect, write, drift-check."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _mirror_module():
    spec = importlib.util.spec_from_file_location(
        "mirror_cookbooks", PROJECT_ROOT / "scripts" / "mirror_cookbooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture_tree(root: Path) -> None:
    page = root / "docs" / "redteam" / "first.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\n"
        "kind: agent-learning.docs-page.v1\n"
        "track: redteam\n"
        "backing:\n"
        "  - examples/backing_module.py\n"
        "---\n# First\n",
        encoding="utf-8",
    )
    backing = root / "examples" / "backing_module.py"
    backing.parent.mkdir(parents=True)
    backing.write_text("def run(path):\n    return None\n", encoding="utf-8")
    (root / "docs" / "llms.txt").write_text("# index\n", encoding="utf-8")


def test_collect_resolves_pages_index_and_backing(tmp_path):
    mirror = _mirror_module()
    _write_fixture_tree(tmp_path)
    files = mirror.collect_mirror_set(tmp_path)
    relatives = {str(path.relative_to(tmp_path)) for path in files}
    assert relatives == {
        "docs/redteam/first.md",
        "docs/llms.txt",
        "examples/backing_module.py",
    }


def test_write_mirror_produces_hash_only_manifest(tmp_path):
    mirror = _mirror_module()
    source = tmp_path / "source"
    source.mkdir()
    _write_fixture_tree(source)
    dest = tmp_path / "dest"
    hashes = mirror.write_mirror(source, dest, mirror.collect_mirror_set(source))
    manifest = json.loads((dest / "MIRROR_MANIFEST.json").read_text())
    assert set(manifest) == {"files"}
    assert manifest["files"] == dict(sorted(hashes.items()))
    assert "generated_at" not in json.dumps(manifest)
    assert (dest / "docs" / "redteam" / "first.md").is_file()
    assert (dest / "README.md").is_file()


def test_check_mirror_flags_tampered_file(tmp_path):
    mirror = _mirror_module()
    source = tmp_path / "source"
    source.mkdir()
    _write_fixture_tree(source)
    dest = tmp_path / "dest"
    files = mirror.collect_mirror_set(source)
    mirror.write_mirror(source, dest, files)
    assert mirror.check_mirror(source, dest, files) == []
    (dest / "docs" / "redteam" / "first.md").write_text("tampered", encoding="utf-8")
    findings = mirror.check_mirror(source, dest, files)
    assert any("drift" in finding or "mismatch" in finding for finding in findings)
