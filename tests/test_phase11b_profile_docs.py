"""Phase 11B — the profile-doc track (5 IO profiles + index, 11B-4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fi.alk import trinity

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "docs/frameworks/profiles"

PROFILE_BACKING = {
    "streaming": "examples/sdk_framework_adapter_streaming.py",
    "typed_output": "examples/sdk_framework_adapter_typed_output.py",
    "nested_method": "examples/sdk_framework_adapter_nested_method.py",
    "message_history": "examples/sdk_framework_adapter_message_history.py",
    "handoff_transcript": "examples/sdk_framework_adapter_handoff_transcript.py",
}

# §7.3 split: the gate that covers each profile's backing example.
BACKING_GATE = {
    "streaming": "framework_adapter_io_readiness",
    "typed_output": "framework_adapter_io_readiness",
    "nested_method": "framework_adapter_io_readiness",
    "message_history": "framework_adapter_probe_readiness",
    "handoff_transcript": "framework_adapter_probe_readiness",
}

IO_SURFACES = {
    contract["surface"] for contract in trinity.V1_FRAMEWORK_ADAPTER_IO_CONTRACTS
}


def _meta(name: str) -> dict:
    return trinity._parse_docs_frontmatter(
        (PROFILES / f"{name}.md").read_text(encoding="utf-8")
    )


def test_five_profile_pages_present():
    for name in PROFILE_BACKING:
        assert (PROFILES / f"{name}.md").is_file()
    assert (PROFILES / "index.md").is_file()


@pytest.mark.parametrize("name", sorted(PROFILE_BACKING))
def test_profile_page_frontmatter(name):
    meta = _meta(name)
    assert meta["track"] == "frameworks"
    assert meta["objective"] == "capability"
    assert meta["stage"] == "simulate"
    assert meta["backing"] == [PROFILE_BACKING[name]]


@pytest.mark.parametrize("name", sorted(PROFILE_BACKING))
def test_profile_backing_coverage_split(name):
    # Each profile page is admitted via its backing example, which is mapped to
    # the gate that covers it (the §7.3 split).
    assert (
        trinity.V1_DOCS_BACKING_COVERAGE[PROFILE_BACKING[name]]
        == BACKING_GATE[name]
    )


def test_profile_index_crosslinks_five():
    meta = _meta("index")
    assert meta["backing"] == []
    body = (PROFILES / "index.md").read_text(encoding="utf-8")
    for name in PROFILE_BACKING:
        assert f"{name}.md" in body


@pytest.mark.parametrize("name", sorted(PROFILE_BACKING))
def test_profiles_reference_existing_io_contract(name):
    assert name in IO_SURFACES
