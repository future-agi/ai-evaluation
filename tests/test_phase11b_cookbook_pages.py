"""Phase 11B — the 19 framework cookbook pages (18 new + a2a EDITED)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fi.alk import trinity

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/frameworks"

FRAMEWORKS = (
    "a2a",
    "agno",
    "beeai",
    "claude_agent_sdk",
    "google_adk",
    "instructor",
    "smolagents",
    "bedrock",
    "cerebras",
    "cohere",
    "deepseek",
    "fireworks",
    "huggingface",
    "litellm",
    "ollama",
    "portkey",
    "strands",
    "together",
    "xai",
)

KEYED = {
    "bedrock",
    "cerebras",
    "cohere",
    "deepseek",
    "fireworks",
    "litellm",
    "portkey",
    "together",
    "xai",
    "instructor",
    "huggingface",
    "strands",
}


def _meta(framework: str) -> dict:
    return trinity._parse_docs_frontmatter(
        (DOCS / f"{framework}.md").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_each_framework_page_frontmatter(framework):
    meta = _meta(framework)
    assert meta is not None
    assert meta["kind"] == trinity.V1_DOCS_PAGE_METADATA_KIND
    assert meta["track"] == "frameworks"
    assert meta["objective"] in trinity.V1_DOCS_OBJECTIVE_AXIS
    assert meta["stage"] in trinity.V1_DOCS_STAGE_AXIS
    for kind in meta.get("artifact_kinds", []):
        assert kind in trinity.V1_DOCS_ALLOWED_ARTIFACT_KINDS


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_each_page_backing_is_its_cert_shim(framework):
    meta = _meta(framework)
    assert meta["backing"] == [
        f"examples/sdk_framework_adapter_cert_{framework}.py"
    ]
    assert (
        trinity.V1_DOCS_BACKING_COVERAGE[
            f"examples/sdk_framework_adapter_cert_{framework}.py"
        ]
        == "framework_adapter_preset_certification_readiness"
    )


def test_a2a_page_crosslinks():
    page = (DOCS / "a2a.md").read_text(encoding="utf-8")
    assert "a2a_lane.py" in page
    assert "sdk_framework_adapter_a2a_protocol_trace.py" in page
    assert _meta("a2a")["backing"] == [
        "examples/sdk_framework_adapter_cert_a2a.py"
    ]


def test_bedrock_page_reconciliation_note():
    page = (DOCS / "bedrock.md").read_text(encoding="utf-8")
    assert "not in traceai" in page.lower()
    assert "inventory artifact" in page.lower()


@pytest.mark.parametrize("framework", sorted(KEYED))
def test_keyed_pages_carry_live_recipe(framework):
    page = (DOCS / f"{framework}.md").read_text(encoding="utf-8")
    assert "--live" in page
    lane = {
        row["framework"]: row
        for row in trinity.V1_FRAMEWORK_PRESET_LIVE_VALIDATION_LANE
    }
    assert lane[framework]["env_var"] in page


def test_ollama_page_optional_daemon_note():
    page = (DOCS / "ollama.md").read_text(encoding="utf-8")
    assert "daemon" in page.lower()
    assert "--live" not in page  # not a ◐ row (11B-A9)


def test_certification_phrase_licensed():
    """Any certification-wording hit must be declared in claims with the gate id."""
    import re

    pattern = (
        r"\b(?:certified[- ]preset|preset[- ]certification|"
        r"first[- ]class[- ]adapter)\b"
    )
    assert (
        trinity.V1_DOCS_CLAIM_PHRASE_GATES[
            r"\b(?:certified[- ]preset|preset[- ]certification|first[- ]class[- ]adapter)\b"
        ]
        == "framework_adapter_preset_certification_readiness"
    )
    for framework in FRAMEWORKS:
        text = (DOCS / f"{framework}.md").read_text(encoding="utf-8")
        body = text[text.find("\n---\n", 4) + len("\n---\n"):]
        meta = _meta(framework)
        declared = {
            (c.get("phrase"), c.get("gate_id"))
            for c in (meta.get("claims") or [])
        }
        for match in re.finditer(pattern, body, re.IGNORECASE):
            phrase = match.group(0)
            assert (
                phrase,
                "framework_adapter_preset_certification_readiness",
            ) in declared, f"{framework}: undeclared phrase {phrase!r}"
