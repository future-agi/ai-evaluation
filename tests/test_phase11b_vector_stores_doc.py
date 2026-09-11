"""Phase 11B — the single vector-stores retrieval-hook cookbook page (11B-3)."""

from __future__ import annotations

from pathlib import Path

from fi.alk import trinity

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs/frameworks/vector-stores.md"

VENDORS = (
    "chromadb",
    "lancedb",
    "milvus",
    "mongodb-vector",
    "pgvector",
    "pinecone",
    "qdrant",
    "redis-vector",
    "weaviate",
)


def test_vector_stores_page_frontmatter():
    meta = trinity._parse_docs_frontmatter(PAGE.read_text(encoding="utf-8"))
    assert meta is not None
    assert meta["track"] == "frameworks"
    assert meta["backing"] == ["examples/sdk_retrieval_hook_optimization.py"]
    assert meta.get("claims", []) == []


def test_vector_stores_page_lists_nine_vendors():
    body = PAGE.read_text(encoding="utf-8")
    for vendor in VENDORS:
        assert vendor in body, f"{vendor} missing from vector-stores page"


def test_nine_vector_dbs_absent_from_presets():
    from fi.simulate.agent.frameworks import FRAMEWORK_PRESETS

    for vendor in VENDORS:
        assert vendor not in FRAMEWORK_PRESETS
        assert vendor.replace("-", "_") not in FRAMEWORK_PRESETS


def test_vector_stores_backing_covered_by_retrieval_hook():
    assert (
        trinity.V1_DOCS_BACKING_COVERAGE[
            "examples/sdk_retrieval_hook_optimization.py"
        ]
        == "retrieval_hook_readiness"
    )
