"""Consolidated promotion for the keyword-inputs certification family.

Covers the agentic frameworks whose preset entrypoint is a keyword ``run``
(agno, beeai, google_adk). Each promotion offers >=2 adapter candidates — the
preset default (``run``/``dict``) versus a deliberately weak alternative
(``respond``/``text``) — and certifies the optimizer selects the preset shape
and builds a runnable manifest with ``require_no_external_service: True``.

Credential-free: local shims only, no real framework import, no network.
Per ARCH §3 / BBG §2.4 the probe shims stay one-per-framework while the
promotions consolidate by IO-surface family; the gate row carries the
framework key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize


FAMILY = "keyword_inputs"
FRAMEWORKS = ("agno", "beeai", "google_adk")
TARGET = f"{Path(__file__).resolve()}:LocalKeywordRunner"


class LocalKeywordRunner:
    """Local keyword-run shim with a weak and a preset (``run``) entrypoint."""

    def respond(self, text: str) -> dict[str, Any]:
        assert text
        return {
            "content": "Weak adapter response without keyword input or tools.",
            "tool_calls": [],
            "metadata": {"framework_conformance": "incomplete"},
        }

    def run(self, *, inputs: dict[str, Any] | None = None, **params: Any) -> dict[str, Any]:
        return {
            "content": "Keyword adapter approved refund with run runtime evidence.",
            "tool_calls": [
                {
                    "id": "kw_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed", "input_key": "inputs"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "keyword.run",
                    "payload": {"input_key": "inputs"},
                }
            ],
        }


def _cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "kw-refund",
            "input": "Approve the refund and emit adapter evidence.",
            "expected_contains": ["approved refund"],
            "required_tools": ["framework_trace_status"],
            "required_events": ["framework_trace"],
            "required_state_keys": ["framework_runtime"],
        }
    ]


def _evaluation_config(framework: str) -> dict[str, Any]:
    return {
        "task_description": (
            f"Promote the certified {framework} keyword-run adapter into a "
            "runnable simulation manifest."
        ),
        "expected_result": (
            "The selected run adapter emits framework_trace_status evidence."
        ),
        "required_tools": ["framework_trace_status"],
        "available_tools": ["framework_trace_status"],
        "framework_adapter_contract_quality": {
            "kind": "agent-learning.framework-adapter-contract.v1",
            "framework": framework,
            "method": "run",
            "input_mode": "dict",
            "require_no_external_service": True,
        },
    }


def build_promotion(framework: str) -> dict[str, Any]:
    return optimize.optimize_framework_adapter_probe(
        name=f"cert-{framework}-promotion",
        framework=framework,
        target=TARGET,
        agent_factory=LocalKeywordRunner,
        adapter_candidates=[
            {"method": "respond", "input_mode": "text"},
            {"method": "run", "input_mode": "dict"},
        ],
        cases=_cases(),
        metadata={"certification": "11B", "io_surface": FAMILY, "framework": framework},
    )


def build_manifest(framework: str, optimization: dict[str, Any]) -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_probe_optimization(
        optimization,
        name=f"cert-{framework}-run",
        evaluation_config=_evaluation_config(framework),
        metadata={"certification": "11B", "io_surface": FAMILY, "framework": framework},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    results: dict[str, Any] = {"family": FAMILY, "frameworks": {}}
    for framework in FRAMEWORKS:
        optimization = build_promotion(framework)
        manifest = build_manifest(framework, optimization)
        selected = (optimization.get("optimization") or {}).get("best_config") or {}
        results["frameworks"][framework] = {
            "selected_adapter": selected.get("adapter") or {},
            "manifest_agent": manifest.get("agent") or {},
        }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


if __name__ == "__main__":
    destination = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts") / "sdk-framework-adapter-cert-keyword-inputs-promotion.json"
    )
    run(destination)
