"""Consolidated promotion for the side_kwargs certification family.

Covers a2a, whose preset entrypoint (``send_message``) carries a message kwarg plus a session side-kwarg.

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


FAMILY = "side_kwargs"
FRAMEWORKS = ('a2a',)
TARGET = f"{Path(__file__).resolve()}:LocalSideKwargsShim"

PRESET_METHODS = {'a2a': ('send_message', 'dict')}
WEAK_METHOD = ('respond', 'text')



class LocalSideKwargsShim:
    """Local side_kwargs shim with a weak and a preset entrypoint."""

    def respond(self, text: str) -> dict[str, Any]:
        assert text
        return {
            "content": "Weak adapter response without tool evidence.",
            "tool_calls": [],
            "metadata": {"framework_conformance": "incomplete"},
        }

    def send_message(self, *, message=None, session=None, **params: Any) -> dict[str, Any]:
        return {
            "content": "send_message adapter approved refund with runtime evidence.",
            "tool_calls": [
                {
                    "id": "cert_status",
                    "name": "framework_trace_status",
                    "arguments": {"status": "passed"},
                }
            ],
            "events": [
                {
                    "type": "framework_trace",
                    "name": "send_message",
                    "payload": {"framework_conformance": "complete"},
                }
            ],
        }


def _cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "side_kwargs-refund",
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
            f"Promote the certified {framework} side_kwargs adapter into a "
            "runnable simulation manifest."
        ),
        "expected_result": (
            "The selected adapter emits framework_trace_status evidence."
        ),
        "required_tools": ["framework_trace_status"],
        "available_tools": ["framework_trace_status"],
        "framework_adapter_contract_quality": {
            "kind": "agent-learning.framework-adapter-contract.v1",
            "framework": framework,
            "method": PRESET_METHODS[framework][0],
            "input_mode": PRESET_METHODS[framework][1],
            "require_no_external_service": True,
        },
    }


def build_promotion(framework: str) -> dict[str, Any]:
    preset_method, preset_mode = PRESET_METHODS[framework]
    return optimize.optimize_framework_adapter_probe(
        name=f"cert-{framework}-promotion",
        framework=framework,
        target=TARGET,
        agent_factory=LocalSideKwargsShim,
        adapter_candidates=[
            {"method": WEAK_METHOD[0], "input_mode": WEAK_METHOD[1]},
            {"method": preset_method, "input_mode": preset_mode},
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
        else Path("artifacts") / "sdk-framework-adapter-cert-side-kwargs-promotion.json"
    )
    run(destination)
