from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_OPTIMIZER_PROFILE_MATRIX_KEY"


def _quiet_backend_logging() -> None:
    """Silence chatty third-party study logs on the deterministic gate path."""

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        pass
    logging.getLogger("optuna").setLevel(logging.WARNING)


def build_manifests() -> dict[str, dict[str, Any]]:
    """One manifest per declared 33-coordinate matrix cell (P4-D2 subset)."""

    manifests = optimize.build_optimizer_profile_matrix_manifests()
    for manifest in manifests.values():
        manifest["required_env"] = [REQUIRED_ENV]
    return manifests


def _routing_check_manifest(
    *,
    routing_table: dict[str, Any] | None,
    optimizer: dict[str, Any] | None,
) -> dict[str, Any]:
    return optimize.build_target_optimization_manifest(
        name="optimizer-profile-matrix-routing-check",
        base_config={
            "agent": {"type": "scripted", "responses": [{"content": "weak"}]},
            "simulation": {
                "engine": "local_text",
                "min_turns": 1,
                "max_turns": 1,
                "auto_execute_tools": True,
                "environments": [],
            },
        },
        target_candidates={"agent.responses.0.content": ["weak", "strong"]},
        evaluation_config={
            "task_description": "Routing engagement contract check.",
            "expected_result": "strong",
        },
        optimizer=optimizer,
        routing_table=routing_table,
        target_metadata={
            "task_kind": "prompt",
            "framework_profile": "llamaindex",
        },
    )


def build_routing_checks(routing_table: dict[str, Any]) -> dict[str, Any]:
    """Exercise the ARCH §2d default-picker engagement contract.

    Omitted optimizer consults the routing table by default; explicit choice
    always overrides (recorded, recommendation stays visible); cold start
    falls back to the static default with a warning finding — exit 0, never
    a ValueError.
    """

    default_manifest = _routing_check_manifest(
        routing_table=routing_table,
        optimizer=None,
    )
    override_manifest = _routing_check_manifest(
        routing_table=routing_table,
        optimizer={"algorithm": "agent"},
    )
    cold_start_manifest = _routing_check_manifest(
        routing_table={
            "kind": "agent-learning.optimizer-routing-table.v1",
            "rows": [],
        },
        optimizer=None,
    )
    return {
        "default": dict(
            default_manifest["optimization"]["optimizer_routing_evidence"]
        ),
        "override": dict(
            override_manifest["optimization"]["optimizer_routing_evidence"]
        ),
        "cold_start": dict(
            cold_start_manifest["optimization"]["optimizer_routing_evidence"]
        ),
    }


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)
    _quiet_backend_logging()

    manifests = build_manifests()
    result = optimize.run_optimizer_profile_matrix(manifests)
    result["required_env"] = [REQUIRED_ENV]
    result["declared_cells"] = [
        list(cell) for cell in optimize.OPTIMIZER_PROFILE_MATRIX_CELLS
    ]
    result["routing_checks"] = build_routing_checks(result["routing_table"])
    result["routing_table_matches_committed"] = (
        optimize.routing_table_matches_committed(
            result["routing_table"],
            Path(__file__).resolve().parent / "optimizer_routing_table.json",
        )
    )
    if output_path is not None:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
