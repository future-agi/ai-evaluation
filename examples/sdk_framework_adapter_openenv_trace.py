import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fi.alk import optimize, simulate


TARGET = f"{Path(__file__).resolve()}:LocalOpenEnvRunner"


class LocalOpenEnvRunner:
    """Local framework adapter that exports an OpenEnv/Gymnasium-style trace."""

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert payload
        openenv = {
            "kind": "openenv_trace",
            "name": "framework-openenv-refund",
            "runtime": "in_process",
            "transport": "local",
            "requires_external_service": False,
            "deterministic_reset": True,
            "action_space": {
                "type": "discrete",
                "actions": ["approve_refund", "probe_policy_drift"],
            },
            "observation_space": {"type": "dict"},
            "initial_observation": {
                "ticket_id": "refund-1001",
                "refund_status": "pending",
                "risk_score": 0.1,
            },
            "current_observation": {
                "ticket_id": "refund-1001",
                "refund_status": "approved",
                "policy_probe": "blocked",
            },
            "state": {
                "ticket_id": "refund-1001",
                "refund_status": "approved",
                "policy_probe_blocked": True,
            },
            "reset_info": {"seed": 7, "source": "local_fixture"},
            "last_info": {"policy_probe": "blocked"},
            "trajectory": [
                {
                    "id": "approve-refund",
                    "step_index": 1,
                    "action": {"type": "approve_refund", "amount": 42.5},
                    "observation": {
                        "ticket_id": "refund-1001",
                        "refund_status": "approved",
                    },
                    "reward": 0.7,
                    "terminated": False,
                    "truncated": False,
                    "done": False,
                    "info": {"policy": "refund_policy_doc"},
                    "metadata": {"route": "approve_refund"},
                    "state": {"refund_status": "approved"},
                },
                {
                    "id": "probe-policy-drift",
                    "step_index": 2,
                    "action": {"type": "probe_policy_drift"},
                    "observation": {
                        "ticket_id": "refund-1001",
                        "refund_status": "approved",
                        "policy_probe": "blocked",
                    },
                    "reward": 0.3,
                    "terminated": True,
                    "truncated": False,
                    "done": True,
                    "info": {"policy_probe": "blocked"},
                    "metadata": {"route": "failure_injection_probe"},
                    "state": {"policy_probe_blocked": True},
                    "failure_injected": True,
                    "failure": {
                        "id": "policy_drift",
                        "type": "adversarial_state",
                    },
                },
            ],
            "sandbox": {
                "enabled": True,
                "isolation": "process",
                "network": "disabled",
            },
            "replay": {
                "mode": "local_fixture",
                "transport": "local",
                "deterministic": True,
            },
            "failure_injections": [
                {"id": "policy_drift", "type": "adversarial_state"}
            ],
            "tool_registry": [{"name": "refund_policy_lookup"}],
            "metadata": {"source": "sdk_framework_adapter_openenv_trace"},
        }
        return {
            "content": "OpenEnv replay completed and approved refund.",
            "openenv": openenv,
        }


def build_manifest() -> dict[str, Any]:
    return optimize.build_framework_run_manifest_from_local_adapter(
        name="sdk-framework-adapter-openenv-trace-run",
        framework="openenv",
        target=TARGET,
        method_candidates=["run"],
        input_mode_candidates=["dict"],
        discovery_max_candidates=3,
        cases=[
            {
                "id": "openenv-framework-refund",
                "input": "Run the OpenEnv refund replay with policy drift probe.",
                "expected_contains": ["approved refund"],
                "required_events": ["openenv"],
                "required_state_keys": ["framework_runtime", "openenv"],
                "required_artifact_types": ["trace"],
            }
        ],
        auto_evaluation_config=True,
        metadata={"cookbook": "sdk-framework-adapter-openenv-trace"},
    )


def run(output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser()
    manifest_path = output.with_suffix(".manifest.json")
    manifest = build_manifest()
    simulate.write_manifest_file(manifest, manifest_path)
    result = asyncio.run(simulate.run_manifest_file(manifest_path))
    result["framework_adapter_openenv_trace_manifest"] = manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    output_arg = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("artifacts/sdk-framework-adapter-openenv-trace.json")
    )
    run(output_arg)
