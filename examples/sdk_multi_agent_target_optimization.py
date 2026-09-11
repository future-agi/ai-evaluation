from __future__ import annotations

import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_MULTI_AGENT_TARGET_OPTIMIZATION_KEY"
TARGET_PATH = "simulation.environments.0.data.participants"


def _multi_agent_example() -> Any:
    example_path = Path(__file__).with_name("sdk_multi_agent_optimization.py")
    spec = importlib.util.spec_from_file_location(
        "sdk_multi_agent_optimization",
        example_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _participant_candidates() -> tuple[dict[str, Any], dict[str, Any]]:
    participants = deepcopy(_multi_agent_example().participants())
    missing_critic = {
        role: participant
        for role, participant in participants.items()
        if role != "critic"
    }
    return missing_critic, participants


def _base_config() -> dict[str, Any]:
    multi_agent_example = _multi_agent_example()
    missing_critic, _ = _participant_candidates()
    room = deepcopy(multi_agent_example.strong_room())
    room["participants"] = missing_critic
    return {
        "agent": multi_agent_example.strong_agent(),
        "simulation": {
            "engine": "local_text",
            "min_turns": 3,
            "max_turns": 3,
            "auto_execute_tools": True,
            "environments": [{"type": "multi_agent_room", "data": room}],
        },
    }


def _evaluation_config() -> dict[str, Any]:
    return _multi_agent_example().evaluation_config()


def _target_candidates() -> dict[str, list[dict[str, Any]]]:
    missing_critic, participants = _participant_candidates()
    return {TARGET_PATH: [missing_critic, participants]}


def build_manifest() -> dict[str, Any]:
    return optimize.build_target_optimization_manifest(
        name="sdk-multi-agent-target-optimization",
        required_env=[REQUIRED_ENV],
        base_config=_base_config(),
        evaluation_config=_evaluation_config(),
        target_candidates=_target_candidates(),
        layers=["multi_agent", "orchestration", "harness", "evaluator"],
        min_turns=3,
        max_turns=3,
        threshold=0.98,
        target_metadata={
            "cookbook": "sdk-multi-agent-target-optimization",
            "optimized_surface": "multi_agent_room_participants",
        },
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    result = optimize.optimize_target(
        name="sdk-multi-agent-target-optimization",
        required_env=[REQUIRED_ENV],
        base_config=_base_config(),
        evaluation_config=_evaluation_config(),
        target_candidates=_target_candidates(),
        layers=["multi_agent", "orchestration", "harness", "evaluator"],
        min_turns=3,
        max_turns=3,
        threshold=0.98,
        target_metadata={
            "cookbook": "sdk-multi-agent-target-optimization",
            "optimized_surface": "multi_agent_room_participants",
        },
        manifest_path=Path(__file__).with_suffix(".json"),
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
