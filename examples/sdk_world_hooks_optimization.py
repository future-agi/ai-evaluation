from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, optimize


REQUIRED_ENV = "AGENT_LEARNING_SDK_WORLD_HOOKS_KEY"


def build_manifest() -> dict[str, Any]:
    return optimize.build_world_hooks_optimization_manifest(
        name="sdk-world-hooks-optimization",
        required_env=[REQUIRED_ENV],
        target_metadata={
            "cookbook": "sdk-world-hooks-optimization",
            "research_sources": [
                {
                    "id": "2606.05558",
                    "source": "arxiv:2606.05558",
                    "url": "https://arxiv.org/abs/2606.05558",
                    "used_for": "offline agent evaluation with world-model rollouts",
                },
                {
                    "id": "2606.03892",
                    "source": "arxiv:2606.03892",
                    "url": "https://arxiv.org/abs/2606.03892",
                    "used_for": "verified stateful execution environments",
                },
                {
                    "id": "2606.02372",
                    "source": "arxiv:2606.02372",
                    "url": "https://arxiv.org/abs/2606.02372",
                    "used_for": "closed-loop world-model and policy co-evolution",
                },
                {
                    "id": "2605.30880",
                    "source": "arxiv:2605.30880",
                    "url": "https://arxiv.org/abs/2605.30880",
                    "used_for": "executable inspectable world models",
                },
            ],
            "original_synthesis": (
                "World hooks should be native executable state hooks, not "
                "HTTP adapters: AgentOptimizer searches complete in-process "
                "world candidates whose transitions, contracts, adversarial "
                "pressure, memory provenance, and replay evidence all close."
            ),
        },
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    result = optimize.optimize_world_hooks(
        name="sdk-world-hooks-optimization",
        required_env=[REQUIRED_ENV],
        target_metadata={
            "cookbook": "sdk-world-hooks-optimization",
            "research_sources": [
                {
                    "id": "2606.05558",
                    "source": "arxiv:2606.05558",
                    "url": "https://arxiv.org/abs/2606.05558",
                },
                {
                    "id": "2606.03892",
                    "source": "arxiv:2606.03892",
                    "url": "https://arxiv.org/abs/2606.03892",
                },
                {
                    "id": "2606.02372",
                    "source": "arxiv:2606.02372",
                    "url": "https://arxiv.org/abs/2606.02372",
                },
                {
                    "id": "2605.30880",
                    "source": "arxiv:2605.30880",
                    "url": "https://arxiv.org/abs/2605.30880",
                },
            ],
            "original_synthesis": (
                "World hooks should be native executable state hooks, not "
                "HTTP adapters: AgentOptimizer searches complete in-process "
                "world candidates whose transitions, contracts, adversarial "
                "pressure, memory provenance, and replay evidence all close."
            ),
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
