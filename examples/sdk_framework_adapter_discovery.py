from __future__ import annotations

import json
from pathlib import Path

from fi.alk import simulate


class LocalRefundOrchestrator:
    """Local framework shim with multiple plausible adapter methods."""

    def run(self, text):
        return f"Weak text-only adapter for {text}"

    async def execute_task(self, payload):
        return {
            "content": "Adapter discovery selected the structured refund task path.",
            "metadata": {"framework": payload["metadata"]["framework"]},
        }


def run(output_path: str | Path) -> dict:
    result = simulate.discover_framework_adapter(
        "custom_refund_orchestrator",
        LocalRefundOrchestrator(),
        target="sdk_framework_adapter_discovery.py:LocalRefundOrchestrator",
        method_candidates=["run", "execute_task"],
        input_mode_candidates=["text", "dict", "agent_input"],
        metadata={"cookbook": "sdk-framework-adapter-discovery"},
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run(Path("artifacts") / "sdk-framework-adapter-discovery.json")
