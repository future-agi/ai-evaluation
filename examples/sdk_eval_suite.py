from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fi.alk import configure, evals, suite


REQUIRED_ENV = "AGENT_LEARNING_SDK_EVAL_SUITE_KEY"


def build_manifest() -> dict[str, Any]:
    return evals.build_eval_suite_manifest(
        name="sdk-local-eval-suite",
        providers=[{"id": "echo", "type": "echo"}],
        prompts=[
            {
                "id": "support-policy-question",
                "template": "{{question}}",
            }
        ],
        tests=[
            {
                "id": "policy-grounding",
                "vars": {"question": "Where is the refund policy?"},
                "assert": [
                    {"type": "contains", "value": "refund policy"},
                    {"type": "not_contains", "value": "private credentials"},
                ],
            }
        ],
        threshold=1.0,
        metadata={"cookbook": "sdk-eval-suite"},
    )


def build_suite_wrapper(eval_path: str | Path) -> dict[str, Any]:
    return suite.build_suite_manifest(
        name="sdk-eval-suite-wrapper",
        jobs=[
            {
                "id": "sdk-eval-suite",
                "command": "eval",
                "path": Path(eval_path).name,
                "name": "sdk-local-eval-suite",
            }
        ],
        required_capabilities={
            "commands": ["eval"],
            "result_kinds": ["agent-learning.eval.v1"],
        },
        metadata={"cookbook": "sdk-eval-suite"},
    )


def run(output_path: str | Path | None = None) -> dict[str, Any]:
    api_key = os.environ.get(REQUIRED_ENV)
    if not api_key:
        raise RuntimeError(f"Set {REQUIRED_ENV} before running this example.")
    configure(api_key=api_key)

    result_path = Path(output_path).expanduser() if output_path else None
    manifest_path = (
        result_path.with_suffix(".manifest.json")
        if result_path
        else Path(__file__).with_suffix(".json")
    )
    wrapper_path = (
        result_path.with_suffix(".suite.json")
        if result_path
        else Path(__file__).with_name("sdk_eval_suite_wrapper.json")
    )

    evals.write_eval_suite_file(build_manifest(), manifest_path)
    suite.write_suite_file(build_suite_wrapper(manifest_path), wrapper_path)
    result = evals.run_eval_suite_file(manifest_path)

    if result_path is not None:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return result


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    payload = run(destination)
    if destination is None:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
