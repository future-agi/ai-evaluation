"""Run hosted ALK authoring inside the Daytona guest without exposing job secrets.

Generation credentials are materialized only into this process environment.  Target-agent values
remain in ``secrets.json`` for the later process-runtime boundary.  Vertex regions are deliberately
split: Gemini's ``GOOGLE_CLOUD_LOCATION`` is agent/runtime configuration, while Claude on Vertex
uses ``CLOUD_ML_REGION`` (default ``us-east5``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .authoring_entrypoint import main as authoring_main

_SECRETS_PATH = Path("/run/futureagi/secrets.json")
_ADC_PATH = Path("/work/.authoring-credentials/google.json")
_PASSTHROUGH = {
    # Not a credential, and the only non-credential here on purpose: the switch that decides
    # whether mailbox scenarios may be written has to reach the process that writes them, and this
    # is the environment authoring runs in.
    "ALK_VOICEMAIL_SCENARIOS",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "ANTHROPIC_VERTEX_REGION",
    "CLOUD_ML_REGION",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "OPENAI_API_KEY",
}


def _load_values(path: Path) -> dict[str, str]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError("hosted secrets document must be an object")
    return {
        str(name): str(value)
        for name, value in body.items()
        if isinstance(name, str) and isinstance(value, str) and value
    }


def _platform_simulator_values(values: dict[str, str]) -> dict[str, str]:
    """Canonicalize only platform-owned aliases; never use customer target values."""
    prefix = "SIMULATOR_"
    return {
        name.removeprefix(prefix): value
        for name, value in values.items()
        if name.startswith(prefix)
    }


def _configure_generation_environment(values: dict[str, str]) -> None:
    for name in _PASSTHROUGH:
        if values.get(name):
            os.environ[name] = values[name]

    adc_json = values.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if adc_json:
        parsed = json.loads(adc_json)
        if not isinstance(parsed, dict):
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS_JSON must contain an object"
            )
        _ADC_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ADC_PATH.write_text(
            json.dumps(parsed, separators=(",", ":")), encoding="utf-8"
        )
        _ADC_PATH.chmod(0o600)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_ADC_PATH)

    project = values.get("ANTHROPIC_VERTEX_PROJECT_ID") or values.get(
        "GOOGLE_CLOUD_PROJECT"
    )
    if project and (adc_json or values.get("GOOGLE_APPLICATION_CREDENTIALS")):
        os.environ.setdefault("CLAUDE_CODE_USE_VERTEX", "1")
        os.environ.setdefault("ANTHROPIC_VERTEX_PROJECT_ID", project)
        # Never derive this from GOOGLE_CLOUD_LOCATION: that is Gemini/agent configuration and
        # commonly us-central1, where Claude Sonnet is not servable.
        os.environ.setdefault(
            "CLOUD_ML_REGION",
            values.get("ANTHROPIC_VERTEX_REGION")
            or values.get("CLOUD_ML_REGION")
            or "us-east5",
        )


def main(argv: list[str] | None = None) -> int:
    values = _platform_simulator_values(_load_values(_SECRETS_PATH))
    _configure_generation_environment(values)
    try:
        return authoring_main(argv)
    finally:
        try:
            _ADC_PATH.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
