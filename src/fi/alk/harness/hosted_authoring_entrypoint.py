"""Run hosted ALK authoring inside the Daytona guest without exposing job secrets.

Generation credentials are materialized only into this process environment.  Target-agent values
remain in ``secrets.json`` for the later process-runtime boundary.  Vertex regions are deliberately
split: Gemini's ``GOOGLE_CLOUD_LOCATION`` is agent/runtime configuration, while Claude on Vertex
uses ``CLOUD_ML_REGION`` (default ``us-east5``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import observability
from .authoring_entrypoint import main as authoring_main

_SECRETS_PATH = Path("/run/futureagi/secrets.json")
_ADC_PATH = Path("/work/.authoring-credentials/google.json")
_TARGET_SECRETS_PATH = Path("/run/futureagi/authoring-target-secrets.json")
_SIMULATOR_SECRETS_PATH = Path("/run/futureagi/simulator-secrets.json")
_TARGET_CREDENTIAL_NAMES = {"RETELL_API_KEY", "VAPI_API_KEY"}
_PASSTHROUGH = {
    "AGENTCC_API_KEY",
    "AGENTCC_BASE_URL",
    # Not a credential: authoring writes the scenarios, so the switch has to reach it.
    "ALK_VOICEMAIL_SCENARIOS",
    "ALK_BACKGROUND_NOISE",
    "ALK_BACKGROUND_NOISE_CATALOG",
    "ALK_CLAUDE_GATEWAY_URL",
    "ALK_CLAUDE_GATEWAY_API_KEY",
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


# Unprefixed, so read from the raw channel rather than the SIMULATOR_ view.
_OBSERVABILITY_PASSTHROUGH = {
    "FI_API_KEY",
    "FI_BASE_URL",
    "FI_HARNESS_PROJECT",
    "FI_SECRET_KEY",
    "HARNESS_OBSERVABILITY",
}


def _configure_observability_environment(all_values: dict[str, str]) -> None:
    # Read without consuming: the run process loads and deletes this file, and needs it intact.
    values = dict(all_values)
    try:
        body = json.loads(_SIMULATOR_SECRETS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        body = {}
    if isinstance(body, dict):
        values.update({str(k): str(v) for k, v in body.items() if v not in (None, "")})
    for name in _OBSERVABILITY_PASSTHROUGH:
        if values.get(name):
            os.environ[name] = values[name]


def _authoring_job_context(forwarded: list[str]) -> tuple[str, str, dict]:
    """The ids authoring reports under. It is a separate process and is handed none of its own."""
    for candidate in forwarded:
        if candidate.startswith("-"):
            continue
        try:
            body = json.loads(Path(candidate).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(body, dict) or "job_id" not in body:
            continue
        telemetry = (body.get("metadata") or {}).get("telemetry") or {}
        return str(body.get("job_id") or ""), str(body.get("run_id") or ""), telemetry
    return "", "", {}


def main(argv: list[str] | None = None) -> int:
    # This process authors the environment but never runs the submitted target. Do not let a
    # launcher-provided parent environment accidentally bypass the target-secrets file boundary.
    for name in _TARGET_CREDENTIAL_NAMES:
        os.environ.pop(name, None)
    all_values = _load_values(_SECRETS_PATH)
    values = _platform_simulator_values(all_values)
    # The launch-time fallback authoring command runs before hosted_entrypoint consumes this
    # control-plane channel. Read only the allowlisted platform-owned authoring values and leave
    # the file for hosted_entrypoint to consume and delete. Never copy customer target credentials
    # here.
    try:
        gateway_values = _load_values(_SIMULATOR_SECRETS_PATH)
    except (OSError, ValueError):
        gateway_values = {}
    values.update(
        {
            name: gateway_values[name]
            for name in (
                "AGENTCC_API_KEY",
                "AGENTCC_BASE_URL",
                "ALK_BACKGROUND_NOISE",
                "ALK_BACKGROUND_NOISE_CATALOG",
            )
            if gateway_values.get(name)
        }
    )
    _configure_generation_environment(values)
    _configure_observability_environment(all_values)
    target_values = {
        name: all_values[name]
        for name in _TARGET_CREDENTIAL_NAMES
        if all_values.get(name)
    }
    forwarded = list(argv) if argv is not None else sys.argv[1:]
    if target_values and "--target-secrets" not in forwarded:
        _TARGET_SECRETS_PATH.write_text(
            json.dumps(target_values, separators=(",", ":")), encoding="utf-8"
        )
        _TARGET_SECRETS_PATH.chmod(0o600)
        forwarded.extend(["--target-secrets", str(_TARGET_SECRETS_PATH)])
    if "--conversation-capabilities" in forwarded:
        os.environ["ALK_FOREGROUND_COORDINATOR"] = "1"
    job_id, run_id, telemetry = _authoring_job_context(forwarded)
    if job_id:
        observability.begin(job_id, run_id, telemetry)
        observability.stage("authoring")
    try:
        from .authoring_runtime_validation import RuntimeValidationError
        from .outbound import redact_outbound_text

        try:
            return authoring_main(forwarded, validate_runtime=True)
        except RuntimeValidationError as exc:
            print(
                "RuntimeValidationError: " + redact_outbound_text(str(exc)),
                file=sys.stderr,
            )
            # EX_CONFIG: deterministic generated-environment failure, not retryable infra.
            return 78
    finally:
        observability.end()
        try:
            _ADC_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            _TARGET_SECRETS_PATH.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
