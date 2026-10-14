"""One disposable chat turn proves ingress, without harness tool replay or judging."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from fi._diagnostics import sanitize_diagnostic
from fi.simulate.agent.wrapper import AgentInput

from .chat_transport import create_chat_wrapper
from .diagnostics import HarnessDiagnostic
from .job import HarnessStage


def _failure_code(trace):
    status = trace.get("status_code", 0)
    stage = trace.get("failure_stage")
    if stage in {"setup", "http"} and status in {400, 404, 405, 415, 422}:
        return "chat_ingress_configuration_invalid"
    if status in {401, 403}:
        return "credential_missing"
    if status in {408, 429} or trace.get("network_error"):
        return "external_service_unavailable"
    return "chat_ingress_failed"


def _usable_tool_call(call):
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = call.get("name") or function.get("name")
    arguments = call.get("arguments", function.get("arguments", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return False
    return isinstance(name, str) and bool(name.strip()) and isinstance(arguments, dict)


async def probe_chat_ingress(
    *,
    interface,
    endpoint,
    document,
    authoring: Path,
    candidate_hash,
    bundle_digest,
    secret_values,
    redaction_values,
    tools=(),
    scenario_id=None,
    scenario_key=None,
    timeout=120.0,
):
    from .authoring_runtime_validation import RuntimeValidationError

    attempt = uuid4().hex
    key = scenario_key or document["scenario_key"]
    persona = document.get("persona") or {}
    opening = persona.get("initial_message") if isinstance(persona, dict) else None
    message = {
        "role": "user",
        "content": opening if isinstance(opening, str) and opening.strip() else "Hello",
    }
    receipt = {
        "attempt": attempt,
        "candidate_hash": candidate_hash,
        "bundle_digest": bundle_digest,
        "scenario_key": key,
        "status": "failed",
    }
    try:
        wrapper = create_chat_wrapper(
            interface,
            endpoint,
            timeout=timeout,
            total_timeout=timeout,
            secret_values=secret_values,
        )
        response = await wrapper.call(
            AgentInput(
                thread_id=key,
                execution_id=scenario_id,
                modality="text",
                tools=list(tools),
                scenario_name=key,
                turn_index=0,
                messages=[message],
                new_message=message,
            )
        )
        trace = (response.metadata or {}).get("external_agent", {})
        receipt["trace"] = {
            name: trace.get(name)
            for name in (
                "status_code",
                "failure_stage",
                "latency_ms",
                "error",
                "response_excerpt",
            )
        }
        valid_tools = all(_usable_tool_call(call) for call in response.tool_calls or [])
        if (
            not trace.get("success")
            or not valid_tools
            or not (response.content.strip() or response.tool_calls)
        ):
            code = (
                _failure_code(trace)
                if not trace.get("success")
                else "chat_ingress_failed"
            )
            detail = trace.get("error") or (
                "Chat returned malformed tool calls"
                if not valid_tools
                else "Chat returned no text or tool calls"
            )
            excerpt = trace.get("response_excerpt")
            if excerpt:
                detail = (
                    f"{detail}\n<untrusted_response>\n{excerpt}\n</untrusted_response>"
                )
            diagnostic = HarnessDiagnostic.create(
                stage=HarnessStage.CONNECTING_AGENT,
                component="chat_ingress",
                code=code,
                message=sanitize_diagnostic(str(detail), redaction_values)[:4000],
            )
            receipt["diagnostic"] = diagnostic.model_dump(mode="json")
            raise RuntimeValidationError(
                "environment", diagnostic.redacted_message, diagnostics=(diagnostic,)
            )
        receipt["status"] = "passed"
    except RuntimeValidationError:
        raise
    except Exception as exc:
        diagnostic = HarnessDiagnostic.create(
            stage=HarnessStage.CONNECTING_AGENT,
            component="chat_ingress",
            code="chat_ingress_failed",
            message=sanitize_diagnostic(str(exc), redaction_values)[:4000],
        )
        receipt["diagnostic"] = diagnostic.model_dump(mode="json")
        raise RuntimeValidationError(
            "environment", diagnostic.redacted_message, diagnostics=(diagnostic,)
        ) from exc
    except asyncio.CancelledError:
        receipt["status"] = "cancelled"
        raise
    finally:
        folder = authoring / "generic-harness" / "chat-readiness"
        folder.mkdir(parents=True, exist_ok=True)

        # Redact evidence before bounding; controller identities must remain exact.
        if "trace" in receipt:
            for name in ("error", "response_excerpt"):
                if isinstance(receipt["trace"].get(name), str):
                    receipt["trace"][name] = sanitize_diagnostic(
                        receipt["trace"][name], redaction_values
                    )[:4000]
        (folder / f"{attempt}.json").write_text(json.dumps(receipt, indent=2) + "\n")
