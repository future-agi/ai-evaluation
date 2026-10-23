"""Private chat ingress and workflow readiness checks."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from fi._diagnostics import _SECRET_FIELD, sanitize_diagnostic
from fi.simulate.agent.wrapper import AgentInput

from .chat_transport import create_chat_wrapper
from .diagnostics import HarnessDiagnostic
from .job import HarnessStage


def _failure_code(trace):
    if trace.get("timeout_error") is True:
        return "chat_ingress_timeout"
    status = trace.get("status_code", 0)
    stage = trace.get("failure_stage")
    if stage in {"setup", "http"} and status in {400, 404, 405, 415, 422}:
        return "chat_ingress_configuration_invalid"
    if status in {401, 403}:
        return "credential_missing"
    if (
        status in {408, 429, 502, 503, 504}
        or trace.get("network_error")
        or (stage == "tool_events" and 500 <= status < 600)
    ):
        return "external_service_unavailable"
    if stage == "tool_events":
        return "chat_tool_evidence_unverified"
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
    document=None,
    opening_message=None,
    authoring: Path,
    candidate_hash,
    bundle_digest,
    secret_values,
    redaction_values,
    tools=(),
    scenario_id=None,
    scenario_key=None,
    timeout=120.0,
    local_auth_values=None,
):
    from .authoring_runtime_validation import RuntimeValidationError

    redaction_values = {**(redaction_values or {}), **(local_auth_values or {})}
    attempt = uuid4().hex
    key = scenario_key or (document or {}).get("scenario_key") or attempt
    persona = (document or {}).get("persona") or {}
    opening = opening_message or (persona.get("initial_message") if isinstance(persona, dict) else None)
    message = {
        "role": "user",
        "content": opening if isinstance(opening, str) and opening.strip() else "Hello",
    }
    receipt = {
        "attempt": attempt,
        "candidate_hash": candidate_hash,
        "bundle_digest": bundle_digest,
        "status": "failed",
        "timeout_seconds": timeout,
    }
    if document is not None or scenario_key is not None:
        receipt["scenario_key"] = key
    try:
        wrapper = create_chat_wrapper(
            interface,
            endpoint,
            timeout=timeout,
            total_timeout=timeout,
            secret_values=secret_values,
            local_auth_values=local_auth_values,
        )
        response = await wrapper.call(
            AgentInput(
                thread_id=key,
                execution_id=scenario_id or key,
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
                "timeout_error",
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
            if code == "chat_ingress_timeout":
                detail = f"Chat ingress request timed out with a {timeout:g}s deadline"
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


class _PrivateWorkflowArtifacts:
    def __init__(self, folder, redactions):
        self.folder = folder
        self.redactions = redactions
        self.trace = {}
        self.transcript = ""
        self.paths = []

    async def upload_artifact(self, data, *, kind, scenario_key=None, deadline=None):
        from .outbound import ArtifactKind

        def scrub(value):
            if isinstance(value, dict):
                return {
                    scrub(key): "<redacted>" if _SECRET_FIELD.search(key) else scrub(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [scrub(item) for item in value]
            text = value if isinstance(value, str) else json.dumps(value)
            redacted = sanitize_diagnostic(text, self.redactions, limit=max(8192, len(data) * 10))
            if not isinstance(value, str):
                try:
                    return json.loads(redacted)
                except ValueError:
                    pass
            return redacted

        # Redact parsed values so replacement quotes and credential literals cannot break JSON.
        if kind == ArtifactKind.TOOL_TRACE:
            text = "\n".join(json.dumps(scrub(json.loads(line))) for line in data.splitlines() if line.strip())
        else:
            text = json.dumps(scrub(json.loads(data)))
        path = self.folder / f"{kind.value}-{uuid4().hex}.json"
        path.write_text(text, encoding="utf-8")
        path.chmod(0o600)
        self.paths.append(path.name)
        if kind == ArtifactKind.TRACE:
            self.trace = json.loads(data)
        elif kind == ArtifactKind.TRANSCRIPT:
            self.transcript = text
        return str(path)


async def probe_chat_workflow(
    *,
    job,
    source: Path,
    bundle: Path,
    authoring: Path,
    runtime,
    scenario,
    candidate_hash,
    bundle_digest,
    secret_values,
    redaction_values,
    timeout=300.0,
):
    """Exercise a private conversation and require observed tool execution evidence."""
    from .authoring_runtime_validation import RuntimeValidationError
    from .call_runner import CallRunnerContext
    from .chat_call_runner import HostedChatCallRunner
    from .hosted_scheduler import CallAborted

    redactions = {
        **(secret_values or {}),
        **(redaction_values or {}),
        **getattr(runtime, "_local_auth_values", {}),
    }
    attempt = uuid4().hex
    folder = authoring / "generic-harness" / "chat-readiness" / attempt
    folder.mkdir(parents=True, mode=0o700)
    artifacts = _PrivateWorkflowArtifacts(folder, redactions)
    receipt = {
        "attempt": attempt,
        "candidate_hash": candidate_hash,
        "bundle_digest": bundle_digest,
        "scenario_key": scenario.scenario_key,
        "status": "failed",
        "timeout_seconds": timeout,
        "tool_call_count": 0,
        "artifacts": artifacts.paths,
    }

    def failure(code, message):
        detail = sanitize_diagnostic(str(message), redactions)[:4000]
        if artifacts.transcript:
            detail += "\n<untrusted_transcript>\n" + artifacts.transcript[:6000] + "\n</untrusted_transcript>"
        diagnostic = HarnessDiagnostic.create(
            stage=HarnessStage.CONNECTING_AGENT,
            component="chat_ingress",
            code=code,
            message=detail,
        )
        receipt["diagnostic"] = diagnostic.model_dump(mode="json")
        return RuntimeValidationError(
            "environment", diagnostic.redacted_message, diagnostics=(diagnostic,)
        )

    async def retain_interrupted_conversation(exc):
        from dataclasses import asdict

        from .call_runner import _collect_file_tool_calls
        from .outbound import ArtifactKind

        transcript = getattr(exc, "partial_transcript", None)
        if transcript is None:
            return
        if not transcript.calls:
            transcript.calls.extend(_collect_file_tool_calls(runtime))
        receipt["tool_call_count"] = len(transcript.calls)
        receipt["turns"] = len(transcript.exchanges)
        await artifacts.upload_artifact(transcript.artifact(), kind=ArtifactKind.TRANSCRIPT)
        if transcript.calls:
            trace = "\n".join(
                json.dumps(asdict(call), sort_keys=True, default=str)
                for call in transcript.calls
            ).encode("utf-8")
            await artifacts.upload_artifact(trace, kind=ArtifactKind.TOOL_TRACE)

    try:
        runner = HostedChatCallRunner(
            artifacts,
            CallRunnerContext(
                job=job,
                bundle_dir=bundle,
                work_directory=folder,
                evidence_seam=None,
                target_provider_secret_values=secret_values,
                attempt_number=1,
                source_directory=source,
            ),
        )
        async with asyncio.timeout(timeout):
            outcome = await runner.run(scenario, runtime)
        receipt["tool_call_count"] = len(outcome.calls)
        receipt["turns"] = outcome.turns
        from .contract import has_executable_tool_workflow

        requires_tool_evidence = getattr(job, "require_tool_workflow", False) or (
            has_executable_tool_workflow(runner._contract)
        )
        if not outcome.calls and requires_tool_evidence:
            raise failure(
                "chat_tool_evidence_unverified",
                "Required tool evidence was not captured. Inspect source-supported "
                "observation mapping. This may be target noncompliance; do not "
                "fabricate evidence or change business logic.",
            )
        if not outcome.calls:
            receipt["tool_evidence_status"] = "not_observed"
        receipt["status"] = "passed"
    except RuntimeValidationError:
        raise
    except TimeoutError as exc:
        await retain_interrupted_conversation(exc.__cause__)
        raise failure(
            "chat_ingress_timeout", f"Private chat workflow exceeded its {timeout:g}s deadline"
        ) from exc
    except CallAborted as exc:
        if exc.partial is not None:
            receipt["tool_call_count"] = len(exc.partial.calls)
            receipt["turns"] = exc.partial.turns
        raise failure(_failure_code(artifacts.trace), str(exc)) from exc
    except asyncio.CancelledError as exc:
        receipt["status"] = "cancelled"
        await retain_interrupted_conversation(exc)
        raise
    except Exception as exc:
        raise failure("chat_ingress_failed", str(exc)) from exc
    finally:
        path = folder / "receipt.json"
        path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        path.chmod(0o600)
