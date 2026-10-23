"""Isolate chat model sessions and generated tool handlers from other leased worlds."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .call_runner import ArtifactUploader, CallRunnerContext
from .hosted_scheduler import CallAborted, CallOutcome, Scenario, World
from .isolated_process import run_json_worker, run_worker
from .job import FailureDomain
from .outbound import ArtifactKind
from .process_runtime import EnvironmentRuntime, _allowlisted_ambient_env
from .usage import UsageDenied, UsageUnavailable, simulator_funding
from .world.runtime import Call

logger = logging.getLogger(__name__)

# Model configuration only: no platform callback capability or source-fetch credential.
_CHAT_ENV = frozenset(
    {
        # Platform-owned model gateway credentials are simulator configuration,
        # not target-agent or callback credentials.  The isolated worker needs
        # them when the Claude SDK wire protocol is routed to a non-Claude model.
        "AGENTCC_API_KEY",
        "AGENTCC_BASE_URL",
        "ALK_CLAUDE_GATEWAY_API_KEY",
        "ALK_CLAUDE_GATEWAY_URL",
        "ALK_HARNESS",
        "ALK_HARNESS_MODEL",
        "ALK_HARNESS_THINKING",
        "ALK_VERTEX_LOCATION",
        "ALK_CHAT_TARGET_TIMEOUT_SECONDS",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "CLOUD_ML_REGION",
        "ALK_STAGE_IDLE_TIMEOUT_SECONDS",
        "ALK_STAGE_IDLE_RETRIES",
        # Not a credential: it tells the worker's usage records who pays for the simulator.
        "ALK_SIMULATOR_FUNDING",
    }
)


class IsolatedChatCallRunner:
    def __init__(self, adapter: ArtifactUploader, context: CallRunnerContext) -> None:
        self._adapter = adapter
        self._context = context

    async def run(
        self,
        scenario: Scenario,
        runtime: EnvironmentRuntime,
        *,
        world: World | None = None,
    ) -> CallOutcome:
        context = self._context
        reporter = context.usage_reporter
        # The worker holds no platform callback capability, so the parent authorizes the
        # call here and journals the usage the worker measured once it returns.
        if reporter is not None and simulator_funding() == "platform":
            try:
                await asyncio.to_thread(reporter.check, "text_call")
            except UsageDenied as exc:
                raise CallAborted(
                    f"text_usage_check_denied: {exc}", code="usage_exhausted"
                ) from exc
            except UsageUnavailable as exc:
                raise CallAborted(
                    f"text_usage_check_failed: {exc}", code="usage_check_failed"
                ) from exc
        environ = _allowlisted_ambient_env(dict(os.environ))
        environ.update(
            {key: value for key, value in os.environ.items() if key in _CHAT_ENV}
        )
        environ.update(
            {
                key: value
                for key, value in context.simulator_provider_secret_values.items()
                if key in _CHAT_ENV
            }
        )
        result = await run_json_worker(
            "fi.alk.harness.chat_worker",
            {
                "job": context.job.model_dump(mode="json"),
                "bundle_dir": str(context.bundle_dir),
                "work_directory": str(context.work_directory),
                "source_directory": str(context.source_directory)
                if context.source_directory
                else None,
                "target_secrets": dict(context.target_provider_secret_values),
                "attempt_number": context.attempt_number,
                "runtime": runtime.model_dump(mode="json"),
                "scenario_key": scenario.scenario_key,
                "source_scenario_key": getattr(scenario, "source_scenario_key", None),
                "scenario_id": scenario.scenario_id,
            },
            environ=environ,
            work_directory=context.work_directory / "chat-workers",
        )
        await _replay_usage(reporter, result.get("usage", ()))
        if "error" in result:
            raise CallAborted(
                f"chat_worker_failed: {result['error']}",
                marker=result.get("error_marker"),
                code=result.get("error_code") or "call_failed",
            )
        artifact_ids = {}
        for artifact in result["artifacts"]:
            artifact_ids[artifact["id"]] = await self._adapter.upload_artifact(
                base64.b64decode(artifact["data"], validate=True),
                kind=ArtifactKind(artifact["kind"]),
                scenario_key=scenario.scenario_key,
            )
        outcome = result["outcome"]
        outcome["calls"] = tuple(Call(**call) for call in outcome["calls"])
        outcome["recording_artifacts"] = ()
        outcome["transcript_artifact"] = artifact_ids.get(
            outcome["transcript_artifact"]
        )
        return CallOutcome(**outcome)


async def _replay_usage(reporter: Any, records: Iterable[dict[str, Any]]) -> None:
    if reporter is None:
        return
    for fields in records:
        domain = fields["failure_domain"]
        try:
            await asyncio.to_thread(
                reporter.record,
                **{
                    **fields,
                    "occurred_at": datetime.fromisoformat(fields["occurred_at"]),
                    "failure_domain": FailureDomain(domain) if domain else None,
                },
            )
        except Exception:  # a metering failure must not hide the call outcome
            logger.exception(
                "Could not record hosted text usage for scenario %s",
                fields.get("scenario_key"),
            )


class _UsageCollector:
    """The worker's usage reporter: it keeps what the call measured for the parent."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        *,
        occurred_at: datetime,
        failure_domain: FailureDomain | None,
        **fields: Any,
    ) -> None:
        self.records.append(
            {
                **fields,
                "occurred_at": occurred_at.isoformat(),
                "failure_domain": failure_domain.value if failure_domain else None,
            }
        )


class _ArtifactCollector:
    def __init__(self) -> None:
        self.artifacts: list[dict[str, Any]] = []

    async def upload_artifact(
        self, data: bytes, *, kind: ArtifactKind, **_kwargs: Any
    ) -> str:
        identifier = str(len(self.artifacts))
        self.artifacts.append(
            {
                "id": identifier,
                "kind": kind.value,
                "data": base64.b64encode(data).decode("ascii"),
            }
        )
        return identifier


async def main() -> None:
    from .chat_call_runner import HostedChatCallRunner
    from .hosted_agent import CONNECTOR as HOSTED_AGENT
    from .hosted_agent import HostedAgentCallRunner
    from .job import HarnessJob
    from .retell_chat_call_runner import RetellChatCallRunner

    payload = json.load(sys.stdin)
    collector = _ArtifactCollector()
    usage = _UsageCollector()
    try:
        job = HarnessJob.model_validate(payload["job"])
        context = CallRunnerContext(
            job=job,
            bundle_dir=Path(payload["bundle_dir"]),
            work_directory=Path(payload["work_directory"]),
            evidence_seam=None,
            target_provider_secret_values=payload["target_secrets"],
            attempt_number=payload["attempt_number"],
            source_directory=Path(payload["source_directory"])
            if payload["source_directory"]
            else None,
            usage_reporter=usage,
        )
        runner_type = {
            "retell_chat": RetellChatCallRunner,
            HOSTED_AGENT: HostedAgentCallRunner,
        }.get(job.agent.connector, HostedChatCallRunner)
        outcome = await runner_type(collector, context).run(
            SimpleNamespace(
                scenario_key=payload["scenario_key"],
                source_scenario_key=payload.get("source_scenario_key"),
                scenario_id=payload["scenario_id"],
            ),
            EnvironmentRuntime.model_validate(payload["runtime"]),
        )
        result = {"outcome": asdict(outcome), "artifacts": collector.artifacts}
    except CallAborted as exc:
        # CallAborted messages and structured failure fields are already designed for
        # result receipts. Preserve them across the worker boundary; returning only the
        # exception class made every chat failure indistinguishable and discarded useful
        # retry/attribution codes. Arbitrary exceptions remain type-only below.
        result = {
            "error": str(exc),
            "error_code": exc.code,
            "error_marker": exc.marker,
        }
    except Exception as exc:  # noqa: BLE001 - return a secret-safe worker failure
        result = {"error": type(exc).__name__}
    result["usage"] = usage.records
    fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(result, stream, default=str)


if __name__ == "__main__":
    run_worker(main)
