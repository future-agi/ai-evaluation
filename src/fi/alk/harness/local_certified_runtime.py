"""Run the certified generic harness pipeline without a hosted provider.

This is the local counterpart of the Daytona execution lane.  It deliberately reuses Bundle V2,
ProcessRuntimeProvider, the hosted scheduler, and the real chat/voice call runners so local tests
cannot silently fall back to the legacy prompt-reconstruction path.
"""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from .authoring_runtime_validation import (
    _make_local_seed_files_readable,
    validate_and_repair,
    validate_once,
)
from .bundle_v2 import load_bundle_v2
from .call_runner import CallRunnerContext
from .hosted_entrypoint import (
    ProcessWorldFactory,
    _default_build_call_runner,
    job_secret_purposes,
    peek_target_provider_secret_values,
    peek_simulator_provider_secret_values,
)
from .hosted_scheduler import HostedScheduler, ResultReceipt, WorldPool
from .job import HarnessJob
from .process_runtime import (
    ProcessRuntimeProvider,
    default_user_resolver,
    fixed_sandbox_user_resolver,
)
from .scenario_source import load_scenarios
from .secrets import runtime_configuration_value


class LocalCertifiedRunError(RuntimeError):
    def __init__(
        self, domain: str, code: str, message: str, *, completed_scenarios: int
    ) -> None:
        self.domain = domain
        self.code = code
        self.completed_scenarios = completed_scenarios
        super().__init__(message)


def _shared_temporary_parent(fallback: Path) -> Path:
    configured = os.environ.get("ALK_SANDBOX_UPLOAD_ROOT", "").strip()
    parent = Path(configured).expanduser().resolve() if configured else fallback
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def _runtime_values(job: HarnessJob) -> dict[str, str]:
    """Resolve only aliases explicitly admitted on this job, never controller secrets."""

    values: dict[str, str] = {}
    for alias in job.agent.secret_refs:
        value = runtime_configuration_value(alias)
        if value:
            values[alias] = value
    return values


class LocalCertifiedSink:
    """Filesystem evidence sink implementing the scheduler/call-runner outbound surface."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.receipts: list[ResultReceipt] = []
        self._events = self.root / "events.jsonl"

    def _event(self, kind: str, payload: Mapping[str, Any]) -> None:
        record = {
            "type": kind,
            "wall_time": datetime.now(timezone.utc).isoformat(),
            "payload": dict(payload),
        }
        with self._events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    async def scenario_started(self, **value: Any) -> None:
        self._event("scenario_started", value)

    async def scenario_retried(self, **value: Any) -> None:
        self._event("scenario_retried", value)

    async def world_unhealthy(self, **value: Any) -> None:
        self._event("world_unhealthy", value)

    async def log(self, **value: Any) -> None:
        self._event("log", value)

    async def receipt(self, receipt: ResultReceipt) -> None:
        self.receipts.append(receipt)
        scenario = self.root / receipt.scenario_key
        scenario.mkdir(parents=True, exist_ok=True)
        (scenario / "result.json").write_text(
            json.dumps(dataclasses.asdict(receipt), indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    async def upload_artifact(
        self,
        data: bytes,
        *,
        kind: Any,
        scenario_key: str | None = None,
        deadline: float | None = None,
    ) -> str:
        del deadline
        label = str(getattr(kind, "value", kind))
        directory = self.root / (scenario_key or "run")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{label}-{hashlib.sha256(data).hexdigest()}.bin"
        path.write_bytes(data)
        return str(path)


async def check_local_runtime_readiness(job: HarnessJob, source: Path, authoring: Path) -> None:
    """Probe the disposable target using the local execution credential boundary."""
    with tempfile.TemporaryDirectory(
        prefix="alk-local-readiness-", dir=_shared_temporary_parent(authoring.parent)
    ) as raw:
        secrets = Path(raw) / "secrets.json"
        secrets.write_text(json.dumps(_runtime_values(job)), encoding="utf-8")
        secrets.chmod(0o600)

        async def local_validate(candidate_job, candidate_source, candidate_authoring):
            return await validate_once(
                candidate_job,
                candidate_source,
                candidate_authoring,
                secrets_path=secrets,
                local_runtime=True,
                readiness_only=True,
            )

        await validate_and_repair(
            job, source, authoring, validate=local_validate, readiness_only=True
        )


async def run_local_certified(
    job: HarnessJob,
    *,
    source: Path,
    authoring: Path,
) -> tuple[int, int]:
    """Validate/repair, then execute the exact certified bytes locally.

    Returns ``(exit_status, completed_scenarios)`` where status 2 means the calls ran but one or
    more behavioral checks failed, matching the existing local CLI contract.
    """

    values = _runtime_values(job)
    purposes = job_secret_purposes(job)
    temporary_parent = _shared_temporary_parent(authoring.parent)
    with tempfile.TemporaryDirectory(
        prefix=f"alk-local-certified-{job.job_id[:8]}-", dir=temporary_parent
    ) as raw:
        work = Path(raw)
        # Runtime children execute as svc-* users and must traverse this controller-owned parent.
        # Per-process trees are chowned by the provider; the credential handoff stays mode 0600.
        work.chmod(0o755)
        (work / "job.json").write_text(job.model_dump_json(), encoding="utf-8")
        secrets = work / "secrets.json"
        secrets.write_text(json.dumps(values), encoding="utf-8")
        secrets.chmod(0o600)
        target_values = peek_target_provider_secret_values(secrets, purposes)
        simulator_values = peek_simulator_provider_secret_values(secrets, purposes)

        async def local_validate(candidate_job, candidate_source, candidate_authoring):
            return await validate_once(
                candidate_job,
                candidate_source,
                candidate_authoring,
                secrets_path=secrets,
                local_runtime=True,
            )

        await validate_and_repair(
            job,
            source,
            authoring,
            validate=local_validate,
        )

        frozen = authoring / "generic-harness" / "certified-bundle"
        bundle_dir = work / "bundle"
        shutil.copytree(frozen, bundle_dir)
        _make_local_seed_files_readable(bundle_dir)
        manifest = load_bundle_v2(bundle_dir)

        runs = authoring / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        run_root = Path(
            tempfile.mkdtemp(
                prefix=datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S-"),
                dir=runs,
            )
        )
        run_id = run_root.name
        sink = LocalCertifiedSink(run_root)
        # Validation reads a private copy, so the original handoff still exists here.  The
        # execution provider consumes and unlinks it on first provision, exactly like hosted.
        provider = ProcessRuntimeProvider(
            secrets_path=secrets,
            secret_purpose_map=purposes,
            user_resolver=(
                fixed_sandbox_user_resolver
                if os.environ.get("ALK_FIXED_SANDBOX_USER") == "1"
                else default_user_resolver
            ),
            require_declared_user=False,
            generic_artifact_root=run_root / "generic-harness",
        )
        from .bundle_v2 import RuntimeKindV2
        from .world.observation import ObservationWorldFactory
        if manifest.runtime.kind is RuntimeKindV2.COMPOSE:
            from .compose_integration import provider as compose_provider
            provider = compose_provider(target_values, run_root / "compose-logs")
            world_factory = ObservationWorldFactory(bundle_dir)
        else:
            world_factory = ProcessWorldFactory(work, bundle_dir)
        pool = WorldPool(
            provider,
            bundle=manifest,
            source=source,
            bundle_dir=bundle_dir,
            work_directory=work,
            instances=job.runtime.parallelism,
            outbound=sink,
        )
        call_runner = None
        try:
            await pool.start()
            scenarios = load_scenarios(bundle_dir)
            context = CallRunnerContext(
                job=job,
                bundle_dir=bundle_dir,
                work_directory=work,
                evidence_seam=manifest.runtime.evidence_seam,
                target_provider_secret_values=target_values,
                simulator_provider_secret_values=simulator_values,
                attempt_number=1,
                source_directory=source,
            )
            call_runner = _default_build_call_runner(sink, context)
            scheduler = HostedScheduler(
                pool=pool,
                world_factory=world_factory,
                call_runner=call_runner,
                outbound=sink,
                job_seed=job.seed or 0,
                require_tool_workflow=job.require_tool_workflow,
            )
            result = await scheduler.run(scenarios)
            summary = {
                "run_id": run_id,
                "aborted": dataclasses.asdict(result.aborted)
                if result.aborted
                else None,
                "receipts": [dataclasses.asdict(item) for item in sink.receipts],
            }
            (sink.root / "run.json").write_text(
                json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
            )
            if result.aborted is not None:
                raise LocalCertifiedRunError(
                    result.aborted.domain,
                    result.aborted.code,
                    result.aborted.message,
                    completed_scenarios=len(sink.receipts),
                )
            for receipt in sink.receipts:
                if receipt.status == "errored":
                    failure = receipt.failure
                    raise LocalCertifiedRunError(
                        failure.domain if failure else "infrastructure",
                        failure.code if failure else "scenario_errored",
                        failure.message
                        if failure
                        else f"Scenario {receipt.scenario_key} errored without failure details",
                        completed_scenarios=len(sink.receipts),
                    )
            failed = any(item.status != "passed" for item in sink.receipts)
            return (2 if failed else 0), len(sink.receipts)
        finally:
            close = getattr(call_runner, "close", None)
            if callable(close):
                await close()
            await pool.close()
            from .process_diagnostics import upload_process_logs

            try:
                await upload_process_logs(run_root / "generic-harness", sink)
            except OSError:
                import logging

                logging.getLogger(__name__).warning(
                    "process diagnostic collection failed"
                )


__all__ = ["LocalCertifiedRunError", "LocalCertifiedSink", "run_local_certified"]
