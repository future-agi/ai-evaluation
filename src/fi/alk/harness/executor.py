"""One ALK harness executor used by the local CLI and hosted sandbox workers."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Callable, Protocol

from .job import (
    FailureDomain,
    HarnessFailure,
    HarnessJob,
    HarnessJobStatus,
    HarnessStage,
    SourceKind,
    SourceVisibility,
)


class SourceAcquirer(Protocol):
    """Materialize job source inside an executor-owned ephemeral workspace."""

    async def acquire(self, job: HarnessJob, workspace: Path) -> Path: ...


class SourceAcquisitionError(RuntimeError):
    pass


class GitHubSourceAcquirer:
    """Clone one platform-authorized repository without exposing its token in argv or logs."""

    _REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

    def __init__(self, installation_token: Callable[[str], str]) -> None:
        self._installation_token = installation_token

    async def acquire(self, job: HarnessJob, workspace: Path) -> Path:
        source = job.source
        repository = str(source.repository or "")
        installation_id = str(source.installation_id or "")
        if not self._REPOSITORY.fullmatch(repository):
            raise SourceAcquisitionError("github_repository_invalid")
        is_public = source.visibility is SourceVisibility.PUBLIC
        if not is_public and not installation_id:
            raise SourceAcquisitionError("github_installation_missing")
        token = "" if is_public else self._installation_token(installation_id)
        if not is_public and not token:
            raise SourceAcquisitionError("github_installation_token_missing")
        if source.ref and (
            ".." in source.ref or not re.fullmatch(r"[A-Za-z0-9._/-]+", source.ref)
        ):
            raise SourceAcquisitionError("github_ref_invalid")
        workspace = workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        destination = workspace / "repository"
        if destination.exists():
            raise SourceAcquisitionError(f"source_destination_not_empty: {destination}")

        command = ["git", "clone", "--depth", "1"]
        if source.ref:
            command.extend(["--branch", source.ref])
        command.extend([f"https://github.com/{repository}.git", str(destination)])
        # Git reads the authorization header from its child environment. It never appears in
        # the process command, exception, persisted job, or event stream.
        environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        if token:
            environment.update(
                {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "http.extraHeader",
                    "GIT_CONFIG_VALUE_0": f"Authorization: Bearer {token}",
                }
            )

        def clone() -> None:
            try:
                completed = subprocess.run(
                    command,
                    env=environment,
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise SourceAcquisitionError(
                    f"github_clone_unavailable: {type(exc).__name__}"
                ) from exc
            if completed.returncode:
                detail = (
                    completed.stderr or completed.stdout or "clone failed"
                ).strip()
                # Git errors should not contain an env-only header, but redact defensively.
                detail = detail.replace(token, "[REDACTED]")[:1000]
                raise SourceAcquisitionError(f"github_clone_failed: {detail}")

            if source.commit_sha:
                verified = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=destination,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                )
                actual = verified.stdout.strip().lower()
                if verified.returncode or actual != source.commit_sha.lower():
                    raise SourceAcquisitionError("github_commit_mismatch")

        await asyncio.to_thread(clone)
        if not destination.is_dir():
            raise SourceAcquisitionError("github_clone_missing_checkout")
        return destination


class HarnessExecutor:
    """Execute the autonomous pipeline without platform or scheduler dependencies.

    A hosted worker first resolves a GitHub installation/archive/image through its
    ``SourceAcquirer``. A local invocation already has a path. From that point onward both call
    exactly the same ``auto`` pipeline and produce the same job, bundle, event, scenario, trace,
    and result artifacts.
    """

    async def run(
        self,
        job: HarnessJob,
        *,
        source: Path,
        output: Path,
        model: str | None = None,
        run_model: str | None = None,
        adjustments_path: Path | None = None,
    ) -> HarnessJobStatus:
        from .cli import _auto

        output = output.expanduser().resolve()
        source = source.expanduser().resolve()
        # A branch name is acquisition input, not immutable provenance. Resolve the checkout to
        # its exact commit before job.json and the environment plan are written. The source
        # content digest remains authoritative for uploads and non-Git sources.
        if job.source.kind is SourceKind.GITHUB and not job.source.commit_sha:
            completed = await asyncio.to_thread(
                subprocess.run,
                ["git", "rev-parse", "HEAD"],
                cwd=source,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            commit_sha = completed.stdout.strip().lower()
            if completed.returncode or not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
                return HarnessJobStatus(
                    job_id=job.job_id,
                    run_id=job.run_id,
                    stage=HarnessStage.FAILED,
                    updated_at=datetime.now(timezone.utc),
                    detail="could not resolve the acquired GitHub revision",
                    failure=HarnessFailure(
                        domain=FailureDomain.INFRASTRUCTURE,
                        stage=HarnessStage.ACQUIRING_SOURCE,
                        code="github_commit_resolution_failed",
                        message="The runner could not resolve the acquired GitHub revision",
                    ),
                    total_scenarios=job.scenario_count,
                )
            job = job.model_copy(
                update={
                    "source": job.source.model_copy(update={"commit_sha": commit_sha})
                }
            )
        # Source acquisition has already materialized GitHub/archive/local inputs as a local
        # checkout.  The understanding registry describes how to inspect that materialized
        # content (``repo``), while the immutable job retains its original source provenance.
        # Passing transport kinds such as ``archive`` into source resolution makes a valid
        # uploaded repository fail before its code is inspected.
        understanding_kind = str(job.metadata.get("source_kind") or "repo")
        if understanding_kind not in {"repo", "spec"}:
            understanding_kind = "repo"
        args = argparse.Namespace(
            path=str(source),
            name=str(job.metadata.get("agent_name") or source.name),
            kind=understanding_kind,
            out=str(output),
            count=job.scenario_count,
            model=model,
            run_model=run_model,
            job=job,
            adjustments_path=str(adjustments_path) if adjustments_path else None,
        )
        status = await _auto(args)
        failure = _failure_from_events(output) if status not in (0, 2) else None
        return HarnessJobStatus(
            job_id=job.job_id,
            run_id=job.run_id,
            stage=HarnessStage.COMPLETED if status in (0, 2) else HarnessStage.FAILED,
            updated_at=datetime.now(timezone.utc),
            detail=(
                "agent checks failed"
                if status == 2
                else None
                if status == 0
                else f"exit {status}"
            ),
            failure=failure,
            completed_scenarios=_scenario_count(output) if status in (0, 2) else 0,
            total_scenarios=_scenario_count(output) or job.scenario_count,
        )

    async def acquire_and_run(
        self,
        job: HarnessJob,
        *,
        acquirer: SourceAcquirer,
        workspace: Path,
        output: Path,
        model: str | None = None,
        run_model: str | None = None,
    ) -> HarnessJobStatus:
        source = await acquirer.acquire(job, workspace)
        return await self.run(
            job,
            source=source,
            output=output,
            model=model,
            run_model=run_model,
        )


def run_sync(job: HarnessJob, *, source: Path, output: Path) -> HarnessJobStatus:
    """Small synchronous adapter for job consumers that do not own an event loop."""
    return asyncio.run(HarnessExecutor().run(job, source=source, output=output))


def _scenario_count(output: Path) -> int:
    try:
        value = json.loads((output / "scenarios.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return len(value) if isinstance(value, list) else 0


def _failure_from_events(output: Path) -> HarnessFailure:
    failed: dict = {}
    path = output / "harness-events.jsonl"
    if path.is_file():
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            if event.get("type") == "harness.stage.failed":
                failed = event.get("payload") or {}
    label = str(failed.get("stage") or "running")
    stage, domain = {
        "understand": (HarnessStage.UNDERSTANDING_AGENT, FailureDomain.AGENT),
        "environment": (
            HarnessStage.VALIDATING_ENVIRONMENT,
            FailureDomain.ENVIRONMENT,
        ),
        "scenarios": (
            HarnessStage.VALIDATING_SCENARIOS,
            FailureDomain.SIMULATOR,
        ),
        "calls": (HarnessStage.RUNNING, FailureDomain.CONNECTIVITY),
        "cleaning_up": (
            HarnessStage.CLEANING_UP,
            FailureDomain.INFRASTRUCTURE,
        ),
        "uploading_artifacts": (
            HarnessStage.UPLOADING_ARTIFACTS,
            FailureDomain.ARTIFACT,
        ),
    }.get(label, (HarnessStage.RUNNING, FailureDomain.INFRASTRUCTURE))
    return HarnessFailure(
        domain=domain,
        stage=stage,
        code=str(failed.get("code") or f"{label}_failed"),
        message=str(failed.get("detail") or f"Harness stage {label} failed"),
        # A job replay can repeat real calls. Only a failing stage that explicitly proves
        # it is safe may opt in to retry; deterministic agent/grading failures never do.
        retryable=bool(failed.get("retryable", False)),
        details={"status": failed.get("status", 1)},
    )


__all__ = [
    "GitHubSourceAcquirer",
    "HarnessExecutor",
    "SourceAcquirer",
    "SourceAcquisitionError",
    "run_sync",
]
