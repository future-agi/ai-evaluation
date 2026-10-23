"""A design inside the hosted sandbox: the job's control process.

It reads the job's `metadata.design` (a design has no agent and no source, so not through
`HarnessJob`), unpacks the template the platform delivered under `--releases` into the workspace
when there is one, and waits while the conversation's builder designs, exactly as an adaptation's
control process waits. The builder's `finish_design`, `stop_design` or a parked question writes
`design-output/result.json` and the finished marker; this process turns that into the exit code.

A resumed design (`resume_from`) starts from the checkpointed workspace the platform uploaded, so
the design is already there and is kept.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..releases import (
    _SLUG_PATTERN,
    RELEASES_SUITE,
    DigestRef,
    ReleaseError,
    environment_directory,
)
from .tools import FINISHED_MARKER, RESULT
from .workspace import open_design

OUTPUT = "design-output"


class DesignTemplateRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment: DigestRef
    suite: DigestRef


class DesignJob(BaseModel):
    """`metadata.design` of a hosted design job, written by the platform."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(pattern=_SLUG_PATTERN)
    requirements: str = ""
    modality: Literal["chat", "voice"] = "chat"
    template: DesignTemplateRef | None = None
    resume_from: str | None = None


def design_job(job: dict) -> DesignJob:
    return DesignJob.model_validate((job.get("metadata") or {}).get("design"))


def failed(output: Path, workspace: Path, code: str, message: str) -> int:
    output.mkdir(parents=True, exist_ok=True)
    (output / RESULT).write_text(
        json.dumps({"status": "failed", "failure": {"code": code, "message": message}}),
        encoding="utf-8",
    )
    (workspace / FINISHED_MARKER).write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    print(f"design failed: {code}: {message}")
    return 2


def run_hosted(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    output = workspace / OUTPUT
    try:
        job = design_job(json.loads(Path(args.job).read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as exc:
        return failed(output, workspace, "design_job_invalid", str(exc)[:600])
    releases = Path(args.releases)
    template = (
        (
            environment_directory(releases, job.template.environment.digest),
            releases / RELEASES_SUITE,
        )
        if job.template is not None
        else None
    )
    try:
        open_design(
            workspace,
            name=job.name,
            requirements=job.requirements,
            modality=job.modality,
            template=template,
        )
    except ReleaseError as exc:
        return failed(output, workspace, exc.code, exc.detail[:600])
    deadline = time.monotonic() + int(
        os.environ.get("ALK_HOSTED_AUTHORING_MAX_DURATION_SECONDS") or 3300
    )
    while time.monotonic() < deadline:
        if (workspace / FINISHED_MARKER).is_file() and (output / RESULT).is_file():
            result = json.loads((output / RESULT).read_text(encoding="utf-8"))
            print(
                json.dumps(
                    {"status": result.get("status"), "failure": result.get("failure")}
                )
            )
            return 0 if result.get("status") == "completed" else 2
        time.sleep(5)
    return failed(
        output, workspace, "design_unfinished", "the design did not finish in time"
    )
