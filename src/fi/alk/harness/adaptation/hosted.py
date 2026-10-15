"""Adaptation inside the hosted sandbox (`environment-releases.md` §7.5, Launch).

The job's command is the control process: it inspects the agent read-only with the target's
provider key (never placed in the process environment, never seen by the model), starts the scratch
PostgreSQL the proofs run on, hands the conversation's builder what it needs in the workspace, and
waits until the builder finishes, exactly as a build waits for its harness. The builder session
does the binding through the workbench; this process turns its `result.json` into the exit code.

Workspace layout (`/work/authoring`): `adaptation/inputs.json` and `adaptation/profile.json` from
this process; `adaptation/result.json` and the sealed files from the workbench; the builder's
`.futureagi-build-finished.json` marker when the work ends either way.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

from ..hosted_authoring_entrypoint import (
    _SECRETS_PATH,
    _TARGET_CREDENTIAL_NAMES,
    _TARGET_SECRETS_PATH,
    _load_values,
)
from ..releases import RELEASES_SUITE, environment_directory
from .errors import AdaptationFailure
from .workbench import (
    INPUTS,
    PROFILE,
    RESULT,
    Workbench,
    failed_result,
    profile_from_document,
    write_json,
)

ADAPTATION_DIRECTORY = "adaptation"
FINISHED_MARKER = ".futureagi-build-finished.json"
_INPUTS_WAIT_SECONDS = 300


def _values(path: Path) -> dict[str, str]:
    try:
        return _load_values(path)
    except (OSError, ValueError):
        return {}

def _target_credentials() -> dict[str, str]:
    """The target's provider key, read once: an adaptation runs no customer process, so nothing
    else in the sandbox needs either file afterwards."""
    values = {**_values(_SECRETS_PATH), **_values(_TARGET_SECRETS_PATH)}
    _SECRETS_PATH.unlink(missing_ok=True)
    _TARGET_SECRETS_PATH.unlink(missing_ok=True)
    return {name: values[name] for name in _TARGET_CREDENTIAL_NAMES if values.get(name)}


def _environments(job: Any, releases: Path) -> tuple[list[Path], Path]:
    adaptation = (job.metadata or {}).get("adaptation")
    if not isinstance(adaptation, dict):
        raise AdaptationFailure(
            "adaptation_job_invalid", "job metadata.adaptation is missing"
        )
    chain = adaptation.get("environment")
    suite = adaptation.get("suite")
    if not isinstance(chain, list) or not chain or not isinstance(suite, dict):
        raise AdaptationFailure(
            "adaptation_job_invalid",
            "metadata.adaptation needs an environment chain and a suite",
        )
    directories = [
        environment_directory(releases, str(ref.get("digest") or "")) for ref in chain
    ]
    suite_root = releases / RELEASES_SUITE
    from ..releases import load_suite_release

    delivered = load_suite_release(suite_root).digest
    if delivered != suite.get("digest"):
        raise AdaptationFailure(
            "release_invalid",
            f"delivered suite {delivered} is not the job's {suite.get('digest')}",
        )
    return directories, suite_root


def failed_before_start(
    output: Path, code: str, message: str, *, hosted: bool = False
) -> int:
    """A failure before the work could start still leaves result.json; in the sandbox it also
    leaves the marker that tells the builder there is nothing left to do."""
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / RESULT, failed_result(code, message))
    if hosted:
        write_json(output.parent / FINISHED_MARKER, {"status": "failed"})
    print(f"adaptation failed: {code}: {message}")
    return 2


def _await_result(output: Path) -> int:
    """Wait while the conversation's builder binds the agent in this sandbox."""
    marker = output.parent / FINISHED_MARKER
    deadline = time.monotonic() + int(
        os.environ.get("ALK_HOSTED_AUTHORING_MAX_DURATION_SECONDS") or 3300
    )
    while time.monotonic() < deadline:
        if marker.is_file() and (output / RESULT).is_file():
            result = json.loads((output / RESULT).read_text(encoding="utf-8"))
            print(json.dumps({"status": result.get("status")}))
            return 0 if result.get("status") == "completed" else 2
        time.sleep(5)
    return failed_before_start(
        output,
        "adaptation_unfinished",
        "the builder did not finish the adaptation in time",
        hosted=True,
    )


def run_hosted(args: argparse.Namespace) -> int:
    from ..job import HarnessJob
    from ..releases import ReleaseError
    from .profile import profile_from_job

    for name in _TARGET_CREDENTIAL_NAMES:
        os.environ.pop(name, None)
    credentials = _target_credentials()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        job = HarnessJob.model_validate_json(Path(args.job).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdaptationFailure("adaptation_job_invalid", str(exc)) from exc
    try:
        environments, suite_root = _environments(job, Path(args.releases))
    except ReleaseError as exc:
        raise AdaptationFailure("release_invalid", f"{exc.code}: {exc.detail}") from exc
    profile = profile_from_job(job, credentials=credentials)
    with contextlib.ExitStack() as stack:
        if args.postgres_dsn:
            dsn = args.postgres_dsn
        else:
            from .scratch_postgres import scratch_cluster

            try:
                dsn = stack.enter_context(scratch_cluster())
            except (OSError, RuntimeError, KeyError) as exc:
                raise AdaptationFailure("postgres_unavailable", str(exc)) from exc
        write_json(output / PROFILE, profile.document())
        # Written last: its presence tells the builder everything else is in place.
        write_json(
            output / INPUTS,
            {
                "environments": [str(one) for one in environments],
                "suite": str(suite_root),
                "postgres_dsn": dsn,
            },
        )
        return _await_result(output)


async def load_workbench(workspace: Path) -> Workbench:
    """The builder's side: the workbench over what the control process prepared."""
    output = workspace / ADAPTATION_DIRECTORY
    deadline = time.monotonic() + _INPUTS_WAIT_SECONDS
    while not (output / INPUTS).is_file():
        if (output / RESULT).is_file():
            result = json.loads((output / RESULT).read_text(encoding="utf-8"))
            failure = result.get("failure") or {}
            raise AdaptationFailure(
                str(failure.get("code") or "adaptation_failed"),
                str(
                    failure.get("message") or "the adaptation failed before it started"
                ),
            )
        if time.monotonic() > deadline:
            raise AdaptationFailure(
                "adaptation_inputs_missing", "the adaptation's inputs never arrived"
            )
        await asyncio.sleep(2)
    inputs = json.loads((output / INPUTS).read_text(encoding="utf-8"))
    profile = profile_from_document(
        json.loads((output / PROFILE).read_text(encoding="utf-8"))
    )
    return await asyncio.to_thread(
        Workbench,
        environments=[Path(one) for one in inputs["environments"]],
        suite_root=Path(inputs["suite"]),
        output=output,
        postgres_dsn=str(inputs["postgres_dsn"]),
        profile=profile,
        finished_marker=workspace / FINISHED_MARKER,
    )
