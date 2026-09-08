"""Run the established ALK authoring stages for one hosted job.

This is intentionally a thin process boundary over :func:`fi.alk.harness.cli._auto`.
Contract creation, logical environment creation, and scenario generation therefore remain the
same implementation used by the local SDK and sandbox flows.  Daytona consumes the frozen output
afterward; this command never executes scenarios itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import tempfile

from .cli import _auto
from .job import HarnessJob, ProviderExecutionMode
from .provider_import import inspect_provider_target
from .scenarios import load as load_written
from .understand import PROVIDER_IMPORT_PROFILE_PATH_ENV


def _persist_authored_scenario_count(
    job_path: Path, job: HarnessJob, output: Path
) -> None:
    """Keep the frozen job in sync with adjustments applied during authoring.

    The control plane may increase ``scenario_count`` while this process is already
    running.  ``_auto`` sees that adjustment and writes the larger validated suite,
    but the following Bundle V2 process reloads this on-disk job document.  Without
    reconciling it here the bundler copies the original number of scenarios and the
    platform correctly rejects preallocation because its expected count is newer.
    """
    authored_count = len(load_written(output))
    if authored_count <= 0 or authored_count == job.scenario_count:
        return
    updated = job.model_copy(update={"scenario_count": authored_count})
    temporary = job_path.with_suffix(f"{job_path.suffix}.tmp")
    temporary.write_text(
        json.dumps(updated.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(job_path)


def _load_provider_import_profile(
    job: HarnessJob,
    secrets_path: Path | None,
    profile_cache_path: Path | None = None,
) -> dict[str, object] | None:
    inspect_connect_only_chat = (
        job.agent.mode is ProviderExecutionMode.CONNECT_ONLY
        and job.agent.connector.strip().lower() == "retell_chat"
    )
    if (
        job.agent.mode is not ProviderExecutionMode.PROVIDER_IMPORT
        and not inspect_connect_only_chat
    ):
        return None
    if profile_cache_path is not None and profile_cache_path.is_file():
        cached = json.loads(profile_cache_path.read_text(encoding="utf-8"))
        if not isinstance(cached, dict):
            raise RuntimeError("provider_import_authoring_profile_invalid")
        return cached
    if secrets_path is None:
        raise RuntimeError("provider_import_authoring_secrets_missing")
    try:
        values = json.loads(secrets_path.read_text(encoding="utf-8"))
    finally:
        # The provider credential is needed only for this read-only inspection. Remove the file
        # before any model session or source/environment process starts.
        secrets_path.unlink(missing_ok=True)
    if not isinstance(values, dict):
        raise RuntimeError("provider_import_authoring_secrets_invalid")
    connector = job.agent.connector.strip().lower()
    provider = "retell" if connector == "retell_chat" else connector
    secret_name = "VAPI_API_KEY" if provider == "vapi" else "RETELL_API_KEY"
    target_key = "assistant_id" if provider == "vapi" else "agent_id"
    profile = inspect_provider_target(
        provider,
        source_target_id=str(job.agent.config.get(target_key) or ""),
        api_key=str(values.get(secret_name) or ""),
        api_base_url=str(job.agent.config.get("provider_api_base_url") or "") or None,
        target_modality="chat" if connector == "retell_chat" else "voice",
    )
    if profile_cache_path is not None:
        profile_cache_path.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return profile


def main(argv: list[str] | None = None, *, validate_runtime: bool = False) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adjustments",
        type=Path,
        help="JSONL inbox for user corrections applied at safe stage boundaries",
    )
    parser.add_argument(
        "--target-secrets",
        type=Path,
        help="One-shot control-process secrets used to inspect an imported provider target",
    )
    parser.add_argument(
        "--provider-profile-cache",
        type=Path,
        help="Control-owned sanitized profile reused across authoring retries",
    )
    args = parser.parse_args(argv)

    job = HarnessJob.model_validate(json.loads(args.job.read_text(encoding="utf-8")))
    # Transport kinds such as ``archive`` and ``github`` describe how the platform acquired the
    # source.  Once extracted, the established ALK authoring pipeline must inspect it as a repo.
    source_kind = str(job.metadata.get("source_kind") or "repo")
    if source_kind not in {"repo", "spec"}:
        source_kind = "repo"
    namespace = argparse.Namespace(
        path=str(args.source.resolve()),
        name=str(job.metadata.get("agent_name") or args.source.name),
        kind=source_kind,
        out=str(args.output.resolve()),
        count=job.scenario_count,
        model=None,
        run_model=None,
        job=job,
        adjustments_path=str(args.adjustments) if args.adjustments else None,
        authoring_only=True,
    )
    profile = _load_provider_import_profile(
        job, args.target_secrets, args.provider_profile_cache
    )
    previous_profile_path = os.environ.get(PROVIDER_IMPORT_PROFILE_PATH_ENV)
    with tempfile.TemporaryDirectory(prefix="alk-provider-profile-") as temporary:
        if profile is not None:
            profile_path = Path(temporary) / "provider-import-profile.json"
            profile_path.write_text(
                json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.environ[PROVIDER_IMPORT_PROFILE_PATH_ENV] = str(profile_path)
        try:
            status = asyncio.run(_auto(namespace))
            if status == 0 and validate_runtime:
                from .authoring_runtime_validation import validate_and_repair

                _persist_authored_scenario_count(args.job, job, args.output.resolve())
                runtime_job = HarnessJob.model_validate_json(args.job.read_text())
                if profile is not None:
                    (args.output / "provider-import-profile.json").write_text(
                        json.dumps(profile) + "\n", encoding="utf-8"
                    )
                asyncio.run(
                    validate_and_repair(
                        runtime_job, args.source.resolve(), args.output.resolve()
                    )
                )
        finally:
            if previous_profile_path is None:
                os.environ.pop(PROVIDER_IMPORT_PROFILE_PATH_ENV, None)
            else:
                os.environ[PROVIDER_IMPORT_PROFILE_PATH_ENV] = previous_profile_path
    if status == 0:
        _persist_authored_scenario_count(args.job, job, args.output.resolve())
        if profile is not None:
            (args.output.resolve() / "provider-import-profile.json").write_text(
                json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
