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
import sys
import shutil
import threading
import time
from pathlib import Path
import tempfile

from .cli import _auto
from .job import HarnessJob, ProviderExecutionMode, SourceKind
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
    if (
        job.agent.mode is ProviderExecutionMode.CONNECT_ONLY
        and job.agent.connector.strip().lower() == "phone"
    ):
        # A dial-in number has no provider API to inspect. The supplied prompt is the only
        # behavioral source of truth; never pretend we discovered tools or implementation code.
        return {
            "provider": "phone",
            "modality": "voice",
            "phone_number": str(job.agent.config["phone_number"]),
            "system_prompt": str(job.agent.config["target_system_prompt"]),
            "tools": [],
        }
    if (
        job.agent.mode is ProviderExecutionMode.CONNECT_ONLY
        and job.agent.connector.strip().lower() == "livekit"
    ):
        if secrets_path is not None:
            secrets_path.unlink(missing_ok=True)
        return {
            "provider": "livekit",
            "modality": "voice",
            "agent_name": str(job.agent.config["agent_name"]),
            "system_prompt": str(job.agent.config["target_system_prompt"]),
            "tools": [],
        }
    inspect_connect_only_provider = (
        job.agent.mode is ProviderExecutionMode.CONNECT_ONLY
        and job.agent.connector.strip().lower() in {"vapi", "retell", "retell_chat"}
    )
    if (
        job.agent.mode is not ProviderExecutionMode.PROVIDER_IMPORT
        and not inspect_connect_only_provider
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


def agent_source_kind(job: HarnessJob, profile: dict[str, object] | None) -> str:
    """How authoring reads the agent: `provider`, `repo` or `spec`.

    Transport kinds such as ``archive`` and ``github`` describe how the platform acquired the
    source. Once extracted, they are repositories. A source-free connect-only provider is the
    exception: its fetched definition is the source of truth and must never be represented by
    the intentionally empty /work/source directory.
    """
    if (
        job.source.kind is SourceKind.PROVIDER
        and job.agent.mode is ProviderExecutionMode.CONNECT_ONLY
        and profile is not None
    ):
        return "provider"
    kind = str(job.metadata.get("source_kind") or "repo")
    return kind if kind in {"repo", "spec"} else "repo"


def _await_harness(output: Path) -> int:
    """Wait while the conversation's harness builds the environment in this sandbox."""
    finished = output / ".futureagi-build-finished.json"
    deadline = time.monotonic() + int(os.environ.get("ALK_HOSTED_AUTHORING_MAX_DURATION_SECONDS") or 3300)
    while time.monotonic() < deadline:
        if finished.is_file():
            return 0
        time.sleep(5)
    print("the harness did not finish the build in time", file=sys.stderr)
    return 1


def _review_while_writing(job: HarnessJob, source: Path, output: Path, done: threading.Event) -> None:
    """Run the source data review against the saved world while the harness writes scenarios.

    Best effort: whatever it does not finish, validation reviews exactly as before."""
    from .authoring_runtime_validation import validate_once
    from .contract import is_data_free_conversation
    from .source_data_invariants import review_while_writing, world_digest
    from .understand import load as load_contract
    from .world.snapshot import saved as world_saved

    failures: dict[str, int] = {}
    while not done.is_set():
        if not world_saved(output):
            time.sleep(5)
            continue
        contract = load_contract(output)
        if contract is None or is_data_free_conversation(contract) or (
            contract.data_store and contract.data_store.kind.strip().lower() == "in_process"
        ):
            return
        world_key = world_digest(output)
        if failures.get(world_key, 0) >= 2:
            time.sleep(5)
            continue
        changed: list[bool] = []
        try:
            print("early source data review: world saved, reviewing", flush=True)
            with tempfile.TemporaryDirectory(prefix="early-review-", dir=output.parent) as root:
                staging = Path(root) / "authoring"
                shutil.copytree(
                    output,
                    staging,
                    ignore=shutil.ignore_patterns(".claude", "scenarios", "environment-bundle"),
                )
                (staging / "scenarios").mkdir()

                async def review(world, endpoints):
                    changed.append(
                        await review_while_writing(
                            source,
                            output,
                            world,
                            endpoints=endpoints,
                            world_key=world_key,
                            done=done,
                            expected=job.scenario_count,
                        )
                    )

                asyncio.run(
                    validate_once(
                        job.model_copy(update={"scenario_count": 0}), source, staging, review=review
                    )
                )
        except Exception as exc:  # noqa: BLE001 - validation reviews whatever this did not
            failures[world_key] = failures.get(world_key, 0) + 1
            note = output / "generic-harness" / "source-data-review-early-error.txt"
            try:
                note.parent.mkdir(parents=True, exist_ok=True)
                with note.open("a", encoding="utf-8") as handle:
                    handle.write(f"{type(exc).__name__}: {exc}"[:4000] + "\n")
            except OSError:
                pass
            time.sleep(30)
            continue
        if not any(changed):
            return


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
    parser.add_argument(
        "--await-harness",
        action="store_true",
        help="The conversation's harness builds the environment; wait for it to finish the build",
    )
    parser.add_argument(
        "--conversation-capabilities",
        type=Path,
        help="Conversation capability document for messages delivered to the current stage",
    )
    args = parser.parse_args(argv)

    job = HarnessJob.model_validate(json.loads(args.job.read_text(encoding="utf-8")))
    profile = _load_provider_import_profile(
        job, args.target_secrets, args.provider_profile_cache
    )
    source_kind = agent_source_kind(job, profile)
    source_free_provider = source_kind == "provider"
    namespace = argparse.Namespace(
        path=str(args.source.resolve()),
        name=str(job.metadata.get("agent_name") or args.source.name),
        kind=source_kind,
        out=str(args.output.resolve()),
        count=job.scenario_count,
        interactive=False,
        model=None,
        guidance=job.authoring_guidance(),
        adjustments_path=str(args.adjustments) if args.adjustments else None,
        conversation_capabilities_path=(
            str(args.conversation_capabilities)
            if args.conversation_capabilities
            else None
        ),
        authoring_only=True,
        run_model=None,
        job=job,
        provider_profile=profile if source_free_provider else None,
    )
    previous_profile_path = os.environ.get(PROVIDER_IMPORT_PROFILE_PATH_ENV)
    with tempfile.TemporaryDirectory(prefix="alk-provider-profile-") as temporary:
        if profile is not None:
            profile_path = Path(temporary) / "provider-import-profile.json"
            profile_path.write_text(
                json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.environ[PROVIDER_IMPORT_PROFILE_PATH_ENV] = str(profile_path)
        early = threading.Event()
        reviewer = None
        if (
            args.await_harness
            and validate_runtime
            and job.metadata.get("generic_harness_v1") is True
            and not source_free_provider
            and job.source.kind is not SourceKind.PROVIDER
        ):
            reviewer = threading.Thread(
                target=_review_while_writing,
                args=(job, args.source.resolve(), args.output.resolve(), early),
                name="early-source-data-review",
                daemon=True,
            )
            reviewer.start()
        try:
            try:
                status = (
                    _await_harness(args.output.resolve())
                    if args.await_harness
                    else asyncio.run(_auto(namespace))
                )
            finally:
                early.set()
                if reviewer is not None:
                    reviewer.join()
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
