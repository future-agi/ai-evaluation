"""Run the established ALK authoring stages for one hosted job.

This is intentionally a thin process boundary over :func:`fi.alk.harness.cli._auto`.
Contract creation, logical environment creation, and scenario generation therefore remain the
same implementation used by the local SDK and sandbox flows.  Daytona consumes the frozen output
afterward; this command never executes scenarios itself.
"""

from __future__ import annotations

import argparse
import asyncio
from functools import partial
import json
import os
import sys
import shutil
import threading
import time
from pathlib import Path
import shutil
import sys
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


CERTIFY_REPORT = "certify-report.json"


def _one_line(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())


def _prune_suite_to_catalogue(output: Path) -> tuple[list[str], dict[str, object]]:
    """Grade each scenario only on sub_goals the environment defines; drop ones left with none.

    Folders keep their names and scenarios their keys. Returns notes and the pruning report.
    """
    from .authoring_runtime_validation import RuntimeValidationError
    from .catalogue import CATALOGUE, load_catalogue
    from .folder import SCENARIOS, read_folder, refresh_check, write_index

    if not (output / CATALOGUE).is_file():
        raise RuntimeValidationError(
            "environment", f"Certify-only: {CATALOGUE} is missing from the environment"
        )
    if not (output / CATALOGUE).read_text(encoding="utf-8").strip():
        raise RuntimeValidationError(
            "environment", f"Certify-only: {CATALOGUE} is empty"
        )
    catalogue = load_catalogue(output)
    if not catalogue.sub_goals:
        raise RuntimeValidationError(
            "environment", f"Certify-only: {CATALOGUE} defines no sub_goals"
        )
    known = catalogue.names()
    root = output / SCENARIOS
    folders = (
        sorted(path for path in root.iterdir() if path.is_dir())
        if root.is_dir()
        else []
    )
    documents: list[tuple[Path, dict, str]] = []
    owners: dict[str, str] = {}
    for folder in folders:
        try:
            if read_folder(output, folder.name) is None:
                raise RuntimeValidationError(
                    "scenarios",
                    f"Certify-only: {folder.name}: scenario.json is missing",
                )
            body = json.loads((folder / "scenario.json").read_text(encoding="utf-8"))
        except RuntimeValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - a silently skipped folder would shrink the suite
            raise RuntimeValidationError(
                "scenarios",
                f"Certify-only: {folder.name}: unreadable scenario: {_one_line(exc)}",
            ) from exc
        name = str(body.get("name") or "")
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise RuntimeValidationError(
                "scenarios",
                f"Certify-only: {folder.name}: invalid scenario name {name!r}",
            )
        # The key the bundle derives; written back so Scenario's slug(name) cannot replace it.
        key = str(body.get("scenario_key") or name or folder.name)
        if key in owners:
            raise RuntimeValidationError(
                "scenarios",
                f"Certify-only: {owners[key]} and {folder.name} share scenario_key {key!r}",
            )
        owners[key] = folder.name
        documents.append((folder, body, key))

    kept: list[tuple[Path, dict]] = []
    untouched: list[Path] = []
    notes: list[str] = []
    dropped: list[str] = []
    narrowed: dict[str, list[str]] = {}
    for folder, body, key in documents:
        if "sub_goals" not in body:
            # Older suites name no sub_goals; they are left as authored.
            untouched.append(folder)
            continue
        goals = list(body.get("sub_goals") or [])
        graded = [goal for goal in goals if goal in known]
        removed = [goal for goal in goals if goal not in known]
        if not graded:
            dropped.append(key)
            notes.append(f"{key}: dropped, no sub_goal left to grade")
            continue
        if removed:
            narrowed[key] = sorted(removed)
            notes.append(
                f"{key}: sub_goals not in the environment catalogue: "
                + ", ".join(removed)
            )
        kept.append((folder, {**body, "scenario_key": key, "sub_goals": graded}))
    if not kept and not untouched:
        raise RuntimeValidationError(
            "scenarios",
            "Certify-only: no scenario grades a sub_goal in the environment catalogue",
        )

    keeping = {folder for folder, _ in kept} | set(untouched)
    for folder, _, _ in documents:
        if folder not in keeping:
            shutil.rmtree(folder)
    for folder, body in kept:
        (folder / "scenario.json").write_text(
            json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (folder / "checks").mkdir(exist_ok=True)
        for stale in (folder / "checks").glob("*.py"):
            if stale.stem not in body["sub_goals"]:
                stale.unlink()
    # Checks come from this environment's catalogue; a judged sub_goal keeps no check file.
    for sub_goal in catalogue.sub_goals:
        refresh_check(output, sub_goal)
    # TODO: write_index names each folder after its scenario, which a kept folder may not match.
    write_index(
        [
            scenario
            for scenario in (
                read_folder(output, folder.name) for folder in sorted(keeping)
            )
            if scenario is not None
        ],
        output,
    )
    return notes, {"dropped": sorted(dropped), "narrowed": narrowed}


def _drop_stale_invariants(source: Path, output: Path) -> list[str]:
    """Hold the saved invariant review to the run's source; return the checks dropped.

    A check whose quoted lines the source still holds is stamped with this source. One quoting
    a line the source no longer has is dropped: certify-only cannot review it again.
    """
    from .source_data_invariants import ARTIFACT, source_files, validate_evidence

    artifact = output / ARTIFACT
    if not artifact.is_file():
        return []
    body = json.loads(artifact.read_text(encoding="utf-8"))
    checks = body.get("checks") if isinstance(body, dict) else None
    if not isinstance(checks, list) or not checks:
        return []
    files = source_files(source)
    kept: list[dict] = []
    stale: list[str] = []
    for check in checks:
        try:
            evidence = validate_evidence(check, files)["evidence"]
        except (ValueError, AttributeError, TypeError):
            stale.append(str(check.get("name") or "a check"))
            continue
        kept.append({**check, "evidence": evidence})
    if kept != checks:
        artifact.write_text(
            json.dumps({**body, "checks": kept}, indent=2) + "\n", encoding="utf-8"
        )
    return stale


def _prepare_source_data_invariants(job: HarnessJob, output: Path) -> list[str]:
    """Reuse the environment's invariant review without asking a model for a new one."""
    from .authoring_runtime_validation import RuntimeValidationError
    from .contract import AgentContract, is_data_free_conversation
    from .source_data_invariants import ARTIFACT, prune_invariant_scopes

    artifact = output / ARTIFACT
    if artifact.is_file():
        body = json.loads(artifact.read_text(encoding="utf-8"))
        if isinstance(body, dict) and body.get("status") == "not_applicable":
            # Regenerated deterministically; contract.json may have been rewritten since.
            artifact.unlink()
            return []
        notes = [
            f"source-data invariant {name} dropped, its scenarios left the suite"
            for name in prune_invariant_scopes(output)
        ]
        if not json.loads(artifact.read_text(encoding="utf-8")).get("checks"):
            raise RuntimeValidationError(
                "environment",
                "Certify-only: no saved source-data invariant applies to the scenarios "
                "left in the suite and the agent's source, so invariants must be authored "
                "again",
            )
        return notes
    if (
        job.source.kind is SourceKind.PROVIDER
        and job.agent.mode is ProviderExecutionMode.CONNECT_ONLY
    ):
        return []
    contract = AgentContract.model_validate_json(
        (output / "contract.json").read_text(encoding="utf-8")
    )
    in_process = bool(
        contract.data_store and contract.data_store.kind.strip().lower() == "in_process"
    )
    if not (is_data_free_conversation(contract) or in_process):
        raise RuntimeValidationError(
            "environment",
            "Certify-only: environment has no saved source-data invariant review",
        )
    return []


def _saved_world_outgrew_world_ir(output: Path) -> bool:
    """Whether world.sqlite holds data the source model its World IR was imported for lacks.

    An IR that differs from a clean import is kept: it may carry a validation repair.
    """
    import sqlite3
    from contextlib import closing

    from .certification import GenericHarnessArtifactStore
    from .world_import.sqlite import SQLiteWorldImportError, import_sqlite_world

    store = GenericHarnessArtifactStore(output / "generic-harness")
    world = output / "world.sqlite"
    try:
        with closing(sqlite3.connect(f"file:{world}?mode=ro", uri=True)) as connection:
            import_sqlite_world(connection, store.read_source_model(), validate=False)
    except SQLiteWorldImportError:
        return True
    return False


def _refresh_source_facts(job: HarnessJob, source: Path, output: Path) -> list[str]:
    """Describe the agent version being certified and the world the environment saved.

    Both are recorded for whichever agent version last built or edited the environment.
    Entrypoints left behind by tools the contract no longer has are dropped.
    """
    from .amend import _persist
    from .cli import after_contract, source_model_is_stale
    from .understand import load

    notes: list[str] = []
    contract = load(output)
    if contract is not None:
        kept = contract.tool_names()
        stale = sorted(
            {e.tool for e in contract.tool_entrypoints if e.tool not in kept}
        )
        if stale:
            contract.tool_entrypoints = [
                e for e in contract.tool_entrypoints if e.tool in kept
            ]
            _persist(contract, output)
            notes.append(
                "entrypoints dropped for tools the contract no longer has: "
                + ", ".join(stale)
            )
    stale_model = source_model_is_stale(output, source)
    world_ir = output / "generic-harness" / "world-ir.json"
    drop_world_ir = (
        world_ir.is_file()
        and (output / "world.sqlite").is_file()
        and (stale_model or _saved_world_outgrew_world_ir(output))
    )
    if stale_model:
        after_contract(output, source, job)
        notes.append("source model rebuilt for this agent version's source")
    if drop_world_ir:
        # The saved world is imported again at seed, against this source's schema.
        world_ir.unlink()
        notes.append("World IR dropped, world.sqlite is seeded instead")
    return notes


def _discard_certify_report(path: Path) -> None:
    """A failed run leaves no report behind."""
    from .authoring_runtime_validation import RuntimeValidationError

    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeValidationError(
            "environment",
            f"Certify-only could not remove {CERTIFY_REPORT}: {_one_line(exc)}",
        ) from exc


def _certify_only(args: argparse.Namespace) -> int:
    """Certify an assembled environment and suite, then seal its bundle. No model calls.

    The provider profile is not read: only authoring stages consume it, and validation and
    bundling work from contract.json and the scenarios. No provider API is called.
    """
    from .authoring_runtime_validation import (
        RuntimeValidationError,
        validate_and_repair,
        validate_once,
    )
    from .bundle_author_v2 import _reuse_certified_bundle
    from .repair_controller import RepairBudgets, RepairController

    source = args.source.resolve()
    output = args.output.resolve()
    bundle_output = args.bundle_output.resolve()
    report_path = output / CERTIFY_REPORT
    try:
        if args.target_secrets is not None:
            # One-shot provider credential; certify-only never uses it, so drop it before any
            # environment process starts.
            args.target_secrets.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeValidationError(
            "environment", f"Certify-only could not clear stale files: {_one_line(exc)}"
        ) from exc
    try:
        job = HarnessJob.model_validate_json(args.job.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeValidationError(
            "environment", f"Certify-only: invalid job document: {_one_line(exc)}"
        ) from exc
    if job.metadata.get("generic_harness_v1") is not True:
        raise RuntimeValidationError(
            "environment", "Certify-only requires a generic_harness_v1 job"
        )
    try:
        notes, report = _prune_suite_to_catalogue(output)
        _persist_authored_scenario_count(args.job, job, output)
        runtime_job = HarnessJob.model_validate_json(
            args.job.read_text(encoding="utf-8")
        )
        stale = _drop_stale_invariants(source, output)
        notes += [
            f"source-data invariant {name} dropped, it quotes a line this agent "
            "version's source no longer has"
            for name in stale
        ]
        if stale:
            report["stale_invariants"] = stale
        notes += _prepare_source_data_invariants(runtime_job, output)
        notes += _refresh_source_facts(runtime_job, source, output)
    except RuntimeValidationError:
        raise
    except Exception as exc:
        raise RuntimeValidationError(
            "environment", f"Certify-only preparation failed: {_one_line(exc)}"
        ) from exc
    for note in notes:
        print(f"certify-only: {note}", file=sys.stderr, flush=True)

    async def refuse_repair(phase: str, guidance: str) -> int:
        raise RuntimeValidationError(phase, "Certify-only does not repair")

    try:
        asyncio.run(
            validate_and_repair(
                runtime_job,
                source,
                output,
                validate=partial(validate_once, reuse_invariants_only=True),
                repair=refuse_repair,
                controller=RepairController(
                    RepairBudgets(
                        compiler_candidates=0,
                        environment_patches=0,
                        scenario_patches=0,
                    )
                ),
            )
        )
    except RuntimeValidationError:
        raise
    except Exception as exc:
        raise RuntimeValidationError(
            "environment", f"Certify-only validation failed: {_one_line(exc)}"
        ) from exc
    try:
        # Written before sealing: the platform's fill gate keys on the sealed certificate.
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        _discard_certify_report(report_path)
        raise RuntimeValidationError(
            "environment",
            f"Certify-only could not write {CERTIFY_REPORT}: {_one_line(exc)}",
        ) from exc
    try:
        bundle_output.parent.mkdir(parents=True, exist_ok=True)
        # Only the exact bytes validation froze; never a fresh compile.
        sealed = _reuse_certified_bundle(
            source=source, job=runtime_job, authoring=output, output=bundle_output
        )
    except Exception as exc:
        _discard_certify_report(report_path)
        raise RuntimeValidationError(
            "bundle", f"Certified bundle could not be sealed: {_one_line(exc)}"
        ) from exc
    if sealed is None:
        _discard_certify_report(report_path)
        raise RuntimeValidationError(
            "bundle", "Certify-only: validation left no certified bundle to seal"
        )
    return 0


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
    parser.add_argument(
        "--certify-only",
        action="store_true",
        help="Certify the assembled authoring directory and seal its bundle; author nothing",
    )
    parser.add_argument(
        "--bundle-output",
        type=Path,
        help="Directory the sealed bundle is written to in --certify-only mode",
    )
    args = parser.parse_args(argv)
    if args.certify_only:
        if args.bundle_output is None:
            parser.error("--certify-only requires --bundle-output")
        return _certify_only(args)

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
        guidance=[],
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
