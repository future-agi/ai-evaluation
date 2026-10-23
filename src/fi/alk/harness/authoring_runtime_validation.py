"""Validate generated setup against the actual hosted runtime before accepting authoring.

This is setup proof, not a claim that a reference tool trajectory executed. Actual
agent tool execution remains evidence collected during the calls.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _artifact_digest(path: Path) -> str:
    """Hash one authoring artifact without depending on its absolute checkout path."""

    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    elif path.is_dir():
        for item in sorted(
            candidate for candidate in path.rglob("*") if candidate.is_file()
        ):
            relative = item.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(item.read_bytes())
    return "sha256:" + digest.hexdigest()


def _runtime_snapshot(job) -> str | None:
    metadata = getattr(job, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("daytona_snapshot", "sandbox_snapshot", "snapshot"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return os.environ.get("ALK_DAYTONA_SNAPSHOT") or None


def _world_isolation_status(build_output: dict):
    """Classify the isolation proof at the runtime's enforced parallelism."""

    from .certification import CheckStatus

    if build_output.get("conformance") is True:
        return CheckStatus.PASSED
    if (
        build_output.get("degrade_reason") == "fixed_port"
        and build_output.get("effective_parallelism") == 1
    ):
        return CheckStatus.NOT_APPLICABLE
    return CheckStatus.NOT_RUN


def _write_runtime_evidence(
    *,
    job,
    authoring: Path,
    manifest,
    count: int,
    external_provider: bool,
    tool_report,
    action_report,
    reset_equivalence,
    world_isolation,
) -> None:
    """Seal facts which would otherwise disappear with the validation sandbox."""

    from .certification import (
        CertificationChecks,
        CheckStatus,
        GenericHarnessArtifactStore,
        RuntimeValidationEvidence,
    )

    store = GenericHarnessArtifactStore(authoring / "generic-harness")
    limitations = [
        "runtime-only agent tool effects are verified during calls, not setup certification",
    ]
    if external_provider:
        source_schema_hash = manifest.provenance.source_digest
        world_ir_hash = _artifact_digest(Path("__external_provider_world__"))
        compiler_version = "external-provider-black-box"
        schema_and_seed = CheckStatus.NOT_APPLICABLE
        source_invariants = CheckStatus.NOT_APPLICABLE
        limitations.append(
            "external provider state and tool implementations were not available for local inspection"
        )
    if not external_provider:
        source_model = store.read_source_model()
        world_ir = store.read_world_ir()
        source_schema_hash = source_model.fingerprint
        world_ir_hash = world_ir.fingerprint
        from .generic_pipeline import default_compiler_registry

        compiler_version = (
            default_compiler_registry().resolve(source_model.engine).version
        )
        schema_and_seed = CheckStatus.PASSED
        source_invariants = CheckStatus.PASSED
    if world_isolation is CheckStatus.NOT_RUN:
        limitations.append(
            "concurrent world isolation was not applicable or runtime parallelism was safely degraded"
        )
    elif world_isolation is CheckStatus.NOT_APPLICABLE and not external_provider:
        limitations.append(
            "concurrent world isolation is not applicable because a fixed-port runtime is "
            "enforced at single-world parallelism; reset equivalence remains certified"
        )
    if action_report.executable != action_report.total:
        limitations.append(
            f"{action_report.total - action_report.executable} actions require runtime or an external sandbox probe"
        )
    if action_report.certified != action_report.executable:
        limitations.append(
            f"{action_report.executable - action_report.certified} safe action probes failed at the runtime boundary"
        )

    contract = authoring / "contract.json"
    scenarios = authoring / "scenarios"
    if not scenarios.exists():
        scenarios = authoring / "scenario"
    evidence = RuntimeValidationEvidence(
        source_digest=manifest.provenance.source_digest,
        source_schema_hash=source_schema_hash,
        world_ir_hash=world_ir_hash,
        compiler_version=compiler_version,
        bundle_digest=manifest.digest,
        contract_hash=_artifact_digest(contract),
        scenario_set_hash=_artifact_digest(scenarios),
        checks=CertificationChecks(
            static=CheckStatus.PASSED,
            schema_and_seed=schema_and_seed,
            processes=CheckStatus.PASSED,
            source_invariants=source_invariants,
            scenario_setup_ready=f"{count}/{count}",
            tool_contract=(
                f"{tool_report.certified_or_runtime_only}/{tool_report.total}"
            ),
            action_behavior=f"{action_report.certified}/{action_report.executable}",
            reset_equivalence=reset_equivalence,
            world_isolation=world_isolation,
        ),
        limitations=tuple(limitations),
    )
    store.write_runtime_evidence(evidence)
    store.write_tool_certification(tool_report)
    store.write_action_certification(action_report)


def _database_lanes() -> int:
    """One lane per CPU by default: each check starts a worker inside its time budget."""
    from .process_runtime import _cgroup_cpu_quota

    configured = os.environ.get("ALK_VALIDATION_DB_LANES")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            raise ValueError(
                f"ALK_VALIDATION_DB_LANES must be a whole number, not {configured!r}"
            ) from None
    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    quota = _cgroup_cpu_quota()
    return max(1, min(cpus, int(quota)) if quota else cpus)


def _world_state_digest(world) -> str:
    """Return a value-level baseline fingerprint without persisting world contents."""

    state = world.read_only().state()
    canonical = {
        table: sorted(
            rows,
            key=lambda row: json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ),
        )
        for table, rows in sorted(state.items())
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _freeze_certified_bundle(bundle: Path, authoring: Path) -> None:
    """Persist the exact bundle bytes exercised by runtime validation."""

    certified_bundle = authoring / "generic-harness" / "certified-bundle"
    shutil.rmtree(certified_bundle, ignore_errors=True)
    shutil.copytree(bundle, certified_bundle)


def _setup_check_keys(bundle: Path, invariants: list, baseline_digest: str) -> list[str]:
    """One key per scenario folder, in load order, over everything its setup check depends on."""

    shared = hashlib.sha256()
    shared.update(baseline_digest.encode("utf-8"))
    shared.update(json.dumps(invariants, sort_keys=True, default=str).encode("utf-8"))
    for item in sorted(path for path in bundle.rglob("*") if path.is_file()):
        relative = item.relative_to(bundle)
        if relative.parts[0] in {"scenarios", "manifest.json"}:
            continue
        shared.update(relative.as_posix().encode("utf-8") + b"\0")
        shared.update(item.read_bytes())
    prefix = shared.hexdigest()
    return [
        "sha256:" + hashlib.sha256((prefix + _artifact_digest(folder)).encode()).hexdigest()
        for folder in sorted((bundle / "scenarios").iterdir())
        if folder.is_dir()
    ]


def _make_local_seed_files_readable(bundle: Path) -> None:
    """Allow an unprivileged data service to read public, generated seed inputs.

    The bundle is created in a private controller workspace.  PostgreSQL is deliberately
    launched as svc-data, so its ``psql -f`` needs search permission on the seed directory
    and read permission on the generated files.  Do not relax modes elsewhere in the bundle:
    credential handoffs and submitted source are outside this seed-only boundary.
    """

    seed = bundle / "seed"
    if not seed.is_dir():
        return
    bundle.chmod(bundle.stat().st_mode | 0o055)
    seed.chmod(seed.stat().st_mode | 0o055)
    for path in seed.rglob("*"):
        if path.is_symlink():
            continue
        path.chmod(path.stat().st_mode | (0o055 if path.is_dir() else 0o044))


def _write_generic_certificate(job, authoring: Path, repairs) -> None:
    from .certification import (
        CertificationAuthoring,
        CertificationCompiler,
        CertificationRuntime,
        CertificationSource,
        CertificationStatus,
        GenericHarnessArtifactStore,
        HarnessCertification,
    )

    store = GenericHarnessArtifactStore(authoring / "generic-harness")
    evidence = store.read_runtime_evidence()
    source = getattr(job, "source", None)
    certificate = HarnessCertification.create(
        status=CertificationStatus.CERTIFIED,
        source=CertificationSource(
            repository=getattr(source, "repository", None),
            commit=getattr(source, "commit_sha", None),
            digest=evidence.source_digest,
            schema_hash=evidence.source_schema_hash,
        ),
        authoring=CertificationAuthoring(
            contract_hash=evidence.contract_hash,
            world_ir_hash=evidence.world_ir_hash,
            scenario_set_hash=evidence.scenario_set_hash,
        ),
        compiler=CertificationCompiler(
            version=evidence.compiler_version,
            bundle_digest=evidence.bundle_digest,
        ),
        runtime=CertificationRuntime(
            snapshot=_runtime_snapshot(job),
            validation_attempts=len(repairs.decisions),
        ),
        checks=evidence.checks,
        repairs=repairs,
        limitations=evidence.limitations,
    )
    store.write_certification(certificate)
    (authoring / "runtime-validation.json").write_text(
        certificate.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def _generic_candidate_hash(source: Path, authoring: Path) -> str:
    from .provision import source_fingerprint

    digest = hashlib.sha256()
    digest.update(source_fingerprint(source).encode("ascii"))
    for path in sorted(item for item in authoring.rglob("*") if item.is_file()):
        relative = path.relative_to(authoring).as_posix()
        if relative == "runtime-validation.json":
            continue
        if relative.startswith("generic-harness/") and relative not in {
            "generic-harness/source-model.json",
            "generic-harness/world-ir.json",
        }:
            continue
        digest.update(f"authoring/{relative}\n".encode("utf-8"))
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _fallback_diagnostic(error):
    from .diagnostics import HarnessDiagnostic
    from .job import HarnessStage

    detail = str(error)
    normalized = detail.casefold()
    provider_lifecycle_failure = "provider_lifecycle/" in normalized
    if (
        provider_lifecycle_failure
        and "http error 408" not in normalized
        and "http error 429" not in normalized
        and re.search(r"http error 4\d\d\b", normalized)
    ):
        # A lifecycle command is submitted source. A definitive external 4xx means
        # that source/config/credential was rejected; changing generated world data
        # cannot make it valid. Keep 408 and 429 in the transient bucket below.
        code = "external_service_request_rejected"
        stage = HarnessStage.CONNECTING_AGENT
    elif provider_lifecycle_failure and (
        re.search(r"http error 5\d\d\b", normalized)
        or "http error 408" in normalized
        or "http error 429" in normalized
        or any(
            marker in normalized
            for marker in (
                "clientconnectordnserror",
                "name resolution",
                "temporary failure in name resolution",
                "connection reset",
                "connection refused",
                "timed out",
                "timeout",
            )
        )
    ):
        code = "external_service_unavailable"
        stage = HarnessStage.CONNECTING_AGENT
    elif error.phase == "scenarios":
        code = "ready_condition_invalid"
        stage = HarnessStage.VALIDATING_SCENARIOS
    elif error.phase == "infrastructure":
        code = "process_dependency_timeout"
        stage = HarnessStage.BUILDING_ENVIRONMENT
    elif error.phase == "runtime" or _looks_like_runtime_plan_failure(normalized):
        code = "generated_runtime_plan_invalid"
        stage = HarnessStage.BUILDING_ENVIRONMENT
    else:
        code = "generated_setup_invalid"
        stage = HarnessStage.VALIDATING_ENVIRONMENT
    return HarnessDiagnostic.create(
        stage=stage,
        component="runtime_validation",
        code=code,
        message=str(error),
    )


def _looks_like_runtime_plan_failure(detail: str) -> bool:
    """Recognize failures in the generated execution plan, not agent behavior/data.

    The concrete exception types are deliberately allowed to vary across Docker, Compose,
    process and HTTP runtimes.  These stable boundary messages are produced before a scenario
    can exercise the agent and are therefore safe to route to repository/runtime inspection.
    """

    return any(
        marker in detail
        for marker in (
            "generated_runtime_",
            "no runnable shipped entrypoint",
            "dependency manifest",
            "failed to build",
            "docker build",
            "process exited",
            "process failed",
            "readiness probe",
            "health check",
            "healthcheck",
            "entrypoint",
            "module not found",
            "modulenotfounderror",
            "cannot find module",
            "command not found",
        )
    )


def _repair_guidance(diagnostics) -> str:
    """Render the complete secret-safe evidence needed for one constrained repair."""

    rendered = []
    for item in sorted(
        diagnostics, key=lambda diagnostic: (diagnostic.code, diagnostic.component)
    ):
        location = (
            item.location.model_dump(mode="json", exclude_none=True)
            if item.location is not None
            else None
        )
        parts = [f"{item.code}@{item.component}"]
        if item.repair_strategy:
            parts.append(f"strategy={item.repair_strategy}")
        if location:
            parts.append(
                "location="
                + json.dumps(location, sort_keys=True, separators=(",", ":"))
            )
        # HarnessDiagnostic.create already redacts this value. Keep the model input
        # bounded as a second line of defence against oversized provider traces.
        parts.append(f"evidence={item.redacted_message[:4000]}")
        rendered.append(" | ".join(parts))
    return "\n".join(rendered)[:12000]


def _needs_runtime_plan_repair(diagnostics) -> bool:
    """Route execution construction failures away from world-data authoring."""

    runtime_codes = {
        "generated_runtime_plan_invalid",
        "process_dependency_timeout",
        "tool_endpoint_unreachable",
    }
    return bool(diagnostics) and all(
        diagnostic.code in runtime_codes for diagnostic in diagnostics
    )


class RuntimeValidationError(RuntimeError):
    def __init__(self, phase: str, detail: str, *, diagnostics=()):
        self.phase = phase
        self.diagnostics = tuple(diagnostics)
        super().__init__(detail)


class _GeneratedWorldActionAdapter:
    """Expose the generated world through the generic action-probe boundary."""

    def __init__(self, world) -> None:
        self.world = world
        self.baseline = world.checkpoint()

    def invoke(self, action, arguments):
        from .action_certification import ActionInvocation
        from .world.handle import WorldUnavailable

        try:
            call = self.world.call(action.name, arguments)
        except WorldUnavailable as exc:
            # A source-owned callable that is reachable only through the running agent is not
            # an executable action at this validation boundary.  Reporting NOT_RUN keeps the
            # evidence honest; treating the absent direct adapter as a broken customer action
            # made every repository-hosted tool fail before scenarios could exercise it.
            return ActionInvocation(not_run=True, error=str(exc))
        return ActionInvocation(
            result=call.result,
            refused=bool(call.refused),
            error=(call.error or "action execution failed")
            if not call.ok and not call.refused
            else None,
        )

    def checkpoint(self) -> str:
        return _world_state_digest(self.world)

    def reset(self) -> None:
        self.world.revert(self.baseline)


async def validate_once(
    job,
    source: Path,
    authoring: Path,
    *,
    secrets_path=Path("/run/futureagi/secrets.json"),
    local_runtime: bool = False,
    review=None,
) -> int:
    from . import outbound
    from .bundle_author_v2 import author_bundle_v2
    from .contract import AgentContract
    from .hosted_entrypoint import (
        ProcessWorldFactory,
        _resolve_hosted_public_url,
        job_secret_purposes,
    )
    from .hosted_scheduler import _classify_ready, _run_phase
    from .job import ProviderExecutionMode, SourceKind
    from .process_preflight import preflight_bundle
    from .process_runtime import (
        ProcessRuntimeProvider,
        default_user_resolver,
        fixed_sandbox_user_resolver,
    )
    from .scenario_source import does_nothing, load_scenarios
    from .source_data_invariants import author_invariants, check_invariants
    from .tool_certification import ToolAvailability, certify_tool_inventory

    external_provider = (
        getattr(getattr(job, "source", None), "kind", None) is SourceKind.PROVIDER
        and getattr(getattr(job, "agent", None), "mode", None)
        is ProviderExecutionMode.CONNECT_ONLY
    )
    generic = bool(
        isinstance(getattr(job, "metadata", None), dict)
        and job.metadata.get("generic_harness_v1") is True
    )
    if generic:
        from .certification import CheckStatus, GenericHarnessArtifactStore

        artifact_root = authoring / "generic-harness"
        # A prior certified bundle is an output of validation, never an input to a fresh
        # validation attempt.  Remove it before compiling so repairs cannot accidentally
        # validate yesterday's environment.
        shutil.rmtree(artifact_root / "certified-bundle", ignore_errors=True)
        (artifact_root / GenericHarnessArtifactStore.RUNTIME_EVIDENCE).unlink(
            missing_ok=True
        )
        (artifact_root / GenericHarnessArtifactStore.CERTIFICATION).unlink(
            missing_ok=True
        )
        (authoring / "runtime-validation.json").unlink(missing_ok=True)

    # The real execution consumes its credential file. Validation gets a private copy,
    # with the same purpose map, so it cannot destroy the execution handoff.
    temporary_parent = authoring.parent
    if local_runtime:
        # The local sandbox itself runs in Docker while ProcessRuntimeProvider may ask the
        # host Docker daemon to build/start submitted Compose services.  A path in the sandbox's
        # private artifact volume is not visible to that daemon.  The upload root is deliberately
        # bind-mounted at the same absolute path on both sides, so use it for disposable runtime
        # candidates when present.  A plain local CLI has no such variable and keeps the historic
        # authoring-adjacent temporary directory.
        shared_root = os.environ.get("ALK_SANDBOX_UPLOAD_ROOT", "").strip()
        if shared_root:
            temporary_parent = Path(shared_root).expanduser().resolve()
            temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="runtime-validation-", dir=temporary_parent
    ) as root:
        work = Path(root)
        runtime = getattr(job, "runtime", None)
        (work / "job.json").write_text(
            json.dumps(
                {
                    "runtime": {
                        "cpu_units": getattr(runtime, "cpu_units", 1),
                        "memory_mb": getattr(runtime, "memory_mb", 512),
                    }
                }
            )
        )
        if local_runtime:
            # Child processes drop to svc-* identities. They need search permission on the
            # disposable workspace parent in order to reach their individually chowned trees;
            # secrets below retain mode 0600 and are injected by the controller.
            work.chmod(0o755)
        secrets = work / "secrets.json"
        secrets.write_bytes(secrets_path.read_bytes())
        secrets.chmod(0o600)
        secret_values = tuple(
            str(value) for value in json.loads(secrets.read_text()).values()
        )
        capabilities = (
            None if local_runtime else outbound.load_capabilities(unlink=False)
        )
        transport = outbound.RequestsTransport()
        provider = ProcessRuntimeProvider(
            secrets_path=secrets,
            secret_purpose_map=job_secret_purposes(job),
            # A local parity run shares the host/container process namespace and must resolve
            # the same unprivileged identities installed in its image. E2B, however, launches
            # the guest as one fixed OS identity and does not grant setuid/chown capabilities.
            user_resolver=(
                default_user_resolver if local_runtime else fixed_sandbox_user_resolver
            ),
            require_declared_user=False,
            public_url_resolver=(
                None
                if capabilities is None
                else lambda port, ttl: _resolve_hosted_public_url(
                    capabilities, transport, port=port, expires_in_seconds=ttl
                )
            ),
            provider_attempt_id=(
                capabilities.attempt_id if capabilities is not None else None
            ),
            provider_expires_at=(
                capabilities.expires_at if capabilities is not None else None
            ),
            generic_artifact_root=(authoring / "generic-harness") if generic else None,
        )
        # Parallel runtime instances inside the one sandbox, each with its own ports and databases.
        requested_lanes = max(1, int(os.environ.get("ALK_VALIDATION_INSTANCES", "4") or 4))
        # Setup checks touch only the world's database, so a single-database world checks each
        # scenario on its own clone of the sealed baseline instead of a whole agent stack.
        database_lanes = _database_lanes()
        executor = ThreadPoolExecutor(
            max_workers=max(requested_lanes, database_lanes),
            thread_name_prefix="runtime-validation",
        )
        started = time.monotonic()
        phase = "runtime"
        try:
            bundle = work / "bundle"
            manifest = await asyncio.to_thread(
                author_bundle_v2,
                source=source,
                job=job,
                authoring=authoring,
                output=bundle,
            )
            if local_runtime:
                _make_local_seed_files_readable(bundle)
            validation_parallelism = (
                1
                if external_provider or review is not None
                else max(requested_lanes, 2 if generic else 1)
            )
            preflight_bundle(
                bundle,
                manifest,
                parallelism=validation_parallelism,
                secret_refs=job_secret_purposes(job),
            )
            runtimes = await provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle,
                work_directory=work,
                instances=validation_parallelism,
            )
            if review is not None:
                # Only the source data review, against this world, while scenarios are written.
                await provider.reset(runtimes[0], work_directory=work)
                world = await ProcessWorldFactory(work).create(
                    runtimes[0], rng=random.Random(job.seed or 0)
                )
                await review(world.read_only(), runtimes[0].endpoints)
                return 0
            # From here onward the generated build/start plan has succeeded. Failures belong to
            # the generated world, tool boundary, or scenarios rather than runtime discovery.
            phase = "environment"
            tool_report = None
            action_report = None
            world_isolation = None
            if generic:
                from .action_certification import certify_actions

                contract = AgentContract.model_validate_json(
                    (authoring / "contract.json").read_text(encoding="utf-8")
                )
                tool_report = certify_tool_inventory(
                    contract, bundle, external_provider=external_provider
                )
                rejected_tools = [
                    item.tool
                    for item in tool_report.tools
                    if item.availability is ToolAvailability.REJECTED
                ]
                if rejected_tools:
                    from .diagnostics import DiagnosticLocation, HarnessDiagnostic
                    from .job import HarnessStage

                    diagnostics = tuple(
                        HarnessDiagnostic.create(
                            stage=HarnessStage.VALIDATING_ENVIRONMENT,
                            component="tool_contract",
                            code="tool_schema_mismatch",
                            message=f"{name}: tool has no single runnable implementation seam",
                            location=DiagnosticLocation(tool=name),
                        )
                        for name in sorted(rejected_tools)
                    )
                    raise RuntimeValidationError(
                        "environment",
                        "Tool contract certification rejected: "
                        + ", ".join(sorted(rejected_tools)),
                        diagnostics=diagnostics,
                    )
                source_model = GenericHarnessArtifactStore(
                    authoring / "generic-harness"
                ).read_source_model()
                action_report = certify_actions(source_model.actions, adapter=None)
                build_output = json.loads(
                    (work / "artifacts" / "build.json").read_text(encoding="utf-8")
                )
                conformance = build_output.get("conformance")
                if conformance is False:
                    raise RuntimeValidationError(
                        "environment",
                        "Concurrent world isolation conformance failed: "
                        + str(
                            build_output.get("conformance_reason") or "unknown reason"
                        ),
                    )
                # A source process that genuinely hard-codes its listen port cannot coexist with
                # a second copy in the same network namespace. The process provider enforces that
                # physical constraint by returning exactly one world. There is no concurrent
                # boundary to certify at that supported parallelism; clean lifecycle reuse is
                # still proved independently by reset_equivalence.
                world_isolation = _world_isolation_status(build_output)
            factory = ProcessWorldFactory(work)
            runtime = runtimes[0]
            lanes = len(runtimes) or 1
            phase = "scenarios"
            scenarios = await asyncio.to_thread(load_scenarios, bundle)
            if len(scenarios) != job.scenario_count:
                raise RuntimeValidationError(
                    phase, "Runtime scenario count differs from the requested count"
                )
            targets: list = list(runtimes)
            if (
                generic
                and not external_provider
                and await provider.clone_world_database(runtime, "lane0") is not None
            ):
                targets = [
                    f"lane{index}"
                    for index in range(max(1, min(database_lanes, len(scenarios))))
                ]
                lanes = len(targets)

            baseline_state_digest: str | None = None
            cleared: list = []

            # A failure under an in-place reset is re-checked once against a full restart.
            in_place = os.environ.get("ALK_VALIDATION_RESET_IN_PLACE", "1") != "0"

            async def check_setups(invariants, suite=None):
                suite = scenarios if suite is None else suite
                failures: list[str] = []

                async def lane(own, against):
                    for scenario in own:
                        try:
                            await check_setup(scenario, invariants, against)
                        except RuntimeValidationError as exc:
                            failures.append(str(exc))
                        else:
                            cleared.append(scenario)

                await asyncio.gather(
                    *(
                        lane(suite[index::lanes], targets[index])
                        for index in range(lanes)
                        if suite[index::lanes]
                    )
                )
                if failures:
                    raise RuntimeValidationError(phase, "\n".join(sorted(failures)))

            async def check_setup(scenario, invariants, against=None):
                try:
                    await check_setup_once(scenario, invariants, against, in_place)
                except Exception:
                    if not in_place:
                        raise
                    await check_setup_once(scenario, invariants, against, False)

            async def check_setup_once(scenario, invariants, against, keep_processes):
                nonlocal baseline_state_digest
                if isinstance(against, str):
                    against = await provider.clone_world_database(runtime, against)
                else:
                    against = runtime if against is None else against
                    await provider.reset(
                        against, work_directory=work, keep_processes=keep_processes
                    )
                world = await factory.create(against, rng=random.Random(job.seed or 0))
                if generic and not external_provider:
                    current_digest = _world_state_digest(world)
                    if baseline_state_digest is None:
                        baseline_state_digest = current_digest
                    elif current_digest != baseline_state_digest:
                        raise RuntimeValidationError(
                            phase,
                            f"{scenario.scenario_key}: reset did not restore the certified baseline",
                        )
                for name, fn, target, timeout in (
                    ("setup", scenario.setup, world, 30.0),
                    ("ready", scenario.ready, world.read_only(), 15.0),
                ):
                    result = await _run_phase(
                        fn, target, timeout=timeout, phase=name, executor=executor
                    )
                    if result.failure:
                        raise RuntimeValidationError(
                            phase, f"{scenario.scenario_key}: {name}: {result.failure}"
                        )
                    if name == "setup":
                        try:
                            await check_invariants(
                                world.read_only(),
                                invariants,
                                scenario_key=scenario.scenario_key,
                            )
                        except Exception as exc:
                            raise RuntimeValidationError(
                                phase, f"{scenario.scenario_key}: source data: {exc}"
                            ) from exc
                    if name == "ready":
                        verdict = _classify_ready(result.value)
                        if verdict.broken or not verdict.held:
                            raise RuntimeValidationError(
                                phase,
                                f"{scenario.scenario_key}: ready precondition did not hold",
                            )

            print(
                f"runtime validation: {len(scenarios)} scenarios across {lanes} lane"
                f"{'s' if lanes != 1 else ''}, environment ready in "
                f"{time.monotonic() - started:.0f}s",
                flush=True,
            )
            invariants: list = []
            if external_provider:
                # A connect-only provider owns its state and executes its tools outside
                # this sandbox. There is no harness-owned source database to probe or
                # seed, so source-data invariant review would invent a local environment.
                print(
                    "runtime validation: external provider black-box mode; "
                    "skipping local source-data invariant review",
                    flush=True,
                )
                active = [
                    scenario
                    for scenario in scenarios
                    if not (does_nothing(scenario.setup) and does_nothing(scenario.ready))
                ]
                print(
                    f"runtime validation: {len(scenarios) - len(active)} of {len(scenarios)} "
                    "scenarios set up nothing and hold by construction; walking the rest",
                    flush=True,
                )
                await check_setups(invariants, active)
                if generic:
                    _write_runtime_evidence(
                        job=job,
                        authoring=authoring,
                        manifest=manifest,
                        count=len(scenarios),
                        external_provider=True,
                        tool_report=tool_report,
                        action_report=action_report,
                        reset_equivalence=CheckStatus.NOT_APPLICABLE,
                        world_isolation=CheckStatus.NOT_APPLICABLE,
                    )
                    _freeze_certified_bundle(bundle, authoring)
                return len(scenarios)
            phase = "environment"
            await provider.reset(runtime, work_directory=work)
            baseline = await factory.create(runtime, rng=random.Random(job.seed or 0))
            print("runtime validation: reviewing source data invariants", flush=True)
            invariants[:] = await author_invariants(
                source, authoring, baseline.read_only(), endpoints=runtime.endpoints
            )
            # Review probes may have effects; none belongs in the test baseline.
            await provider.reset(runtime, work_directory=work)
            baseline = await factory.create(runtime, rng=random.Random(job.seed or 0))
            await check_invariants(baseline.read_only(), invariants)
            if generic:
                from .action_certification import certify_actions

                source_model = GenericHarnessArtifactStore(
                    authoring / "generic-harness"
                ).read_source_model()
                action_report = certify_actions(
                    source_model.actions,
                    adapter=_GeneratedWorldActionAdapter(baseline),
                )
                GenericHarnessArtifactStore(
                    authoring / "generic-harness"
                ).write_action_certification(action_report)
                from .action_certification import ActionProbeStatus
                from .diagnostics import DiagnosticLocation, HarnessDiagnostic
                from .job import HarnessStage

                failed_actions = [
                    item
                    for item in action_report.actions
                    if item.status is ActionProbeStatus.FAILED
                ]
                if failed_actions:
                    raise RuntimeValidationError(
                        "environment",
                        "Action behavior certification failed: "
                        + ", ".join(item.action for item in failed_actions),
                        diagnostics=tuple(
                            HarnessDiagnostic.create(
                                stage=HarnessStage.VALIDATING_ENVIRONMENT,
                                component="action_behavior",
                                code="action_probe_failed",
                                message=f"{item.action}: {item.reason}",
                                location=DiagnosticLocation(tool=item.action),
                            )
                            for item in failed_actions
                        ),
                    )
                # Probes may mutate the disposable world. Restore before scenario checks.
                await provider.reset(runtime, work_directory=work)
                baseline = await factory.create(
                    runtime, rng=random.Random(job.seed or 0)
                )
            phase = "scenarios"
            walk = time.monotonic()
            pending = scenarios
            if generic:
                # A scenario that already passed against this exact baseline, bundle and
                # invariants passes again, so a repair re-checks only what failed or changed.
                baseline_state_digest = _world_state_digest(baseline)
                passed_path = authoring / "generic-harness" / "setup-checks-passed.json"
                passed = (
                    set(json.loads(passed_path.read_text(encoding="utf-8")))
                    if passed_path.is_file()
                    else set()
                )
                key_of = dict(
                    zip(
                        map(id, scenarios),
                        _setup_check_keys(bundle, invariants, baseline_state_digest),
                    )
                )
                pending = [item for item in scenarios if key_of[id(item)] not in passed]
            try:
                await check_setups(invariants, pending)
            finally:
                if generic:
                    passed.update(key_of[id(item)] for item in cleared)
                    passed_path.write_text(
                        json.dumps(sorted(passed)) + "\n", encoding="utf-8"
                    )
            print(
                f"runtime validation: setup pass {time.monotonic() - walk:.0f}s for "
                f"{len(pending)} of {len(scenarios)} scenarios with {len(invariants)} "
                f"invariants, complete in {time.monotonic() - started:.0f}s",
                flush=True,
            )
            if generic:
                await provider.reset(runtime, work_directory=work)
                reset_world = await factory.create(
                    runtime, rng=random.Random(job.seed or 0)
                )
                if (
                    baseline_state_digest is None
                    or _world_state_digest(reset_world) != baseline_state_digest
                ):
                    raise RuntimeValidationError(
                        "environment",
                        "Final reset did not restore the certified baseline",
                    )
                _write_runtime_evidence(
                    job=job,
                    authoring=authoring,
                    manifest=manifest,
                    count=len(scenarios),
                    external_provider=False,
                    tool_report=tool_report,
                    action_report=action_report,
                    reset_equivalence=CheckStatus.PASSED,
                    world_isolation=world_isolation,
                )
                # Freeze the exact bytes that passed the real runtime checks.  Recompiling from
                # the authoring archive in the execution sandbox can otherwise drift on inputs
                # that are deliberately outside the source fingerprint (generated build/cache
                # artifacts are one example).  The consumer re-verifies the certificate and all
                # authoring fingerprints before accepting this directory.
                _freeze_certified_bundle(bundle, authoring)
            return len(scenarios)
        except RuntimeValidationError as exc:
            raise RuntimeValidationError(
                exc.phase,
                outbound.redact_outbound_text(
                    str(exc), extra_secret_values=secret_values
                ),
                diagnostics=exc.diagnostics,
            ) from None
        except Exception as exc:
            if "CERTIFICATE_VERIFY_FAILED" in str(exc):
                # Generated data cannot repair the infrastructure trust store.
                phase = "infrastructure"
            raise RuntimeValidationError(
                phase,
                outbound.redact_outbound_text(
                    f"{type(exc).__name__}: {exc}", extra_secret_values=secret_values
                ),
                diagnostics=tuple(getattr(exc, "diagnostics", ())),
            ) from None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            await provider.close(work_directory=work)


def _invariants_the_ir_still_breaks(world, authoring: Path) -> list[str]:
    """Declared invariants the patched rows still violate, judged without building anything."""
    from .source_data_invariants import ARTIFACT, violations_in_ir

    document = authoring / ARTIFACT
    if not document.is_file():
        return []
    try:
        checks = json.loads(document.read_text(encoding="utf-8")).get("checks") or []
        return violations_in_ir(world, checks)
    except Exception:  # noqa: BLE001 - a check we cannot run is the real environment's to judge
        return []


async def validate_and_repair(
    job,
    source: Path,
    authoring: Path,
    *,
    validate=validate_once,
    repair=None,
    controller=None,
    sleeper=asyncio.sleep,
) -> None:
    """Two repairs per phase; environment repairs cannot exhaust setup's budget."""
    generic = bool(
        job is not None
        and isinstance(getattr(job, "metadata", None), dict)
        and job.metadata.get("generic_harness_v1") is True
    )
    if repair is None:

        async def repair(phase, guidance):
            from .cli import _build, _scenarios

            if phase == "environment":
                artifact_root = authoring / "generic-harness"
                if generic and _needs_runtime_plan_repair(_active_repair_diagnostics):
                    from .contract import AgentContract
                    from .runtime_repair import request_runtime_plan_patch

                    contract = AgentContract.model_validate_json(
                        (authoring / "contract.json").read_text(encoding="utf-8")
                    )
                    return await request_runtime_plan_patch(
                        source, contract, _active_repair_diagnostics
                    )
                if generic and all(
                    (artifact_root / name).is_file()
                    for name in ("source-model.json", "world-ir.json")
                ):
                    from .certification import GenericHarnessArtifactStore
                    from .repair_authoring import (
                        RepairPatchNotSubmittedError,
                        request_world_ir_patch,
                    )

                    store = GenericHarnessArtifactStore(artifact_root)
                    # Diagnostics are already included in guidance, but the constrained author
                    # consumes their typed form. The closure is called only by the generic loop,
                    # which replaces this placeholder through `_active_repair_diagnostics`.
                    try:
                        return await request_world_ir_patch(
                            source,
                            store.read_source_model(),
                            store.read_world_ir(),
                            _active_repair_diagnostics,
                        )
                    except RepairPatchNotSubmittedError:
                        return 1
                return await _build(
                    argparse.Namespace(
                        name=source.name,
                        path=str(source),
                        out=str(authoring),
                        interactive=False,
                        guidance=[guidance],
                        skip_source_provision=True,
                        external_runtime=(
                            job.source.kind.value == "provider"
                            and getattr(job.agent.mode, "value", None) == "connect_only"
                        ),
                    )
                )
            return await _scenarios(
                argparse.Namespace(
                    name=source.name,
                    out=str(authoring),
                    count=job.scenario_count,
                    interactive=False,
                    guidance=[guidance],
                    job=job,
                )
            )

    _active_repair_diagnostics = ()
    if generic:
        from .certification import GenericHarnessArtifactStore
        from .repair_controller import (
            CandidateObservation,
            RepairAction,
            RepairController,
            RepairOutcome,
            RepairPhase,
        )

        repair_controller = controller or RepairController()
        artifacts = GenericHarnessArtifactStore(authoring / "generic-harness")
        budgets = repair_controller.budgets
        candidate_repairs = (
            budgets.compiler_candidates
            + budgets.environment_patches
            + budgets.scenario_patches
        )
        # Every materially changed candidate may first consume its transient retry allowance.
        # Derive the guard from the controller's real budgets so increasing autonomous repair
        # depth cannot make this outer loop terminate before the final candidate is validated.
        decision_bound = (
            1
            + candidate_repairs
            + (candidate_repairs + 1) * budgets.infrastructure_retries_per_candidate
        )
        for _attempt in range(decision_bound):
            candidate_hash = _generic_candidate_hash(source, authoring)
            try:
                count = await validate(job, source, authoring)
            except RuntimeValidationError as exc:
                diagnostics = exc.diagnostics or (_fallback_diagnostic(exc),)
                # The repair history deliberately fingerprints diagnostics without persisting
                # their potentially high-cardinality text. Keep the redacted evidence visible in
                # the guest log as well, otherwise a bounded repair can appear to hang while the
                # actionable cause is available only inside the model session.
                print(
                    f"runtime validation: {exc.phase}: {exc}",
                    flush=True,
                )
                phase = (
                    RepairPhase.SCENARIOS
                    if exc.phase == "scenarios"
                    else RepairPhase.ENVIRONMENT
                )
                decision = repair_controller.decide(
                    CandidateObservation(
                        candidate_hash=candidate_hash,
                        phase=phase,
                        diagnostics=diagnostics,
                    )
                )
                artifacts.write_repair_history(repair_controller.history)
                if decision.action is RepairAction.REJECT:
                    raise
                if decision.action is RepairAction.RETRY_INFRASTRUCTURE:
                    await sleeper(decision.retry_after_seconds or 0)
                    repair_controller.record_result(
                        decision.sequence,
                        after_candidate_hash=candidate_hash,
                        outcome=RepairOutcome.APPLIED,
                    )
                    artifacts.write_repair_history(repair_controller.history)
                    continue
                if decision.action is RepairAction.RECOMPILE:
                    # Compilation is deterministic. Re-evaluate once; an unchanged fingerprint
                    # is rejected by the controller rather than handed to a model as data repair.
                    repair_controller.record_result(
                        decision.sequence,
                        after_candidate_hash=candidate_hash,
                        outcome=RepairOutcome.NO_MATERIAL_CHANGE,
                    )
                    artifacts.write_repair_history(repair_controller.history)
                    continue
                repair_phase = (
                    "scenarios"
                    if decision.action is RepairAction.PATCH_SCENARIOS
                    else "environment"
                )
                guidance = (
                    "Apply one constrained repair using submitted source evidence only. "
                    "Do not modify source, disable constraints, weaken checks, drop scenarios, "
                    "or invent services/tools. Resolve this complete, redacted diagnostic set:\n"
                    + _repair_guidance(diagnostics)
                )
                _active_repair_diagnostics = diagnostics
                scenarios_before = None
                if repair_phase == "scenarios":
                    from .scenario_tools import load_scenarios

                    scenarios_before = load_scenarios(authoring)
                repair_result = await repair(repair_phase, guidance)
                if repair_phase == "environment":
                    from .repair_patch import (
                        WorldIRRepairPatch,
                        apply_world_ir_repair_patch,
                    )
                    from .runtime_repair import (
                        RuntimePlanPatch,
                        apply_runtime_plan_patch,
                    )

                    if isinstance(repair_result, WorldIRRepairPatch):
                        source_model = artifacts.read_source_model()
                        world_ir = artifacts.read_world_ir()
                        artifacts.write_repair_patch(
                            repair_result, sequence=decision.sequence
                        )
                        repaired_world = apply_world_ir_repair_patch(
                            world_ir,
                            source_model,
                            repair_result,
                            allowed_reason_codes={item.code for item in diagnostics},
                        )
                        artifacts.write_world_ir(repaired_world)
                        still_broken = _invariants_the_ir_still_breaks(
                            repaired_world, authoring
                        )
                        if still_broken:
                            print(
                                "runtime validation: patch did not close the invariants, "
                                "repairing again without rebuilding: "
                                + "; ".join(still_broken)[:400],
                                flush=True,
                            )
                        status = 0
                    elif isinstance(repair_result, RuntimePlanPatch):
                        apply_runtime_plan_patch(
                            source,
                            authoring / "contract.json",
                            repair_result,
                            allowed_diagnostic_codes={
                                item.code for item in diagnostics
                            },
                        )
                        patch_path = (
                            authoring
                            / "generic-harness"
                            / f"runtime-plan-repair-{decision.sequence}.json"
                        )
                        patch_path.write_text(
                            repair_result.model_dump_json(indent=2) + "\n",
                            encoding="utf-8",
                        )
                        status = 0
                    else:
                        status = repair_result
                else:
                    from .scenario_repair import certify_scenario_repair
                    from .scenario_tools import load_scenarios, write_scenarios

                    status = repair_result
                    for correction in range(budgets.scenario_patches + 1):
                        if status:
                            break
                        assert scenarios_before is not None
                        scenarios_after = load_scenarios(authoring)
                        try:
                            receipt = certify_scenario_repair(
                                scenarios_before,
                                scenarios_after,
                                expected_count=job.scenario_count,
                                diagnostic_codes={item.code for item in diagnostics},
                            )
                        except Exception as rejected:
                            # Restore the last certified candidate; a rejected patch never becomes
                            # the input to another repair attempt.
                            write_scenarios(scenarios_before, authoring)
                            if correction == budgets.scenario_patches:
                                raise
                            print(
                                f"runtime validation: scenario repair rejected, repairing again: {rejected}",
                                flush=True,
                            )
                            status = await repair(
                                repair_phase,
                                guidance
                                + "\nThe previous repair was rejected and undone: "
                                + f"{rejected}. Repair again within these rules.",
                            )
                            continue
                        artifacts.write_scenario_repair(
                            receipt, sequence=decision.sequence
                        )
                        break
                after_hash = _generic_candidate_hash(source, authoring)
                repair_controller.record_result(
                    decision.sequence,
                    after_candidate_hash=after_hash,
                    outcome=(RepairOutcome.FAILED if status else RepairOutcome.APPLIED),
                )
                artifacts.write_repair_history(repair_controller.history)
                if status:
                    raise
            else:
                decision = repair_controller.decide(
                    CandidateObservation(
                        candidate_hash=candidate_hash,
                        phase=RepairPhase.ENVIRONMENT,
                    )
                )
                if decision.action is not RepairAction.CERTIFY:
                    raise RuntimeValidationError(
                        "environment", "generic pipeline refused passing candidate"
                    )
                artifacts.write_repair_history(repair_controller.history)
                evidence_path = (
                    authoring
                    / "generic-harness"
                    / GenericHarnessArtifactStore.RUNTIME_EVIDENCE
                )
                if evidence_path.is_file():
                    _write_generic_certificate(
                        job, authoring, repair_controller.history
                    )
                else:
                    # Injected validators in unit/integration consumers may not implement the
                    # runtime-evidence seam. Keep their legacy proof shape without allowing the
                    # production validator to emit a false full certificate.
                    (authoring / "runtime-validation.json").write_text(
                        json.dumps(
                            {
                                "status": "passed",
                                "attempts": len(repair_controller.history.decisions),
                                "setup_ready_scenarios": count,
                                "reference_tools_proven": False,
                                "generic_harness": "v1",
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                return
        raise RuntimeValidationError(
            "environment", "generic repair controller decision bound exhausted"
        )

    repairs = {"environment": 0, "scenarios": 0}
    for attempt in range(5):
        print(f"runtime validation: attempt {attempt + 1}/5", flush=True)
        try:
            count = await validate(job, source, authoring)
        except RuntimeValidationError as exc:
            print(f"runtime validation: {exc.phase}: {exc}", flush=True)
            if exc.phase not in repairs or repairs[exc.phase] >= 2:
                raise
            repairs[exc.phase] += 1
            guidance = (
                "Actual hosted runtime validation failed. Repair the generated environment/data "
                "or scenario setup using the submitted source as authority. Do not modify source, "
                "disable database constraints, weaken checks or ready conditions, drop scenarios, "
                "or replace tools with invented implementations. Preserve scenario count and intent. "
                f"Diagnostic: {str(exc)[:4000]}"
            )
            if await repair(exc.phase, guidance):
                raise
        else:
            (authoring / "runtime-validation.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "attempts": attempt + 1,
                        "setup_ready_scenarios": count,
                        "reference_tools_proven": False,
                    },
                    indent=2,
                )
                + "\n"
            )
            return
