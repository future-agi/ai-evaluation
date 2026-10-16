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
    chat_ingress,
    native_observation: bool = False,
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
        "chat readiness proves one representative ingress turn, not task success or every conversation path",
    ]
    if chat_ingress is CheckStatus.NOT_APPLICABLE:
        limitations.append(
            "chat ingress is not applicable to voice or external connect-only runtimes"
        )
    if external_provider or native_observation:
        source_schema_hash = manifest.provenance.source_digest
        world_ir_hash = _artifact_digest(Path("__external_provider_world__"))
        compiler_version = "external-provider-black-box"
        schema_and_seed = CheckStatus.NOT_APPLICABLE
        source_invariants = CheckStatus.NOT_APPLICABLE
        if external_provider:
            limitations.append(
                "external provider state and tool implementations were not available for local inspection"
            )
    if native_observation:
        compiler_version = "native-compose-observation-v1"
        limitations.append("Native application state is opaque: schema, seed, source invariants, value-level reset equivalence and state isolation are not certified. Compose lifecycle rebuilds private source copies and volumes; checks may observe only native responses. Source actions are not probed by the harness.")
    if not external_provider and not native_observation:
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
        from .source_data_invariants import ARTIFACT

        invariant_review = authoring / ARTIFACT
        if (
            invariant_review.is_file()
            and json.loads(invariant_review.read_text()).get("status") == "not_applicable"
        ):
            source_invariants = CheckStatus.NOT_APPLICABLE
            limitations.append(
                "No runtime SQL business tables were available for source invariants; "
                "private application state is not certified by this check"
            )
    if world_isolation is CheckStatus.NOT_RUN:
        limitations.append(
            "concurrent world isolation was not applicable or runtime parallelism was safely degraded"
        )
    elif world_isolation is CheckStatus.NOT_APPLICABLE and not external_provider and not native_observation:
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
            chat_ingress=chat_ingress,
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
    workspace_snapshot = getattr(world, "_workspace_snapshot", None)
    if workspace_snapshot is not None:
        workspaces = workspace_snapshot()
        if workspaces:
            canonical = {
                "database": canonical,
                "workspaces": {
                    name: {
                        path: {
                            "mode": entry.mode,
                            "digest": hashlib.sha256(entry.data).hexdigest()
                            if entry.data is not None
                            else None,
                        }
                        for path, entry in sorted(entries.items())
                    }
                    for name, entries in sorted(workspaces.items())
                },
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


def _fallback_diagnostic(error, *, authoring: Path | None = None):
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
    elif (
        detail.removeprefix("BundleAuthorError: ")
        == "generic_pipeline_source_schema_required: no source-owned PostgreSQL schema found"
        and authoring is not None
        and (authoring / "world.sqlite").is_file()
        and json.loads((authoring / "contract.json").read_text(encoding="utf-8")).get(
            "data_store"
        )
        is None
    ):
        code = "generated_storage_declaration_missing"
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

    from .repair_controller import is_chat_dependency_investigation

    if is_chat_dependency_investigation(diagnostics):
        return True
    runtime_codes = {
        "generated_runtime_plan_invalid",
        "chat_ingress_configuration_invalid",
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
        self.process_evidence = ""
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


async def _probe_runtime_readiness(
    *, job, source, authoring, manifest, contract, provider, runtime, factory,
    work, values, redactions, compare_baseline=True,
):
    from .chat_call_runner import _tools, _chat_target_timeout_seconds
    from .chat_readiness import probe_chat_ingress

    interface = contract.runtime.interface if contract.runtime else None
    endpoint = runtime.endpoints.get("target_http")
    if interface is None or interface.kind not in {"http", "callable"} or endpoint is None:
        raise RuntimeValidationError(
            "runtime", "generated_runtime_plan_invalid: unsupported chat ingress"
        )
    baseline = await factory.create(runtime, rng=random.Random(job.seed or 0))
    baseline_digest = _world_state_digest(baseline) if compare_baseline else None
    receipt = {
        "status": "failed",
        "candidate_hash": _generic_candidate_hash(source, authoring),
        "bundle_digest": manifest.digest,
        "reset": "failed",
    }
    try:
        try:
            await probe_chat_ingress(
                interface=interface, endpoint=endpoint.address, opening_message="Hello",
                authoring=authoring, candidate_hash=receipt["candidate_hash"],
                bundle_digest=manifest.digest, secret_values=values,
                redaction_values=redactions,
                tools=_tools(contract) if interface.include_tools else (),
                timeout=_chat_target_timeout_seconds(),
            )
        finally:
            try:
                await provider.reset(runtime, work_directory=work)
                if compare_baseline:
                    restored = await factory.create(runtime, rng=random.Random(job.seed or 0))
                    if _world_state_digest(restored) != baseline_digest:
                        raise RuntimeValidationError(
                            "environment", "Readiness reset did not restore the baseline"
                        )
            except RuntimeValidationError:
                raise
            except Exception as exc:
                raise RuntimeValidationError(
                    "environment", f"Readiness reset failed: {exc}"
                ) from exc
            receipt["reset"] = "equivalent" if compare_baseline else "restarted"
        receipt["status"] = "passed"
    finally:
        path = authoring / "generic-harness" / "runtime-readiness.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


async def validate_once(
    job,
    source: Path,
    authoring: Path,
    *,
    secrets_path=Path("/run/futureagi/secrets.json"),
    local_runtime: bool = False,
    readiness_only: bool = False,
) -> int:
    from . import outbound
    from .bundle_author_v2 import author_bundle_v2
    from .compose_integration import enabled as compose_enabled
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

    if getattr(job, "require_tool_workflow", False):
        from .contract import validate_tool_workflow

        try:
            validate_tool_workflow(
                AgentContract.model_validate_json(
                    (authoring / "contract.json").read_text(encoding="utf-8")
                )
            )
        except (ValueError, OSError) as exc:
            raise RuntimeValidationError("environment", str(exc)) from exc

    external_provider = (
        getattr(getattr(job, "source", None), "kind", None) is SourceKind.PROVIDER
        and getattr(getattr(job, "agent", None), "mode", None)
        is ProviderExecutionMode.CONNECT_ONLY
    )
    generic = bool(
        isinstance(getattr(job, "metadata", None), dict)
        and job.metadata.get("generic_harness_v1") is True
    )
    if readiness_only and not generic:
        raise RuntimeValidationError("runtime", "Early readiness requires the generic runtime")
    if readiness_only:
        (authoring / "generic-harness" / "runtime-readiness.json").unlink(missing_ok=True)
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
        redaction_values = json.loads(secrets.read_text())
        process_logs = authoring / "generic-harness" / "process-logs"
        logs_before = set(process_logs.glob("*.log"))
        validation_error = None
        from .hosted_entrypoint import peek_target_provider_secret_values

        target_secret_values = peek_target_provider_secret_values(
            secrets, job_secret_purposes(job)
        )
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
                default_user_resolver
                if local_runtime and os.environ.get("ALK_FIXED_SANDBOX_USER") != "1"
                else fixed_sandbox_user_resolver
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
        requested_lanes = max(
            1, int(os.environ.get("ALK_VALIDATION_INSTANCES", "4") or 4)
        )
        executor = ThreadPoolExecutor(
            max_workers=requested_lanes, thread_name_prefix="runtime-validation"
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
                include_scenarios=not readiness_only,
            )
            if local_runtime:
                _make_local_seed_files_readable(bundle)
            validation_parallelism = (
                1 if external_provider or readiness_only else max(requested_lanes, 2 if generic else 1)
            )
            preflight_bundle(
                bundle,
                manifest,
                parallelism=validation_parallelism,
                secret_refs=job_secret_purposes(job),
                allow_compose=local_runtime and compose_enabled(),
            )
            from .bundle_v2 import RuntimeKindV2
            if manifest.runtime.kind is RuntimeKindV2.COMPOSE:
                from .compose_integration import validate_native
                return await validate_native(job=job, source=source, authoring=authoring,
                    bundle=bundle, manifest=manifest, work=work, values=target_secret_values,
                    redactions=redaction_values, executor=executor, readiness_only=readiness_only)

            runtimes = await provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle,
                work_directory=work,
                instances=validation_parallelism,
            )
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
            factory = ProcessWorldFactory(work, bundle)
            runtime = runtimes[0]
            if readiness_only:
                await _probe_runtime_readiness(
                    job=job, source=source, authoring=authoring, manifest=manifest,
                    contract=contract, provider=provider, runtime=runtime, factory=factory,
                    work=work, values=target_secret_values, redactions=redaction_values,
                )
                return 0
            lanes = len(runtimes) or 1
            phase = "scenarios"
            scenarios = await asyncio.to_thread(load_scenarios, bundle)
            if len(scenarios) != job.scenario_count:
                raise RuntimeValidationError(
                    phase, "Runtime scenario count differs from the requested count"
                )

            baseline_state_digest: str | None = None

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

                await asyncio.gather(
                    *(
                        lane(suite[index::lanes], runtimes[index])
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

            from .local_certified_runtime import LocalCertifiedSink

            phase_sink = LocalCertifiedSink(
                authoring / "generic-harness" / "phase-observations"
            )

            async def check_setup_once(scenario, invariants, against, keep_processes):
                nonlocal baseline_state_digest
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
                        fn, target, timeout=timeout, phase=name, executor=executor,
                        audit_log=phase_sink.log,
                        audit_context={"scenario_key": scenario.scenario_key, "world_index": against.world_index, "stage": "validation"},
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
                    if not (
                        does_nothing(scenario.setup) and does_nothing(scenario.ready)
                    )
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
                        chat_ingress=CheckStatus.NOT_APPLICABLE,
                    )
                    _freeze_certified_bundle(bundle, authoring)
                return len(scenarios)
            if generic:
                from .chat_call_runner import (
                    _scenario_document,
                    _with_scenario_edits,
                    _tools,
                    _chat_target_timeout_seconds,
                )
                from .chat_readiness import probe_chat_ingress

                chat_ingress = CheckStatus.NOT_APPLICABLE
                interface = contract.runtime.interface if contract.runtime else None
                if contract.modality != "voice":
                    if interface is None or interface.kind not in {"http", "callable"}:
                        raise RuntimeValidationError(
                            "runtime",
                            "generated_runtime_plan_invalid: unsupported chat ingress",
                        )
                    if not scenarios:
                        raise RuntimeValidationError(
                            "environment",
                            "Chat readiness requires a representative scenario",
                        )
                    await check_setup(scenarios[0], [])
                    endpoint = runtime.endpoints.get("target_http")
                    if endpoint is None:
                        raise RuntimeValidationError(
                            "runtime",
                            "generated_runtime_plan_invalid: target_http is absent",
                        )
                    source_key = (
                        getattr(scenarios[0], "source_scenario_key", None)
                        or scenarios[0].scenario_key
                    )
                    document = _with_scenario_edits(
                        _scenario_document(bundle, source_key), job.metadata, source_key
                    )
                    await probe_chat_ingress(
                        interface=interface,
                        endpoint=endpoint.address,
                        document=document,
                        scenario_key=scenarios[0].scenario_key,
                        authoring=authoring,
                        candidate_hash=_generic_candidate_hash(source, authoring),
                        bundle_digest=manifest.digest,
                        secret_values=target_secret_values,
                        redaction_values=redaction_values,
                        tools=_tools(contract) if interface.include_tools else (),
                        scenario_id=scenarios[0].scenario_id,
                        timeout=_chat_target_timeout_seconds(),
                    )
                    chat_ingress = CheckStatus.PASSED
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
            await check_setups(invariants)
            print(
                f"runtime validation: setup pass {time.monotonic() - walk:.0f}s for "
                f"{len(scenarios)} scenarios with {len(invariants)} invariants, "
                f"complete in {time.monotonic() - started:.0f}s",
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
                    chat_ingress=chat_ingress,
                )
                # Freeze the exact bytes that passed the real runtime checks.  Recompiling from
                # the authoring archive in the execution sandbox can otherwise drift on inputs
                # that are deliberately outside the source fingerprint (generated build/cache
                # artifacts are one example).  The consumer re-verifies the certificate and all
                # authoring fingerprints before accepting this directory.
                _freeze_certified_bundle(bundle, authoring)
            return len(scenarios)
        except RuntimeValidationError as exc:
            validation_error = RuntimeValidationError(
                exc.phase,
                outbound.redact_outbound_text(
                    str(exc), extra_secret_values=secret_values
                ),
                diagnostics=exc.diagnostics,
            )
            raise validation_error from None
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
            if validation_error is not None:
                from .process_diagnostics import retained_process_evidence

                try:
                    validation_error.process_evidence = retained_process_evidence(
                        process_logs, logs_before, redaction_values
                    )
                except OSError:
                    pass


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
    validate=None,
    repair=None,
    controller=None,
    sleeper=asyncio.sleep,
    readiness_only: bool = False,
) -> None:
    """Two repairs per phase; environment repairs cannot exhaust setup's budget."""
    from .job import HarnessJob

    base_guidance = (
        "\n".join(job.authoring_guidance()) if isinstance(job, HarnessJob) else ""
    )
    if validate is None:
        if readiness_only:
            async def validate(job, source, authoring):
                return await validate_once(job, source, authoring, readiness_only=True)
        else:
            validate = validate_once
    generic = bool(
        job is not None
        and isinstance(getattr(job, "metadata", None), dict)
        and job.metadata.get("generic_harness_v1") is True
    )
    if readiness_only:
        if not generic:
            raise RuntimeValidationError("runtime", "Early readiness requires the generic runtime")
        (authoring / "generic-harness" / "runtime-readiness.json").unlink(missing_ok=True)
    if repair is None:

        async def repair(phase, guidance):
            from .cli import _build, _scenarios

            if phase == "environment":
                artifact_root = authoring / "generic-harness"
                if generic and {item.code for item in _active_repair_diagnostics} == {
                    "generated_storage_declaration_missing"
                }:
                    from .cli import _understand
                    from .contract import AgentContract

                    contract_path = authoring / "contract.json"
                    contract = AgentContract.model_validate_json(
                        contract_path.read_text(encoding="utf-8")
                    )
                    if contract.data_store is not None:
                        return 1
                    with tempfile.TemporaryDirectory(
                        prefix="alk-storage-repair-"
                    ) as candidate_dir:
                        status = await _understand(
                            argparse.Namespace(
                                kind="repo",
                                name=source.name,
                                path=str(source),
                                out=candidate_dir,
                                interactive=False,
                                job=job,
                                guidance=[
                                    guidance,
                                    (
                                        "Inspect source to recover the missing data_store declaration. "
                                        "Record explicit filesystem, in_memory, none, or database storage "
                                        "only when source supports it. Do not invent a schema or change "
                                        "source behavior. Only data_store will be promoted. Current contract:\n"
                                        + contract.model_dump_json(indent=2)
                                    ),
                                ],
                            )
                        )
                        candidate_path = Path(candidate_dir) / "contract.json"
                        if status != 0 or not candidate_path.is_file():
                            return 1
                        candidate = AgentContract.model_validate_json(
                            candidate_path.read_text(encoding="utf-8")
                        )
                        if (
                            candidate.data_store is None
                            or not candidate.data_store.kind.strip()
                        ):
                            return 1
                        payload = contract.model_dump()
                        payload["data_store"] = candidate.data_store.model_dump()
                        repaired = AgentContract.model_validate(payload)
                        contract_path.write_text(
                            repaired.model_dump_json(indent=2) + "\n", encoding="utf-8"
                        )
                    return 0
                if generic and _needs_runtime_plan_repair(_active_repair_diagnostics):
                    from .contract import AgentContract
                    from .runtime_repair import request_runtime_plan_patch

                    contract = AgentContract.model_validate_json(
                        (authoring / "contract.json").read_text(encoding="utf-8")
                    )
                    return await request_runtime_plan_patch(
                        source, contract, _active_repair_diagnostics, guidance=guidance
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
                            guidance=guidance,
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

        def write_history():
            if readiness_only:
                path = authoring / "generic-harness" / "readiness-repair-history.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(repair_controller.history.model_dump_json(indent=2) + "\n")
            else:
                artifacts.write_repair_history(repair_controller.history)

        budgets = repair_controller.budgets
        candidate_repairs = (
            budgets.compiler_candidates
            + budgets.runtime_patches
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
        candidate_hash = _generic_candidate_hash(source, authoring)
        for _attempt in range(decision_bound):
            try:
                count = await validate(job, source, authoring)
            except RuntimeValidationError as exc:
                if readiness_only and exc.phase == "scenarios":
                    raise
                diagnostics = exc.diagnostics or (
                    _fallback_diagnostic(exc, authoring=authoring),
                )
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
                    else RepairPhase.RUNTIME
                    if exc.phase == "runtime"
                    else RepairPhase.ENVIRONMENT
                )
                decision = repair_controller.decide(
                    CandidateObservation(
                        candidate_hash=candidate_hash,
                        phase=phase,
                        diagnostics=diagnostics,
                    )
                )
                write_history()
                if decision.action is RepairAction.REJECT:
                    raise
                if decision.action is RepairAction.RETRY_INFRASTRUCTURE:
                    await sleeper(decision.retry_after_seconds or 0)
                    repair_controller.record_result(
                        decision.sequence,
                        after_candidate_hash=candidate_hash,
                        outcome=RepairOutcome.APPLIED,
                    )
                    write_history()
                    continue
                if decision.action is RepairAction.RECOMPILE:
                    # Compilation is deterministic. Re-evaluate once; an unchanged fingerprint
                    # is rejected by the controller rather than handed to a model as data repair.
                    repair_controller.record_result(
                        decision.sequence,
                        after_candidate_hash=candidate_hash,
                        outcome=RepairOutcome.NO_MATERIAL_CHANGE,
                    )
                    write_history()
                    continue
                if readiness_only and decision.action is RepairAction.PATCH_SCENARIOS:
                    raise
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
                    + "\n\n"
                    + base_guidance
                )
                if exc.process_evidence:
                    guidance += (
                        "\n<untrusted_current_attempt_process_logs>\n"
                        + exc.process_evidence
                        + "\n</untrusted_current_attempt_process_logs>"
                    )
                _active_repair_diagnostics = diagnostics
                scenarios_before = None
                if repair_phase == "scenarios":
                    from .scenario_tools import load_scenarios

                    scenarios_before = load_scenarios(authoring)
                before_repair_hash = _generic_candidate_hash(source, authoring)
                try:
                    repair_result = await repair(repair_phase, guidance)
                except Exception as repair_error:
                    print(
                        f"runtime validation: repair failed ({type(repair_error).__name__})",
                        flush=True,
                    )
                    repair_controller.record_result(
                        decision.sequence,
                        after_candidate_hash=candidate_hash,
                        outcome=RepairOutcome.FAILED,
                    )
                    write_history()
                    raise exc from None
                if repair_phase == "environment":
                    from .repair_patch import (
                        WorldIRRepairPatch,
                        apply_world_ir_repair_patch,
                    )
                    from .runtime_repair import (
                        RuntimePlanPatch,
                        apply_runtime_plan_patch,
                    )

                    from .repair_controller import is_chat_dependency_investigation

                    if is_chat_dependency_investigation(diagnostics) and not isinstance(
                        repair_result, RuntimePlanPatch
                    ):
                        status = 1
                    elif isinstance(repair_result, WorldIRRepairPatch):
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
                    status = repair_result
                    if not status:
                        from .scenario_repair import certify_scenario_repair
                        from .scenario_tools import load_scenarios, write_scenarios

                        assert scenarios_before is not None
                        scenarios_after = load_scenarios(authoring)
                        try:
                            receipt = certify_scenario_repair(
                                scenarios_before,
                                scenarios_after,
                                expected_count=job.scenario_count,
                                diagnostic_codes={item.code for item in diagnostics},
                            )
                        except Exception:
                            # Restore the last certified candidate; a rejected patch never becomes
                            # the input to another repair attempt.
                            write_scenarios(scenarios_before, authoring)
                            raise
                        artifacts.write_scenario_repair(
                            receipt, sequence=decision.sequence
                        )
                after_hash = _generic_candidate_hash(source, authoring)
                if status or after_hash == before_repair_hash:
                    after_hash = candidate_hash
                repair_controller.record_result(
                    decision.sequence,
                    after_candidate_hash=after_hash,
                    outcome=(RepairOutcome.FAILED if status else RepairOutcome.APPLIED),
                )
                write_history()
                if status:
                    raise
                candidate_hash = after_hash
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
                write_history()
                if readiness_only:
                    return
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
                f"Diagnostic: {str(exc)[:4000]}" + "\n\n" + base_guidance
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
