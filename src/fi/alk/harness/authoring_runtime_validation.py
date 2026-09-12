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
import tempfile
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


def _write_runtime_evidence(
    *,
    job,
    authoring: Path,
    manifest,
    count: int,
    external_provider: bool,
    tool_report,
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
    from .compile.postgres import POSTGRES_COMPILER_VERSION

    store = GenericHarnessArtifactStore(authoring / "generic-harness")
    limitations = [
        "runtime-only agent tool effects are verified during calls, not setup certification",
    ]
    if external_provider:
        source_schema_hash = manifest.provenance.source_digest
        world_ir_hash = _artifact_digest(Path("__external_provider_world__"))
        compiler_version = "external-provider-black-box"
        schema_and_seed = CheckStatus.NOT_RUN
        source_invariants = CheckStatus.NOT_RUN
        limitations.append(
            "external provider state and tool implementations were not available for local inspection"
        )
    if world_isolation is CheckStatus.NOT_RUN:
        limitations.append(
            "concurrent world isolation was not applicable or runtime parallelism was safely degraded"
        )
    else:
        source_model = store.read_source_model()
        world_ir = store.read_world_ir()
        source_schema_hash = source_model.fingerprint
        world_ir_hash = world_ir.fingerprint
        compiler_version = POSTGRES_COMPILER_VERSION
        schema_and_seed = CheckStatus.PASSED
        source_invariants = CheckStatus.PASSED

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
            reset_equivalence=reset_equivalence,
            world_isolation=world_isolation,
        ),
        limitations=tuple(limitations),
    )
    store.write_runtime_evidence(evidence)
    store.write_tool_certification(tool_report)


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
        if (
            relative.startswith("generic-harness/")
            or relative == "runtime-validation.json"
        ):
            continue
        digest.update(f"authoring/{relative}\n".encode("utf-8"))
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _fallback_diagnostic(error):
    from .diagnostics import HarnessDiagnostic
    from .job import HarnessStage

    if error.phase == "scenarios":
        code = "ready_condition_invalid"
        stage = HarnessStage.VALIDATING_SCENARIOS
    elif error.phase == "infrastructure":
        code = "process_dependency_timeout"
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


class RuntimeValidationError(RuntimeError):
    def __init__(self, phase: str, detail: str, *, diagnostics=()):
        self.phase = phase
        self.diagnostics = tuple(diagnostics)
        super().__init__(detail)


async def validate_once(
    job,
    source: Path,
    authoring: Path,
    *,
    secrets_path=Path("/run/futureagi/secrets.json"),
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
    from .process_runtime import ProcessRuntimeProvider
    from .scenario_source import load_scenarios
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
        (artifact_root / GenericHarnessArtifactStore.RUNTIME_EVIDENCE).unlink(
            missing_ok=True
        )
        (artifact_root / GenericHarnessArtifactStore.CERTIFICATION).unlink(
            missing_ok=True
        )
        (authoring / "runtime-validation.json").unlink(missing_ok=True)

    # The real execution consumes its credential file. Validation gets a private copy,
    # with the same purpose map, so it cannot destroy the execution handoff.
    with tempfile.TemporaryDirectory(
        prefix="runtime-validation-", dir=authoring.parent
    ) as root:
        work = Path(root)
        secrets = work / "secrets.json"
        secrets.write_bytes(secrets_path.read_bytes())
        secrets.chmod(0o600)
        secret_values = tuple(
            str(value) for value in json.loads(secrets.read_text()).values()
        )
        capabilities = outbound.load_capabilities(unlink=False)
        transport = outbound.RequestsTransport()
        provider = ProcessRuntimeProvider(
            secrets_path=secrets,
            secret_purpose_map=job_secret_purposes(job),
            user_resolver=lambda _name: None,
            require_declared_user=False,
            public_url_resolver=lambda port, ttl: _resolve_hosted_public_url(
                capabilities, transport, port=port, expires_in_seconds=ttl
            ),
            provider_attempt_id=capabilities.attempt_id,
            provider_expires_at=capabilities.expires_at,
            generic_artifact_root=(authoring / "generic-harness") if generic else None,
        )
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="runtime-validation"
        )
        phase = "environment"
        try:
            bundle = work / "bundle"
            manifest = await asyncio.to_thread(
                author_bundle_v2,
                source=source,
                job=job,
                authoring=authoring,
                output=bundle,
            )
            validation_parallelism = 1 if external_provider or not generic else 2
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
            tool_report = None
            world_isolation = None
            if generic:
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
                build_output = json.loads(
                    (work / "artifacts" / "build.json").read_text(encoding="utf-8")
                )
                conformance = build_output.get("conformance")
                if conformance is False:
                    raise RuntimeValidationError(
                        "environment",
                        "Concurrent world isolation conformance failed: "
                        + str(build_output.get("conformance_reason") or "unknown reason"),
                    )
                world_isolation = (
                    CheckStatus.PASSED if conformance is True else CheckStatus.NOT_RUN
                )
            factory = ProcessWorldFactory(work)
            runtime = runtimes[0]
            phase = "scenarios"
            scenarios = await asyncio.to_thread(load_scenarios, bundle)
            if len(scenarios) != job.scenario_count:
                raise RuntimeValidationError(
                    phase, "Runtime scenario count differs from the requested count"
                )

            baseline_state_digest: str | None = None

            async def check_setups(invariants):
                failures = []
                for scenario in scenarios:
                    try:
                        await check_setup(scenario, invariants)
                    except RuntimeValidationError as exc:
                        failures.append(str(exc))
                if failures:
                    raise RuntimeValidationError(phase, "\n".join(failures))

            async def check_setup(scenario, invariants):
                nonlocal baseline_state_digest
                await provider.reset(runtime, work_directory=work)
                world = await factory.create(runtime, rng=random.Random(job.seed or 0))
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

            # Collect all executable setup errors before spending a model review or
            # a repair attempt. Each scenario still gets an independent clean world.
            await check_setups([])
            if external_provider:
                # A connect-only provider owns its state and executes its tools outside
                # this sandbox. There is no harness-owned source database to probe or
                # seed, so source-data invariant review would invent a local environment.
                print(
                    "runtime validation: external provider black-box mode; "
                    "skipping local source-data invariant review",
                    flush=True,
                )
                if generic:
                    _write_runtime_evidence(
                        job=job,
                        authoring=authoring,
                        manifest=manifest,
                        count=len(scenarios),
                        external_provider=True,
                            tool_report=tool_report,
                        reset_equivalence=CheckStatus.NOT_RUN,
                        world_isolation=CheckStatus.NOT_RUN,
                    )
                return len(scenarios)
            phase = "environment"
            await provider.reset(runtime, work_directory=work)
            baseline = await factory.create(runtime, rng=random.Random(job.seed or 0))
            print("runtime validation: reviewing source data invariants", flush=True)
            invariants = await author_invariants(
                source, authoring, baseline.read_only(), endpoints=runtime.endpoints
            )
            # Review probes may have effects; none belongs in the test baseline.
            await provider.reset(runtime, work_directory=work)
            baseline = await factory.create(runtime, rng=random.Random(job.seed or 0))
            await check_invariants(baseline.read_only(), invariants)
            phase = "scenarios"
            if invariants:
                await check_setups(invariants)
            if generic:
                await provider.reset(runtime, work_directory=work)
                reset_world = await factory.create(
                    runtime, rng=random.Random(job.seed or 0)
                )
                if baseline_state_digest is None or _world_state_digest(
                    reset_world
                ) != baseline_state_digest:
                    raise RuntimeValidationError(
                        "environment", "Final reset did not restore the certified baseline"
                    )
                _write_runtime_evidence(
                    job=job,
                    authoring=authoring,
                    manifest=manifest,
                    count=len(scenarios),
                    external_provider=False,
                    tool_report=tool_report,
                    reset_equivalence=CheckStatus.PASSED,
                    world_isolation=world_isolation,
                )
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
    if repair is None:

        async def repair(phase, guidance):
            from .cli import _build, _scenarios

            if phase == "environment":
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
                )
            )

    generic = bool(
        job is not None
        and isinstance(getattr(job, "metadata", None), dict)
        and job.metadata.get("generic_harness_v1") is True
    )
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
        for _attempt in range(12):
            candidate_hash = _generic_candidate_hash(source, authoring)
            try:
                count = await validate(job, source, authoring)
            except RuntimeValidationError as exc:
                diagnostics = exc.diagnostics or (_fallback_diagnostic(exc),)
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
                    "or invent services/tools. Resolve this complete diagnostic set: "
                    + ", ".join(
                        sorted(f"{item.code}@{item.component}" for item in diagnostics)
                    )
                )
                status = await repair(repair_phase, guidance)
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
