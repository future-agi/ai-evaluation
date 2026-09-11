"""Validate generated setup against the actual hosted runtime before accepting authoring.

This is setup proof, not a claim that a reference tool trajectory executed. Actual
agent tool execution remains evidence collected during the calls.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


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

    external_provider = (
        getattr(getattr(job, "source", None), "kind", None) is SourceKind.PROVIDER
        and getattr(getattr(job, "agent", None), "mode", None)
        is ProviderExecutionMode.CONNECT_ONLY
    )

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
            preflight_bundle(
                bundle, manifest, parallelism=1, secret_refs=job_secret_purposes(job)
            )
            runtimes = await provider.provision(
                manifest,
                source=source,
                bundle_dir=bundle,
                work_directory=work,
                instances=1,
            )
            factory = ProcessWorldFactory(work)
            runtime = runtimes[0]
            phase = "scenarios"
            scenarios = await asyncio.to_thread(load_scenarios, bundle)
            if len(scenarios) != job.scenario_count:
                raise RuntimeValidationError(
                    phase, "Runtime scenario count differs from the requested count"
                )

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
                await provider.reset(runtime, work_directory=work)
                world = await factory.create(runtime, rng=random.Random(job.seed or 0))
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
