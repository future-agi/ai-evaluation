"""Explicit local Compose selection with native response observation."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

from .bundle_v2 import (
    BUNDLE_V2_SCHEMA_VERSION,
    BundleProvenanceV2,
    BundleRuntimeV2,
    CapabilityV2,
    EnvironmentBundleV2,
    EvidenceSeam,
    ReadinessProbeV2,
    RuntimeKindV2,
    load_bundle_v2,
    seal_bundle_v2,
)
from .process_preflight import preflight_bundle
from .runtime_selection import ComposeSelection, select_runtime


def enabled() -> bool:
    return bool(
        os.environ.get("ALK_COMPOSE_BROKER_SOCKET")
        or os.environ.get("ALK_TRUSTED_COMPOSE_SOCKET")
    )


def runtime_capability_briefing() -> str:
    heading = (
        "\n\n## Available execution capabilities\n\n"
        "Generated native process execution supports Python or source-declared pinned Bun, "
        "not arbitrary native Node.js startup. Do not translate a Node.js project to Bun "
        "without source evidence or invent a replacement entrypoint or runtime. "
    )
    if not enabled():
        return heading + (
            "Managed source-owned Compose deployment is not enabled in this session. "
            "Inspect deployment files as evidence for supported native source startup; "
            "do not assume container execution or select published Compose images."
        )
    return heading + (
        "Managed source-owned Compose deployment is available for chat agents with an HTTP ingress. "
        "Choose supported native startup or Compose from repository deployment evidence; "
        "neither mode is required. For Compose, "
        "declare runtime.execution=compose, the repository-relative runtime.compose_file, "
        "and runtime.service. ALK runs the selected service and its declared dependency closure, "
        "including source-defined startup and migrations. Source-owned overlays may be selected "
        "with runtime.compose_overrides; additional source-declared workflow roots use "
        "runtime.additional_services. Use runtime.compose_image_services only for an already "
        "selected service's source-declared image. Never invent deployment files or images. "
        "Prefer source-declared Compose when native startup would otherwise recreate services "
        "or migration steps that the repository already assembles. "
        "For agents with multiple model providers, inspect the source-owned provider selection "
        "mechanism and declare its public selector in runtime.config_files; forwarding provider "
        "credentials alone is not configuration. Bind confidential values with secret_aliases. "
        "When startup exceeds available resources, inspect and prefer a source-declared minimal "
        "or development deployment that still provides the selected chat service and its dependencies. "
        "Auto selection discovers conventional Compose filenames only at the repository root; "
        "nested or ambiguous deployment files require an explicit selection. Explicit native "
        "execution does not auto-select Compose. This capability does not add standalone "
        "Dockerfile execution. Source validation, deployment policy, and real chat readiness "
        "still must pass; capability availability is not evidence that an agent works."
    )


def provider(values, artifact_root):
    from .certified_compose_runtime import CertifiedComposeRuntimeProvider

    broker_socket = os.environ.get("ALK_COMPOSE_BROKER_SOCKET")
    if broker_socket:
        from .compose_broker import ComposeBrokerClient

        return ComposeBrokerClient(Path(broker_socket))
    socket = os.environ.get("ALK_TRUSTED_COMPOSE_SOCKET")
    if not socket:
        raise RuntimeError("compose_trusted_socket_required")
    return CertifiedComposeRuntimeProvider(
        socket_path=Path(socket), configuration=values, artifact_root=artifact_root
    )


def validate_source(
    source: Path, admissions=(), *, fixtures=(), source_digest=None, paths=None
) -> None:
    from .bundle import _inspect_file
    from .certified_compose_runtime import validate_source_tree

    source = Path(source).resolve()
    validate_source_tree(source)
    from .provision import source_fingerprint

    if admissions or fixtures or source_digest is not None:
        actual_digest = source_fingerprint(source)
        if source_digest is not None and actual_digest != source_digest:
            raise ValueError("public_material_admission_source_changed")
        if any(item.source_digest != actual_digest for item in [*admissions, *fixtures]):
            raise ValueError("public_material_admission_source_changed")
    by_path = {}
    for item in admissions:
        by_path.setdefault(item.path, []).append(item)
    fixture_paths = {item.path: item for item in fixtures}
    if len(fixture_paths) != len(fixtures):
        raise ValueError("public_test_fixture_duplicate_path")
    admission_paths = set(by_path) | set(fixture_paths)
    for relative in admission_paths:
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("public_material_admission_file_missing")
    inspected = source.rglob("*") if paths is None else [
        *paths, *(source / relative for relative in admission_paths)
    ]
    for path in sorted(set(inspected)):
        if not path.resolve().is_relative_to(source):
            raise ValueError("compose_source_path_escape")
        if not path.is_symlink() and path.is_file():
            relative = path.relative_to(source).as_posix()
            _inspect_file(
                source, path, admissions=by_path.get(relative, ()),
                fixture=fixture_paths.get(relative),
            )


def compile_bundle(*, source, job, authoring, output, contract, include_scenarios=True):
    from .bundle_author_v2 import (
        _copy_scenarios,
        _copy_sub_goal_catalogue,
        _copy_chat_authoring,
        _compile_source_tool_handlers,
        _files,
        source_fingerprint,
    )

    runtime = contract.get("runtime") or {}
    from .contract import (
        RuntimeConfigFile,
        validate_runtime_resource_paths,
        runtime_public_environment,
    )

    resources = {
        path: RuntimeConfigFile.model_validate(value)
        for path, value in runtime.get("config_files", {}).items()
    }
    directories = runtime.get("directories", [])
    validate_runtime_resource_paths(resources, directories)
    public_environment = runtime_public_environment(runtime)
    for resource in resources.values():
        for alias in resource.secret_aliases.values():
            reference = job.agent.secret_refs.get(alias)
            if reference is None or reference.purpose != "target_provider":
                raise ValueError("compose_config_target_reference_required")
    selection = select_runtime(
        source,
        runtime,
        validate_environment_dependencies=False,
        public_environment=public_environment,
    )
    if not isinstance(selection, ComposeSelection):
        return None
    interface = runtime.get("interface") or {}
    if contract.get("modality") != "chat" or interface.get("kind") != "http":
        raise ValueError("compose_native_http_chat_required")
    if not selection.control_service or not interface.get("port"):
        raise ValueError("compose_control_service_and_port_required")
    body = selection.document
    unsupported = set(body) - {"services", "volumes", "networks", "name", "version"}
    if any(not name.startswith("x-") for name in unsupported):
        raise ValueError("compose_top_level_unsupported")
    document = {
        key: copy.deepcopy(value)
        for key, value in body.items()
        if key in {"volumes", "networks"}
    }
    document["services"] = copy.deepcopy(selection.services)
    if resources:
        document["x-alk-config-files"] = {
            path: value.model_dump() for path, value in resources.items()
        }
    if directories:
        document["x-alk-directories"] = directories
    base = selection.compose_file.parent

    from .contract import RuntimeSecretFile

    for service_name, bindings in runtime.get("secret_files", {}).items():
        if service_name not in document["services"] or len(bindings) > 32:
            raise ValueError("compose_secret_file_service_invalid")
        parsed = [RuntimeSecretFile.model_validate(item) for item in bindings]
        for binding in parsed:
            reference = job.agent.secret_refs.get(binding.alias)
            if reference is None or reference.purpose != "target_provider":
                raise ValueError("compose_secret_file_target_reference_required")
        document["services"][service_name]["x-alk-secret-files"] = [
            item.model_dump(exclude_none=True) for item in parsed
        ]

    def relative(value):
        if not isinstance(value, str) or "$" in value or Path(value).is_absolute():
            raise ValueError("compose_source_path_invalid")
        path = (base / value).resolve()
        if not path.is_relative_to(source):
            raise ValueError("compose_source_path_escape")
        return "./" + path.relative_to(source).as_posix()

    for service in document["services"].values():
        # Explicit roots and their dependency closures are enabled regardless of discovery
        # profiles; unrelated profiled services never enter this document.
        service.pop("profiles", None)
        build = service.get("build")
        if build:
            build = {"context": build} if isinstance(build, str) else build
            build["context"] = relative(build.get("context", "."))
            # Selection already resolved build values; Compose must preserve literal dollars.
            args = build.get("args")
            if isinstance(args, dict):
                build["args"] = {
                    key: value.replace("$", "$$") if isinstance(value, str) else value
                    for key, value in args.items()
                }
            elif isinstance(args, list):
                build["args"] = [value.replace("$", "$$") for value in args]
            service["build"] = build
        envfiles = service.get("env_file", [])
        envfiles = [envfiles] if isinstance(envfiles, str) else envfiles
        for index, entry in enumerate(envfiles):
            if isinstance(entry, str):
                envfiles[index] = relative(entry)
            else:
                entry["path"] = relative(entry["path"])
        if envfiles:
            service["env_file"] = envfiles
        mounts = service.get("volumes", [])
        for index, mount in enumerate(mounts):
            if isinstance(mount, str):
                parts = mount.split(":")
                if parts[0].startswith((".", "/", "~")):
                    parts[0] = relative(parts[0])
                    mounts[index] = ":".join(parts)
            elif mount.get("type") == "bind":
                mount["source"] = relative(mount["source"])
    temporary = Path(tempfile.mkdtemp(prefix=".compose-", dir=output.parent))
    try:
        if include_scenarios:
            _copy_scenarios(authoring, temporary, count=job.scenario_count)
            _copy_sub_goal_catalogue(authoring, temporary)
        _copy_chat_authoring(authoring, temporary, include_simulator=include_scenarios)
        _compile_source_tool_handlers(contract, temporary)
        (temporary / "compose.json").write_text(json.dumps(document, indent=2))
        manifest = EnvironmentBundleV2(
            schema_version=BUNDLE_V2_SCHEMA_VERSION,
            digest="sha256:" + "0" * 64,
            name=str(job.metadata.get("name") or source.name),
            runtime=BundleRuntimeV2(
                kind=RuntimeKindV2.COMPOSE,
                document="compose.json",
                control_service=selection.control_service,
                evidence_seam=EvidenceSeam.TOOL_TRACE,
            ),
            capabilities={
                "target_http": CapabilityV2(
                    protocol="http",
                    service=selection.control_service,
                    container_port=interface["port"],
                )
            },
            readiness=[
                ReadinessProbeV2(
                    capability="target_http", path=interface.get("health_path") or "/"
                )
            ],
            files=_files(temporary),
            provenance=BundleProvenanceV2(
                public_material_admissions=job.public_material_admissions,
                public_test_fixture_admissions=job.public_test_fixture_admissions,
                source_kind=job.source.kind.value,
                repository=job.source.repository,
                commit=job.source.commit_sha,
                source_digest=source_fingerprint(source),
                generator="fi.alk.harness.compose_integration",
                generator_version="1",
                adopted_files=(["scenarios/"] if include_scenarios else []) + ["contract.json"],
                generated_files=["compose.json", "manifest.json"],
            ),
            metadata={
                "observation_mode": "native",
                "generic_harness": "v1",
                "context_excludes": list(runtime.get("context_excludes") or ()),
            },
        )
        from .compose_source import source_inputs

        inputs = source_inputs(
            source,
            document,
            capabilities=manifest.capabilities,
            context_excludes=tuple(runtime.get("context_excludes") or ()),
        )
        validate_source(
            source, job.public_material_admissions,
            fixtures=job.public_test_fixture_admissions,
            paths=(*inputs.files, *selection.source_files),
        )
        manifest = manifest.model_copy(update={"digest": seal_bundle_v2(manifest)})
        from .compose_image_lock import lock_document

        if any(not service.get("build") for service in document["services"].values()):
            identities = provider({}, None).resolve_images(
                manifest=manifest, document=(temporary / "compose.json").read_text(),
                runtime=runtime, source=source,
            )
            lock_document(document, identities)
            (temporary / "compose.json").write_text(json.dumps(document, indent=2))
            (temporary / "image-lock.json").write_text(json.dumps({
                "version": 1,
                "images": [item.model_dump(mode="json") for item in identities],
                "source_built_services": sorted(
                    name for name, service in document["services"].items() if service.get("build")
                ),
                "source_build_reproducibility": "not_verified",
            }, indent=2))
            manifest = manifest.model_copy(update={
                "files": _files(temporary),
                "provenance": manifest.provenance.model_copy(update={
                    "generated_files": [*manifest.provenance.generated_files, "image-lock.json"],
                }),
            })
            manifest = manifest.model_copy(update={"digest": seal_bundle_v2(manifest)})
        (temporary / "manifest.json").write_text(manifest.model_dump_json(indent=2))
        manifest = load_bundle_v2(temporary)
        preflight_bundle(
            temporary,
            manifest,
            parallelism=job.runtime.parallelism,
            secret_refs={k: v.purpose for k, v in job.agent.secret_refs.items()},
            allow_compose=True,
        )
        if output.exists():
            shutil.rmtree(output)
        temporary.rename(output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


async def validate_native(
    *, job, source, authoring, bundle, manifest, work, values, redactions, executor,
    readiness_only=False,
):
    import random
    from .action_certification import certify_actions
    from .authoring_runtime_validation import (
        RuntimeValidationError,
        _write_runtime_evidence,
        _freeze_certified_bundle,
        _generic_candidate_hash,
    )
    from .certification import CheckStatus
    from .chat_call_runner import (
        _scenario_document,
        _with_scenario_edits,
        _tools,
        _chat_target_timeout_seconds,
    )
    from .chat_readiness import probe_chat_ingress
    from .contract import AgentContract
    from .hosted_scheduler import _run_phase, _classify_ready, _classify_check
    from .scenario_source import load_scenarios
    from .tool_certification import certify_tool_inventory, ToolAvailability
    from .world.observation import ObservationWorldFactory

    contract = AgentContract.model_validate_json(
        (authoring / "contract.json").read_text()
    )
    report = certify_tool_inventory(
        contract, bundle, external_provider=False, native_observation=True
    )
    if any(item.availability is ToolAvailability.REJECTED for item in report.tools):
        raise RuntimeValidationError(
            "environment", "native tool inventory contains unreachable declarations"
        )
    scenarios = [] if readiness_only else load_scenarios(bundle)
    if not readiness_only and (not scenarios or len(scenarios) != job.scenario_count):
        raise RuntimeValidationError("environment", "native scenario count mismatch")
    owner = provider(values, authoring / "generic-harness" / "compose-logs")
    try:
        runtimes = await owner.provision(
            manifest,
            source=source,
            bundle_dir=bundle,
            work_directory=work,
            instances=job.runtime.parallelism,
        )
        factory = ObservationWorldFactory(bundle)
        if readiness_only:
            from .authoring_runtime_validation import _probe_runtime_readiness

            await _probe_runtime_readiness(
                job=job, source=source, authoring=authoring, manifest=manifest,
                contract=contract, provider=owner, runtime=runtimes[0], factory=factory,
                work=work, values=values, redactions=redactions,
                compare_baseline=False,
            )
            return 0
        from .local_certified_runtime import LocalCertifiedSink

        phase_sink = LocalCertifiedSink(
            authoring / "generic-harness" / "phase-observations"
        )
        baseline_evidence = []
        for scenario in scenarios:
            runtime = runtimes[0]
            await owner.reset(runtime, work_directory=work)
            world = await factory.create(runtime, rng=random.Random(job.seed or 0))
            for name, fn in (("setup", scenario.setup), ("ready", scenario.ready)):
                result = await _run_phase(
                    fn,
                    world.read_only() if name == "ready" else world,
                    timeout=30,
                    phase=name,
                    executor=executor,
                    audit_log=phase_sink.log,
                    audit_context={"scenario_key": scenario.scenario_key, "world_index": runtime.world_index, "stage": "validation"},
                )
                if result.failure:
                    raise RuntimeValidationError(
                        "scenarios",
                        f"{scenario.scenario_key}: {name}: {result.failure}",
                    )
                if name == "ready":
                    verdict = _classify_ready(result.value)
                    if verdict.broken or not verdict.held:
                        raise RuntimeValidationError(
                            "scenarios", f"{scenario.scenario_key}: ready failed"
                        )
            deterministic = [goal for goal in scenario.sub_goals if not goal.judged]
            held_without_action = []
            for goal in deterministic:
                result = await _run_phase(
                    goal.check,
                    world.read_only(),
                    [],
                    timeout=30,
                    phase="check",
                    executor=executor,
                    audit_log=phase_sink.log,
                    audit_context={"scenario_key": scenario.scenario_key, "world_index": runtime.world_index, "stage": "validation", "check": goal.name},
                )
                if result.failure:
                    raise RuntimeValidationError(
                        "scenarios",
                        f"{scenario.scenario_key}: {goal.name}: {result.failure}",
                    )
                verdict = _classify_check(result.value)
                if verdict.broken:
                    raise RuntimeValidationError(
                        "scenarios",
                        f"{scenario.scenario_key}: {goal.name}: invalid verdict",
                    )
                if verdict.held:
                    held_without_action.append(goal.name)
            if deterministic and len(held_without_action) == len(deterministic):
                raise RuntimeValidationError(
                    "scenarios",
                    f"{scenario.scenario_key}: all deterministic checks pass without agent action",
                )
            baseline_evidence.append(
                {
                    "scenario_key": scenario.scenario_key,
                    "no_action": "passed" if deterministic else "not_applicable",
                    "checks_held_without_action": held_without_action,
                    "solvability": "not_run",
                    "limitation": "Reference actions were not executed. Judged goals require conversation grading.",
                }
            )
            if scenario is scenarios[0]:
                key = (
                    getattr(scenario, "source_scenario_key", None)
                    or scenario.scenario_key
                )
                document = _with_scenario_edits(
                    _scenario_document(bundle, key), job.metadata, key
                )
                await probe_chat_ingress(
                    interface=contract.runtime.interface,
                    endpoint=runtime.endpoints["target_http"].address,
                    document=document,
                    scenario_key=scenario.scenario_key,
                    authoring=authoring,
                    candidate_hash=_generic_candidate_hash(source, authoring),
                    bundle_digest=manifest.digest,
                    secret_values=values,
                    redaction_values=redactions,
                    tools=_tools(contract)
                    if contract.runtime.interface.include_tools
                    else (),
                    scenario_id=scenario.scenario_id,
                    timeout=_chat_target_timeout_seconds(),
                )
        evidence_path = authoring / "generic-harness" / "native-no-action.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(baseline_evidence, indent=2) + "\n")
        _write_runtime_evidence(
            job=job,
            authoring=authoring,
            manifest=manifest,
            count=len(scenarios),
            external_provider=False,
            native_observation=True,
            tool_report=report,
            action_report=certify_actions([], adapter=None),
            reset_equivalence=CheckStatus.NOT_APPLICABLE,
            world_isolation=CheckStatus.NOT_APPLICABLE,
            chat_ingress=CheckStatus.PASSED,
        )
        _freeze_certified_bundle(bundle, authoring)
        return len(scenarios)
    finally:
        await owner.close(work_directory=work)
