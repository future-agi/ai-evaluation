from __future__ import annotations

from fi.simulate.runtime.capabilities import CapabilitySet
from fi.simulate.runtime.ids import new_plan_id
from fi.simulate.runtime.plan import (
    AdapterRef,
    ArtifactPlan,
    EvidencePlan,
    SimulationPlan,
)
from fi.simulate.runtime.spec import SimulationSpec

# Import for the registration side effect: builtin endpoint profiles populate
# the endpoint_registry that build_plan reads capabilities from, and builtin
# simulator descriptors populate the simulator_registry that build_plan validates.
from fi.simulate.endpoints import profiles as _endpoint_profiles  # noqa: F401
from fi.simulate.simulator import builtins as _simulator_builtins  # noqa: F401
from fi.simulate.registry import (
    AdapterNotFound,
    endpoint_registry,
    environment_registry,
    simulator_registry,
)


class UnsupportedWorldKind(ValueError):
    """Raised when a spec's ``world_kind`` isn't one the environment declares."""

    def __init__(self, adapter: str, world_kind: str, supported: list[str]) -> None:
        self.adapter = adapter
        self.world_kind = world_kind
        self.supported = sorted(supported)
        super().__init__(
            f"world_kind_unsupported: {world_kind!r} is not supported by "
            f"environment {adapter!r}; supported: {self.supported}"
        )


def build_plan(spec: SimulationSpec) -> SimulationPlan:
    # Validate the simulator adapter against the registry (typo → clear error).
    # Only enforce when the registry is populated with named builtins, so a
    # third-party simulator registered by an integrator still passes.
    if simulator_registry.get_or_none(spec.simulator.adapter) is None:
        known = simulator_registry.names()
        if known:
            raise AdapterNotFound("simulator", spec.simulator.adapter, known)
    # Validate world_kind against what the environment plugin declares. Empty
    # declaration = unrestricted (third-party plugins that don't declare stay
    # unbroken); a declared, non-empty list is enforced (typo → clear error).
    env_factory = environment_registry.get_or_none(spec.environment.adapter)
    supported_world_kinds = list(
        getattr(getattr(env_factory, "manifest", None), "world_kinds", []) or []
    )
    if supported_world_kinds and spec.environment.world_kind not in supported_world_kinds:
        raise UnsupportedWorldKind(
            spec.environment.adapter,
            spec.environment.world_kind,
            supported_world_kinds,
        )
    profile = endpoint_registry.get_or_none(spec.target.adapter)
    supported = (
        sorted(profile.manifest.capabilities.supported()) if profile else []
    )
    root_directory = spec.artifacts.root_directory or f".fagi/runs/{spec.run_id}"
    return SimulationPlan(
        plan_id=new_plan_id(),
        run_id=spec.run_id,
        spec_hash=spec.spec_hash or spec.content_hash(),
        environment_adapter=AdapterRef(
            name=spec.environment.adapter,
            version=spec.environment.adapter_version,
            config=spec.environment.config,
        ),
        target_adapter=AdapterRef(
            name=spec.target.adapter,
            version=spec.target.adapter_version,
            config=spec.target.config,
        ),
        simulator_adapter=AdapterRef(
            name=spec.simulator.adapter,
            version=spec.simulator.adapter_version,
            config=spec.simulator.config,
        ),
        negotiated_capabilities=CapabilitySet(
            required=spec.target.required_capabilities,
            supported=supported,
        ),
        runtime_requirements=spec.execution.runtime,
        timeout_policy=spec.execution.timeout,
        retry_policy=spec.execution.retry,
        cleanup_policy=spec.execution.cleanup,
        evidence_plan=EvidencePlan(
            source_ids=[source.source_id for source in spec.evidence.sources],
            required_capabilities=spec.evidence.required_capabilities,
        ),
        artifact_plan=ArtifactPlan(
            enabled=spec.artifacts.enabled,
            root_directory=root_directory,
            record_audio=spec.artifacts.record_audio,
            required_types=spec.artifacts.required_types,
        ),
    )
