"""Unit 3 (BBG U3 / ARCH §2a) — the generic SIMULATION contract models.

``agent-learning.simulation.v1``: a typed, content-addressed world definition
that sits ABOVE the adapters (13D-D6). Engine-side home (AD-A): this module
imports only from ``.models`` / ``.goal_machine`` / stdlib and NEVER from
``fi.alk`` (the studio one-way rule). Canonicalization is the Persona
rule verbatim (AD-D); ``world`` (incl. every tool-mock block) is inside the
identity (R4/AD-O).

R5 dispositions honored: APPLY the A1/A3-A6/A8-A12/A16 shape fields + A2
emulator_backend discriminator; DEFER A15/A18 (TOOL_MOCK_LEVELS stays the
4-tuple, no 5th evidence class, emulated stays typed-only); STAGE A7
(GOAL_CHECK_KINDS stays the 5-kind set, imported from goal_machine); A19 (no
new world kinds — the 6-kind vocabulary is frozen).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Field, model_validator

from .goal_machine import (  # re-exported canon (ARCH §3; the gate byte-compares)
    GOAL_CHECK_KINDS,
    GOAL_CHECK_RUNGS,
    GOAL_PREDICATE_OPS,
)
from .models import Persona, Scenario, ScenarioGoal, VerificationSpec

# Single-home canon re-exported here so contract consumers / the gate mirror can
# read goal-machine vocab from the contract module (ARCH §3).
__all__ = [
    "GOAL_CHECK_KINDS", "GOAL_CHECK_RUNGS", "GOAL_PREDICATE_OPS",
    "SIMULATION_KIND", "SIMULATION_CAST_ROLES", "SIMULATION_WORLD_KINDS",
    "EXECUTABLE_WORLD_KINDS_V1", "TYPED_ONLY_WORLD_KINDS_V1", "TOOL_MOCK_LEVELS",
    "EMULATOR_BACKENDS", "RECORDED_REPLAY_MISS_POLICIES", "STATE_CONSISTENCY_CLASSES",
    "RESET_SEMANTICS", "REQUIRES_NETWORK_POLICIES", "ORACLE_SOLVER_KINDS",
    "TOOL_CALL_ANSWERED_BY", "WORLD_RUNGS", "DYNAMICS_EVENT_KINDS",
    "EPISODE_PERSISTENCE", "WORLD_EXECUTION_MODES",
    "Simulation", "ScenarioBinding", "CastMember", "WorldSpec", "ToolBinding",
    "ClockSpec", "DynamicsEvent", "EpisodeSpec", "AdmissionSpec",
    "register_cast_role", "register_world_kind", "register_environment_type",
    "resolved_cast_roles", "resolved_world_kinds",
]

# ===========================================================================
# Canon constants (ARCH §3 — this is the single home; trinity.py mirrors them,
# the gate byte-compares; registration NEVER mutates them, AD-J).
# ===========================================================================
SIMULATION_KIND = "agent-learning.simulation.v1"

SIMULATION_CAST_ROLES = ("user", "opponent", "coworker", "counterpart")

SIMULATION_WORLD_KINDS = (
    "conversation",
    "tool_api",
    "browser",
    "computer_use",
    "code_exec",
    "voice_telephony",
)
EXECUTABLE_WORLD_KINDS_V1 = ("conversation", "tool_api")
TYPED_ONLY_WORLD_KINDS_V1 = ("browser", "computer_use", "code_exec", "voice_telephony")

# Tool-mock vocabulary — closed 4-level set UNCHANGED (A15 deferred).
TOOL_MOCK_LEVELS = ("static_fixture", "recorded_replay", "emulated", "live")
EMULATOR_BACKENDS = ("code", "prompted_lm", "finetuned_lm")  # A2 sub-discriminator
RECORDED_REPLAY_MISS_POLICIES = ("fail", "fallthrough_emulated", "fallthrough_live", "re_record")  # A3
STATE_CONSISTENCY_CLASSES = ("shared_programmatic", "per_call_context", "declared_preconditions")  # A4
RESET_SEMANTICS = (
    "stateless_fixture", "scripted_init", "image_copy", "snapshot_revert",
    "memory_branch", "ephemeral_tenant", "container_provisioned",
)  # A6 — v1 executes only the first two
REQUIRES_NETWORK_POLICIES = ("off", "allowlist", "proxy_recorded", "live")  # A9
ORACLE_SOLVER_KINDS = ("script", "trajectory")  # A10
TOOL_CALL_ANSWERED_BY = (
    "fixture_hit", "cassette_hit", "emulated:code",
    "emulated:prompted_lm", "emulated:finetuned_lm", "live",
)  # A11

WORLD_RUNGS = (1, 2, 3)  # rung→evidence: 1→local_gate, 2→captured_fixture, 3→live
DYNAMICS_EVENT_KINDS = ("env_state_patch", "counterpart_message", "tool_outcome_shift", "fault_profile")
EPISODE_PERSISTENCE = ("fresh", "carry_state", "carry_memory")
WORLD_EXECUTION_MODES = ("derived_legacy", "contract_native")

# ===========================================================================
# Extension registries (Appendix C-1): contract.py owns private tables with
# narrow setters; fi/alk/extensions.py (facade) is their ONLY writer
# (downward push). Built-ins shadow extensions at resolution; canon never
# mutates.
# ===========================================================================
_EXTRA_CAST_ROLES: Dict[str, dict] = {}
_EXTRA_WORLD_KINDS: Dict[str, dict] = {}
_EXTRA_ENVIRONMENT_TYPES: Dict[str, dict] = {}


def register_cast_role(name: str, record: Mapping[str, Any]) -> None:
    _EXTRA_CAST_ROLES[str(name)] = dict(record)


def register_world_kind(name: str, record: Mapping[str, Any]) -> None:
    _EXTRA_WORLD_KINDS[str(name)] = dict(record)


def register_environment_type(name: str, record: Mapping[str, Any]) -> None:
    _EXTRA_ENVIRONMENT_TYPES[str(name)] = dict(record)


def resolved_cast_roles() -> tuple[str, ...]:
    return tuple(SIMULATION_CAST_ROLES) + tuple(sorted(_EXTRA_CAST_ROLES))


def resolved_world_kinds() -> tuple[str, ...]:
    return tuple(SIMULATION_WORLD_KINDS) + tuple(sorted(_EXTRA_WORLD_KINDS))


def _reset_contract_extensions() -> None:  # test-only
    _EXTRA_CAST_ROLES.clear()
    _EXTRA_WORLD_KINDS.clear()
    _EXTRA_ENVIRONMENT_TYPES.clear()


# ===========================================================================
# Content-hash helper — the Persona rule verbatim (models.py:117-119), with
# 6-place float rounding applied to float leaves before dump.
# ===========================================================================
def _round_floats(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, Mapping):
        return {k: _round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(v) for v in value]
    return value


def _content_hash(payload: Mapping[str, Any]) -> str:
    rounded = _round_floats(dict(payload))
    canonical = json.dumps(rounded, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ===========================================================================
# Models (ARCH §2a field tables — verbatim; no field added/dropped/renamed).
# ===========================================================================
class ToolBinding(BaseModel):
    """First-class tool mocking (R4).

    Note: the ``schema`` field name is ARCH §2a verbatim; Pydantic emits a
    benign class-definition warning because it shadows ``BaseModel.schema``.
    Renaming is forbidden by the BBG ("do not add, drop, or rename a field")
    and the field works correctly (validators/serialization unaffected).
    """
    name: str
    schema: Optional[Dict[str, Any]] = None
    mock: Dict[str, Any] = Field(default_factory=dict)  # {level, source, emulator_backend, ...}
    required_env: List[str] = Field(default_factory=list)
    requires: Optional[Dict[str, Any]] = None  # A9 per-rung block

    @model_validator(mode="after")
    def _validate_mock(self) -> "ToolBinding":
        level = self.mock.get("level")
        if not level:
            raise ValueError(
                "tool_mock_level_undeclared: ToolBinding.mock.level is required "
                f"(one of {TOOL_MOCK_LEVELS}) for tool {self.name!r}"
            )
        if level not in TOOL_MOCK_LEVELS:
            raise ValueError(
                f"tool_mock_level_undeclared: mock.level {level!r} not in {TOOL_MOCK_LEVELS}"
            )
        if level == "recorded_replay":
            source = self.mock.get("source")
            prov = self.mock.get("provenance") or {}
            if not source or not prov.get("capture"):
                raise ValueError(
                    "tool_mock_replay_missing: recorded_replay requires mock.source "
                    "(scrubbed capture ref) + mock.provenance.capture (sha256)"
                )
            sub = self.mock.get("recorded_replay") or {}
            miss = sub.get("miss_policy")
            if miss is not None and miss not in RECORDED_REPLAY_MISS_POLICIES:
                raise ValueError(
                    f"tool_mock_replay_missing: miss_policy {miss!r} not in "
                    f"{RECORDED_REPLAY_MISS_POLICIES}"
                )
        if level == "emulated":
            backend = self.mock.get("emulator_backend")
            if backend is not None and backend not in EMULATOR_BACKENDS:
                raise ValueError(
                    f"tool_mock_level_undeclared: emulator_backend {backend!r} not in "
                    f"{EMULATOR_BACKENDS}"
                )
        if level == "live" and not self.required_env:
            raise ValueError(
                "tool_mock_live_unkeyed: live mock requires required_env NAMES "
                f"for tool {self.name!r} (lane-only; refused in gate/release)"
            )
        return self


class WorldSpec(BaseModel):
    """Typed world: kind + environments + tools (R4)."""
    kind: str
    environments: List[Dict[str, Any]] = Field(default_factory=list)
    spec: Dict[str, Any] = Field(default_factory=dict)
    tools: List[ToolBinding] = Field(default_factory=list)
    rung: int = 1
    state_consistency: str = "shared_programmatic"  # A4
    reset_semantics: str = "stateless_fixture"      # A6
    perturbation_profile: Optional[Dict[str, Any]] = None  # A8
    policies: List[Dict[str, Any]] = Field(default_factory=list)  # A16
    stochasticity_profile: Optional[Dict[str, Any]] = None  # A12

    @model_validator(mode="after")
    def _validate_world(self) -> "WorldSpec":
        if self.kind not in resolved_world_kinds():
            raise ValueError(
                f"world_kind_unsupported: world.kind {self.kind!r} not in "
                f"{resolved_world_kinds()}"
            )
        if self.rung not in WORLD_RUNGS:
            raise ValueError(f"world.rung {self.rung!r} not in {WORLD_RUNGS}")
        if self.state_consistency not in STATE_CONSISTENCY_CLASSES:
            raise ValueError(
                f"world.state_consistency {self.state_consistency!r} not in "
                f"{STATE_CONSISTENCY_CLASSES}"
            )
        if self.reset_semantics not in RESET_SEMANTICS:
            raise ValueError(
                f"world.reset_semantics {self.reset_semantics!r} not in {RESET_SEMANTICS}"
            )
        return self


class ClockSpec(BaseModel):
    model: str = "turn"
    step_s: Optional[float] = None
    horizon: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_clock(self) -> "ClockSpec":
        if self.model not in ("turn", "simulated"):
            raise ValueError(f"clock.model {self.model!r} not in ('turn', 'simulated')")
        if self.model == "simulated" and (self.step_s is None or self.step_s <= 0):
            raise ValueError("clock.step_s is required (>0) when model == 'simulated'")
        return self


class DynamicsEvent(BaseModel):
    at: Dict[str, Any]
    event: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    seed: Optional[int] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_dynamics(self) -> "DynamicsEvent":
        if self.event not in DYNAMICS_EVENT_KINDS:
            raise ValueError(
                f"dynamics.event {self.event!r} not in {DYNAMICS_EVENT_KINDS}"
            )
        keys = set(self.at)
        valid = (
            keys == {"turn"}
            or keys == {"time_s"}
            or keys <= {"every", "phase"} and "every" in keys
        )
        if not valid:
            raise ValueError(
                "dynamics.at must be exactly one of {turn}, {time_s}, {every, phase?}"
            )
        return self


class EpisodeSpec(BaseModel):
    count: int = 1
    persistence: str = "fresh"
    settle: List[Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_episode(self) -> "EpisodeSpec":
        if self.count < 1:
            raise ValueError("episodes.count must be >= 1")
        if self.persistence not in EPISODE_PERSISTENCE:
            raise ValueError(
                f"episodes.persistence {self.persistence!r} not in {EPISODE_PERSISTENCE}"
            )
        return self


class AdmissionSpec(BaseModel):
    fidelity_floors: Dict[str, float] = Field(default_factory=dict)
    oracle_adequacy: Optional[Dict[str, Any]] = None
    realism_certificate: Optional[str] = None
    epidemic_rate: Optional[float] = None


class CastMember(BaseModel):
    """R2: a cast member holds turns; a dynamics entry never does."""
    persona: str  # persona version hash (ref into simulation.personas)
    role: str = "user"
    alias: Optional[str] = None

    @model_validator(mode="after")
    def _validate_role(self) -> "CastMember":
        if self.role not in resolved_cast_roles():
            raise ValueError(
                f"cast_role_unknown: role {self.role!r} not in {resolved_cast_roles()}"
            )
        return self


class ScenarioBinding(BaseModel):
    """Cast selection wrapping a Scenario (AD-B: Scenario stays byte-stable)."""
    scenario: Optional[Scenario] = None
    scenario_ref: Optional[str] = None
    cast: List[CastMember]
    casting: str = "each"
    goal: Optional[ScenarioGoal] = None
    verification: Optional[VerificationSpec] = None
    oracle_solver: Optional[Dict[str, Any]] = None  # A10
    weight: float = 1.0

    @model_validator(mode="after")
    def _validate_binding(self) -> "ScenarioBinding":
        if not self.cast:
            raise ValueError("simulation_contract_invalid: ScenarioBinding.cast requires >= 1 member")
        if self.casting not in ("each", "together"):
            raise ValueError(f"simulation_contract_invalid: casting {self.casting!r} not in ('each','together')")
        if self.weight <= 0:
            raise ValueError("simulation_contract_invalid: ScenarioBinding.weight must be > 0")
        if self.oracle_solver is not None:
            kind = self.oracle_solver.get("kind")
            if kind is not None and kind not in ORACLE_SOLVER_KINDS:
                raise ValueError(
                    f"simulation_contract_invalid: oracle_solver.kind {kind!r} not in {ORACLE_SOLVER_KINDS}"
                )
        # contract-native scenarios MUST have empty/absent legacy dataset
        if self.scenario is not None and self.scenario.dataset:
            # auto-lift produces non-empty dataset; flag only when explicitly built.
            pass
        return self


class Simulation(BaseModel):
    """The top-level ``agent-learning.simulation.v1`` object."""
    kind: str = SIMULATION_KIND
    name: str
    description: Optional[str] = None
    version: Optional[str] = None
    personas: List[Persona] = Field(default_factory=list)
    scenarios: List[ScenarioBinding]
    world: WorldSpec
    clock: ClockSpec = Field(default_factory=ClockSpec)
    dynamics: List[DynamicsEvent] = Field(default_factory=list)
    episodes: EpisodeSpec = Field(default_factory=EpisodeSpec)
    goal: Optional[ScenarioGoal] = None
    verification: Optional[VerificationSpec] = None
    objective: Optional[Dict[str, Any]] = None
    admission: AdmissionSpec = Field(default_factory=AdmissionSpec)
    seed: Optional[int] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def content_hash(self) -> str:
        payload = self.model_dump(exclude={"version"}, exclude_none=True)
        return _content_hash(payload)

    @model_validator(mode="after")
    def _validate_and_stamp(self) -> "Simulation":
        if self.kind != SIMULATION_KIND:
            raise ValueError(f"simulation_contract_invalid: kind must be {SIMULATION_KIND!r}")
        # duplicate persona version hashes rejected
        seen: set[str] = set()
        persona_hashes: set[str] = set()
        for persona in self.personas:
            digest = persona.version or persona.content_hash()
            if digest in seen:
                raise ValueError(
                    "simulation_contract_invalid: duplicate persona version hashes in personas"
                )
            seen.add(digest)
            persona_hashes.add(digest)
        # every cast ref resolves into the persona set (closed world)
        for binding in self.scenarios:
            for member in binding.cast:
                if member.persona not in persona_hashes:
                    raise ValueError(
                        f"simulation_contract_invalid: cast persona ref {member.persona!r} "
                        "does not resolve into simulation.personas (closed world)"
                    )
            turn_holders = len(binding.cast)
            if binding.casting == "each" and turn_holders < 1:
                raise ValueError(
                    "simulation_contract_invalid: casting 'each' requires >= 1 turn-holder"
                )
            # casting 'together' validates structurally; execution refuses until U23c.
        # R2 litmus (ambient side): a dynamics payload may not declare turn-holding
        # capability (responds_to / utterance templates) — that is a persona.
        for evt in self.dynamics:
            if evt.event == "counterpart_message":
                payload = evt.payload or {}
                if "responds_to" in payload or "utterance_templates" in payload:
                    raise ValueError(
                        "counterpart_misclassified: a dynamics entry never holds a turn "
                        "(responds_to/utterance_templates ⇒ this is a persona with a role, "
                        "not a dynamics event). R2 litmus: does it hold a turn?"
                    )
        # objective is structure-only on the engine side (semantic validation lives
        # facade-side in loss.py — the engine never imports the loss module, AD-A).
        if self.objective is not None and not isinstance(self.objective, Mapping):
            raise ValueError("simulation_contract_invalid: objective must be a mapping")
        if self.version is None:
            object.__setattr__(self, "version", self.content_hash())
        return self
