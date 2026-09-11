from typing import List, Dict, Any, Union, Optional, Literal
from pydantic import BaseModel, Field, validator, model_validator
import hashlib
import pandas as pd
import json
from fi.simulate.agent.wrapper import SimulationArtifact, SimulationEvent

PERSONA_TEMPERAMENT_AXES = ("rajas", "sattva", "tamas")
# Byte-equal to fi.opt.optimizers.council.GUNA_AXES (council.py:40) — pinned by
# a cross-equality unit test, never imported (fi.simulate must not depend on
# fi.opt). Scholarly design device used as deterministic engineering metadata —
# same framing as council.py:36-38; the four honest limits of Phase-7
# RESEARCH §3.4 are binding (design metaphor, not a psychometric claim about
# simulated users; axes ship ONLY with a transcript-observable realization
# metric — see behavior_policy.py).

PERSONA_EVIDENCE_CLASSES = (
    "hand_written", "schema_sampled", "policy_evolved",
    "trace_mined", "cloud_downloaded", "legacy",
)
SCENARIO_KINDS = ("task", "adversarial", "regression", "perturbation", "composed")


class PersonaIdentity(BaseModel):
    """Layer 1 — minimal, behavioral-first. Demographics optional, non-default,
    lint-flagged (P7-D4; SCOPE: demographics explain ~1.5% of behavioral
    variance). NO field here ever backs a realism claim."""
    name: Optional[str] = None
    role: Optional[str] = None
    summary: Optional[str] = None
    language: Optional[str] = None                  # locale for bias-lint re-runs
    demographics: Dict[str, Any] = Field(default_factory=dict)  # ALWAYS lint-flagged
    style_notes: List[str] = Field(default_factory=list)        # verbatim vendor/platform text


class PersonaTemperament(BaseModel):
    """Layer 2 — continuous axes that COMPILE into layer 3 (behavior_policy.py);
    never prose adjectives in a prompt (PPol / 2604.00026)."""
    rajas: float = Field(0.5, ge=0.0, le=1.0)    # activation/urgency dial
    sattva: float = Field(0.5, ge=0.0, le=1.0)   # clarity/cooperation dial
    tamas: float = Field(0.5, ge=0.0, le=1.0)    # inertia/withdrawal dial


class BehaviorPolicy(BaseModel):
    """Layer 3 — the executable, searchable representation (PPol). The six
    parameters map 1:1 onto the canon axes V1_PERSONA_BEHAVIOR_AXES
    ["patience", "disclosure", "interruption", "escalation", "cooperation",
    "repair"] (ARCH §4), each paired with its transcript-observable
    realization metric (behavior_policy.py); a parameter without one DOES NOT
    SHIP (R§3.4 limit 4). verbosity/tempo dials are POST-v1.x — they exceed
    the closed axis set and ship no realization metric in v1 (ARCH Decision 4)."""
    patience_curve: List[float] = Field(default_factory=lambda: [1.0])  # axis: patience — per-turn patience 0..1
    disclosure_policy: float = Field(0.7, ge=0.0, le=1.0)   # axis: disclosure — fraction of known facts volunteered
    interruption_propensity: float = Field(0.1, ge=0.0, le=1.0)  # axis: interruption
    escalation_schedule: List[float] = Field(default_factory=lambda: [0.0])  # axis: escalation — per-turn pressure 0..1
    cooperation_bounds: float = Field(0.8, ge=0.0, le=1.0)  # axis: cooperation — ceiling on helpfulness (anti-cooperative-bias)
    repair_propensity: float = Field(0.5, ge=0.0, le=1.0)   # axis: repair — good-faith repair probability after misunderstanding


class PersonaFact(BaseModel):
    """Layer 4 — retrievable knowledge store (2603.19313: retrieved, not
    prompt-stuffed). Goals are NOT here: the Scenario owns the task
    (2601.15290 separation)."""
    key: str
    value: str
    disclosure: Literal["volunteer", "on_request", "withhold"] = "on_request"


class AttackConditioning(BaseModel):
    """Optional red-team conditioning (PCAP). Values must be members of the
    gate-enforced 10x6 taxonomy — membership is asserted FACADE-side
    (studio.validate_persona) and by the gate, not here (fi.simulate must not
    import fi.alk.trinity)."""
    strategies: List[str] = Field(default_factory=list)   # ⊆ V1_REDTEAM_RESEARCH_ATTACK_TYPES
    surfaces: List[str] = Field(default_factory=list)     # ⊆ V1_REDTEAM_RESEARCH_SURFACES
    in_character_floor: float = Field(0.6, ge=0.0, le=1.0)


class PersonaProvenance(BaseModel):
    """Layer 5 — how the persona was made + what it is calibrated FOR.
    No class ever claims population representativeness (2602.18462 hard
    limit — stated here in the schema, not just docs)."""
    evidence_class: Literal[
        "hand_written", "schema_sampled", "policy_evolved",
        "trace_mined", "cloud_downloaded", "legacy",
    ] = "legacy"
    calibrated: bool = False
    calibration_ref: Optional[str] = None         # content hash of the calibration artifact
    source_format: Optional[str] = None           # "vapi" | "retell" | "futureagi" | None
    raw: Optional[str] = None                     # verbatim vendor/source text (ARCH Decision 8 losslessness)
    pin: Dict[str, Any] = Field(default_factory=dict)   # download pin block (studio/_download.py)
    representativeness_claim: Literal["none"] = "none"  # frozen; the schema-level hard limit


class Persona(BaseModel):
    """
    A single test case defining a customer persona, situation, and desired outcome.
    """
    persona: Dict[str, Any] = Field(..., description="Characteristics of the simulated customer (e.g., name, age, communication_style).")
    situation: str = Field(..., description="The context or reason for the customer's call.")
    outcome: str = Field(..., description="The desired goal or resolution for the conversation.")
    # ---- Phase 7 typed layers (ALL optional => full back-compat) ----------
    identity: Optional[PersonaIdentity] = None
    temperament: Optional[PersonaTemperament] = None
    behavior_policy: Optional[BehaviorPolicy] = None
    knowledge: List[PersonaFact] = Field(default_factory=list)
    attack: Optional[AttackConditioning] = None
    provenance: Optional[PersonaProvenance] = None
    version: Optional[str] = None                 # content address, ARCH §2d

    @property
    def is_typed(self) -> bool:
        """True when the persona carries an executable layer-3 policy —
        the precondition for fidelity measurement (fidelity.py)."""
        return self.behavior_policy is not None

    def content_hash(self) -> str:
        payload = self.model_dump(exclude={"version"}, exclude_none=True)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def _stamp_version(self) -> "Persona":
        if self.version is None and self.is_typed:
            object.__setattr__(self, "version", self.content_hash())
        return self


class ScenarioGoal(BaseModel):
    """Goal/state progression — the task-state half of the 2601.15290 split."""
    states: List[str] = Field(default_factory=list)          # ordered milestone names
    success_state: Optional[str] = None
    failure_states: List[str] = Field(default_factory=list)


class VerificationSpec(BaseModel):
    checks: List[Dict[str, Any]] = Field(default_factory=list)  # eval-template refs / predicates
    threshold: float = Field(0.7, ge=0.0, le=1.0)


class CoverageDeclaration(BaseModel):
    """Declared coverage axes (2605.26521 obligations, not counts)."""
    intents: List[str] = Field(default_factory=list)
    personas: List[str] = Field(default_factory=list)            # persona version hashes
    perturbations: List[str] = Field(default_factory=list)
    tool_obligations: List[str] = Field(default_factory=list)    # "allow:<tool>" / "deny:<tool>"
    delegation_obligations: List[str] = Field(default_factory=list)


class ScenarioConstraints(BaseModel):
    """The tau^2 move: the scenario BOUNDS persona freedom so fidelity is
    checkable — declared tools, observable state, goal machine."""
    declared_tools: List[str] = Field(default_factory=list)
    observable_state: Dict[str, Any] = Field(default_factory=dict)
    max_user_knowledge: List[str] = Field(default_factory=list)  # PersonaFact keys usable here


class EscalationStep(BaseModel):
    turn: int = Field(..., ge=1)
    pressure: float = Field(..., ge=0.0, le=1.0)
    tactic: str                                    # free label, e.g. "reframe", "urgency", "authority"


class EscalationArc(BaseModel):
    """Turn-wise in-character escalation (Crescendo finding, R§1 2605.04019)."""
    steps: List[EscalationStep]
    hold_character: bool = True


class Scenario(BaseModel):
    """
    Defines a collection of test cases for a simulation.
    """
    name: str = Field(..., description="A unique name for the scenario.")
    description: Optional[str] = Field(None, description="A brief description of what this scenario tests.")
    dataset: List[Persona] = Field(..., description="A list of personas defining the test cases.")
    # ---- Phase 7 typing (ALL optional; kind=None == legacy untyped — NEVER
    # silently retyped; studio-created scenarios must carry an explicit kind;
    # the gate requires kinds only on studio-library scenarios — ARCH §2a) ----
    kind: Optional[Literal["task", "adversarial", "regression", "perturbation", "composed"]] = None
    goal: Optional[ScenarioGoal] = None
    verification: Optional[VerificationSpec] = None
    coverage: Optional[CoverageDeclaration] = None
    constraints: Optional[ScenarioConstraints] = None
    escalation: Optional[EscalationArc] = None
    attack_type: Optional[str] = None              # adversarial kind: required
    attack_surface: Optional[str] = None           # adversarial kind: required
    version: Optional[str] = None                  # content address, ARCH §2d
    parent_version: Optional[str] = None           # expansion lineage (studio/_coverage.py)

    @validator('dataset', pre=True)
    def load_dataset(cls, v: Union[List[Dict], str]) -> List[Dict]:
        if isinstance(v, str):
            if v.endswith('.csv'):
                return pd.read_csv(v).to_dict('records')
            elif v.endswith('.json'):
                with open(v, 'r') as f:
                    return json.load(f)
            else:
                raise ValueError("Unsupported file type for dataset. Please use .csv or .json.")
        return v

    def content_hash(self) -> str:                 # same canonicalization as Persona
        payload = self.model_dump(exclude={"version"}, exclude_none=True)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def _kind_contract(self) -> "Scenario":
        if self.kind == "adversarial":
            if not (self.attack_type and self.attack_surface and self.escalation):
                raise ValueError(
                    "adversarial scenarios must declare attack_type, "
                    "attack_surface, and an escalation arc"
                )
        if self.kind is not None and self.version is None:
            object.__setattr__(self, "version", self.content_hash())
        return self

class TestCaseResult(BaseModel):
    """
    Represents the result of a single test case.
    """
    persona: Persona = Field(..., description="The original persona that was run.")
    transcript: str = Field(..., description="The full transcript of the conversation.")
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Normalized message trajectory including user, assistant, and tool turns.",
    )
    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tool calls observed during the run, when wrappers expose them.",
    )
    artifacts: List[SimulationArtifact] = Field(
        default_factory=list,
        description="Multimodal artifacts observed during the run, such as audio, images, screenshots, files, and traces.",
    )
    events: List[SimulationEvent] = Field(
        default_factory=list,
        description="Normalized simulation events, including tools, memory, voice, browser/CUA, and framework spans.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Engine, scenario, timing, stop reason, and other run metadata.",
    )
    evaluation: dict | None = Field(
        default=None,
        description="Optional evaluation results (scores, reasons) keyed by template.",
    )
    audio_input_path: str | None = Field(
        default=None,
        description="Optional path to recorded customer (input) audio for this test.",
    )
    audio_output_path: str | None = Field(
        default=None,
        description="Optional path to recorded agent (output) audio for this test.",
    )
    audio_combined_path: str | None = Field(
        default=None,
        description="Optional path to a single WAV containing the mixed conversation.",
    )
    audio_stereo_path: str | None = Field(
        default=None,
        description="Optional path to a 2-channel WAV: ch0 customer, ch1 assistant.",
    )

class TestReport(BaseModel):
    """
    A comprehensive report aggregating the results of all test cases in a scenario.
    """
    results: List[TestCaseResult] = Field(default_factory=list, description="A list of results for each test case.")

    def admissible_results(self) -> List[TestCaseResult]:
        return [r for r in self.results
                if r.metadata.get("admission", {}).get("admissible", True)]

    def inconclusive_results(self) -> List[TestCaseResult]:
        return [r for r in self.results
                if r.metadata.get("admission", {}).get("verdict") == "inconclusive"]
