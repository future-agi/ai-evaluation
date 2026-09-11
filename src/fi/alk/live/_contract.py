"""Live-lane contract: vocabularies, lane specs, budgets, flag discipline.

Imports: stdlib only. Every substrate module must remain importable (and
its unit tests green) in an environment with no framework extra installed —
the live_lane_boundary gate scans them like any other release module.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Mapping

# --- artifact kind (same kind simulate/cli emit — never a parallel kind) ----
AGENT_LEARNING_RUN_KIND = "agent-learning.run.v1"

# --- evidence classes (R§3.2; PRD §4.1) -----------------------------------
EVIDENCE_CLASSES = ("local_gate", "live_lane", "live_stressed", "captured_fixture")
RELEASE_ADMISSIBLE_EVIDENCE_CLASSES = ("local_gate", "captured_fixture")

# --- failure layers (R§1 #1 HarnessFix; PRD §4.1) --------------------------
FAILURE_LAYERS = ("lane_infra", "framework_runtime", "provider", "agent_behavior")

# --- per-scenario verdicts (R§3.4) ------------------------------------------
VERDICTS = ("pass", "fail", "unstable", "void")

# --- env-flag conventions (PRD §4.1: AGENT_LEARNING_LIVE_<LANE>=1) ---------
LANE_ENV_FLAGS = {
    "livekit": "AGENT_LEARNING_LIVE_LIVEKIT",
    "pipecat": "AGENT_LEARNING_LIVE_PIPECAT",
    "langchain": "AGENT_LEARNING_LIVE_LANGCHAIN",
    "mcp": "AGENT_LEARNING_LIVE_MCP",
    "a2a": "AGENT_LEARNING_LIVE_A2A",
    "credentialed": "AGENT_LEARNING_LIVE_CREDENTIALED",
}

# --- lane → extra map (skip lines and import errors name these) ------------
LANE_EXTRAS = {
    "livekit": "livekit",
    "pipecat": "pipecat",
    "langchain": "langchain",
    "mcp": "mcp",
    "a2a": "a2a",
}

# --- budget caps (P3-D2): 600 s default; voice lanes 900 s -----------------
LANE_BUDGET_S_DEFAULT = 600.0
LANE_BUDGET_S = {"livekit": 900.0, "pipecat": 900.0}

# --- repeat policy (P3-D2) ---------------------------------------------------
DEFAULT_REPEATS = 8
UNSTABLE_ICC_FLOOR = 0.5


def lane_budget_s(lane: str) -> float:
    """Hard wall-clock cap for one lane run (P3-D2)."""

    return LANE_BUDGET_S.get(lane, LANE_BUDGET_S_DEFAULT)


class LaneDisabledError(RuntimeError):
    """Raised when a lane entry point runs without its env flag."""


def require_lane_enabled(lane: str) -> None:
    """Gate every lane entry on its env flag (PRD §4.1).

    The live_lane_boundary gate statically asserts every public lane module
    calls this (unit 4.2 check 3) — the dynamic raise and the static scan
    are two halves of the same discipline.
    """

    flag = LANE_ENV_FLAGS[lane]
    if os.environ.get(flag) != "1":
        raise LaneDisabledError(
            f"live lane '{lane}' is opt-in: set {flag}=1 to run it "
            "(never set in release flows)"
        )


@dataclasses.dataclass(frozen=True)
class LaneSpec:
    """What a lane run was asked to do — shared by runner and lanes."""

    lane: str
    scenario: Mapping[str, Any]
    rung: int = 1
    required_env: tuple[str, ...] = ()
    version_requirement: str | None = None
    repeats: int = DEFAULT_REPEATS
    budget_s: float | None = None
    evidence_class: str = "live_lane"

    def resolved_budget_s(self) -> float:
        if self.budget_s is not None:
            return float(self.budget_s)
        return lane_budget_s(self.lane)


@dataclasses.dataclass
class LaneRun:
    """One repeat of one scenario inside a lane run (a per_repeat row)."""

    index: int
    passed: bool | None
    score: float | None
    failure_layer: str | None
    quarantined: bool
    evidence_class: str
    detail: str = ""
    void_reason: str | None = None
    transcript_path: str | None = None
    transcript_complete: bool | None = None
    transcript_sha256: str | None = None
    step_signature: tuple[str, ...] = ()

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "repeat": self.index,
            "passed": self.passed,
            "score": self.score,
            "failure_layer": self.failure_layer,
            "quarantined": self.quarantined,
            "evidence_class": self.evidence_class,
            "transcript_path": self.transcript_path,
            "transcript_complete": self.transcript_complete,
            "transcript_sha256": self.transcript_sha256,
            "step_signature": list(self.step_signature),
        }
        if self.detail:
            row["detail"] = self.detail
        if self.void_reason:
            row["void_reason"] = self.void_reason
        return row
