"""Unit 8 (BBG U8 / ARCH §3) — practice vocabularies + canon constants.

Every constant ARCH §3 freezes for the Practice Loop, verbatim. RU-1 numeric
defaults. Evidence/verdict vocab is IMPORTED from live/_contract.py, never
redeclared (ARCH §1.7). The Unit-20 gate byte-compares these.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..live._contract import (  # noqa: F401 (re-exported canon)
    DEFAULT_REPEATS,
    EVIDENCE_CLASSES,
    RELEASE_ADMISSIBLE_EVIDENCE_CLASSES,
    UNSTABLE_ICC_FLOOR,
    VERDICTS,
)
from ..loss import (  # noqa: F401 (re-export the objective/loss-report kinds)
    AGENT_LEARNING_LOSS_REPORT_KIND,
    AGENT_LEARNING_OBJECTIVE_KIND,
)

# --- artifact kinds (RU-4) -------------------------------------------------
AGENT_LEARNING_PRACTICE_LOOP_KIND = "agent-learning.practice-loop.v1"
AGENT_LEARNING_PRACTICE_RESULT_KIND = "agent-learning.practice-result.v1"
AGENT_LEARNING_PRACTICE_REPORT_KIND = "agent-learning.practice-report.v1"
AGENT_LEARNING_PRACTICE_DEFICITS_KIND = "agent-learning.practice-deficits.v1"
AGENT_LEARNING_PRACTICE_DRILL_KIND = "agent-learning.practice-drill.v1"
AGENT_LEARNING_PRACTICE_UPDATE_KIND = "agent-learning.practice-update.v1"
AGENT_LEARNING_CONSOLIDATED_LESSON_KIND = "agent-learning.consolidated-lesson.v1"
AGENT_LEARNING_PRACTICE_CALIBRATION_KIND = "agent-learning.practice-calibration.v1"

PRACTICE_ARTIFACT_KINDS = (
    AGENT_LEARNING_PRACTICE_LOOP_KIND,
    AGENT_LEARNING_PRACTICE_RESULT_KIND,
    AGENT_LEARNING_PRACTICE_REPORT_KIND,
    AGENT_LEARNING_PRACTICE_DEFICITS_KIND,
    AGENT_LEARNING_PRACTICE_DRILL_KIND,
    AGENT_LEARNING_PRACTICE_UPDATE_KIND,
    AGENT_LEARNING_CONSOLIDATED_LESSON_KIND,
    AGENT_LEARNING_PRACTICE_CALIBRATION_KIND,
)

# --- phases + vocabularies (ARCH §3) ---------------------------------------
PRACTICE_PHASES = ("assess", "diagnose", "drill", "update", "consolidate", "calibrate")
SCAFFOLD_TYPES = ("world_simplification", "hint_tool", "worked_example", "relaxed_success")
ZPD_VERDICTS = ("in_band", "vygotsky_form", "below_band", "above_band", "unstable")
CALIBRATION_VERDICTS = ("learned", "fluent_not_learned", "in_rotation", "plateaued", "zpd_exited")
LADDER_STATES = ("episodic", "instruction", "skill")
PRACTICE_REPLAY_INTERVALS = (1, 2, 4, 8, 16)  # cap 16
STORE_STATUSES = ("active", "retired")
RETIREMENT_REASONS = ("repeated_failure", "obsolete")
LESSON_KINDS = ("instruction_block", "config_patch", "skill")

# --- 13D-5 capstone ablation knobs (additive; the experiment path only) -----
# Real trainer config flags that change run_practice_loop behaviour (never
# labels): A1 disables ZPD filtering, A2 disables standing spaced reviews
# (replay only at promotion), A3 skips the consolidate phase entirely, A4
# disables the calibration learned-gate (fixed-k, never stop early).
PRACTICE_ABLATIONS = ("a1_no_zpd", "a2_no_spacing", "a3_no_consolidation", "a4_no_calibration")

# --- RU-1 defaults ---------------------------------------------------------
ZPD_BAND = (0.2, 0.7)
REVIEW_RATIO = 0.25
BUDGET_PLAN = (0.25, 0.35, 0.25, 0.15)  # assess / drill / update / review
PRACTICE_STORE_ACTIVE_CAP = 64
SCAFFOLD_FADE_DEFAULT = (1.0, 0.5, 0.0)  # MUST end at 0.0
MAX_REPLAY_INTERVAL = 16
DEFAULT_MAX_ROUNDS = 8
DEFAULT_INNER_OPERATOR_BACKEND = "society"

# --- store placement (AD-G — the Phase-8 ledger precedent) -----------------
LESSON_ID_PREFIX = "lesson_"
PRACTICE_STORE_PATH_ENV = "AGENT_LEARNING_PRACTICE_STORE_PATH"
PRACTICE_STORE_HOME_ENV = "AGENT_LEARNING_HOME"
PRACTICE_STORE_DIR_NAME = "practice"
PRACTICE_STORE_FILE_NAME = "records.jsonl"


def practice_store_path(override: str | Path | None = None) -> Path:
    """Resolve the consolidation store path (AD-G). Precedence: explicit arg >
    AGENT_LEARNING_PRACTICE_STORE_PATH > ${AGENT_LEARNING_HOME:-~/.agent-learning}
    /practice/records.jsonl."""
    if override is not None:
        return Path(override)
    env_override = os.environ.get(PRACTICE_STORE_PATH_ENV)
    if env_override:
        return Path(env_override)
    home = os.environ.get(PRACTICE_STORE_HOME_ENV) or (Path.home() / ".agent-learning")
    return Path(home) / PRACTICE_STORE_DIR_NAME / PRACTICE_STORE_FILE_NAME
