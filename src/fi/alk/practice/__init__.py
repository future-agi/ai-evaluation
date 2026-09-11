"""Phase 13D — the Practice Loop trainer (facade only; mirrors live/ style).

Lazy exports so ``import fi.alk.practice`` stays cheap. The trainer
employs the existing 13C operators; it adds no new step API and emits standard
``agent-learning.run.v1`` rows through ``run_manifest``/``public_payload`` so
every episode lands a telemetry ledger row with zero new telemetry code.
"""
from __future__ import annotations

import importlib
from typing import Any

# public name → home submodule (resolved lazily)
_LAZY_EXPORTS = {
    # contract constants
    "PRACTICE_PHASES": "_contract",
    "PRACTICE_ARTIFACT_KINDS": "_contract",
    "SCAFFOLD_TYPES": "_contract",
    "LADDER_STATES": "_contract",
    "PRACTICE_REPLAY_INTERVALS": "_contract",
    "ZPD_BAND": "_contract",
    "REVIEW_RATIO": "_contract",
    "BUDGET_PLAN": "_contract",
    "PRACTICE_STORE_ACTIVE_CAP": "_contract",
    "SCAFFOLD_FADE_DEFAULT": "_contract",
    "AGENT_LEARNING_PRACTICE_LOOP_KIND": "_contract",
    "AGENT_LEARNING_PRACTICE_RESULT_KIND": "_contract",
    "practice_store_path": "_contract",
    # budget
    "BudgetMeter": "_budget",
    "BudgetExhausted": "_budget",
    # trainer surface
    "run_practice_loop": "_trainer",
    "practice_report": "_assess",
    "ladder_state": "_store",
    "run_due_reviews": "_schedule",
}


def __getattr__(name: str) -> Any:
    home = _LAZY_EXPORTS.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{home}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
