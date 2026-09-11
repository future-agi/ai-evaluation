"""Unit 8 (BBG U8 / ARCH §2d, AD-I) — the single budget meter.

ONE unit = one scored episode evaluation. Every assess row, ZPD repeat,
scaffolded/unscaffolded drill evaluation, inner-operator evaluation, scheduled
review row, and promotion-sweep row charges THIS meter — there is no second
currency. Soft per-phase enforcement of budget_plan with carry-over.
"""
from __future__ import annotations

from typing import Dict

from ._contract import BUDGET_PLAN, PRACTICE_PHASES

# Map the 4-fraction budget_plan onto phases. assess / drill / update / review;
# diagnose+consolidate+calibrate draw from their adjacent phase allocations.
_BUDGET_PLAN_PHASES = ("assess", "drill", "update", "review")


class BudgetExhausted(RuntimeError):
    """Raised when the meter has no remaining budget (trainer stop)."""


class BudgetMeter:
    """The single eval-unit meter (AD-I)."""

    def __init__(self, total: int, *, budget_plan: tuple[float, ...] = BUDGET_PLAN) -> None:
        if not isinstance(total, int) or isinstance(total, bool) or total < 1:
            raise ValueError("budget total must be an int >= 1")
        self.total = int(total)
        self.consumed = 0
        self._by_phase: Dict[str, int] = {}
        self._plan = tuple(budget_plan)
        # per-phase soft caps (allocation of total) keyed by the 4 plan phases.
        self._caps = {
            phase: int(round(self.total * frac))
            for phase, frac in zip(_BUDGET_PLAN_PHASES, self._plan)
        }

    def _plan_phase(self, phase: str) -> str:
        if phase in _BUDGET_PLAN_PHASES:
            return phase
        if phase == "diagnose":
            return "assess"
        if phase in ("consolidate", "calibrate"):
            return "update"
        return "drill"

    def charge(self, phase: str, n: int = 1) -> int:
        if phase not in PRACTICE_PHASES and phase not in ("review", "promotion_sweep"):
            raise ValueError(f"unknown budget phase {phase!r}")
        if n < 0:
            raise ValueError("charge n must be >= 0")
        if self.consumed + n > self.total:
            raise BudgetExhausted(
                f"budget exhausted: consumed={self.consumed} + {n} > total={self.total}"
            )
        self.consumed += n
        self._by_phase[phase] = self._by_phase.get(phase, 0) + n
        return self.consumed

    def remaining(self) -> int:
        return self.total - self.consumed

    def slice(self, phase: str, fraction: float) -> int:
        """Return an integer sub-budget handed to inner operators (their declared
        eval_budget IS the slice). Bounded by remaining budget."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("slice fraction must be in [0, 1]")
        want = int(self.total * fraction)
        return max(0, min(want, self.remaining()))

    def ledger(self) -> dict:
        """Per-phase consumption; conservation: sum(phase) == consumed <= total."""
        by_phase = {p: self._by_phase.get(p, 0) for p in sorted(self._by_phase)}
        assert sum(by_phase.values()) == self.consumed <= self.total
        return {
            "total": self.total,
            "consumed": self.consumed,
            "remaining": self.remaining(),
            "by_phase": by_phase,
        }
