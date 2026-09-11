"""
Unified result types for all evaluations.

EvalResult is the ONE result type returned by evaluate() and all engines.
BatchResult wraps multiple EvalResults when running several evals at once.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class EvalResult:
    """The unified result type returned by evaluate() and all engines."""

    eval_name: str
    score: Optional[float] = None
    passed: Optional[bool] = None
    reason: str = ""
    latency_ms: float = 0.0
    status: str = "completed"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Auto-derive passed from score if not explicitly set
        if self.passed is None and self.score is not None:
            self.passed = self.score >= 0.5


@dataclass
class BatchResult:
    """Returned when multiple evals are run via evaluate()."""

    results: List[EvalResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        completed = sum(1 for r in self.results if r.status == "completed")
        return completed / len(self.results)

    def get(self, name: str) -> Optional[EvalResult]:
        """Get result by eval name."""
        for r in self.results:
            if r.eval_name == name:
                return r
        return None

    def __iter__(self) -> Iterator[EvalResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)
