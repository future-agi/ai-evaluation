"""
Dual-Judge System for AI Code Security.

Combines pattern-based and LLM-based detection for high-accuracy
vulnerability detection with configurable consensus modes.

Architecture:
    ┌─────────────────────────────────────────┐
    │           Dual-Judge System             │
    ├─────────────────────────────────────────┤
    │  ┌─────────────┐    ┌─────────────┐    │
    │  │   Pattern   │    │     LLM     │    │
    │  │    Judge    │    │    Judge    │    │
    │  │  (< 10ms)   │    │  (accurate) │    │
    │  └─────────────┘    └─────────────┘    │
    │          ↓                ↓             │
    │      ┌───────────────────────┐         │
    │      │   Consensus Engine    │         │
    │      │  • any: high recall   │         │
    │      │  • both: high prec.   │         │
    │      │  • weighted: balanced │         │
    │      │  • cascade: efficient │         │
    │      └───────────────────────┘         │
    └─────────────────────────────────────────┘

Usage:
    from fi.evals.metrics.code_security.judges import (
        DualJudge,
        PatternJudge,
        LLMJudge,
    )

    # Pattern-only (fast, <10ms)
    judge = PatternJudge()
    result = judge.judge(code, "python")

    # LLM-only (accurate, semantic understanding)
    judge = LLMJudge(model="gemini/gemini-2.5-flash")
    result = judge.judge(code, "python")

    # Dual judge (best of both)
    judge = DualJudge(
        pattern_judge=PatternJudge(),
        llm_judge=LLMJudge(),
        consensus_mode=ConsensusMode.WEIGHTED,
    )
    result = judge.judge(code, "python")

    # Factory methods for common configurations
    judge = DualJudge.pattern_only()      # Fast, no API
    judge = DualJudge.high_recall()       # Catches more
    judge = DualJudge.high_precision()    # Fewer false positives
    judge = DualJudge.balanced()          # Default weighted
    judge = DualJudge.efficient()         # CASCADE mode
"""

from .base import (
    BaseJudge,
    JudgeResult,
    JudgeFinding,
    ConsensusMode,
)

from .pattern_judge import PatternJudge, PatternRule
from .llm_judge import LLMJudge, MockLLMJudge
from .dual_judge import DualJudge


__all__ = [
    # Base
    "BaseJudge",
    "JudgeResult",
    "JudgeFinding",
    "ConsensusMode",
    # Pattern Judge
    "PatternJudge",
    "PatternRule",
    # LLM Judge
    "LLMJudge",
    "MockLLMJudge",
    # Dual Judge
    "DualJudge",
]
