"""
Dual-Judge System for AI Code Security.

Combines pattern-based and LLM-based detection with configurable
consensus modes for optimal precision/recall tradeoffs.
"""

import time
from typing import List, Dict, Any, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from .base import BaseJudge, JudgeResult, JudgeFinding, ConsensusMode
from .pattern_judge import PatternJudge
from .llm_judge import LLMJudge
from ..types import Severity


class DualJudge(BaseJudge):
    """
    Dual-judge system combining pattern and LLM analysis.

    Provides multiple consensus modes to balance precision and recall:

    - ANY: Flag if either judge flags (high recall, may have false positives)
    - BOTH: Flag only if both agree (high precision, may miss some)
    - WEIGHTED: Weighted combination of confidences (balanced)
    - CASCADE: Pattern first, LLM only for uncertain cases (efficient)

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
        # Default dual judge
        judge = DualJudge()
        result = judge.judge(code, "python")

        # High precision mode
        judge = DualJudge(consensus_mode=ConsensusMode.BOTH)

        # High recall mode
        judge = DualJudge(consensus_mode=ConsensusMode.ANY)

        # Efficient cascade mode
        judge = DualJudge(consensus_mode=ConsensusMode.CASCADE)

        # Custom judges
        judge = DualJudge(
            pattern_judge=PatternJudge.with_strict_rules(),
            llm_judge=LLMJudge.with_gpt4(),
            consensus_mode=ConsensusMode.WEIGHTED,
        )
    """

    judge_type = "dual"

    def __init__(
        self,
        pattern_judge: Optional[PatternJudge] = None,
        llm_judge: Optional[LLMJudge] = None,
        consensus_mode: ConsensusMode = ConsensusMode.WEIGHTED,
        severity_threshold: Severity = Severity.HIGH,
        min_confidence: float = 0.7,
        pattern_weight: float = 0.4,
        llm_weight: float = 0.6,
        cascade_threshold: float = 0.6,
        parallel: bool = True,
        llm_timeout: float = 30.0,
    ):
        """
        Initialize the dual judge.

        Args:
            pattern_judge: Pattern-based judge (default: PatternJudge())
            llm_judge: LLM-based judge (default: None, pattern-only)
            consensus_mode: How to combine judge results
            severity_threshold: Minimum severity to flag as insecure
            min_confidence: Minimum confidence to include findings
            pattern_weight: Weight for pattern judge in WEIGHTED mode
            llm_weight: Weight for LLM judge in WEIGHTED mode
            cascade_threshold: Confidence threshold for CASCADE mode
            parallel: Run judges in parallel
            llm_timeout: Timeout for LLM judge
        """
        super().__init__(severity_threshold, min_confidence)

        self.pattern_judge = pattern_judge or PatternJudge(
            severity_threshold=severity_threshold,
            min_confidence=min_confidence,
        )
        self.llm_judge = llm_judge
        self.consensus_mode = consensus_mode
        self.pattern_weight = pattern_weight
        self.llm_weight = llm_weight
        self.cascade_threshold = cascade_threshold
        self.parallel = parallel
        self.llm_timeout = llm_timeout

    def judge(
        self,
        code: str,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None,
    ) -> JudgeResult:
        """
        Judge code using dual-judge system.

        Args:
            code: Source code to analyze
            language: Programming language
            context: Optional context

        Returns:
            JudgeResult with combined findings
        """
        start_time = time.time()

        # Get pattern result (always run)
        pattern_result = self.pattern_judge.judge(code, language, context)

        # Get LLM result if configured
        llm_result = None
        if self.llm_judge is not None:
            if self.consensus_mode == ConsensusMode.CASCADE:
                # Only run LLM if pattern result is uncertain
                if self._should_cascade(pattern_result):
                    llm_result = self._run_llm_judge(code, language, context)
            elif self.parallel:
                # Run in parallel (but we already have pattern result)
                llm_result = self._run_llm_judge(code, language, context)
            else:
                llm_result = self._run_llm_judge(code, language, context)

        # Combine results based on consensus mode
        combined_result = self._combine_results(
            pattern_result, llm_result, language
        )

        execution_time = (time.time() - start_time) * 1000
        combined_result.execution_time_ms = execution_time

        return combined_result

    def _run_llm_judge(
        self,
        code: str,
        language: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[JudgeResult]:
        """Run LLM judge with timeout handling."""
        if self.llm_judge is None:
            return None

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.llm_judge.judge, code, language, context
                )
                return future.result(timeout=self.llm_timeout)
        except TimeoutError:
            return None
        except Exception:
            return None

    def _should_cascade(self, pattern_result: JudgeResult) -> bool:
        """Determine if LLM should be invoked in CASCADE mode."""
        # Invoke LLM if:
        # 1. Pattern found findings but confidence is moderate
        # 2. Pattern found no findings (might be false negative)

        if not pattern_result.findings:
            # No findings - let LLM check for false negatives
            return True

        # Check if any findings have moderate confidence
        avg_confidence = sum(f.confidence for f in pattern_result.findings) / len(
            pattern_result.findings
        )
        return avg_confidence < self.cascade_threshold

    def _combine_results(
        self,
        pattern_result: JudgeResult,
        llm_result: Optional[JudgeResult],
        language: str,
    ) -> JudgeResult:
        """Combine results based on consensus mode."""
        # If no LLM result, return pattern result
        if llm_result is None:
            return JudgeResult(
                is_secure=pattern_result.is_secure,
                security_score=pattern_result.security_score,
                findings=pattern_result.findings,
                judge_type=self.judge_type,
                language=language,
                pattern_result=pattern_result,
                llm_result=None,
                consensus_mode=self.consensus_mode,
            )

        # Combine based on mode
        if self.consensus_mode == ConsensusMode.ANY:
            return self._combine_any(pattern_result, llm_result, language)
        elif self.consensus_mode == ConsensusMode.BOTH:
            return self._combine_both(pattern_result, llm_result, language)
        elif self.consensus_mode == ConsensusMode.WEIGHTED:
            return self._combine_weighted(pattern_result, llm_result, language)
        elif self.consensus_mode == ConsensusMode.CASCADE:
            return self._combine_cascade(pattern_result, llm_result, language)
        else:
            raise ValueError(f"Unknown consensus mode: {self.consensus_mode}")

    def _combine_any(
        self,
        pattern_result: JudgeResult,
        llm_result: JudgeResult,
        language: str,
    ) -> JudgeResult:
        """ANY mode: Include findings from either judge (union)."""
        # Merge all findings
        all_findings = list(pattern_result.findings) + list(llm_result.findings)

        # Deduplicate by CWE and location
        unique_findings = self._deduplicate_findings(all_findings)

        # Insecure if either judge says insecure
        is_secure = pattern_result.is_secure and llm_result.is_secure

        # Score is minimum of both
        security_score = min(
            pattern_result.security_score, llm_result.security_score
        )

        return JudgeResult(
            is_secure=is_secure,
            security_score=security_score,
            findings=unique_findings,
            judge_type=self.judge_type,
            language=language,
            pattern_result=pattern_result,
            llm_result=llm_result,
            consensus_mode=self.consensus_mode,
        )

    def _combine_both(
        self,
        pattern_result: JudgeResult,
        llm_result: JudgeResult,
        language: str,
    ) -> JudgeResult:
        """BOTH mode: Only include findings both judges agree on (intersection)."""
        # Find matching findings
        agreed_findings = self._find_agreed_findings(
            pattern_result.findings, llm_result.findings
        )

        # Insecure only if both say insecure
        is_secure = pattern_result.is_secure or llm_result.is_secure

        # Score is maximum of both (more conservative)
        security_score = max(
            pattern_result.security_score, llm_result.security_score
        )

        return JudgeResult(
            is_secure=self._is_secure(agreed_findings),
            security_score=self._compute_score(agreed_findings),
            findings=agreed_findings,
            judge_type=self.judge_type,
            language=language,
            pattern_result=pattern_result,
            llm_result=llm_result,
            consensus_mode=self.consensus_mode,
        )

    def _combine_weighted(
        self,
        pattern_result: JudgeResult,
        llm_result: JudgeResult,
        language: str,
    ) -> JudgeResult:
        """WEIGHTED mode: Weighted combination of confidences."""
        # Merge findings with weighted confidence
        weighted_findings = []

        # Group findings by CWE
        pattern_by_cwe = self._group_by_cwe(pattern_result.findings)
        llm_by_cwe = self._group_by_cwe(llm_result.findings)

        all_cwes = set(pattern_by_cwe.keys()) | set(llm_by_cwe.keys())

        for cwe in all_cwes:
            pattern_findings = pattern_by_cwe.get(cwe, [])
            llm_findings = llm_by_cwe.get(cwe, [])

            if pattern_findings and llm_findings:
                # Both found - combine confidence
                pattern_conf = max(f.confidence for f in pattern_findings)
                llm_conf = max(f.confidence for f in llm_findings)
                combined_conf = (
                    self.pattern_weight * pattern_conf
                    + self.llm_weight * llm_conf
                )

                # Use LLM finding as base (better description/reasoning)
                best_llm = max(llm_findings, key=lambda f: f.confidence)
                combined = JudgeFinding(
                    cwe_id=best_llm.cwe_id,
                    vulnerability_type=best_llm.vulnerability_type,
                    description=best_llm.description,
                    severity=best_llm.severity,
                    confidence=combined_conf,
                    location=best_llm.location or (
                        pattern_findings[0].location if pattern_findings else None
                    ),
                    suggested_fix=best_llm.suggested_fix,
                    judge_type="dual",
                    reasoning=best_llm.reasoning,
                )
                weighted_findings.append(combined)

            elif pattern_findings:
                # Only pattern found - use pattern weight
                for f in pattern_findings:
                    adjusted = JudgeFinding(
                        cwe_id=f.cwe_id,
                        vulnerability_type=f.vulnerability_type,
                        description=f.description,
                        severity=f.severity,
                        confidence=f.confidence * self.pattern_weight,
                        location=f.location,
                        suggested_fix=f.suggested_fix,
                        judge_type="pattern",
                        reasoning=f.reasoning,
                    )
                    weighted_findings.append(adjusted)

            else:
                # Only LLM found - use LLM weight
                for f in llm_findings:
                    adjusted = JudgeFinding(
                        cwe_id=f.cwe_id,
                        vulnerability_type=f.vulnerability_type,
                        description=f.description,
                        severity=f.severity,
                        confidence=f.confidence * self.llm_weight,
                        location=f.location,
                        suggested_fix=f.suggested_fix,
                        judge_type="llm",
                        reasoning=f.reasoning,
                    )
                    weighted_findings.append(adjusted)

        # Filter by confidence threshold
        filtered = self._filter_findings(weighted_findings)

        # Weighted score
        weighted_score = (
            self.pattern_weight * pattern_result.security_score
            + self.llm_weight * llm_result.security_score
        )

        return JudgeResult(
            is_secure=self._is_secure(filtered),
            security_score=weighted_score,
            findings=filtered,
            judge_type=self.judge_type,
            language=language,
            pattern_result=pattern_result,
            llm_result=llm_result,
            consensus_mode=self.consensus_mode,
        )

    def _combine_cascade(
        self,
        pattern_result: JudgeResult,
        llm_result: JudgeResult,
        language: str,
    ) -> JudgeResult:
        """CASCADE mode: Pattern first, LLM validates/refines."""
        # LLM result refines pattern result
        # - Confirms or rejects pattern findings
        # - May add new findings pattern missed

        final_findings = []

        # Check which pattern findings LLM confirms
        pattern_by_cwe = self._group_by_cwe(pattern_result.findings)
        llm_by_cwe = self._group_by_cwe(llm_result.findings)

        for cwe, pattern_findings in pattern_by_cwe.items():
            if cwe in llm_by_cwe:
                # LLM confirms - boost confidence
                for f in pattern_findings:
                    confirmed = JudgeFinding(
                        cwe_id=f.cwe_id,
                        vulnerability_type=f.vulnerability_type,
                        description=f.description,
                        severity=f.severity,
                        confidence=min(1.0, f.confidence * 1.3),  # Boost
                        location=f.location,
                        suggested_fix=llm_by_cwe[cwe][0].suggested_fix or f.suggested_fix,
                        judge_type="dual",
                        reasoning=llm_by_cwe[cwe][0].reasoning,
                    )
                    final_findings.append(confirmed)
            else:
                # LLM doesn't confirm - reduce confidence
                for f in pattern_findings:
                    unconfirmed = JudgeFinding(
                        cwe_id=f.cwe_id,
                        vulnerability_type=f.vulnerability_type,
                        description=f.description,
                        severity=f.severity,
                        confidence=f.confidence * 0.5,  # Reduce
                        location=f.location,
                        suggested_fix=f.suggested_fix,
                        judge_type="pattern",
                        reasoning="LLM did not confirm this finding",
                    )
                    final_findings.append(unconfirmed)

        # Add LLM-only findings (pattern false negatives)
        for cwe, llm_findings in llm_by_cwe.items():
            if cwe not in pattern_by_cwe:
                for f in llm_findings:
                    final_findings.append(f)

        # Filter
        filtered = self._filter_findings(final_findings)

        return JudgeResult(
            is_secure=self._is_secure(filtered),
            security_score=self._compute_score(filtered),
            findings=filtered,
            judge_type=self.judge_type,
            language=language,
            pattern_result=pattern_result,
            llm_result=llm_result,
            consensus_mode=self.consensus_mode,
        )

    def _deduplicate_findings(
        self,
        findings: List[JudgeFinding],
    ) -> List[JudgeFinding]:
        """Remove duplicate findings, keeping highest confidence."""
        # Group by (CWE, line)
        by_key: Dict[Tuple[str, Optional[int]], JudgeFinding] = {}

        for f in findings:
            line = f.location.line if f.location else None
            key = (f.cwe_id, line)

            if key not in by_key or f.confidence > by_key[key].confidence:
                by_key[key] = f

        return list(by_key.values())

    def _find_agreed_findings(
        self,
        pattern_findings: List[JudgeFinding],
        llm_findings: List[JudgeFinding],
    ) -> List[JudgeFinding]:
        """Find findings that both judges agree on."""
        pattern_cwes = {f.cwe_id for f in pattern_findings}
        llm_cwes = {f.cwe_id for f in llm_findings}
        agreed_cwes = pattern_cwes & llm_cwes

        # Return LLM findings for agreed CWEs (better descriptions)
        return [f for f in llm_findings if f.cwe_id in agreed_cwes]

    def _group_by_cwe(
        self,
        findings: List[JudgeFinding],
    ) -> Dict[str, List[JudgeFinding]]:
        """Group findings by CWE ID."""
        by_cwe: Dict[str, List[JudgeFinding]] = {}
        for f in findings:
            if f.cwe_id not in by_cwe:
                by_cwe[f.cwe_id] = []
            by_cwe[f.cwe_id].append(f)
        return by_cwe

    @classmethod
    def pattern_only(cls) -> "DualJudge":
        """Factory for pattern-only mode (fast, no API calls)."""
        return cls(
            pattern_judge=PatternJudge(),
            llm_judge=None,
        )

    @classmethod
    def high_recall(cls, llm_model: str = "gpt-4") -> "DualJudge":
        """Factory for high recall mode (catches more vulnerabilities)."""
        return cls(
            pattern_judge=PatternJudge.with_strict_rules(),
            llm_judge=LLMJudge(model=llm_model),
            consensus_mode=ConsensusMode.ANY,
        )

    @classmethod
    def high_precision(cls, llm_model: str = "gpt-4") -> "DualJudge":
        """Factory for high precision mode (fewer false positives)."""
        return cls(
            pattern_judge=PatternJudge.with_high_precision(),
            llm_judge=LLMJudge(model=llm_model),
            consensus_mode=ConsensusMode.BOTH,
        )

    @classmethod
    def balanced(cls, llm_model: str = "gpt-4") -> "DualJudge":
        """Factory for balanced mode (default)."""
        return cls(
            pattern_judge=PatternJudge(),
            llm_judge=LLMJudge(model=llm_model),
            consensus_mode=ConsensusMode.WEIGHTED,
        )

    @classmethod
    def efficient(cls, llm_model: str = "gpt-3.5-turbo") -> "DualJudge":
        """Factory for efficient mode (pattern first, LLM only when needed)."""
        return cls(
            pattern_judge=PatternJudge(),
            llm_judge=LLMJudge(model=llm_model),
            consensus_mode=ConsensusMode.CASCADE,
        )
