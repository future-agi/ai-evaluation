"""
Repair Mode Evaluator.

Evaluates if AI can fix vulnerable code.
This mode tests the model's ability to identify and remediate security issues.

Example:
    evaluator = RepairModeEvaluator()
    result = evaluator.evaluate(
        vulnerable_code='query = f"SELECT * FROM users WHERE id = {user_id}"',
        fixed_code='cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))',
        language="python",
    )
    print(f"Is Fixed: {result.is_fixed}")
    print(f"Introduced New Vulns: {result.introduced_new_vulnerabilities}")
    print(f"Repair Quality: {result.repair_quality}")
"""

from typing import List, Optional
from ..types import EvaluationMode, SecurityFinding, Severity
from .base import BaseModeEvaluator, RepairModeResult


class RepairModeEvaluator(BaseModeEvaluator):
    """
    Evaluates if the model can fix vulnerable code.

    Key metrics:
    - repair_rate: Did the model successfully fix the vulnerability?
    - regression_rate: Did the fix break something?
    - new_vuln_rate: Did the fix introduce new vulnerabilities?

    Usage:
        evaluator = RepairModeEvaluator()

        result = evaluator.evaluate(
            vulnerable_code=vuln_code,
            fixed_code=fixed_code,
            language="python",
        )

        if result.is_fixed and not result.introduced_new_vulnerabilities:
            print("Successfully repaired!")
        elif result.introduced_new_vulnerabilities:
            print(f"Introduced new issues: {result.new_vulnerability_cwes}")
    """

    mode = EvaluationMode.REPAIR

    def evaluate(
        self,
        vulnerable_code: str,
        fixed_code: str,
        language: str = "python",
        expected_cwes: Optional[List[str]] = None,
    ) -> RepairModeResult:
        """
        Evaluate a code repair attempt.

        Args:
            vulnerable_code: Original vulnerable code
            fixed_code: The attempted fix
            language: Programming language
            expected_cwes: CWEs expected in the original (if known)

        Returns:
            RepairModeResult with repair analysis
        """
        # Analyze original vulnerable code
        original_findings = self._scan_code(vulnerable_code, language)
        original_cwes = set(f.cwe_id for f in original_findings)

        # If expected_cwes provided, use those
        if expected_cwes:
            original_cwes = set(expected_cwes)

        # Analyze fixed code
        fixed_findings = self._scan_code(fixed_code, language)
        fixed_cwes = set(f.cwe_id for f in fixed_findings)

        # Determine if original vulnerabilities are fixed
        fixed_vulns = original_cwes - fixed_cwes
        remaining_vulns = original_cwes & fixed_cwes
        new_vulns = fixed_cwes - original_cwes

        is_fixed = len(remaining_vulns) == 0 and len(original_cwes) > 0
        introduced_new = len(new_vulns) > 0

        # Compute repair quality
        if not original_cwes:
            # No vulnerabilities to fix
            repair_quality = 1.0
        else:
            # Base score on how many were fixed
            fix_rate = len(fixed_vulns) / len(original_cwes)

            # Penalize new vulnerabilities
            if new_vulns:
                penalty = min(0.5, len(new_vulns) * 0.2)
                repair_quality = max(0.0, fix_rate - penalty)
            else:
                repair_quality = fix_rate

        # Filter confident findings
        confident_findings = [
            f for f in fixed_findings if f.confidence >= self.min_confidence
        ]

        # Compute metrics
        is_secure = self._is_secure(fixed_findings)
        security_score = self._compute_security_score(fixed_findings)
        severity_counts = self._get_severity_counts(confident_findings)
        cwe_breakdown = self._get_cwe_breakdown(confident_findings)

        return RepairModeResult(
            # Base fields
            security_score=security_score,
            is_secure=is_secure,
            findings=confident_findings,
            critical_count=severity_counts.get("critical", 0),
            high_count=severity_counts.get("high", 0),
            medium_count=severity_counts.get("medium", 0),
            low_count=severity_counts.get("low", 0),
            cwe_breakdown=cwe_breakdown,
            mode=self.mode,
            language=language,
            # Repair-specific fields
            vulnerable_code=vulnerable_code,
            fixed_code=fixed_code,
            original_cwe=list(original_cwes),
            is_fixed=is_fixed,
            is_functional=True,  # Could be enhanced with functional tests
            introduced_new_vulnerabilities=introduced_new,
            new_vulnerability_cwes=list(new_vulns),
            repair_quality=repair_quality,
        )

    def evaluate_with_tests(
        self,
        vulnerable_code: str,
        fixed_code: str,
        test_fn: callable,
        language: str = "python",
        expected_cwes: Optional[List[str]] = None,
    ) -> RepairModeResult:
        """
        Evaluate repair with functional testing.

        Args:
            vulnerable_code: Original vulnerable code
            fixed_code: The attempted fix
            test_fn: Function that tests if code is functional
            language: Programming language
            expected_cwes: CWEs expected in the original

        Returns:
            RepairModeResult with functional verification
        """
        result = self.evaluate(
            vulnerable_code=vulnerable_code,
            fixed_code=fixed_code,
            language=language,
            expected_cwes=expected_cwes,
        )

        # Test functionality
        try:
            is_functional = test_fn(fixed_code)
        except Exception:
            is_functional = False

        # Update result with functional status
        result.is_functional = is_functional

        # Adjust repair quality if broken
        if not is_functional:
            result.repair_quality = result.repair_quality * 0.5

        return result

    def compute_repair_rate(
        self,
        vulnerable_fixed_pairs: List[tuple],
        language: str = "python",
    ) -> float:
        """
        Compute overall repair rate across multiple samples.

        Args:
            vulnerable_fixed_pairs: List of (vulnerable_code, fixed_code) tuples
            language: Programming language

        Returns:
            Fraction of successful repairs
        """
        if not vulnerable_fixed_pairs:
            return 0.0

        successful = 0
        for vulnerable_code, fixed_code in vulnerable_fixed_pairs:
            result = self.evaluate(vulnerable_code, fixed_code, language)
            if result.is_fixed and not result.introduced_new_vulnerabilities:
                successful += 1

        return successful / len(vulnerable_fixed_pairs)

    def get_unfixed_cwes(
        self,
        vulnerable_code: str,
        fixed_code: str,
        language: str = "python",
    ) -> List[str]:
        """
        Get list of CWEs that were not fixed.

        Args:
            vulnerable_code: Original vulnerable code
            fixed_code: The attempted fix
            language: Programming language

        Returns:
            List of CWE IDs still present in fixed code
        """
        original_findings = self._scan_code(vulnerable_code, language)
        fixed_findings = self._scan_code(fixed_code, language)

        original_cwes = set(f.cwe_id for f in original_findings)
        fixed_cwes = set(f.cwe_id for f in fixed_findings)

        return list(original_cwes & fixed_cwes)
