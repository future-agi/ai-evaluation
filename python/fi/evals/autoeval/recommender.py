"""Evaluation recommender for AutoEval.

Maps application requirements to specific evaluations and scanners.
"""

from typing import List, Dict, Tuple, Set
from .types import AppAnalysis, RiskLevel, DomainSensitivity
from .config import EvalConfig, ScannerConfig


# Mapping from requirement eval names to actual class names
EVAL_MAPPINGS: Dict[str, str] = {
    # Semantic evaluations
    "coherence": "CoherenceEval",
    "CoherenceEval": "CoherenceEval",
    # Agentic evaluations
    "action_safety": "ActionSafetyEval",
    "ActionSafetyEval": "ActionSafetyEval",
    "reasoning_quality": "ReasoningQualityEval",
    "ReasoningQualityEval": "ReasoningQualityEval",
}

# Mapping from requirement scanner names to actual scanner names
SCANNER_MAPPINGS: Dict[str, str] = {
    # Security scanners
    "jailbreak": "JailbreakScanner",
    "JailbreakScanner": "JailbreakScanner",
    "code_injection": "CodeInjectionScanner",
    "CodeInjectionScanner": "CodeInjectionScanner",
    "secrets": "SecretsScanner",
    "SecretsScanner": "SecretsScanner",
    "prompt_injection": "JailbreakScanner",  # Use jailbreak scanner for prompt injection
    # Safety scanners (via EvalDelegateScanner)
    "toxicity": "ToxicityScanner",
    "ToxicityScanner": "ToxicityScanner",
    "bias": "BiasScanner",
    "BiasScanner": "BiasScanner",
    "pii": "PIIScanner",
    "PIIScanner": "PIIScanner",
    # Content scanners
    "malicious_url": "MaliciousURLScanner",
    "MaliciousURLScanner": "MaliciousURLScanner",
    "invisible_chars": "InvisibleCharScanner",
    "InvisibleCharScanner": "InvisibleCharScanner",
    "language": "LanguageScanner",
    "LanguageScanner": "LanguageScanner",
    "topic_restriction": "TopicRestrictionScanner",
    "TopicRestrictionScanner": "TopicRestrictionScanner",
    "regex": "RegexScanner",
    "RegexScanner": "RegexScanner",
}

# Risk-based thresholds
RISK_THRESHOLDS: Dict[RiskLevel, float] = {
    RiskLevel.LOW: 0.6,
    RiskLevel.MEDIUM: 0.7,
    RiskLevel.HIGH: 0.8,
    RiskLevel.CRITICAL: 0.9,
}

# Scanner actions by importance
SCANNER_ACTIONS: Dict[str, str] = {
    "required": "block",
    "recommended": "flag",
    "optional": "warn",
}


class EvalRecommender:
    """
    Maps application requirements to specific evaluations and scanners.

    Example:
        recommender = EvalRecommender()
        evals, scanners = recommender.recommend(analysis)
    """

    def __init__(self):
        """Initialize the recommender."""
        self.eval_mappings = EVAL_MAPPINGS
        self.scanner_mappings = SCANNER_MAPPINGS

    def recommend(
        self,
        analysis: AppAnalysis,
    ) -> Tuple[List[EvalConfig], List[ScannerConfig]]:
        """
        Generate evaluation and scanner recommendations.

        Args:
            analysis: Result from AppAnalyzer

        Returns:
            Tuple of (eval configs, scanner configs)
        """
        evals: List[EvalConfig] = []
        scanners: List[ScannerConfig] = []

        # Base threshold from risk level
        base_threshold = RISK_THRESHOLDS.get(analysis.risk_level, 0.7)

        # Track what we've added to avoid duplicates
        added_evals: Set[str] = set()
        added_scanners: Set[str] = set()

        # Process each requirement
        for req in analysis.requirements:
            # Add suggested evaluations
            for eval_name in req.suggested_evals:
                mapped_name = self.eval_mappings.get(eval_name)
                if mapped_name and mapped_name not in added_evals:
                    threshold = base_threshold
                    weight = 1.0

                    # Increase threshold for required items
                    if req.importance == "required":
                        threshold = min(0.95, threshold + 0.05)
                        weight = 1.5

                    evals.append(
                        EvalConfig(
                            name=mapped_name,
                            threshold=threshold,
                            weight=weight,
                        )
                    )
                    added_evals.add(mapped_name)

            # Add suggested scanners
            for scanner_name in req.suggested_scanners:
                mapped_name = self.scanner_mappings.get(scanner_name)
                if mapped_name and mapped_name not in added_scanners:
                    action = SCANNER_ACTIONS.get(req.importance, "flag")

                    scanners.append(
                        ScannerConfig(
                            name=mapped_name,
                            threshold=base_threshold,
                            action=action,
                        )
                    )
                    added_scanners.add(mapped_name)

        # Add domain-specific recommendations
        domain_evals, domain_scanners = self._add_domain_recommendations(
            analysis, base_threshold, added_evals, added_scanners
        )
        evals.extend(domain_evals)
        scanners.extend(domain_scanners)

        return evals, scanners

    def _add_domain_recommendations(
        self,
        analysis: AppAnalysis,
        base_threshold: float,
        added_evals: Set[str],
        added_scanners: Set[str],
    ) -> Tuple[List[EvalConfig], List[ScannerConfig]]:
        """Add domain-specific recommendations."""
        evals: List[EvalConfig] = []
        scanners: List[ScannerConfig] = []

        # PII-sensitive domains always need PII scanner
        pii_domains = {
            DomainSensitivity.PII_SENSITIVE,
            DomainSensitivity.HEALTHCARE,
            DomainSensitivity.FINANCIAL,
        }

        if analysis.domain_sensitivity in pii_domains:
            if "PIIScanner" not in added_scanners:
                scanners.append(
                    ScannerConfig(
                        name="PIIScanner",
                        threshold=base_threshold,
                        action="redact",  # Redact PII instead of blocking
                    )
                )
            if "SecretsScanner" not in added_scanners:
                scanners.append(
                    ScannerConfig(
                        name="SecretsScanner",
                        threshold=base_threshold,
                        action="block",
                    )
                )

        # Healthcare needs extra safety
        if analysis.domain_sensitivity == DomainSensitivity.HEALTHCARE:
            if "ToxicityScanner" not in added_scanners:
                scanners.append(
                    ScannerConfig(
                        name="ToxicityScanner",
                        threshold=base_threshold + 0.1,
                        action="block",
                    )
                )

        # Children's content needs strict safety
        if analysis.domain_sensitivity == DomainSensitivity.CHILDREN:
            for scanner_name in ["ToxicityScanner", "BiasScanner"]:
                if scanner_name not in added_scanners:
                    scanners.append(
                        ScannerConfig(
                            name=scanner_name,
                            threshold=0.9,
                            action="block",
                        )
                    )

        # High/critical risk always needs jailbreak protection
        if analysis.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            if "JailbreakScanner" not in added_scanners:
                scanners.append(
                    ScannerConfig(
                        name="JailbreakScanner",
                        threshold=base_threshold,
                        action="block",
                    )
                )

        return evals, scanners

    def get_available_evals(self) -> List[str]:
        """Get list of available evaluation names."""
        return list(set(self.eval_mappings.values()))

    def get_available_scanners(self) -> List[str]:
        """Get list of available scanner names."""
        return list(set(self.scanner_mappings.values()))
