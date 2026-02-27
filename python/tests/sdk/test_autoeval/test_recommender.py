"""Tests for evaluation recommender."""

import pytest
from fi.evals.autoeval.recommender import (
    EvalRecommender,
    EVAL_MAPPINGS,
    SCANNER_MAPPINGS,
    RISK_THRESHOLDS,
)
from fi.evals.autoeval.types import (
    AppAnalysis,
    AppCategory,
    RiskLevel,
    DomainSensitivity,
    AppRequirement,
)


class TestEvalMappings:
    """Tests for evaluation and scanner mappings."""

    def test_eval_mappings_exist(self):
        """Should have eval mappings defined."""
        assert len(EVAL_MAPPINGS) > 0

    def test_common_evals_mapped(self):
        """Should map common evaluation names."""
        assert "coherence" in EVAL_MAPPINGS
        assert "answer_relevancy" in EVAL_MAPPINGS
        assert EVAL_MAPPINGS["coherence"] == "answer_relevancy"

    def test_scanner_mappings_exist(self):
        """Should have scanner mappings defined."""
        assert len(SCANNER_MAPPINGS) > 0

    def test_common_scanners_mapped(self):
        """Should map common scanner names."""
        assert "jailbreak" in SCANNER_MAPPINGS
        assert "JailbreakScanner" in SCANNER_MAPPINGS
        assert SCANNER_MAPPINGS["jailbreak"] == "JailbreakScanner"

    def test_risk_thresholds_defined(self):
        """Should define thresholds for all risk levels."""
        assert RiskLevel.LOW in RISK_THRESHOLDS
        assert RiskLevel.MEDIUM in RISK_THRESHOLDS
        assert RiskLevel.HIGH in RISK_THRESHOLDS
        assert RiskLevel.CRITICAL in RISK_THRESHOLDS

    def test_risk_thresholds_increase(self):
        """Higher risk should have higher thresholds."""
        assert RISK_THRESHOLDS[RiskLevel.LOW] < RISK_THRESHOLDS[RiskLevel.MEDIUM]
        assert RISK_THRESHOLDS[RiskLevel.MEDIUM] < RISK_THRESHOLDS[RiskLevel.HIGH]
        assert RISK_THRESHOLDS[RiskLevel.HIGH] < RISK_THRESHOLDS[RiskLevel.CRITICAL]


class TestEvalRecommender:
    """Tests for EvalRecommender class."""

    @pytest.fixture
    def recommender(self):
        """Create recommender instance."""
        return EvalRecommender()

    def test_recommend_from_empty_requirements(self, recommender):
        """Should handle analysis with no requirements."""
        analysis = AppAnalysis(
            category=AppCategory.CHATBOT,
            risk_level=RiskLevel.MEDIUM,
            domain_sensitivity=DomainSensitivity.GENERAL,
            requirements=[],
            detected_features=[],
            confidence=0.7,
            explanation="Test",
        )
        evals, scanners = recommender.recommend(analysis)
        # May still have domain-specific recommendations
        assert isinstance(evals, list)
        assert isinstance(scanners, list)

    def test_recommend_maps_eval_names(self, recommender):
        """Should map requirement eval names to class names."""
        analysis = AppAnalysis(
            category=AppCategory.RAG_SYSTEM,
            risk_level=RiskLevel.MEDIUM,
            domain_sensitivity=DomainSensitivity.GENERAL,
            requirements=[
                AppRequirement(
                    category="quality",
                    importance="required",
                    reason="Test",
                    suggested_evals=["coherence", "action_safety"],
                ),
            ],
            detected_features=[],
            confidence=0.7,
            explanation="Test",
        )
        evals, scanners = recommender.recommend(analysis)
        eval_names = [e.name for e in evals]
        assert "answer_relevancy" in eval_names
        assert "action_safety" in eval_names

    def test_recommend_maps_scanner_names(self, recommender):
        """Should map requirement scanner names to class names."""
        analysis = AppAnalysis(
            category=AppCategory.CHATBOT,
            risk_level=RiskLevel.MEDIUM,
            domain_sensitivity=DomainSensitivity.GENERAL,
            requirements=[
                AppRequirement(
                    category="safety",
                    importance="required",
                    reason="Test",
                    suggested_scanners=["jailbreak", "toxicity"],
                ),
            ],
            detected_features=[],
            confidence=0.7,
            explanation="Test",
        )
        evals, scanners = recommender.recommend(analysis)
        scanner_names = [s.name for s in scanners]
        assert "JailbreakScanner" in scanner_names
        assert "ToxicityScanner" in scanner_names

    def test_recommend_sets_thresholds_by_risk(self, recommender):
        """Should set thresholds based on risk level."""
        for risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]:
            analysis = AppAnalysis(
                category=AppCategory.CHATBOT,
                risk_level=risk_level,
                domain_sensitivity=DomainSensitivity.GENERAL,
                requirements=[
                    AppRequirement(
                        category="quality",
                        importance="recommended",
                        reason="Test",
                        suggested_evals=["coherence"],
                    ),
                ],
                detected_features=[],
                confidence=0.7,
                explanation="Test",
            )
            evals, _ = recommender.recommend(analysis)
            if evals:
                expected_base = RISK_THRESHOLDS[risk_level]
                assert evals[0].threshold >= expected_base - 0.01

    def test_recommend_higher_threshold_for_required(self, recommender):
        """Should increase threshold for required items."""
        analysis = AppAnalysis(
            category=AppCategory.CHATBOT,
            risk_level=RiskLevel.MEDIUM,
            domain_sensitivity=DomainSensitivity.GENERAL,
            requirements=[
                AppRequirement(
                    category="quality",
                    importance="required",
                    reason="Test",
                    suggested_evals=["coherence"],
                ),
            ],
            detected_features=[],
            confidence=0.7,
            explanation="Test",
        )
        evals, _ = recommender.recommend(analysis)
        if evals:
            # Required items get +0.05 threshold
            assert evals[0].threshold >= 0.75

    def test_recommend_higher_weight_for_required(self, recommender):
        """Should increase weight for required items."""
        analysis = AppAnalysis(
            category=AppCategory.CHATBOT,
            risk_level=RiskLevel.MEDIUM,
            domain_sensitivity=DomainSensitivity.GENERAL,
            requirements=[
                AppRequirement(
                    category="quality",
                    importance="required",
                    reason="Test",
                    suggested_evals=["coherence"],
                ),
            ],
            detected_features=[],
            confidence=0.7,
            explanation="Test",
        )
        evals, _ = recommender.recommend(analysis)
        if evals:
            assert evals[0].weight >= 1.5

    def test_recommend_no_duplicates(self, recommender):
        """Should not add duplicate evals or scanners."""
        analysis = AppAnalysis(
            category=AppCategory.CHATBOT,
            risk_level=RiskLevel.MEDIUM,
            domain_sensitivity=DomainSensitivity.GENERAL,
            requirements=[
                AppRequirement(
                    category="quality",
                    importance="required",
                    reason="Test 1",
                    suggested_evals=["coherence"],
                ),
                AppRequirement(
                    category="safety",
                    importance="recommended",
                    reason="Test 2",
                    suggested_evals=["coherence"],  # Duplicate
                ),
            ],
            detected_features=[],
            confidence=0.7,
            explanation="Test",
        )
        evals, _ = recommender.recommend(analysis)
        eval_names = [e.name for e in evals]
        assert eval_names.count("answer_relevancy") == 1

    def test_recommend_adds_pii_scanner_for_healthcare(self, recommender):
        """Should add PII scanner for healthcare domain."""
        analysis = AppAnalysis(
            category=AppCategory.CUSTOMER_SUPPORT,
            risk_level=RiskLevel.HIGH,
            domain_sensitivity=DomainSensitivity.HEALTHCARE,
            requirements=[],
            detected_features=[],
            confidence=0.7,
            explanation="Test",
        )
        _, scanners = recommender.recommend(analysis)
        scanner_names = [s.name for s in scanners]
        assert "PIIScanner" in scanner_names

    def test_recommend_adds_pii_scanner_for_financial(self, recommender):
        """Should add PII scanner for financial domain."""
        analysis = AppAnalysis(
            category=AppCategory.CUSTOMER_SUPPORT,
            risk_level=RiskLevel.HIGH,
            domain_sensitivity=DomainSensitivity.FINANCIAL,
            requirements=[],
            detected_features=[],
            confidence=0.7,
            explanation="Test",
        )
        _, scanners = recommender.recommend(analysis)
        scanner_names = [s.name for s in scanners]
        assert "PIIScanner" in scanner_names
        assert "SecretsScanner" in scanner_names

    def test_recommend_adds_scanners_for_financial(self, recommender):
        """Should add PII and secrets scanners for financial domain."""
        analysis = AppAnalysis(
            category=AppCategory.CUSTOMER_SUPPORT,
            risk_level=RiskLevel.HIGH,
            domain_sensitivity=DomainSensitivity.FINANCIAL,
            requirements=[],
            detected_features=[],
            confidence=0.7,
            explanation="Test",
        )
        _, scanners = recommender.recommend(analysis)
        scanner_names = [s.name for s in scanners]
        assert "PIIScanner" in scanner_names
        assert "SecretsScanner" in scanner_names

    def test_recommend_strict_for_children(self, recommender):
        """Should add strict safety for children's content."""
        analysis = AppAnalysis(
            category=AppCategory.CHATBOT,
            risk_level=RiskLevel.HIGH,
            domain_sensitivity=DomainSensitivity.CHILDREN,
            requirements=[],
            detected_features=[],
            confidence=0.7,
            explanation="Test",
        )
        _, scanners = recommender.recommend(analysis)
        scanner_names = [s.name for s in scanners]
        assert "ToxicityScanner" in scanner_names
        assert "BiasScanner" in scanner_names
        # Children's content should have high thresholds
        for scanner in scanners:
            if scanner.name in ["ToxicityScanner", "BiasScanner"]:
                assert scanner.threshold >= 0.9

    def test_recommend_jailbreak_for_high_risk(self, recommender):
        """Should add jailbreak scanner for high-risk apps."""
        analysis = AppAnalysis(
            category=AppCategory.CHATBOT,
            risk_level=RiskLevel.HIGH,
            domain_sensitivity=DomainSensitivity.GENERAL,
            requirements=[],
            detected_features=[],
            confidence=0.7,
            explanation="Test",
        )
        _, scanners = recommender.recommend(analysis)
        scanner_names = [s.name for s in scanners]
        assert "JailbreakScanner" in scanner_names

    def test_get_available_evals(self, recommender):
        """Should return list of available eval names."""
        evals = recommender.get_available_evals()
        assert isinstance(evals, list)
        assert len(evals) > 0
        assert "answer_relevancy" in evals

    def test_get_available_scanners(self, recommender):
        """Should return list of available scanner names."""
        scanners = recommender.get_available_scanners()
        assert isinstance(scanners, list)
        assert len(scanners) > 0
        assert "JailbreakScanner" in scanners
