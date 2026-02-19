"""Rule-based analyzer for AutoEval.

Provides keyword-based analysis when LLM is unavailable.
"""

from typing import List, Dict, Set, Tuple
from .types import (
    AppAnalysis,
    AppCategory,
    RiskLevel,
    DomainSensitivity,
    AppRequirement,
)


# Keyword mappings for category detection
CATEGORY_KEYWORDS: Dict[AppCategory, List[str]] = {
    AppCategory.CUSTOMER_SUPPORT: [
        "customer support",
        "help desk",
        "support ticket",
        "customer service",
        "faq",
        "help bot",
        "service desk",
        "customer query",
        "support chat",
        "helpdesk",
    ],
    AppCategory.RAG_SYSTEM: [
        "rag",
        "retrieval",
        "knowledge base",
        "document q&a",
        "document search",
        "context retrieval",
        "knowledge retrieval",
        "semantic search",
        "vector search",
        "document qa",
        "retrieval augmented",
    ],
    AppCategory.CODE_ASSISTANT: [
        "code",
        "programming",
        "developer",
        "ide",
        "code generation",
        "code review",
        "debugging",
        "software development",
        "coding assistant",
        "code completion",
        "copilot",
    ],
    AppCategory.CONTENT_MODERATION: [
        "moderation",
        "content filter",
        "content safety",
        "inappropriate content",
        "review content",
        "harmful content",
        "policy violation",
        "content policy",
        "safe content",
        "nsfw",
        "toxicity detection",
    ],
    AppCategory.AGENT_WORKFLOW: [
        "agent",
        "tool use",
        "function calling",
        "autonomous",
        "workflow",
        "multi-step",
        "execute tasks",
        "api calls",
        "agentic",
        "tool calling",
        "action execution",
    ],
    AppCategory.CHATBOT: [
        "chatbot",
        "conversational",
        "chat bot",
        "virtual assistant",
        "dialogue system",
        "chat interface",
    ],
    AppCategory.SUMMARIZATION: [
        "summarization",
        "summarize",
        "summary",
        "tldr",
        "digest",
        "condensing",
    ],
    AppCategory.TRANSLATION: [
        "translation",
        "translate",
        "multilingual",
        "language conversion",
        "localization",
    ],
    AppCategory.CREATIVE_WRITING: [
        "creative writing",
        "content generation",
        "story writing",
        "copywriting",
        "blog writing",
        "marketing copy",
    ],
    AppCategory.DATA_EXTRACTION: [
        "data extraction",
        "information extraction",
        "entity extraction",
        "parsing",
        "structured data",
        "json extraction",
    ],
    AppCategory.SEARCH: [
        "search engine",
        "search system",
        "information retrieval",
        "query processing",
    ],
    AppCategory.QUESTION_ANSWERING: [
        "question answering",
        "qa system",
        "q&a",
        "answer questions",
    ],
}

# Domain sensitivity keywords
DOMAIN_SENSITIVITY_KEYWORDS: Dict[DomainSensitivity, List[str]] = {
    DomainSensitivity.HEALTHCARE: [
        "healthcare",
        "medical",
        "patient",
        "health",
        "clinical",
        "hipaa",
        "diagnosis",
        "prescription",
        "treatment",
        "hospital",
        "doctor",
        "nurse",
        "pharmacy",
        "medication",
        "symptoms",
    ],
    DomainSensitivity.FINANCIAL: [
        "financial",
        "banking",
        "payment",
        "transaction",
        "account",
        "credit",
        "loan",
        "investment",
        "trading",
        "money",
        "fintech",
        "pci",
        "wallet",
        "stock",
    ],
    DomainSensitivity.LEGAL: [
        "legal",
        "law",
        "attorney",
        "court",
        "litigation",
        "contract",
        "compliance",
        "regulation",
        "lawsuit",
        "lawyer",
        "paralegal",
    ],
    DomainSensitivity.PII_SENSITIVE: [
        "personal information",
        "user data",
        "profile",
        "identity",
        "ssn",
        "social security",
        "address",
        "phone number",
        "email address",
        "private data",
        "gdpr",
        "ccpa",
        "pii",
    ],
    DomainSensitivity.CHILDREN: [
        "children",
        "kids",
        "educational",
        "school",
        "minor",
        "coppa",
        "young",
        "student",
        "classroom",
        "learning",
        "teen",
        "child",
    ],
    DomainSensitivity.GOVERNMENT: [
        "government",
        "federal",
        "state agency",
        "public sector",
        "fedramp",
        "classified",
        "official",
    ],
}

# Feature detection patterns
FEATURE_PATTERNS: Dict[str, List[str]] = {
    "tool_use": [
        "tool",
        "function call",
        "api",
        "execute",
        "action",
        "plugin",
        "integration",
    ],
    "rag": [
        "retrieval",
        "rag",
        "context",
        "knowledge base",
        "document",
        "vector",
        "embedding",
    ],
    "multi_turn": [
        "conversation",
        "chat",
        "dialogue",
        "multi-turn",
        "history",
        "session",
        "follow-up",
    ],
    "code_generation": [
        "code",
        "generate code",
        "programming",
        "script",
        "implement",
    ],
    "image_processing": [
        "image",
        "vision",
        "visual",
        "picture",
        "photo",
        "screenshot",
        "diagram",
    ],
    "safety_critical": [
        "safety",
        "critical",
        "harm",
        "dangerous",
        "risk",
        "sensitive",
    ],
    "real_time": [
        "real-time",
        "realtime",
        "low latency",
        "streaming",
        "live",
    ],
    "batch_processing": [
        "batch",
        "bulk",
        "large scale",
        "mass processing",
    ],
}

# Requirement generators based on category
CATEGORY_REQUIREMENTS: Dict[AppCategory, List[AppRequirement]] = {
    AppCategory.CUSTOMER_SUPPORT: [
        AppRequirement(
            category="quality",
            importance="required",
            reason="Customer support needs clear, coherent responses",
            suggested_evals=["CoherenceEval", "SemanticSimilarityEval"],
        ),
        AppRequirement(
            category="safety",
            importance="required",
            reason="Customer interactions must be safe and appropriate",
            suggested_scanners=["jailbreak", "toxicity"],
        ),
        AppRequirement(
            category="pii_protection",
            importance="recommended",
            reason="Customer data should be protected",
            suggested_scanners=["pii", "secrets"],
        ),
    ],
    AppCategory.RAG_SYSTEM: [
        AppRequirement(
            category="faithfulness",
            importance="required",
            reason="RAG systems must provide grounded, faithful responses",
            suggested_evals=["FactualConsistencyEval", "EntailmentEval"],
        ),
        AppRequirement(
            category="quality",
            importance="required",
            reason="Responses should be coherent and relevant",
            suggested_evals=["CoherenceEval", "SemanticSimilarityEval"],
        ),
        AppRequirement(
            category="safety",
            importance="required",
            reason="Prevent prompt injection attacks",
            suggested_scanners=["jailbreak", "prompt_injection"],
        ),
    ],
    AppCategory.CODE_ASSISTANT: [
        AppRequirement(
            category="quality",
            importance="required",
            reason="Code responses should be coherent",
            suggested_evals=["CoherenceEval"],
        ),
        AppRequirement(
            category="code_safety",
            importance="required",
            reason="Prevent malicious code generation",
            suggested_scanners=["code_injection", "secrets", "jailbreak"],
        ),
    ],
    AppCategory.CONTENT_MODERATION: [
        AppRequirement(
            category="safety",
            importance="required",
            reason="Content moderation requires comprehensive safety checks",
            suggested_scanners=["toxicity", "bias", "invisible_chars"],
        ),
    ],
    AppCategory.AGENT_WORKFLOW: [
        AppRequirement(
            category="tool_use",
            importance="required",
            reason="Agent workflows need tool use validation",
            suggested_evals=[
                "ToolUseCorrectnessEval",
                "TrajectoryEfficiencyEval",
                "GoalCompletionEval",
            ],
        ),
        AppRequirement(
            category="safety",
            importance="required",
            reason="Agent actions must be safe",
            suggested_evals=["ActionSafetyEval"],
            suggested_scanners=["jailbreak", "code_injection"],
        ),
        AppRequirement(
            category="reasoning",
            importance="recommended",
            reason="Agent reasoning should be high quality",
            suggested_evals=["ReasoningQualityEval"],
        ),
    ],
}

# Default requirements for all apps
DEFAULT_REQUIREMENTS: List[AppRequirement] = [
    AppRequirement(
        category="quality",
        importance="recommended",
        reason="All LLM applications benefit from basic quality evaluation",
        suggested_evals=["CoherenceEval"],
    ),
]


class RuleBasedAnalyzer:
    """Rule-based fallback analyzer for when LLM is unavailable.

    Uses keyword matching to detect:
    - Application category
    - Domain sensitivity
    - Risk level
    - Required features

    Example:
        analyzer = RuleBasedAnalyzer()
        analysis = analyzer.analyze(
            "A customer support chatbot for healthcare that answers patient questions."
        )
        print(analysis.category)  # AppCategory.CUSTOMER_SUPPORT
        print(analysis.risk_level)  # RiskLevel.HIGH (due to healthcare)
    """

    def analyze(self, description: str) -> AppAnalysis:
        """
        Analyze description using keyword matching and rules.

        Args:
            description: Application description

        Returns:
            AppAnalysis with detected attributes
        """
        description_lower = description.lower()

        # Detect category
        category = self._detect_category(description_lower)

        # Detect domain sensitivity
        sensitivity = self._detect_sensitivity(description_lower)

        # Determine risk level based on sensitivity
        risk_level = self._determine_risk_level(sensitivity, description_lower)

        # Detect features
        features = self._detect_features(description_lower)

        # Generate requirements based on analysis
        requirements = self._generate_requirements(
            category, sensitivity, risk_level, features
        )

        return AppAnalysis(
            category=category,
            risk_level=risk_level,
            domain_sensitivity=sensitivity,
            requirements=requirements,
            detected_features=features,
            confidence=0.6,  # Rule-based has lower confidence
            explanation=self._generate_explanation(category, sensitivity, features),
        )

    def _detect_category(self, text: str) -> AppCategory:
        """Detect app category from keywords."""
        scores: Dict[AppCategory, int] = {}

        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores, key=scores.get)
        return AppCategory.UNKNOWN

    def _detect_sensitivity(self, text: str) -> DomainSensitivity:
        """Detect domain sensitivity from keywords."""
        scores: Dict[DomainSensitivity, int] = {}

        for sensitivity, keywords in DOMAIN_SENSITIVITY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[sensitivity] = score

        if scores:
            return max(scores, key=scores.get)
        return DomainSensitivity.GENERAL

    def _determine_risk_level(
        self,
        sensitivity: DomainSensitivity,
        text: str,
    ) -> RiskLevel:
        """Determine risk level based on sensitivity and keywords."""
        # High-risk domains
        high_risk_domains = {
            DomainSensitivity.HEALTHCARE,
            DomainSensitivity.FINANCIAL,
            DomainSensitivity.LEGAL,
            DomainSensitivity.CHILDREN,
            DomainSensitivity.GOVERNMENT,
        }

        if sensitivity in high_risk_domains:
            # Check for critical keywords
            critical_keywords = ["critical", "life-saving", "emergency", "classified"]
            if any(kw in text for kw in critical_keywords):
                return RiskLevel.CRITICAL
            return RiskLevel.HIGH

        # Medium risk for PII-sensitive
        if sensitivity == DomainSensitivity.PII_SENSITIVE:
            return RiskLevel.MEDIUM

        # Context-based risk detection
        if any(kw in text for kw in ["production", "public", "customer", "external"]):
            return RiskLevel.MEDIUM

        if any(kw in text for kw in ["internal", "development", "testing", "prototype"]):
            return RiskLevel.LOW

        return RiskLevel.MEDIUM

    def _detect_features(self, text: str) -> List[str]:
        """Detect application features from keywords."""
        features = []

        for feature, patterns in FEATURE_PATTERNS.items():
            if any(p in text for p in patterns):
                features.append(feature)

        return features

    def _generate_requirements(
        self,
        category: AppCategory,
        sensitivity: DomainSensitivity,
        risk_level: RiskLevel,
        features: List[str],
    ) -> List[AppRequirement]:
        """Generate requirements based on analysis."""
        requirements: List[AppRequirement] = []

        # Add category-specific requirements
        if category in CATEGORY_REQUIREMENTS:
            requirements.extend(CATEGORY_REQUIREMENTS[category])

        # Add default requirements if not already covered
        eval_names = {
            e for r in requirements for e in r.suggested_evals
        }
        for default_req in DEFAULT_REQUIREMENTS:
            if not any(e in eval_names for e in default_req.suggested_evals):
                requirements.append(default_req)

        # Add sensitivity-based requirements
        if sensitivity in {
            DomainSensitivity.PII_SENSITIVE,
            DomainSensitivity.HEALTHCARE,
            DomainSensitivity.FINANCIAL,
        }:
            requirements.append(
                AppRequirement(
                    category="pii_protection",
                    importance="required",
                    reason=f"{sensitivity.value} applications must protect sensitive data",
                    suggested_scanners=["pii", "secrets"],
                )
            )

        # Add risk-level based requirements
        if risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            requirements.append(
                AppRequirement(
                    category="safety",
                    importance="required",
                    reason="High-risk applications require comprehensive safety evaluation",
                    suggested_scanners=["jailbreak", "toxicity"],
                )
            )

        # Add feature-based requirements
        if "image_processing" in features:
            requirements.append(
                AppRequirement(
                    category="multimodal",
                    importance="recommended",
                    reason="Image processing needs multimodal evaluation",
                    suggested_evals=[
                        "ImageTextConsistencyEval",
                        "ImageSafetyEval",
                    ],
                )
            )

        return requirements

    def _generate_explanation(
        self,
        category: AppCategory,
        sensitivity: DomainSensitivity,
        features: List[str],
    ) -> str:
        """Generate explanation for the analysis."""
        parts = []

        if category != AppCategory.UNKNOWN:
            parts.append(f"Detected application type: {category.value}")
        else:
            parts.append("Could not determine specific application type")

        if sensitivity != DomainSensitivity.GENERAL:
            parts.append(f"Domain sensitivity: {sensitivity.value}")

        if features:
            parts.append(f"Detected features: {', '.join(features)}")

        parts.append("Analysis performed using rule-based keyword matching.")

        return " ".join(parts)
