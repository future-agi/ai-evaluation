"""Interactive configuration mode for AutoEval.

Provides clarifying questions when analysis confidence is low,
helping to fine-tune evaluation pipeline configuration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .types import AppAnalysis, RiskLevel, DomainSensitivity
from .config import AutoEvalConfig


@dataclass
class ClarificationQuestion:
    """A clarifying question for the user."""

    question: str
    options: List[str]
    impact: str
    selected: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractiveSession:
    """Holds state for an interactive configuration session."""

    analysis: AppAnalysis
    questions: List[ClarificationQuestion]
    answers: Dict[str, str] = field(default_factory=dict)
    completed: bool = False


class InteractiveConfigurator:
    """
    Interactive configuration helper for AutoEval.

    Generates clarifying questions when analysis confidence is low,
    and adjusts configuration based on user answers.

    Example:
        configurator = InteractiveConfigurator()
        session = configurator.start_session(analysis)

        for question in session.questions:
            print(f"Q: {question.question}")
            for i, opt in enumerate(question.options):
                print(f"  {i+1}. {opt}")
            answer = input("Select option: ")
            configurator.answer(session, question.question, answer)

        config = configurator.finalize(session)
    """

    # Confidence threshold for requiring clarification
    CONFIDENCE_THRESHOLD = 0.7

    def __init__(self):
        """Initialize the configurator."""
        pass

    def needs_clarification(self, analysis: AppAnalysis) -> bool:
        """
        Check if the analysis needs clarification.

        Args:
            analysis: AppAnalysis from the analyzer

        Returns:
            True if clarification questions should be asked
        """
        return analysis.confidence < self.CONFIDENCE_THRESHOLD

    def start_session(self, analysis: AppAnalysis) -> InteractiveSession:
        """
        Start an interactive configuration session.

        Args:
            analysis: AppAnalysis from the analyzer

        Returns:
            InteractiveSession with questions to ask
        """
        questions = self._generate_questions(analysis)
        return InteractiveSession(
            analysis=analysis,
            questions=questions,
        )

    def _generate_questions(self, analysis: AppAnalysis) -> List[ClarificationQuestion]:
        """Generate clarifying questions based on analysis."""
        questions = []

        # Always ask about deployment environment if not certain
        if analysis.confidence < 0.8:
            questions.append(
                ClarificationQuestion(
                    question="What is the deployment environment?",
                    options=[
                        "Internal tool (development/testing)",
                        "Production (internal users only)",
                        "Production (external/public users)",
                        "Safety-critical system",
                    ],
                    impact="Affects risk level and threshold settings",
                    metadata={"affects": "risk_level"},
                )
            )

        # Ask about data sensitivity if unclear
        if analysis.domain_sensitivity == DomainSensitivity.GENERAL:
            questions.append(
                ClarificationQuestion(
                    question="What type of data does this application handle?",
                    options=[
                        "General content (no sensitive data)",
                        "Personal information (names, emails, addresses)",
                        "Financial data (payments, transactions)",
                        "Healthcare/medical information",
                        "Content for children/minors",
                    ],
                    impact="Affects scanner selection and PII protection",
                    metadata={"affects": "domain_sensitivity"},
                )
            )

        # Ask about specific features if uncertain
        if analysis.confidence < 0.7:
            questions.append(
                ClarificationQuestion(
                    question="Which features does your application use?",
                    options=[
                        "Simple Q&A (no external data)",
                        "RAG/retrieval from documents",
                        "Tool/function calling",
                        "Code generation/execution",
                        "Image processing/generation",
                    ],
                    impact="Affects evaluation selection",
                    metadata={"affects": "evaluations", "multi_select": True},
                )
            )

        # Ask about compliance requirements
        if analysis.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            questions.append(
                ClarificationQuestion(
                    question="Are there specific compliance requirements?",
                    options=[
                        "No specific requirements",
                        "HIPAA (healthcare)",
                        "PCI-DSS (payments)",
                        "GDPR/CCPA (privacy)",
                        "SOC 2 (security)",
                    ],
                    impact="Adds compliance-specific scanners",
                    metadata={"affects": "scanners"},
                )
            )

        return questions

    def answer(
        self,
        session: InteractiveSession,
        question: str,
        answer: str,
    ) -> None:
        """
        Record an answer to a question.

        Args:
            session: The interactive session
            question: The question being answered
            answer: The selected answer
        """
        session.answers[question] = answer

        # Find and mark the question as answered
        for q in session.questions:
            if q.question == question:
                q.selected = answer
                break

    def finalize(self, session: InteractiveSession) -> AutoEvalConfig:
        """
        Finalize the configuration based on answers.

        Args:
            session: Completed interactive session

        Returns:
            Adjusted AutoEvalConfig
        """
        from .recommender import EvalRecommender

        # Create a copy of the analysis to modify
        analysis = session.analysis

        # Adjust analysis based on answers
        adjusted_analysis = self._adjust_analysis(analysis, session.answers)

        # Generate new recommendations with adjusted analysis
        recommender = EvalRecommender()
        evals, scanners = recommender.recommend(adjusted_analysis)

        # Build config
        config = AutoEvalConfig(
            name=f"autoeval_{adjusted_analysis.category.value}",
            description=f"Interactive configuration for {adjusted_analysis.category.value}",
            app_category=adjusted_analysis.category.value,
            risk_level=adjusted_analysis.risk_level.value,
            domain_sensitivity=adjusted_analysis.domain_sensitivity.value,
            evaluations=evals,
            scanners=scanners,
        )

        session.completed = True
        return config

    def _adjust_analysis(
        self,
        analysis: AppAnalysis,
        answers: Dict[str, str],
    ) -> AppAnalysis:
        """Adjust analysis based on user answers."""
        # Start with current values
        risk_level = analysis.risk_level
        domain_sensitivity = analysis.domain_sensitivity
        features = list(analysis.detected_features)

        # Process deployment environment answer
        env_answer = answers.get("What is the deployment environment?")
        if env_answer:
            if "development" in env_answer.lower() or "internal tool" in env_answer.lower():
                risk_level = RiskLevel.LOW
            elif "internal users" in env_answer.lower():
                risk_level = RiskLevel.MEDIUM
            elif "external" in env_answer.lower() or "public" in env_answer.lower():
                risk_level = RiskLevel.HIGH
            elif "safety-critical" in env_answer.lower():
                risk_level = RiskLevel.CRITICAL

        # Process data sensitivity answer
        data_answer = answers.get("What type of data does this application handle?")
        if data_answer:
            if "personal information" in data_answer.lower():
                domain_sensitivity = DomainSensitivity.PII_SENSITIVE
            elif "financial" in data_answer.lower():
                domain_sensitivity = DomainSensitivity.FINANCIAL
            elif "healthcare" in data_answer.lower() or "medical" in data_answer.lower():
                domain_sensitivity = DomainSensitivity.HEALTHCARE
            elif "children" in data_answer.lower():
                domain_sensitivity = DomainSensitivity.CHILDREN

        # Process feature answer
        feature_answer = answers.get("Which features does your application use?")
        if feature_answer:
            if "rag" in feature_answer.lower() or "retrieval" in feature_answer.lower():
                if "rag" not in features:
                    features.append("rag")
            if "tool" in feature_answer.lower() or "function" in feature_answer.lower():
                if "tool_use" not in features:
                    features.append("tool_use")
            if "code" in feature_answer.lower():
                if "code_generation" not in features:
                    features.append("code_generation")
            if "image" in feature_answer.lower():
                if "image_processing" not in features:
                    features.append("image_processing")

        # Create adjusted analysis
        return AppAnalysis(
            category=analysis.category,
            risk_level=risk_level,
            domain_sensitivity=domain_sensitivity,
            requirements=analysis.requirements,
            detected_features=features,
            confidence=0.9,  # Higher confidence after clarification
            explanation=f"Adjusted after user clarification: {analysis.explanation}",
        )

