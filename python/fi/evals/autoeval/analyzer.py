"""Application analyzer for AutoEval.

Analyzes application descriptions to understand evaluation needs.
Supports both LLM-powered and rule-based analysis.
"""

import json
import logging
from typing import Optional, Any, Protocol, runtime_checkable

from .types import (
    AppAnalysis,
    AppCategory,
    RiskLevel,
    DomainSensitivity,
    AppRequirement,
)
from .rules import RuleBasedAnalyzer
from .prompts import ANALYSIS_SYSTEM_PROMPT, ANALYSIS_USER_PROMPT

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a completion for the given prompt."""
        ...


class AppAnalyzer:
    """
    Analyzes application descriptions to understand evaluation needs.

    Supports both LLM-powered and rule-based analysis. When LLM analysis
    fails, automatically falls back to rule-based analysis.

    Example:
        # With LLM provider
        from fi.evals.llm import LiteLLMProvider
        analyzer = AppAnalyzer(llm_provider=LiteLLMProvider())
        analysis = analyzer.analyze(
            "A customer support chatbot for a healthcare company. "
            "It retrieves patient information and answers questions."
        )

        # Without LLM (rule-based only)
        analyzer = AppAnalyzer(use_llm=False)
        analysis = analyzer.analyze("A RAG-based document Q&A system.")
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        use_llm: bool = True,
        temperature: float = 0.1,
    ):
        """
        Initialize the analyzer.

        Args:
            llm_provider: LLM provider for intelligent analysis (optional)
            model: Model to use for LLM analysis
            use_llm: If False, always use rule-based fallback
            temperature: Temperature for LLM generation (lower = more deterministic)
        """
        self.llm_provider = llm_provider
        self.model = model
        self.use_llm = use_llm and llm_provider is not None
        self.temperature = temperature
        self.rule_analyzer = RuleBasedAnalyzer()

    def analyze(self, description: str) -> AppAnalysis:
        """
        Analyze an application description.

        Args:
            description: Natural language description of the application

        Returns:
            AppAnalysis with detected requirements and recommendations
        """
        if not description or not description.strip():
            return AppAnalysis(
                category=AppCategory.UNKNOWN,
                risk_level=RiskLevel.MEDIUM,
                domain_sensitivity=DomainSensitivity.GENERAL,
                requirements=[],
                detected_features=[],
                confidence=0.0,
                explanation="Empty description provided.",
            )

        if self.use_llm:
            try:
                return self._analyze_with_llm(description)
            except Exception as e:
                logger.warning(f"LLM analysis failed, falling back to rules: {e}")
                return self._analyze_with_rules(description)
        else:
            return self._analyze_with_rules(description)

    def _analyze_with_llm(self, description: str) -> AppAnalysis:
        """Use LLM to analyze the description."""
        if not self.llm_provider:
            raise ValueError("LLM provider not configured")

        prompt = ANALYSIS_USER_PROMPT.format(description=description)

        # Try different ways to call the LLM provider
        try:
            if hasattr(self.llm_provider, "complete"):
                response = self.llm_provider.complete(
                    prompt=prompt,
                    system_prompt=ANALYSIS_SYSTEM_PROMPT,
                    model=self.model,
                    temperature=self.temperature,
                )
            elif hasattr(self.llm_provider, "generate"):
                response = self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt=ANALYSIS_SYSTEM_PROMPT,
                    model=self.model,
                    temperature=self.temperature,
                )
            elif callable(self.llm_provider):
                response = self.llm_provider(
                    prompt=prompt,
                    system_prompt=ANALYSIS_SYSTEM_PROMPT,
                )
            else:
                raise ValueError(f"Unknown LLM provider type: {type(self.llm_provider)}")
        except TypeError as e:
            # Only retry with simpler args if it's a signature mismatch
            if "unexpected keyword argument" in str(e) or "positional argument" in str(e):
                if hasattr(self.llm_provider, "complete"):
                    response = self.llm_provider.complete(prompt)
                else:
                    raise
            else:
                raise

        # Parse JSON response
        return self._parse_llm_response(response, description)

    def _parse_llm_response(self, response: str, description: str) -> AppAnalysis:
        """Parse LLM response into AppAnalysis."""
        # Clean up response - remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Fall back to rule-based
            return self._analyze_with_rules(description)

        # Parse with defaults
        try:
            category = AppCategory(data.get("category", "unknown"))
        except ValueError:
            category = AppCategory.UNKNOWN

        try:
            risk_level = RiskLevel(data.get("risk_level", "medium"))
        except ValueError:
            risk_level = RiskLevel.MEDIUM

        try:
            domain_sensitivity = DomainSensitivity(
                data.get("domain_sensitivity", "general")
            )
        except ValueError:
            domain_sensitivity = DomainSensitivity.GENERAL

        requirements = []
        for req_data in data.get("requirements", []):
            requirements.append(
                AppRequirement(
                    category=req_data.get("category", ""),
                    importance=req_data.get("importance", "recommended"),
                    reason=req_data.get("reason", ""),
                    suggested_evals=req_data.get("suggested_evals", []),
                    suggested_scanners=req_data.get("suggested_scanners", []),
                )
            )

        return AppAnalysis(
            category=category,
            risk_level=risk_level,
            domain_sensitivity=domain_sensitivity,
            requirements=requirements,
            detected_features=data.get("detected_features", []),
            confidence=0.85,  # LLM analysis has higher confidence
            explanation=data.get("explanation", "Analysis performed by LLM."),
        )

    def _analyze_with_rules(self, description: str) -> AppAnalysis:
        """Use rule-based analysis."""
        return self.rule_analyzer.analyze(description)
