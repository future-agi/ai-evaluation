"""
Eval Delegate Scanner for Guardrails.

Delegates scanning to existing evaluation templates for overlapping functionality:
- PII Detection → Template 14, 22 (PII, DataPrivacyCompliance)
- Toxicity → Template 15 (Toxicity)
- Prompt Injection → Template 18 (PromptInjection)
- Bias Detection → Template 69, 77-79 (BiasDetection, NoRacialBias, etc.)
- Content Safety → Template 93 (ContentSafety)
- NSFW/Sexist → Template 17, 20 (Sexist, SafeForWorkText)

This scanner bridges the guardrails system with the evaluation framework,
allowing you to use battle-tested LLM-based evaluations as safety scanners.
"""

import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from fi.evals.guardrails.scanners.base import (
    BaseScanner,
    ScanResult,
    ScanMatch,
    ScannerAction,
    register_scanner,
)


class EvalCategory(str, Enum):
    """Categories of evals that can be delegated to."""
    PII = "pii"
    TOXICITY = "toxicity"
    PROMPT_INJECTION = "prompt_injection"
    BIAS = "bias"
    RACIAL_BIAS = "racial_bias"
    GENDER_BIAS = "gender_bias"
    AGE_BIAS = "age_bias"
    CONTENT_SAFETY = "content_safety"
    NSFW = "nsfw"
    SEXIST = "sexist"


# Mapping from category to eval template info
EVAL_TEMPLATE_MAP: Dict[EvalCategory, Dict[str, Any]] = {
    EvalCategory.PII: {
        "eval_id": "14",
        "eval_name": "pii",
        "description": "Detects personally identifiable information",
        "threshold": 0.5,  # Score above this = PII detected
        "invert": True,  # High score = bad (PII found)
    },
    EvalCategory.TOXICITY: {
        "eval_id": "15",
        "eval_name": "toxicity",
        "description": "Detects toxic or harmful content",
        "threshold": 0.5,
        "invert": True,
    },
    EvalCategory.PROMPT_INJECTION: {
        "eval_id": "18",
        "eval_name": "prompt_injection",
        "description": "Detects prompt injection attempts",
        "threshold": 0.5,
        "invert": True,
    },
    EvalCategory.BIAS: {
        "eval_id": "69",
        "eval_name": "bias_detection",
        "description": "Detects various forms of bias",
        "threshold": 0.5,
        "invert": True,
    },
    EvalCategory.RACIAL_BIAS: {
        "eval_id": "77",
        "eval_name": "no_racial_bias",
        "description": "Detects racial bias",
        "threshold": 0.5,
        "invert": False,  # High score = good (no bias)
    },
    EvalCategory.GENDER_BIAS: {
        "eval_id": "78",
        "eval_name": "no_gender_bias",
        "description": "Detects gender bias",
        "threshold": 0.5,
        "invert": False,
    },
    EvalCategory.AGE_BIAS: {
        "eval_id": "79",
        "eval_name": "no_age_bias",
        "description": "Detects age bias",
        "threshold": 0.5,
        "invert": False,
    },
    EvalCategory.CONTENT_SAFETY: {
        "eval_id": "93",
        "eval_name": "content_safety_violation",
        "description": "Detects content safety violations",
        "threshold": 0.5,
        "invert": True,
    },
    EvalCategory.NSFW: {
        "eval_id": "20",
        "eval_name": "safe_for_work_text",
        "description": "Detects NSFW content",
        "threshold": 0.5,
        "invert": False,  # High score = safe
    },
    EvalCategory.SEXIST: {
        "eval_id": "17",
        "eval_name": "sexist",
        "description": "Detects sexist content",
        "threshold": 0.5,
        "invert": True,
    },
}


@dataclass
class EvalDelegateConfig:
    """Configuration for EvalDelegateScanner."""
    # Categories to check
    categories: List[EvalCategory] = field(default_factory=lambda: [EvalCategory.TOXICITY])

    # Thresholds per category (overrides defaults)
    thresholds: Dict[EvalCategory, float] = field(default_factory=dict)

    # Use local evaluator if available (faster, no API calls)
    prefer_local: bool = True

    # API key for cloud evaluation (if not using local)
    api_key: Optional[str] = None

    # Timeout for evaluation calls (seconds)
    timeout: int = 30

    # Whether to run categories in parallel
    parallel: bool = True

    # Aggregation mode: "any" (fail if any category fails) or "all" (fail only if all fail)
    aggregation: str = "any"


@register_scanner("eval_delegate")
class EvalDelegateScanner(BaseScanner):
    """
    Scanner that delegates detection to existing evaluation templates.

    This scanner bridges guardrails with the evaluation framework, using
    proven LLM-based evaluations for safety detection.

    Supported Categories:
    - pii: Detect personally identifiable information
    - toxicity: Detect toxic or harmful content
    - prompt_injection: Detect prompt injection attempts
    - bias: Detect various forms of bias
    - racial_bias, gender_bias, age_bias: Specific bias detection
    - content_safety: Detect content safety violations
    - nsfw: Detect not-safe-for-work content
    - sexist: Detect sexist content

    Usage:
        # Single category
        scanner = EvalDelegateScanner(categories=[EvalCategory.TOXICITY])
        result = scanner.scan("Your content here")

        # Multiple categories
        scanner = EvalDelegateScanner(
            categories=[EvalCategory.TOXICITY, EvalCategory.PII, EvalCategory.BIAS]
        )

        # With custom thresholds
        scanner = EvalDelegateScanner(
            categories=[EvalCategory.TOXICITY],
            thresholds={EvalCategory.TOXICITY: 0.7}
        )

        # Factory methods
        scanner = EvalDelegateScanner.for_toxicity()
        scanner = EvalDelegateScanner.for_pii()
        scanner = EvalDelegateScanner.for_safety()  # Multiple categories

    Note:
        This scanner requires either:
        - API key for cloud-based evaluation (more accurate, slower)
        - Local LLM setup for local evaluation (faster, may be less accurate)
    """

    name = "eval_delegate"
    category = "eval_delegate"
    description = "Delegates to evaluation templates"
    default_action = ScannerAction.BLOCK

    def __init__(
        self,
        categories: Optional[List[EvalCategory]] = None,
        thresholds: Optional[Dict[EvalCategory, float]] = None,
        prefer_local: bool = True,
        api_key: Optional[str] = None,
        timeout: int = 30,
        parallel: bool = True,
        aggregation: str = "any",
        action: Optional[ScannerAction] = None,
        enabled: bool = True,
    ):
        """
        Initialize the eval delegate scanner.

        Args:
            categories: List of eval categories to check
            thresholds: Custom thresholds per category
            prefer_local: Use local evaluator if available
            api_key: API key for cloud evaluation
            timeout: Timeout in seconds
            parallel: Run categories in parallel
            aggregation: "any" or "all" for failure aggregation
            action: Action to take on detection
            enabled: Whether scanner is enabled
        """
        super().__init__(action=action, enabled=enabled)

        self.categories = categories or [EvalCategory.TOXICITY]
        self.thresholds = thresholds or {}
        self.prefer_local = prefer_local
        self.api_key = api_key
        self.timeout = timeout
        self.parallel = parallel
        self.aggregation = aggregation

        # Lazy-loaded evaluators
        self._local_evaluator = None
        self._cloud_evaluator = None

    @classmethod
    def for_toxicity(cls, threshold: float = 0.5, **kwargs) -> "EvalDelegateScanner":
        """Create a scanner for toxicity detection."""
        return cls(
            categories=[EvalCategory.TOXICITY],
            thresholds={EvalCategory.TOXICITY: threshold},
            **kwargs
        )

    @classmethod
    def for_pii(cls, threshold: float = 0.5, **kwargs) -> "EvalDelegateScanner":
        """Create a scanner for PII detection."""
        return cls(
            categories=[EvalCategory.PII],
            thresholds={EvalCategory.PII: threshold},
            **kwargs
        )

    @classmethod
    def for_prompt_injection(cls, threshold: float = 0.5, **kwargs) -> "EvalDelegateScanner":
        """Create a scanner for prompt injection detection."""
        return cls(
            categories=[EvalCategory.PROMPT_INJECTION],
            thresholds={EvalCategory.PROMPT_INJECTION: threshold},
            **kwargs
        )

    @classmethod
    def for_bias(cls, include_specific: bool = True, threshold: float = 0.5, **kwargs) -> "EvalDelegateScanner":
        """
        Create a scanner for bias detection.

        Args:
            include_specific: Include racial, gender, age bias checks
            threshold: Detection threshold
        """
        categories = [EvalCategory.BIAS]
        if include_specific:
            categories.extend([
                EvalCategory.RACIAL_BIAS,
                EvalCategory.GENDER_BIAS,
                EvalCategory.AGE_BIAS,
            ])

        thresholds = {cat: threshold for cat in categories}
        return cls(categories=categories, thresholds=thresholds, **kwargs)

    @classmethod
    def for_safety(cls, threshold: float = 0.5, **kwargs) -> "EvalDelegateScanner":
        """
        Create a comprehensive safety scanner.

        Includes: toxicity, PII, prompt injection, content safety, NSFW
        """
        categories = [
            EvalCategory.TOXICITY,
            EvalCategory.PII,
            EvalCategory.PROMPT_INJECTION,
            EvalCategory.CONTENT_SAFETY,
            EvalCategory.NSFW,
        ]
        thresholds = {cat: threshold for cat in categories}
        return cls(categories=categories, thresholds=thresholds, **kwargs)

    @classmethod
    def for_content_moderation(cls, threshold: float = 0.5, **kwargs) -> "EvalDelegateScanner":
        """
        Create a content moderation scanner.

        Includes: toxicity, NSFW, sexist, content safety
        """
        categories = [
            EvalCategory.TOXICITY,
            EvalCategory.NSFW,
            EvalCategory.SEXIST,
            EvalCategory.CONTENT_SAFETY,
        ]
        thresholds = {cat: threshold for cat in categories}
        return cls(categories=categories, thresholds=thresholds, **kwargs)

    def _get_local_evaluator(self):
        """Get or create local evaluator."""
        if self._local_evaluator is None:
            try:
                from fi.evals.local import LocalEvaluator
                self._local_evaluator = LocalEvaluator()
            except ImportError:
                self._local_evaluator = None
        return self._local_evaluator

    def _get_cloud_evaluator(self):
        """Get or create cloud evaluator."""
        if self._cloud_evaluator is None:
            try:
                from fi.evals.evaluator import Evaluator
                self._cloud_evaluator = Evaluator(fi_api_key=self.api_key)
            except Exception:
                self._cloud_evaluator = None
        return self._cloud_evaluator

    def _get_threshold(self, category: EvalCategory) -> float:
        """Get threshold for a category."""
        if category in self.thresholds:
            return self.thresholds[category]
        return EVAL_TEMPLATE_MAP[category].get("threshold", 0.5)

    def _evaluate_category(
        self,
        content: str,
        category: EvalCategory,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate content for a single category.

        Returns:
            Dict with keys: passed, score, reason, latency_ms
        """
        template_info = EVAL_TEMPLATE_MAP[category]
        eval_name = template_info["eval_name"]
        threshold = self._get_threshold(category)
        invert = template_info.get("invert", True)

        start_time = time.perf_counter()

        # Prepare input
        eval_input = {"response": content}
        if context:
            eval_input["context"] = context

        # Try local evaluation first if preferred
        if self.prefer_local:
            local_eval = self._get_local_evaluator()
            if local_eval and local_eval.can_run_locally(eval_name):
                try:
                    result = local_eval.evaluate(
                        metric_name=eval_name,
                        inputs=[eval_input],
                    )
                    if result.results.eval_results:
                        eval_result = result.results.eval_results[0]
                        score = eval_result.output or 0.0
                        latency = (time.perf_counter() - start_time) * 1000

                        # Determine pass/fail
                        if invert:
                            passed = score < threshold
                        else:
                            passed = score >= threshold

                        return {
                            "passed": passed,
                            "score": score,
                            "reason": eval_result.reason or f"{eval_name}: {score:.2f}",
                            "latency_ms": latency,
                            "source": "local",
                        }
                except Exception as e:
                    # Fall through to cloud evaluation
                    pass

        # Try cloud evaluation
        cloud_eval = self._get_cloud_evaluator()
        if cloud_eval:
            try:
                result = cloud_eval.evaluate(
                    eval_templates=[eval_name],
                    inputs=[eval_input],
                )
                if result.eval_results:
                    eval_result = result.eval_results[0]
                    score = eval_result.output or 0.0
                    latency = (time.perf_counter() - start_time) * 1000

                    if invert:
                        passed = score < threshold
                    else:
                        passed = score >= threshold

                    return {
                        "passed": passed,
                        "score": score,
                        "reason": eval_result.reason or f"{eval_name}: {score:.2f}",
                        "latency_ms": latency,
                        "source": "cloud",
                    }
            except Exception as e:
                latency = (time.perf_counter() - start_time) * 1000
                return {
                    "passed": True,  # Default to pass on error
                    "score": 0.0,
                    "reason": f"Evaluation error: {str(e)}",
                    "latency_ms": latency,
                    "source": "error",
                }

        # No evaluator available
        latency = (time.perf_counter() - start_time) * 1000
        return {
            "passed": True,
            "score": 0.0,
            "reason": "No evaluator available",
            "latency_ms": latency,
            "source": "none",
        }

    def scan(self, content: str, context: Optional[str] = None) -> ScanResult:
        """
        Scan content using delegated evaluations.

        Args:
            content: Content to scan
            context: Optional context

        Returns:
            ScanResult with aggregated results from all categories
        """
        if not self.enabled:
            return self._create_result(
                passed=True,
                reason="Scanner disabled",
            )

        start_time = time.perf_counter()
        matches = []
        category_results = {}

        # Evaluate each category
        if self.parallel and len(self.categories) > 1:
            # Parallel execution
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=len(self.categories)) as executor:
                futures = {
                    executor.submit(self._evaluate_category, content, cat, context): cat
                    for cat in self.categories
                }

                for future in as_completed(futures, timeout=self.timeout):
                    cat = futures[future]
                    try:
                        category_results[cat] = future.result()
                    except Exception as e:
                        category_results[cat] = {
                            "passed": True,
                            "score": 0.0,
                            "reason": f"Timeout/error: {str(e)}",
                            "latency_ms": 0,
                            "source": "error",
                        }
        else:
            # Sequential execution
            for cat in self.categories:
                category_results[cat] = self._evaluate_category(content, cat, context)

        # Aggregate results
        failed_categories = []
        total_score = 0.0

        for cat, result in category_results.items():
            total_score += result["score"]

            if not result["passed"]:
                failed_categories.append(cat)
                template_info = EVAL_TEMPLATE_MAP[cat]
                matches.append(ScanMatch(
                    pattern_name=f"eval:{cat.value}",
                    matched_text=content[:100] + "..." if len(content) > 100 else content,
                    start=0,
                    end=len(content),
                    confidence=result["score"],
                    metadata={
                        "category": cat.value,
                        "eval_name": template_info["eval_name"],
                        "threshold": self._get_threshold(cat),
                        "source": result.get("source", "unknown"),
                    },
                ))

        # Determine overall pass/fail
        if self.aggregation == "any":
            passed = len(failed_categories) == 0
        else:  # "all"
            passed = len(failed_categories) < len(self.categories)

        # Calculate average score
        avg_score = total_score / len(self.categories) if self.categories else 0.0

        # Generate reason
        if passed:
            reason = f"All {len(self.categories)} eval checks passed"
        else:
            failed_names = [c.value for c in failed_categories]
            reason = f"Failed eval checks: {', '.join(failed_names)}"

        total_latency = (time.perf_counter() - start_time) * 1000

        return self._create_result(
            passed=passed,
            matches=matches,
            score=avg_score,
            reason=reason,
            latency_ms=total_latency,
            metadata={
                "categories_checked": [c.value for c in self.categories],
                "categories_failed": [c.value for c in failed_categories],
                "category_results": {
                    c.value: {
                        "passed": r["passed"],
                        "score": r["score"],
                        "source": r.get("source", "unknown"),
                    }
                    for c, r in category_results.items()
                },
            },
        )


# Convenience aliases
PIIScanner = lambda **kwargs: EvalDelegateScanner.for_pii(**kwargs)
ToxicityScanner = lambda **kwargs: EvalDelegateScanner.for_toxicity(**kwargs)
PromptInjectionScanner = lambda **kwargs: EvalDelegateScanner.for_prompt_injection(**kwargs)
BiasScanner = lambda **kwargs: EvalDelegateScanner.for_bias(**kwargs)
SafetyScanner = lambda **kwargs: EvalDelegateScanner.for_safety(**kwargs)
ContentModerationScanner = lambda **kwargs: EvalDelegateScanner.for_content_moderation(**kwargs)
