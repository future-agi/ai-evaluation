"""
Turing Backend for Guardrails.

Uses FutureAGI's Turing API for content screening.
- turing_flash: Fast binary classification
- turing_safety: Higher accuracy, detailed categories
"""

import time
from typing import Dict, List, Optional, Any

from fi.evals.guardrails.backends.base import BaseBackend
from fi.evals.guardrails.config import RailType, GuardrailModel
from fi.evals.guardrails.types import GuardrailResult
from fi.evals.protect import Protect
from fi.utils.utils import get_keys_from_env


class TuringBackend(BaseBackend):
    """
    Backend for Turing models (turing_flash, turing_safety).

    Uses the existing Protect class for API calls to FutureAGI.
    """

    def __init__(
        self,
        model: GuardrailModel,
        fi_api_key: Optional[str] = None,
        fi_secret_key: Optional[str] = None,
        fi_base_url: Optional[str] = None,
    ):
        """
        Initialize Turing backend.

        Args:
            model: The Turing model to use
            fi_api_key: FutureAGI API key
            fi_secret_key: FutureAGI secret key
            fi_base_url: Base URL for API
        """
        super().__init__(model)

        # Get keys from environment if not provided
        if not fi_api_key or not fi_secret_key:
            env_api_key, env_secret_key = get_keys_from_env()
            fi_api_key = fi_api_key or env_api_key
            fi_secret_key = fi_secret_key or env_secret_key

        self._protect = Protect(
            fi_api_key=fi_api_key,
            fi_secret_key=fi_secret_key,
            fi_base_url=fi_base_url,
        )

        # Determine if this is flash or safety mode
        self._use_flash = model == GuardrailModel.TURING_FLASH

    def classify(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[GuardrailResult]:
        """
        Classify content using Turing API.

        Args:
            content: Content to classify
            rail_type: Type of rail
            context: Optional context
            metadata: Optional metadata

        Returns:
            List of GuardrailResult objects
        """
        start_time = time.time()
        results = []

        # Handle empty or whitespace-only content
        if not content or not content.strip():
            return [
                GuardrailResult(
                    passed=True,
                    category="empty",
                    score=0.0,
                    model=self.model_name,
                    reason="Empty or whitespace-only content",
                    action="pass",
                    latency_ms=(time.time() - start_time) * 1000,
                )
            ]

        try:
            if self._use_flash:
                # Use ProtectFlash for fast binary classification
                response = self._protect.protect(
                    inputs=content,
                    use_flash=True,
                    timeout=30000,
                )
            else:
                # Use full Protect with content moderation rules
                response = self._protect.protect(
                    inputs=content,
                    protect_rules=[
                        {"metric": "content_moderation"},
                        {"metric": "security"},
                        {"metric": "bias_detection"},
                    ],
                    timeout=30000,
                )

            elapsed_ms = (time.time() - start_time) * 1000

            # Parse response
            is_harmful = response.get("status") == "failed"
            failed_rule = response.get("failed_rule", [])
            reasons = response.get("reasons", [])

            if is_harmful:
                # Determine categories from failed rules
                categories = self._extract_categories(failed_rule, reasons)
                for category in categories:
                    results.append(
                        GuardrailResult(
                            passed=False,
                            category=category,
                            score=1.0,
                            model=self.model_name,
                            reason=reasons[0] if reasons else None,
                            action="block",
                            latency_ms=elapsed_ms,
                        )
                    )

                # If no specific categories, use generic harmful_content
                if not results:
                    results.append(
                        GuardrailResult(
                            passed=False,
                            category="harmful_content",
                            score=1.0,
                            model=self.model_name,
                            reason=reasons[0] if reasons else "Content flagged as harmful",
                            action="block",
                            latency_ms=elapsed_ms,
                        )
                    )
            else:
                # Content passed
                results.append(
                    GuardrailResult(
                        passed=True,
                        category="safe",
                        score=0.0,
                        model=self.model_name,
                        reason="Content passed all checks",
                        action="pass",
                        latency_ms=elapsed_ms,
                    )
                )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            results.append(
                GuardrailResult(
                    passed=False,  # Fail closed by default
                    category="error",
                    score=0.0,
                    model=self.model_name,
                    reason=f"Error during classification: {str(e)}",
                    action="block",
                    latency_ms=elapsed_ms,
                )
            )

        return results

    def _extract_categories(
        self,
        failed_rules: List[str],
        reasons: List[str],
    ) -> List[str]:
        """
        Extract category names from failed rules and reasons.

        Args:
            failed_rules: List of failed rule names
            reasons: List of reason strings

        Returns:
            List of category names
        """
        categories = set()

        # Map rule names to categories
        rule_to_category = {
            "content_moderation": "toxicity",
            "security": "prompt_injection",
            "bias_detection": "hate_speech",
            "data_privacy_compliance": "pii",
            "ProtectFlash": "harmful_content",
        }

        if isinstance(failed_rules, list):
            for rule in failed_rules:
                if rule in rule_to_category:
                    categories.add(rule_to_category[rule])
                else:
                    categories.add(rule.lower().replace(" ", "_"))
        elif isinstance(failed_rules, str):
            if failed_rules in rule_to_category:
                categories.add(rule_to_category[failed_rules])
            else:
                categories.add(failed_rules.lower().replace(" ", "_"))

        # Also parse reasons for additional context
        for reason in reasons:
            reason_lower = reason.lower() if reason else ""
            if "jailbreak" in reason_lower:
                categories.add("jailbreak")
            if "injection" in reason_lower:
                categories.add("prompt_injection")
            if "harm" in reason_lower or "violence" in reason_lower:
                categories.add("violence")
            if "hate" in reason_lower:
                categories.add("hate_speech")
            if "self-harm" in reason_lower or "suicide" in reason_lower:
                categories.add("self_harm")
            if "sexual" in reason_lower:
                categories.add("sexual_content")

        return list(categories) if categories else ["harmful_content"]
