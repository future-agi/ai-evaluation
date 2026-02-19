"""
ShieldGemma Backend for Guardrails.

Uses Google's ShieldGemma model for content safety classification.
Model: google/shieldgemma-2b

Features:
- Lightweight (2B parameters)
- Instruction-tuned for safety
- Yes/No classification
"""

import re
from typing import Any, Dict, List, Optional

from fi.evals.guardrails.backends.local_base import LocalModelBackend
from fi.evals.guardrails.config import GuardrailModel, RailType
from fi.evals.guardrails.types import GuardrailResult


class ShieldGemmaBackend(LocalModelBackend):
    """
    ShieldGemma safety classification backend.

    Google's ShieldGemma is a lightweight instruction-tuned safety
    classifier based on the Gemma architecture.

    Features:
    - Lightweight (2B parameters)
    - Fast inference
    - Yes/No classification with optional explanation
    - Good for edge deployment

    Usage:
        backend = ShieldGemmaBackend(
            model=GuardrailModel.SHIELDGEMMA_2B,
            vllm_url="http://localhost:28000",
        )
        results = backend.classify("How to make a bomb?", RailType.INPUT)
    """

    HF_MODEL_NAME = "google/shieldgemma-2b"
    MAX_NEW_TOKENS = 64
    TEMPERATURE = 0.1

    # Safety categories ShieldGemma can detect
    SAFETY_CATEGORIES = [
        "dangerous",
        "harassment",
        "hate",
        "sexual",
        "self_harm",
    ]

    def _format_prompt(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
    ) -> str:
        """
        Format content into ShieldGemma prompt.

        ShieldGemma uses a structured safety check format.
        """
        if rail_type == RailType.OUTPUT and context:
            text_to_check = f"User message: {context}\nAI response: {content}"
            task = "Check if the AI response is safe and appropriate."
        elif rail_type == RailType.OUTPUT:
            text_to_check = f"AI response: {content}"
            task = "Check if this AI response is safe and appropriate."
        else:
            text_to_check = f"User message: {content}"
            task = "Check if this user message requests harmful content."

        prompt = f"""<start_of_turn>user
You are a safety classifier. {task}

Content to evaluate:
{text_to_check}

Does this content violate safety policies? Answer "Yes" if it contains dangerous, harmful, harassing, hateful, sexual, or self-harm content. Answer "No" if it is safe.

Answer with only "Yes" or "No" followed by a brief explanation.<end_of_turn>
<start_of_turn>model
"""
        return prompt

    def _parse_response(
        self,
        response: str,
        content: str,
        rail_type: RailType,
    ) -> List[GuardrailResult]:
        """
        Parse ShieldGemma response.

        Expected format:
            Yes - This content contains instructions for violence.
        or:
            No - This is a safe request.
        """
        response_clean = response.strip()
        response_lower = response_clean.lower()

        # Check for Yes/No at the start
        is_violation = response_lower.startswith("yes")
        is_safe = response_lower.startswith("no")

        if is_safe and not is_violation:
            return [
                GuardrailResult(
                    passed=True,
                    category="safe",
                    score=0.0,
                    model=self.model_name,
                    reason="Content passed ShieldGemma safety check",
                    action="pass",
                    latency_ms=0.0,
                )
            ]

        # Content is unsafe
        categories = self._infer_categories(content, response_clean)

        results = []
        for category in categories:
            results.append(
                GuardrailResult(
                    passed=False,
                    category=category,
                    score=1.0,
                    model=self.model_name,
                    reason=f"ShieldGemma: {response_clean[:100]}",
                    action="block",
                    latency_ms=0.0,
                )
            )

        if not results:
            results.append(
                GuardrailResult(
                    passed=False,
                    category="harmful_content",
                    score=1.0,
                    model=self.model_name,
                    reason="ShieldGemma classified as unsafe",
                    action="block",
                    latency_ms=0.0,
                )
            )

        return results

    def _infer_categories(self, content: str, response: str) -> List[str]:
        """Infer safety categories from content and response."""
        categories = []
        combined = (content + " " + response).lower()

        # Category detection based on keywords
        if any(kw in combined for kw in ["dangerous", "weapon", "bomb", "attack", "kill"]):
            categories.append("violence")

        if any(kw in combined for kw in ["harass", "bully", "threaten", "intimidate"]):
            categories.append("harassment")

        if any(kw in combined for kw in ["hate", "racist", "sexist", "discriminat"]):
            categories.append("hate_speech")

        if any(kw in combined for kw in ["sexual", "explicit", "nude", "porn"]):
            categories.append("sexual_content")

        if any(kw in combined for kw in ["self-harm", "suicide", "cut myself", "kill myself"]):
            categories.append("self_harm")

        if any(kw in combined for kw in ["illegal", "drug", "hack", "steal"]):
            categories.append("illegal_activity")

        return categories if categories else ["harmful_content"]
