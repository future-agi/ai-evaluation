"""
Qwen3Guard Backend for Guardrails.

Uses Alibaba's Qwen3Guard model for content safety classification.
Models: Qwen/Qwen3Guard-8B, Qwen/Qwen3Guard-4B

Features:
- 119 language support
- Safe/Controversial/Unsafe classification
- Streaming support (Qwen3Guard-Stream)
"""

import re
from typing import Any, Dict, List, Optional

from fi.evals.guardrails.backends.local_base import LocalModelBackend
from fi.evals.guardrails.config import GuardrailModel, RailType
from fi.evals.guardrails.types import GuardrailResult


class Qwen3GuardBackend(LocalModelBackend):
    """
    Qwen3Guard safety classification backend.

    Alibaba's Qwen3Guard is a multilingual safety classifier
    supporting 119 languages with three-level classification.

    Features:
    - 119 language support
    - Safe/Controversial/Unsafe classification
    - Severity levels
    - Available in 8B, 4B, and streaming variants

    Usage:
        backend = Qwen3GuardBackend(
            model=GuardrailModel.QWEN3GUARD_8B,
            vllm_url="http://localhost:28000",
        )
        results = backend.classify("How to make a bomb?", RailType.INPUT)
    """

    MODEL_VARIANTS = {
        "qwen3guard-8b": "Qwen/Qwen3Guard-8B",
        "qwen3guard-4b": "Qwen/Qwen3Guard-4B",
    }

    MAX_NEW_TOKENS = 64
    TEMPERATURE = 0.1

    # Qwen3Guard classification to our categories
    SEVERITY_THRESHOLDS = {
        "safe": 0.0,
        "controversial": 0.5,
        "unsafe": 1.0,
    }

    def _get_hf_model_name(self) -> str:
        """Get the HuggingFace model name based on variant."""
        model_value = self.model.value
        return self.MODEL_VARIANTS.get(model_value, self.MODEL_VARIANTS["qwen3guard-8b"])

    def _format_prompt(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
    ) -> str:
        """
        Format content into Qwen3Guard prompt.

        Qwen3Guard uses a simple classification prompt.
        """
        if rail_type == RailType.OUTPUT and context:
            text_to_classify = f"User: {context}\nAssistant: {content}"
        elif rail_type == RailType.OUTPUT:
            text_to_classify = f"Assistant: {content}"
        else:
            text_to_classify = content

        prompt = f"""<|im_start|>system
You are a content safety classifier. Classify the following content as:
- Safe: Content is appropriate and harmless
- Controversial: Content may be sensitive but not clearly harmful
- Unsafe: Content is harmful, dangerous, or violates policies

Respond with only the classification and a brief reason.<|im_end|>
<|im_start|>user
Classify this content:

{text_to_classify}<|im_end|>
<|im_start|>assistant
"""
        return prompt

    def _parse_response(
        self,
        response: str,
        content: str,
        rail_type: RailType,
    ) -> List[GuardrailResult]:
        """
        Parse Qwen3Guard response.

        Expected format:
            Unsafe: This content contains instructions for creating weapons.
        or:
            Safe: This is a normal conversation.
        """
        response_lower = response.lower().strip()

        # Determine classification
        if response_lower.startswith("unsafe"):
            classification = "unsafe"
            score = 1.0
            passed = False
        elif response_lower.startswith("controversial"):
            classification = "controversial"
            score = 0.5
            passed = False  # Treat controversial as failed for safety
        else:
            classification = "safe"
            score = 0.0
            passed = True

        if passed:
            return [
                GuardrailResult(
                    passed=True,
                    category="safe",
                    score=score,
                    model=self.model_name,
                    reason="Content passed Qwen3Guard safety check",
                    action="pass",
                    latency_ms=0.0,
                )
            ]

        # Infer categories from content and response
        categories = self._infer_categories(content, response)

        results = []
        for category in categories:
            results.append(
                GuardrailResult(
                    passed=False,
                    category=category,
                    score=score,
                    model=self.model_name,
                    reason=f"Qwen3Guard: {classification}",
                    action="block" if classification == "unsafe" else "flag",
                    latency_ms=0.0,
                )
            )

        if not results:
            results.append(
                GuardrailResult(
                    passed=False,
                    category="harmful_content",
                    score=score,
                    model=self.model_name,
                    reason=f"Qwen3Guard: {classification}",
                    action="block" if classification == "unsafe" else "flag",
                    latency_ms=0.0,
                )
            )

        return results

    def _infer_categories(self, content: str, response: str) -> List[str]:
        """Infer safety categories from content and response."""
        categories = []
        combined = (content + " " + response).lower()

        # Check for various categories
        category_keywords = {
            "violence": ["violence", "violent", "kill", "murder", "attack", "weapon", "harm"],
            "self_harm": ["suicide", "self-harm", "self harm", "cut myself", "end my life"],
            "hate_speech": ["hate", "racist", "sexist", "discrimination", "slur"],
            "sexual_content": ["sexual", "explicit", "nude", "porn"],
            "illegal_activity": ["illegal", "drug", "hack", "steal", "fraud", "crime"],
            "jailbreak": ["jailbreak", "ignore", "bypass", "pretend"],
            "prompt_injection": ["injection", "override", "system prompt"],
        }

        for category, keywords in category_keywords.items():
            if any(kw in combined for kw in keywords):
                categories.append(category)

        return categories if categories else ["harmful_content"]
