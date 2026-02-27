"""
Generic LLM Guard Backend — use any chat model as a safety classifier.

Prompts the model with a system message asking for safe/unsafe classification.
Works with any LLM that follows instructions (llama3.2, qwen3, mistral, etc.).
"""

import re
from typing import Any, Dict, List, Optional

from fi.evals.guardrails.backends.local_base import LocalModelBackend
from fi.evals.guardrails.config import GuardrailModel, RailType
from fi.evals.guardrails.types import GuardrailResult


SAFETY_SYSTEM_PROMPT = (
    "You are a content safety classifier. For every user message, respond with "
    "ONLY one word on the first line: 'safe' or 'unsafe'. "
    "If unsafe, add a second line listing categories from: "
    "violence, self_harm, hate_speech, sexual_content, harassment, "
    "illegal_activity, jailbreak, prompt_injection. "
    "Do not explain or add anything else."
)


class GenericLLMGuardBackend(LocalModelBackend):
    """
    Use any chat LLM as a safety classifier via prompting.

    Unlike dedicated guard models (LlamaGuard, Qwen3Guard), this backend
    prompts a general-purpose LLM to classify content as safe/unsafe.
    Less accurate than purpose-built models but works with any LLM.

    Usage:
        backend = GenericLLMGuardBackend(
            model=GuardrailModel.LLAMA_3_2_3B,
            vllm_url="http://localhost:11434",
        )
        results = backend.classify("How to make a bomb?", RailType.INPUT)
    """

    HF_MODEL_NAME = ""  # Not used — relies on VLLM/ollama model resolution
    MAX_NEW_TOKENS = 64
    TEMPERATURE = 0.1

    def classify(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[GuardrailResult]:
        """Classify using chat endpoint with a safety system prompt."""
        import time

        start_time = time.time()

        if not content or not content.strip():
            return [
                GuardrailResult(
                    passed=True,
                    category="empty",
                    score=0.0,
                    model=self.model_name,
                    reason="Empty content",
                    action="pass",
                    latency_ms=0.0,
                )
            ]

        try:
            messages = [
                {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]

            if self._use_vllm and self._vllm_client:
                response = self._vllm_client.chat(
                    messages=messages,
                    max_tokens=self.MAX_NEW_TOKENS,
                    temperature=self.TEMPERATURE,
                )
                response_text = response.text
            else:
                # Fallback: format as a single prompt for transformers
                prompt = self._format_prompt(content, rail_type, context)
                response_text = self._generate_with_transformers(prompt)

            elapsed_ms = (time.time() - start_time) * 1000
            results = self._parse_response(response_text, content, rail_type)
            for r in results:
                r.latency_ms = elapsed_ms
            return results

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return [
                GuardrailResult(
                    passed=False,
                    category="error",
                    score=0.0,
                    model=self.model_name,
                    reason=f"Generic LLM guard error: {e}",
                    action="block",
                    latency_ms=elapsed_ms,
                )
            ]

    def _format_prompt(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
    ) -> str:
        """Format for transformers fallback (not used with VLLM/ollama)."""
        return f"{SAFETY_SYSTEM_PROMPT}\n\nUser message: {content}\n\nClassification:"

    def _parse_response(
        self,
        response: str,
        content: str,
        rail_type: RailType,
    ) -> List[GuardrailResult]:
        """Parse safe/unsafe response from a general LLM."""
        response_lower = response.strip().lower()
        lines = response_lower.split("\n")
        first_line = lines[0].strip()

        if "unsafe" in first_line:
            # Try to extract categories from second line
            categories = []
            if len(lines) > 1:
                cat_line = lines[1].strip()
                for cat in self._CATEGORY_KEYWORDS:
                    if cat in cat_line:
                        categories.append(cat)

            if not categories:
                categories = self._infer_categories(content, response_lower)

            return [
                GuardrailResult(
                    passed=False,
                    category=cat,
                    score=1.0,
                    model=self.model_name,
                    reason=f"LLM classified as unsafe",
                    action="block",
                    latency_ms=0.0,
                )
                for cat in categories
            ]

        return [
            GuardrailResult(
                passed=True,
                category="safe",
                score=0.0,
                model=self.model_name,
                reason="LLM classified as safe",
                action="pass",
                latency_ms=0.0,
            )
        ]
