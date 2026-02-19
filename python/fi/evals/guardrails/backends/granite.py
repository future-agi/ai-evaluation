"""
Granite Guardian Backend for Guardrails.

Uses IBM's Granite Guardian model for content safety classification.
Models: ibm-granite/granite-guardian-3.3-8b, ibm-granite/granite-guardian-3.2-5b

Features:
- Multiple risk dimensions
- Probability scores
- Reasoning traces (with think mode)
"""

import re
from typing import Any, Dict, List, Optional

from fi.evals.guardrails.backends.local_base import LocalModelBackend
from fi.evals.guardrails.config import GuardrailModel, RailType
from fi.evals.guardrails.types import GuardrailResult


# Granite Guardian risk dimensions to our categories
GRANITE_RISK_MAP = {
    "harm": "harmful_content",
    "social_bias": "hate_speech",
    "profanity": "toxicity",
    "sexual_content": "sexual_content",
    "violence": "violence",
    "jailbreak": "jailbreak",
    "unethical_behavior": "illegal_activity",
    "groundedness": "hallucination",
    "relevance": "off_topic",
}


class GraniteGuardianBackend(LocalModelBackend):
    """
    Granite Guardian safety classification backend.

    IBM's Granite Guardian provides multi-dimensional risk assessment
    with probability scores and optional reasoning traces.

    Features:
    - Multiple risk dimensions (harm, bias, profanity, etc.)
    - Probability scores for each dimension
    - Optional "think mode" for reasoning traces
    - Available in 8B and 5B variants

    Usage:
        backend = GraniteGuardianBackend(
            model=GuardrailModel.GRANITE_GUARDIAN_8B,
            vllm_url="http://localhost:28000",
        )
        results = backend.classify("How to make a bomb?", RailType.INPUT)
    """

    MODEL_VARIANTS = {
        "granite-guardian-3.3-8b": "ibm-granite/granite-guardian-3.3-8b",
        "granite-guardian-3.2-5b": "ibm-granite/granite-guardian-3.2-5b",
    }

    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.1

    # Risk dimensions to evaluate
    RISK_DIMENSIONS = [
        "harm",
        "social_bias",
        "profanity",
        "sexual_content",
        "violence",
        "jailbreak",
        "unethical_behavior",
    ]

    def __init__(
        self,
        model: GuardrailModel,
        vllm_url: Optional[str] = None,
        device: str = "auto",
        hf_token: Optional[str] = None,
        use_think_mode: bool = False,
        **kwargs,
    ):
        """
        Initialize Granite Guardian backend.

        Args:
            model: The guardrail model enum value
            vllm_url: VLLM server URL
            device: Device to use
            hf_token: HuggingFace token
            use_think_mode: Enable reasoning traces
        """
        super().__init__(model, vllm_url, device, hf_token, **kwargs)
        self._use_think_mode = use_think_mode

    def _get_hf_model_name(self) -> str:
        """Get the HuggingFace model name based on variant."""
        model_value = self.model.value
        return self.MODEL_VARIANTS.get(model_value, self.MODEL_VARIANTS["granite-guardian-3.3-8b"])

    def _format_prompt(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
    ) -> str:
        """
        Format content into Granite Guardian prompt.

        Uses chat format with guardian_config for risk dimensions.
        """
        # Build the conversation
        if rail_type == RailType.OUTPUT and context:
            messages = f"User: {context}\nAssistant: {content}"
        elif rail_type == RailType.OUTPUT:
            messages = f"Assistant: {content}"
        else:
            messages = f"User: {content}"

        # Build risk config
        risk_config = ", ".join(self.RISK_DIMENSIONS)

        if self._use_think_mode:
            think_instruction = "\nProvide step-by-step reasoning before your final answer."
        else:
            think_instruction = ""

        prompt = f"""<|start_of_role|>system<|end_of_role|>You are a safety guardian. Evaluate the following content for potential risks.

Risk dimensions to evaluate: {risk_config}

For each dimension, provide:
1. A yes/no assessment
2. A probability score (0.0 to 1.0)
{think_instruction}

Format your response as:
DIMENSION: yes/no (probability)
...<|end_of_text|>
<|start_of_role|>user<|end_of_role|>{messages}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>"""

        return prompt

    def _parse_response(
        self,
        response: str,
        content: str,
        rail_type: RailType,
    ) -> List[GuardrailResult]:
        """
        Parse Granite Guardian response.

        Expected format:
            harm: yes (0.85)
            social_bias: no (0.12)
            ...
        """
        results = []
        response_lower = response.lower()

        # Parse each dimension
        for dimension in self.RISK_DIMENSIONS:
            # Look for pattern: dimension: yes/no (probability)
            pattern = rf"{dimension}:\s*(yes|no)\s*\(?([\d.]+)?\)?"
            match = re.search(pattern, response_lower)

            if match:
                is_risky = match.group(1) == "yes"
                prob_str = match.group(2)
                probability = float(prob_str) if prob_str else (1.0 if is_risky else 0.0)

                our_category = GRANITE_RISK_MAP.get(dimension, "harmful_content")

                if is_risky:
                    results.append(
                        GuardrailResult(
                            passed=False,
                            category=our_category,
                            score=probability,
                            model=self.model_name,
                            reason=f"Granite Guardian: {dimension} ({probability:.2f})",
                            action="block",
                            latency_ms=0.0,
                        )
                    )

        # Check for simple yes/no responses
        if not results:
            if "yes" in response_lower[:50]:
                results.append(
                    GuardrailResult(
                        passed=False,
                        category="harmful_content",
                        score=1.0,
                        model=self.model_name,
                        reason="Granite Guardian classified as risky",
                        action="block",
                        latency_ms=0.0,
                    )
                )
            else:
                results.append(
                    GuardrailResult(
                        passed=True,
                        category="safe",
                        score=0.0,
                        model=self.model_name,
                        reason="Content passed Granite Guardian safety check",
                        action="pass",
                        latency_ms=0.0,
                    )
                )

        return results
