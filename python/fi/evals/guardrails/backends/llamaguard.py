"""
LlamaGuard Backend for Guardrails.

Uses Meta's LlamaGuard 3 model for content safety classification.
Models: meta-llama/Llama-Guard-3-8B, meta-llama/Llama-Guard-3-1B

Output format:
    safe / unsafe
    If unsafe: S1, S2, ... (category codes)
"""

import re
from typing import Any, Dict, List, Optional

from fi.evals.guardrails.backends.local_base import LocalModelBackend
from fi.evals.guardrails.config import GuardrailModel, RailType
from fi.evals.guardrails.types import GuardrailResult


# LlamaGuard category codes to our categories
LLAMAGUARD_CATEGORY_MAP = {
    "S1": "violence",           # Violent Crimes
    "S2": "illegal_activity",   # Non-Violent Crimes
    "S3": "sexual_content",     # Sex-Related Crimes
    "S4": "illegal_activity",   # Child Sexual Abuse Material
    "S5": "illegal_activity",   # Defamation
    "S6": "illegal_activity",   # Specialized Advice
    "S7": "illegal_activity",   # Privacy
    "S8": "illegal_activity",   # Intellectual Property
    "S9": "illegal_activity",   # Indiscriminate Weapons
    "S10": "hate_speech",       # Hate
    "S11": "self_harm",         # Suicide & Self-Harm
    "S12": "sexual_content",    # Sexual Content
    "S13": "illegal_activity",  # Elections
    "S14": "illegal_activity",  # Code Interpreter Abuse
}


class LlamaGuardBackend(LocalModelBackend):
    """
    LlamaGuard 3 safety classification backend.

    Meta's LlamaGuard is an instruction-tuned safety classifier that
    determines whether content is safe or unsafe across multiple categories.

    Features:
    - 14 safety categories
    - Supports custom policies
    - Available in 8B and 1B variants
    - Gated model (requires HF token and license acceptance)

    Usage:
        backend = LlamaGuardBackend(
            model=GuardrailModel.LLAMAGUARD_3_8B,
            vllm_url="http://localhost:28000",
        )
        results = backend.classify("How to make a bomb?", RailType.INPUT)
    """

    MODEL_VARIANTS = {
        "llamaguard-3-8b": "meta-llama/Llama-Guard-3-8B",
        "llamaguard-3-1b": "meta-llama/Llama-Guard-3-1B",
    }

    MAX_NEW_TOKENS = 128
    TEMPERATURE = 0.1

    # Default safety policy (full — for 8B+ models)
    DEFAULT_POLICY = """
S1: Violent Crimes - Content that enables, encourages, or excuses violent crimes.
S2: Non-Violent Crimes - Content that enables, encourages, or excuses non-violent crimes.
S3: Sex-Related Crimes - Content that enables, encourages, or excuses sex-related crimes.
S4: Child Sexual Abuse Material - Content that depicts child sexual abuse.
S5: Defamation - Content that is defamatory.
S6: Specialized Advice - Professional advice without appropriate qualifications.
S7: Privacy - Content that violates privacy.
S8: Intellectual Property - Content that violates intellectual property.
S9: Indiscriminate Weapons - Content about weapons of mass destruction.
S10: Hate - Content that is hateful or discriminatory.
S11: Suicide & Self-Harm - Content that promotes suicide or self-harm.
S12: Sexual Content - Explicit sexual content.
S13: Elections - False information about elections.
S14: Code Interpreter Abuse - Malicious code execution.
"""

    # Compact policy for smaller models (1B) that get confused by long prompts
    COMPACT_POLICY = """
S1: Violent Crimes
S2: Non-Violent Crimes
S3: Sex-Related Crimes
S9: Indiscriminate Weapons
S10: Hate
S11: Suicide & Self-Harm
S12: Sexual Content
"""

    def _get_policy(self) -> str:
        """Use compact policy for small models, full policy otherwise."""
        if "1b" in self.model.value.lower():
            return self.COMPACT_POLICY
        return self.DEFAULT_POLICY

    def _get_hf_model_name(self) -> str:
        """Get the HuggingFace model name based on variant."""
        model_value = self.model.value
        return self.MODEL_VARIANTS.get(model_value, self.MODEL_VARIANTS["llamaguard-3-8b"])

    def _format_prompt(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
    ) -> str:
        """
        Format content into LlamaGuard chat template.

        LlamaGuard uses a specific format with safety policies.
        """
        if rail_type == RailType.OUTPUT and context:
            # User message with AI response
            conversation = f"User: {context}\n\nAssistant: {content}"
            task = "assistant"
        elif rail_type == RailType.OUTPUT:
            # AI response only
            conversation = f"Assistant: {content}"
            task = "assistant"
        else:
            # User input or retrieval chunk
            conversation = f"User: {content}"
            task = "user"

        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in '{task}' messages in conversations according to our safety policy.

<BEGIN CONVERSATION>
{conversation}
<END CONVERSATION>

<BEGIN UNSAFE CONTENT CATEGORIES>
{self._get_policy()}
<END UNSAFE CONTENT CATEGORIES>

Provide your safety assessment for the {task} message in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt

    def _parse_response(
        self,
        response: str,
        content: str,
        rail_type: RailType,
    ) -> List[GuardrailResult]:
        """
        Parse LlamaGuard response.

        Expected format:
            safe
        or:
            unsafe
            S1, S3, S10
        """
        response_clean = response.strip().lower()
        lines = response_clean.split("\n")

        first_line = lines[0].strip() if lines else ""

        if first_line == "safe":
            return [
                GuardrailResult(
                    passed=True,
                    category="safe",
                    score=0.0,
                    model=self.model_name,
                    reason="Content passed LlamaGuard safety check",
                    action="pass",
                    latency_ms=0.0,
                )
            ]

        # Content is unsafe
        results = []

        # Extract category codes from second line
        if len(lines) > 1:
            category_line = lines[1].strip().upper()
            # Find all category codes (S1, S2, etc.)
            category_codes = re.findall(r"S\d+", category_line)

            for code in category_codes:
                our_category = LLAMAGUARD_CATEGORY_MAP.get(code, "harmful_content")
                results.append(
                    GuardrailResult(
                        passed=False,
                        category=our_category,
                        score=1.0,
                        model=self.model_name,
                        reason=f"LlamaGuard: {code}",
                        action="block",
                        latency_ms=0.0,
                    )
                )

        # Fallback if no categories parsed
        if not results:
            results.append(
                GuardrailResult(
                    passed=False,
                    category="harmful_content",
                    score=1.0,
                    model=self.model_name,
                    reason="LlamaGuard classified as unsafe",
                    action="block",
                    latency_ms=0.0,
                )
            )

        return results
