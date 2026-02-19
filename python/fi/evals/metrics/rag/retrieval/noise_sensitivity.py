"""
Noise Sensitivity Metric.

Measures RAG system robustness to irrelevant/noisy context
mixed with relevant context.
"""

from typing import Any, Dict, List, Optional

from ...base_metric import BaseMetric
from ..types import NoiseSensitivityInput
from ..utils import (
    split_into_sentences,
    extract_claims,
    check_claim_supported,
    compute_semantic_similarity,
)


class NoiseSensitivity(BaseMetric[NoiseSensitivityInput]):
    """
    Measures RAG system robustness to irrelevant context.

    Compares response quality when noise is added to the context.
    Lower sensitivity (higher score) means more robust system.

    Requires both clean and noisy response variants to compare.

    Approach:
    1. Calculate faithfulness of clean response (relevant contexts only)
    2. Calculate faithfulness of noisy response (relevant + irrelevant contexts)
    3. Sensitivity = degradation when noise added
    4. Robustness = 1 - sensitivity (returned as output)

    Score: 0.0 (very sensitive to noise) to 1.0 (robust to noise)

    Example:
        >>> noise_sens = NoiseSensitivity()
        >>> result = noise_sens.evaluate([{
        ...     "query": "What is the capital of France?",
        ...     "response_clean": "The capital of France is Paris.",
        ...     "response_noisy": "The capital of France is Paris, a beautiful city.",
        ...     "relevant_contexts": ["Paris is the capital of France."],
        ...     "irrelevant_contexts": ["The Eiffel Tower is 330 meters tall."],
        ... }])
    """

    @property
    def metric_name(self) -> str:
        return "noise_sensitivity"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.faithfulness_threshold = self.config.get("faithfulness_threshold", 0.5)

    def compute_one(self, inputs: NoiseSensitivityInput) -> Dict[str, Any]:
        # Validate inputs
        if not inputs.relevant_contexts:
            return {
                "output": 0.0,
                "reason": "No relevant contexts provided",
            }

        # Calculate faithfulness of clean response
        clean_faithfulness = self._calculate_faithfulness(
            inputs.response_clean, inputs.relevant_contexts
        )

        # Calculate faithfulness of noisy response
        all_contexts = inputs.relevant_contexts + inputs.irrelevant_contexts
        noisy_faithfulness = self._calculate_faithfulness(
            inputs.response_noisy, all_contexts
        )

        # Calculate semantic similarity between responses
        response_similarity = compute_semantic_similarity(
            inputs.response_clean, inputs.response_noisy
        )

        # Calculate sensitivity metrics
        # Sensitivity = degradation when noise added
        faithfulness_degradation = max(0, clean_faithfulness - noisy_faithfulness)

        # Also check if noisy response introduced irrelevant info
        irrelevant_contamination = self._check_contamination(
            inputs.response_noisy,
            inputs.irrelevant_contexts,
            inputs.relevant_contexts
        )

        # Combined sensitivity score
        sensitivity = (
            0.5 * faithfulness_degradation +
            0.3 * irrelevant_contamination +
            0.2 * (1.0 - response_similarity)
        )

        # Convert to robustness score (1 - sensitivity)
        robustness = max(0.0, 1.0 - sensitivity)

        return {
            "output": round(robustness, 4),
            "reason": f"Robustness: {robustness:.2f} (clean={clean_faithfulness:.2f}, noisy={noisy_faithfulness:.2f})",
            "clean_faithfulness": round(clean_faithfulness, 4),
            "noisy_faithfulness": round(noisy_faithfulness, 4),
            "faithfulness_degradation": round(faithfulness_degradation, 4),
            "irrelevant_contamination": round(irrelevant_contamination, 4),
            "response_similarity": round(response_similarity, 4),
            "sensitivity": round(sensitivity, 4),
        }

    def _calculate_faithfulness(
        self, response: str, contexts: List[str]
    ) -> float:
        """Calculate faithfulness score for a response given contexts."""
        claims = extract_claims(response)

        if not claims:
            return 1.0  # No claims to verify

        supported = 0
        for claim in claims:
            is_supported, score, _ = check_claim_supported(
                claim, contexts, self.faithfulness_threshold
            )
            if is_supported:
                supported += 1

        return supported / len(claims)

    def _check_contamination(
        self,
        response: str,
        irrelevant_contexts: List[str],
        relevant_contexts: List[str]
    ) -> float:
        """
        Check if response contains information from irrelevant contexts
        that is NOT in relevant contexts.
        """
        if not irrelevant_contexts:
            return 0.0

        claims = extract_claims(response)
        if not claims:
            return 0.0

        contaminated_claims = 0

        for claim in claims:
            # Check if claim is supported by irrelevant contexts
            irrel_supported, irrel_score, _ = check_claim_supported(
                claim, irrelevant_contexts, 0.4
            )

            # But NOT supported by relevant contexts
            rel_supported, rel_score, _ = check_claim_supported(
                claim, relevant_contexts, 0.4
            )

            if irrel_supported and not rel_supported:
                contaminated_claims += 1

        return contaminated_claims / len(claims)
