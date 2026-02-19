"""
Multi-Hop Reasoning Metric.

Evaluates whether the response correctly combines information
from multiple contexts to answer complex queries.
"""

import re
from typing import Any, Dict, List, Optional, Set

from ...base_metric import BaseMetric
from ..types import RAGMultiHopInput
from ..utils import (
    extract_entities,
    extract_key_phrases,
    extract_keywords,
    compute_semantic_similarity,
    compute_word_overlap,
    split_into_sentences,
)


class MultiHopReasoning(BaseMetric[RAGMultiHopInput]):
    """
    Evaluates multi-hop reasoning in RAG responses.

    Checks whether the response correctly synthesizes information
    from multiple context passages to answer complex queries.

    Multi-hop queries require:
    1. Finding relevant info in context A
    2. Using that to find related info in context B
    3. Combining both to form the answer

    Example:
        Query: "What award did the director of Inception win?"
        - Context A: "Inception was directed by Christopher Nolan"
        - Context B: "Christopher Nolan won the Academy Award"
        - Requires connecting both pieces of information

    Evaluation approach:
    1. Identify information units in each context
    2. Check which contexts are utilized in response
    3. Verify reasoning chain if provided
    4. Check for synthesis indicators

    Score: 0.0 (no multi-hop) to 1.0 (excellent multi-hop reasoning)
    """

    @property
    def metric_name(self) -> str:
        return "multi_hop_reasoning"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_contexts_for_bonus = self.config.get("min_contexts_for_bonus", 2)
        self.synthesis_weight = self.config.get("synthesis_weight", 0.3)
        self.coverage_weight = self.config.get("coverage_weight", 0.4)
        self.chain_weight = self.config.get("chain_weight", 0.3)

    def compute_one(self, inputs: RAGMultiHopInput) -> Dict[str, Any]:
        response = inputs.response
        contexts = inputs.contexts
        hop_chain = inputs.hop_chain

        if not response or not response.strip():
            return {"output": 0.0, "reason": "Empty response"}

        if not contexts:
            return {"output": 0.0, "reason": "No contexts provided"}

        # 1. Extract information units from each context
        context_info = []
        for i, ctx in enumerate(contexts):
            info = self._extract_information_units(ctx)
            context_info.append({
                "context_id": i,
                "entities": info["entities"],
                "phrases": info["phrases"],
                "keywords": info["keywords"],
            })

        # 2. Check which contexts are utilized in response
        contexts_used = []
        context_usage_scores = []

        for ctx_info in context_info:
            usage_score = self._check_context_usage(response, ctx_info)
            context_usage_scores.append(usage_score)
            if usage_score > 0.2:
                contexts_used.append(ctx_info["context_id"])

        # 3. Calculate multi-context coverage score
        coverage_score = self._calculate_coverage_score(
            len(contexts_used), len(contexts)
        )

        # 4. Check for synthesis indicators
        synthesis_score = self._check_synthesis_indicators(response)

        # 5. Verify reasoning chain if provided
        chain_score = 1.0
        chain_verified = None
        if hop_chain:
            chain_score = self._verify_reasoning_chain(response, hop_chain)
            chain_verified = chain_score > 0.7

        # 6. Check for cross-context connections
        connection_score = self._check_cross_context_connections(
            response, context_info
        )

        # Calculate final score
        final_score = (
            self.coverage_weight * coverage_score +
            self.synthesis_weight * synthesis_score +
            self.chain_weight * chain_score
        )

        # Bonus for actually using multiple contexts
        if len(contexts_used) >= self.min_contexts_for_bonus:
            final_score = min(1.0, final_score * 1.1)

        return {
            "output": round(final_score, 4),
            "reason": f"Used {len(contexts_used)}/{len(contexts)} contexts, synthesis={synthesis_score:.2f}",
            "contexts_utilized": contexts_used,
            "context_usage_scores": [round(s, 3) for s in context_usage_scores],
            "coverage_score": round(coverage_score, 4),
            "synthesis_score": round(synthesis_score, 4),
            "chain_score": round(chain_score, 4) if hop_chain else None,
            "chain_verified": chain_verified,
            "connection_score": round(connection_score, 4),
        }

    def _extract_information_units(self, text: str) -> Dict[str, Set[str]]:
        """Extract atomic information units from text."""
        entities = extract_entities(text)
        phrases = set(extract_key_phrases(text))
        keywords = extract_keywords(text)

        return {
            "entities": entities,
            "phrases": phrases,
            "keywords": keywords,
        }

    def _check_context_usage(
        self, response: str, ctx_info: Dict[str, Set[str]]
    ) -> float:
        """Check how much of context's information appears in response."""
        response_lower = response.lower()

        # Check entities
        entity_score = 0.0
        if ctx_info["entities"]:
            matched = sum(
                1 for e in ctx_info["entities"]
                if e.lower() in response_lower
            )
            entity_score = matched / len(ctx_info["entities"])

        # Check keywords
        keyword_score = 0.0
        if ctx_info["keywords"]:
            matched = sum(
                1 for k in ctx_info["keywords"]
                if k.lower() in response_lower
            )
            keyword_score = matched / len(ctx_info["keywords"])

        return 0.6 * entity_score + 0.4 * keyword_score

    def _calculate_coverage_score(
        self, contexts_used: int, total_contexts: int
    ) -> float:
        """Calculate score based on how many contexts were used."""
        if total_contexts == 0:
            return 0.0

        if total_contexts == 1:
            return 1.0 if contexts_used == 1 else 0.0

        # For multi-hop, using multiple contexts is good
        # But using ALL contexts might indicate noise sensitivity issues
        ideal_usage = min(total_contexts, 3)  # Ideal is 2-3 contexts
        if contexts_used >= ideal_usage:
            return 1.0
        else:
            return contexts_used / ideal_usage

    def _check_synthesis_indicators(self, response: str) -> float:
        """Check for linguistic indicators of information synthesis."""
        response_lower = response.lower()

        synthesis_patterns = [
            # Causal connectors
            (r"\b(therefore|thus|hence|consequently|as a result)\b", 0.3),
            (r"\b(because|since|due to|owing to)\b", 0.2),
            # Additive connectors
            (r"\b(additionally|furthermore|moreover|also)\b", 0.2),
            (r"\b(and|as well as|along with)\b", 0.1),
            # Comparative/contrast
            (r"\b(however|although|while|whereas|but)\b", 0.2),
            # Conclusion indicators
            (r"\b(in conclusion|to summarize|overall|in summary)\b", 0.3),
            # Reference to multiple sources
            (r"\b(both|together|combined|combining)\b", 0.3),
            (r"\b(according to|based on)\b", 0.2),
            # Reasoning indicators
            (r"\b(which means|this indicates|this shows|this suggests)\b", 0.3),
            (r"\b(we can see|we can conclude|it follows)\b", 0.3),
        ]

        total_score = 0.0
        for pattern, weight in synthesis_patterns:
            if re.search(pattern, response_lower):
                total_score += weight

        return min(1.0, total_score)

    def _verify_reasoning_chain(
        self, response: str, chain: List[str]
    ) -> float:
        """Verify that the response follows the expected reasoning chain."""
        if not chain:
            return 1.0

        response_lower = response.lower()
        steps_found = 0

        for step in chain:
            step_keywords = extract_keywords(step)

            # Check if step keywords appear in response
            if step_keywords:
                matched = sum(
                    1 for kw in step_keywords
                    if kw.lower() in response_lower
                )
                coverage = matched / len(step_keywords)
                if coverage >= 0.5:
                    steps_found += 1

        return steps_found / len(chain)

    def _check_cross_context_connections(
        self, response: str, context_info: List[Dict[str, Set[str]]]
    ) -> float:
        """
        Check if response makes connections between contexts.

        Look for entities/concepts from one context being mentioned
        in relation to entities/concepts from another context.
        """
        if len(context_info) < 2:
            return 1.0  # No cross-context needed

        response_sentences = split_into_sentences(response)

        connection_score = 0.0
        total_pairs = 0

        # Check pairs of contexts
        for i in range(len(context_info)):
            for j in range(i + 1, len(context_info)):
                total_pairs += 1

                ctx_i_entities = context_info[i]["entities"]
                ctx_j_entities = context_info[j]["entities"]

                # Check if any sentence mentions entities from both contexts
                for sent in response_sentences:
                    sent_lower = sent.lower()
                    has_i = any(e.lower() in sent_lower for e in ctx_i_entities)
                    has_j = any(e.lower() in sent_lower for e in ctx_j_entities)

                    if has_i and has_j:
                        connection_score += 1.0
                        break

        if total_pairs == 0:
            return 1.0

        return min(1.0, connection_score / total_pairs)
