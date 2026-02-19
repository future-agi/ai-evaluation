"""
Context Entity Recall Metric.

Measures entity-level retrieval coverage - what fraction of
entities from the reference appear in retrieved contexts.
"""

from typing import Any, Dict, List, Optional, Set

from ...base_metric import BaseMetric
from ..types import RAGRetrievalInput
from ..utils import extract_entities, entities_match


class ContextEntityRecall(BaseMetric[RAGRetrievalInput]):
    """
    Measures entity-level retrieval completeness.

    Uses NER to extract entities from reference and contexts,
    then calculates what fraction of reference entities appear
    in the retrieved contexts.

    Critical for fact-based use cases where specific entities
    (names, dates, locations, amounts) must not be missed.

    Formula:
        Context Entity Recall = |Entities in Context ∩ Entities in Reference| / |Entities in Reference|

    Score: 0.0 (no entities recalled) to 1.0 (all entities recalled)

    Example:
        >>> entity_recall = ContextEntityRecall()
        >>> result = entity_recall.evaluate([{
        ...     "query": "When did Einstein win the Nobel Prize?",
        ...     "contexts": ["Albert Einstein received the Nobel Prize in Physics in 1921."],
        ...     "reference": "Albert Einstein won the Nobel Prize in 1921."
        ... }])
    """

    @property
    def metric_name(self) -> str:
        return "context_entity_recall"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.entity_types = self.config.get("entity_types", None)  # None = all types
        self.fuzzy_match = self.config.get("fuzzy_match", True)
        self.case_sensitive = self.config.get("case_sensitive", False)

    def compute_one(self, inputs: RAGRetrievalInput) -> Dict[str, Any]:
        # Handle empty inputs
        if not inputs.reference or not inputs.reference.strip():
            return {
                "output": 1.0,
                "reason": "No reference provided - entity recall is trivially 1.0",
            }

        # Extract entities from reference
        reference_entities = extract_entities(
            inputs.reference,
            entity_types=self.entity_types,
        )

        if not reference_entities:
            return {
                "output": 1.0,
                "reason": "No entities found in reference",
                "reference_entities": [],
            }

        if not inputs.contexts:
            return {
                "output": 0.0,
                "reason": "No contexts provided - cannot recall any entities",
                "reference_entities": list(reference_entities),
                "recalled_entities": [],
                "missing_entities": list(reference_entities),
            }

        # Extract entities from all contexts
        context_entities: Set[str] = set()
        for ctx in inputs.contexts:
            ctx_ents = extract_entities(ctx, entity_types=self.entity_types)
            context_entities.update(ctx_ents)

        # Calculate entity overlap
        recalled = set()
        missing = set()

        for ref_entity in reference_entities:
            found = False

            if self.fuzzy_match:
                # Check for fuzzy matches
                for ctx_entity in context_entities:
                    if entities_match(ref_entity, ctx_entity):
                        found = True
                        break
            else:
                # Exact match (case-insensitive by default)
                if self.case_sensitive:
                    found = ref_entity in context_entities
                else:
                    found = ref_entity.lower() in {e.lower() for e in context_entities}

            if found:
                recalled.add(ref_entity)
            else:
                missing.add(ref_entity)

        # Calculate recall
        recall = len(recalled) / len(reference_entities)

        return {
            "output": round(recall, 4),
            "reason": f"{len(recalled)}/{len(reference_entities)} entities recalled",
            "total_reference_entities": len(reference_entities),
            "recalled_count": len(recalled),
            "missing_count": len(missing),
            "reference_entities": list(reference_entities),
            "recalled_entities": list(recalled),
            "missing_entities": list(missing),
            "context_entities": list(context_entities)[:20],  # Limit for readability
        }
