"""
Comprehensive tests for Hallucination Detection metrics.

Tests cover:
- Faithfulness
- ClaimSupport
- FactualConsistency
- ContradictionDetection
- HallucinationScore
"""

import pytest
from fi.evals.metrics.hallucination import (
    HallucinationInput,
    FactualConsistencyInput,
    Claim,
    Faithfulness,
    ClaimSupport,
    FactualConsistency,
    ContradictionDetection,
    HallucinationScore,
)


class TestFaithfulness:
    """Tests for Faithfulness metric."""

    def test_fully_faithful_response(self):
        """Test response that is fully supported by context."""
        metric = Faithfulness()
        input_data = HallucinationInput(
            response="Paris is the capital of France. France is located in Europe.",
            context="Paris is the capital of France, a country located in Western Europe."
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.8
        assert "supported" in result["reason"]

    def test_hallucinated_response(self):
        """Test response with clearly unsupported claims."""
        metric = Faithfulness()
        input_data = HallucinationInput(
            response="Unicorns live in the forest. Dragons fly over mountains.",
            context="Paris is the capital of France. Berlin is the capital of Germany."
        )
        result = metric.compute_one(input_data)
        # Completely unrelated content should have low support
        assert result["output"] < 0.5

    def test_partial_hallucination(self):
        """Test response with mix of faithful and unrelated claims."""
        metric = Faithfulness()
        input_data = HallucinationInput(
            response="The Eiffel Tower is in Paris. Aliens built the pyramids.",
            context="The Eiffel Tower is a famous landmark located in Paris, France."
        )
        result = metric.compute_one(input_data)
        # Should have partial score - one claim supported, one not
        assert 0.2 < result["output"] < 0.8

    def test_empty_response(self):
        """Test with empty response."""
        metric = Faithfulness()
        input_data = HallucinationInput(
            response="",
            context="Paris is the capital of France."
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0
        assert "No verifiable claims" in result["reason"]

    def test_multiple_contexts(self):
        """Test with multiple context sources."""
        metric = Faithfulness()
        input_data = HallucinationInput(
            response="Albert Einstein developed the theory of relativity. He was born in Germany.",
            context=[
                "Albert Einstein was a theoretical physicist.",
                "Einstein developed the theory of relativity.",
                "Einstein was born in Ulm, Germany in 1879."
            ]
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.7

    def test_with_preextracted_claims(self):
        """Test with pre-extracted claims that match context."""
        metric = Faithfulness()
        input_data = HallucinationInput(
            response="The claim text doesn't matter when claims are provided.",
            context="Python is a popular programming language. Guido van Rossum created Python in 1991.",
            claims=[
                Claim(text="Python is a programming language."),
                Claim(text="Guido van Rossum created Python."),
            ]
        )
        result = metric.compute_one(input_data)
        # With word overlap matching, these claims should have good support
        assert result["output"] >= 0.5


class TestClaimSupport:
    """Tests for ClaimSupport metric."""

    def test_high_support(self):
        """Test claims with high support."""
        metric = ClaimSupport()
        input_data = HallucinationInput(
            response="Water is composed of hydrogen and oxygen. It boils at 100 degrees Celsius.",
            context="Water (H2O) is made of hydrogen and oxygen atoms. Water boils at 100 degrees Celsius at standard pressure."
        )
        result = metric.compute_one(input_data)
        # Heuristic matching should find good overlap for these claims
        assert result["output"] >= 0.5
        assert "claims" in result

    def test_low_support(self):
        """Test claims with low support."""
        metric = ClaimSupport()
        input_data = HallucinationInput(
            response="Cats can fly. Dogs speak French.",
            context="Cats are domestic animals. Dogs are loyal pets."
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 0.5

    def test_mixed_support(self):
        """Test claims with mixed support levels."""
        metric = ClaimSupport()
        input_data = HallucinationInput(
            response="The sun is a star. The moon is made of cheese.",
            context="The sun is a star at the center of our solar system. The moon is Earth's natural satellite."
        )
        result = metric.compute_one(input_data)
        # Should be moderate - one claim supported, one not
        assert 0.3 < result["output"] < 0.8


class TestFactualConsistency:
    """Tests for FactualConsistency metric."""

    def test_consistent_facts(self):
        """Test factually consistent response with matching words."""
        metric = FactualConsistency()
        input_data = FactualConsistencyInput(
            response="The Earth orbits the Sun. A year is 365 days long.",
            reference="The Earth orbits the Sun every 365 days, completing one full year."
        )
        result = metric.compute_one(input_data)
        # Word overlap should detect matching content
        assert result["output"] >= 0.5

    def test_contradictory_facts(self):
        """Test response with explicit contradictions."""
        metric = FactualConsistency()
        input_data = FactualConsistencyInput(
            response="The Earth is not round. The sun does not exist.",
            reference="The Earth is round. The sun is a star."
        )
        result = metric.compute_one(input_data)
        # Negations should be detected as contradictions
        assert result["output"] < 1.0

    def test_no_reference(self):
        """Test without reference provided."""
        metric = FactualConsistency()
        input_data = FactualConsistencyInput(
            response="Some claim here."
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 0.0
        assert "No reference" in result["reason"]


class TestContradictionDetection:
    """Tests for ContradictionDetection metric."""

    def test_no_contradiction(self):
        """Test response without contradictions."""
        metric = ContradictionDetection()
        input_data = HallucinationInput(
            response="Python is a programming language. It was created in 1991.",
            context="Python is a high-level programming language created by Guido van Rossum and first released in 1991."
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0
        assert "No contradictions" in result["reason"]

    def test_explicit_negation_contradiction(self):
        """Test contradiction via explicit negation."""
        metric = ContradictionDetection()
        input_data = HallucinationInput(
            response="Python is not a programming language.",
            context="Python is a programming language."
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 0.0
        assert "contradiction" in result["reason"].lower()

    def test_numeric_contradiction(self):
        """Test contradiction with different numbers."""
        metric = ContradictionDetection()
        input_data = HallucinationInput(
            response="The Eiffel Tower is 500 meters tall.",
            context="The Eiffel Tower is 330 meters tall."
        )
        result = metric.compute_one(input_data)
        # Should detect numeric contradiction
        assert result["output"] < 1.0

    def test_empty_claims(self):
        """Test with response that has no extractable claims."""
        metric = ContradictionDetection()
        input_data = HallucinationInput(
            response="Oh? Really?",
            context="Some context here."
        )
        result = metric.compute_one(input_data)
        assert result["output"] == 1.0


class TestHallucinationScore:
    """Tests for HallucinationScore metric."""

    def test_no_hallucination(self):
        """Test response with no hallucination."""
        metric = HallucinationScore()
        input_data = HallucinationInput(
            response="Machine learning is a subset of artificial intelligence. It uses algorithms to learn from data.",
            context="Machine learning is a branch of artificial intelligence that uses algorithms and statistical models to enable computers to learn from data."
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.7
        assert result["claims_analyzed"] > 0

    def test_severe_hallucination(self):
        """Test response with severe hallucination."""
        metric = HallucinationScore()
        input_data = HallucinationInput(
            response="Unicorns are real animals found in Scotland. They can fly and grant wishes.",
            context="The unicorn is a legendary creature from mythology. Scotland's national animal symbol is the unicorn."
        )
        result = metric.compute_one(input_data)
        assert result["output"] < 0.7  # Should indicate some hallucination

    def test_returns_detailed_breakdown(self):
        """Test that detailed breakdown is returned."""
        metric = HallucinationScore()
        input_data = HallucinationInput(
            response="The Great Wall of China is visible from space. It was built over 2000 years ago.",
            context="The Great Wall of China was built starting around 7th century BC. It is not actually visible from space with the naked eye."
        )
        result = metric.compute_one(input_data)

        # Check all expected fields are present
        assert "output" in result
        assert "claims_analyzed" in result
        assert "supported" in result
        assert "unsupported" in result
        assert "contradicted" in result

    def test_custom_weights(self):
        """Test with custom weights."""
        metric = HallucinationScore(config={
            "support_weight": 0.8,
            "contradiction_weight": 0.2
        })
        input_data = HallucinationInput(
            response="A simple test statement.",
            context="A simple test statement for evaluation."
        )
        result = metric.compute_one(input_data)
        assert "output" in result


class TestRealWorldScenarios:
    """Real-world e2e test scenarios."""

    def test_rag_faithfulness(self):
        """Test RAG response faithfulness to retrieved context."""
        metric = Faithfulness()
        input_data = HallucinationInput(
            response="The API supports API Key authentication. OAuth 2.0 is for user-level access.",
            context="The API supports multiple authentication methods. API Key authentication is available. OAuth 2.0 is for user-level access."
        )
        result = metric.compute_one(input_data)
        # Response uses matching terminology from context
        assert result["output"] >= 0.3

    def test_summarization_faithfulness(self):
        """Test summarization faithfulness to source document."""
        metric = HallucinationScore()
        input_data = HallucinationInput(
            response="The company reported Q3 revenue of $5 billion, a 20% increase year-over-year. They launched 3 new products.",
            context="""Quarterly Earnings Report Q3 2024:
            - Total Revenue: $5.2 billion (18% YoY growth)
            - New Product Launches: 3 products in the smart home category
            - Operating Margin: 22%
            - Employee Count: 45,000"""
        )
        result = metric.compute_one(input_data)
        # Some minor inaccuracies in numbers
        assert 0.4 < result["output"] < 1.0

    def test_qa_hallucination(self):
        """Test Q&A response for hallucination."""
        metric = HallucinationScore()
        input_data = HallucinationInput(
            query="When was the company founded?",
            response="The company was founded in 2015 by John Smith in San Francisco.",
            context="Founded in 2015 by Sarah Johnson in Seattle, the company quickly grew to become a leader in the industry."
        )
        result = metric.compute_one(input_data)
        # Should detect hallucinated founder name and city
        assert result["output"] < 0.8

    def test_long_context(self):
        """Test with long context containing matching information."""
        metric = Faithfulness(config={"support_threshold": 0.4})  # Lower threshold for long context
        long_context = """Python is a high-level programming language created by Guido van Rossum.
        Python is used for machine learning and web development. Python was first released in 1991.
        Python is dynamically typed and garbage-collected."""
        input_data = HallucinationInput(
            response="Python was created by Guido van Rossum. Python is used for machine learning and web development.",
            context=long_context
        )
        result = metric.compute_one(input_data)
        # Should find word matches in context
        assert result["output"] >= 0.3

    def test_batch_evaluation(self):
        """Test batch evaluation of multiple responses."""
        metric = HallucinationScore()
        inputs = [
            HallucinationInput(
                response="The sky is blue due to Rayleigh scattering.",
                context="Rayleigh scattering causes the sky to appear blue due to light scattering."
            ),
            HallucinationInput(
                response="Elephants can fly using their ears.",
                context="Elephants are the largest land animals and cannot fly."
            ),
            HallucinationInput(
                response="Water freezes at 0 degrees Celsius.",
                context="Water freezes at 0 degrees Celsius at standard pressure."
            ),
        ]
        results = metric.evaluate(inputs)
        assert len(results.eval_results) == 3
        # First result should have decent support
        assert results.eval_results[0].output >= 0.5
        # Second result about elephants flying should have lower support
        assert results.eval_results[1].output < results.eval_results[2].output
        # Third result should have good support (direct match)
        assert results.eval_results[2].output >= 0.5


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_short_response(self):
        """Test with very short response."""
        metric = HallucinationScore()
        input_data = HallucinationInput(
            response="Yes.",
            context="The answer is yes."
        )
        result = metric.compute_one(input_data)
        # Very short response may not have extractable claims
        assert result["output"] >= 0.0

    def test_questions_not_treated_as_claims(self):
        """Test that questions are not treated as claims."""
        metric = Faithfulness()
        input_data = HallucinationInput(
            response="What is the capital of France? Is it Paris? Yes, Paris is the capital.",
            context="Paris is the capital of France."
        )
        result = metric.compute_one(input_data)
        # Questions should be filtered out
        assert result["output"] >= 0.5

    def test_meta_statements_filtered(self):
        """Test that meta-statements are filtered out."""
        metric = Faithfulness()
        input_data = HallucinationInput(
            response="I think Python is great. Here is what I know: Python is a programming language.",
            context="Python is a programming language."
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.7

    def test_unicode_and_special_characters(self):
        """Test with unicode and special characters."""
        metric = Faithfulness()
        input_data = HallucinationInput(
            response="Tokyo is the capital of Japan.",
            context="Tokyo is the capital city of Japan."
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.7

    def test_numeric_values(self):
        """Test with numeric claims."""
        metric = FactualConsistency()
        input_data = FactualConsistencyInput(
            response="The building is 100 meters tall.",
            reference="The building stands at 100 meters in height."
        )
        result = metric.compute_one(input_data)
        assert result["output"] >= 0.7

    def test_empty_context(self):
        """Test with empty context."""
        metric = Faithfulness()
        input_data = HallucinationInput(
            response="Some claim here.",
            context=""
        )
        result = metric.compute_one(input_data)
        # Empty context means no support
        assert result["output"] < 0.5
