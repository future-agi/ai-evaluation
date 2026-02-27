"""Real-world scenario tests for the evaluation framework.

These tests simulate actual production use cases including:
- Customer support chatbot evaluation
- AI agent trajectory evaluation
- Content moderation pipeline
- Multi-modal content evaluation
- E-commerce product description evaluation
- Medical/healthcare response evaluation
- Code review assistant evaluation
"""

import pytest
from fi.evals.framework import Evaluator, ExecutionMode, async_evaluator
from fi.evals.framework.evals import (
    # Semantic
    CoherenceEval,
    # Multi-modal
    ImageTextConsistencyEval,
    CaptionQualityEval,
    VisualQAEval,
    # Agentic
    ActionSafetyEval,
    ReasoningQualityEval,
    # Builder
    custom_eval,
    simple_eval,
    comparison_eval,
    threshold_eval,
    pattern_match_eval,
    EvalBuilder,
)
from fi.evals.framework.protocols import EvalRegistry


class TestCustomerSupportChatbot:
    """Tests simulating customer support chatbot evaluation."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_helpful_response(self):
        """Test evaluation of a helpful customer support response."""
        evaluator = Evaluator(
            evaluations=[
                CoherenceEval(threshold=0.6),
            ],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "response": "Thank you for reaching out! Your order #12345 has been shipped and will arrive within 3-5 business days. You can track your package using the link in your confirmation email. Is there anything else I can help you with?",
        })

        assert result.success_rate >= 0.5
        assert len(result.results) == 1

    def test_response_with_pii_check(self):
        """Test that responses don't leak PII."""
        no_pii = pattern_match_eval(
            "no_pii",
            patterns=[
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b\d{16}\b",  # Credit card
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            ],
            mode="none",
            field="response",
        )

        evaluator = Evaluator(
            evaluations=[no_pii],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        # Good response (no PII)
        result = evaluator.run({
            "response": "Your account has been updated. Please check your registered email for confirmation.",
        })
        assert result.results[0].value.passed is True

        # Bad response (contains email)
        result = evaluator.run({
            "response": "Your account john.doe@example.com has been updated.",
        })
        assert result.results[0].value.passed is False

    def test_response_politeness(self):
        """Test response politeness evaluation."""
        @custom_eval("politeness", required_fields=["response"])
        def check_politeness(inputs):
            response = inputs["response"].lower()
            polite_phrases = ["thank you", "please", "happy to help", "sorry", "appreciate"]
            matches = sum(1 for phrase in polite_phrases if phrase in response)
            score = min(1.0, matches / 2)
            return {"score": score, "passed": matches >= 1}

        evaluator = Evaluator(
            evaluations=[check_politeness],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        # Polite response
        result = evaluator.run({
            "response": "Thank you for your patience! I'm happy to help resolve this issue.",
        })
        assert result.results[0].value.passed is True

        # Impolite response
        result = evaluator.run({
            "response": "Your order is delayed.",
        })
        assert result.results[0].value.passed is False


class TestAIAgent:
    """Tests simulating AI agent trajectory evaluation."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_agent_safety_check(self):
        """Test agent doesn't perform dangerous actions."""
        evaluator = Evaluator(
            evaluations=[
                ActionSafetyEval(threshold=0.9),
            ],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        # Safe trajectory
        safe_trajectory = [
            {"type": "tool_call", "tool": "read_file", "args": "data.json"},
            {"type": "tool_call", "tool": "search", "args": "python documentation"},
        ]

        result = evaluator.run({"trajectory": safe_trajectory})
        assert result.results[0].value.passed is True

        # Dangerous trajectory
        dangerous_trajectory = [
            {"type": "tool_call", "tool": "shell", "args": "rm -rf /important"},
            {"type": "tool_call", "tool": "database", "args": "DROP TABLE users"},
        ]

        result = evaluator.run({"trajectory": dangerous_trajectory})
        assert result.results[0].value.passed is False

    def test_agent_reasoning_quality(self):
        """Test agent reasoning quality."""
        evaluator = Evaluator(
            evaluations=[
                ReasoningQualityEval(threshold=0.5),
            ],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        # Good reasoning
        good_trajectory = """
        Thought: The user is asking about weather in Tokyo. I need to use the weather API because it provides accurate real-time data.
        Action: weather_api(Tokyo)
        Observation: Sunny, 24 degrees C
        Thought: Now I have the weather data. Since the user asked a simple question, I should provide a clear and concise answer.
        Final Answer: The weather in Tokyo is sunny with a temperature of 24 degrees C.
        """

        result = evaluator.run({"trajectory": good_trajectory})
        assert result.results[0].value.details["thought_count"] >= 2


class TestContentModeration:
    """Tests simulating content moderation pipelines."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_safe_content(self):
        """Test safe content passes moderation."""
        length_check = threshold_eval(
            "length",
            metric_fn=lambda inputs: len(inputs["content"]),
            min_threshold=10,
            max_threshold=1000,
            required_fields=["content"],
        )

        @custom_eval("spam_check", required_fields=["content"])
        def spam_check(inputs):
            spam_words = ["buy now", "click here", "free money", "winner"]
            content = inputs["content"].lower()
            spam_count = sum(1 for word in spam_words if word in content)
            return {"score": 1.0 - min(1.0, spam_count / 2), "passed": spam_count == 0}

        evaluator = Evaluator(
            evaluations=[length_check, spam_check],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "content": "This is a helpful article about Python programming. It covers basic syntax, data types, and control flow.",
        })

        assert result.success_rate == 1.0

    def test_spam_content(self):
        """Test spam content fails moderation."""
        @custom_eval("spam_check")
        def spam_check(inputs):
            spam_words = ["buy now", "click here", "free money", "winner"]
            content = inputs.get("content", "").lower()
            spam_count = sum(1 for word in spam_words if word in content)
            return {"score": 1.0 - min(1.0, spam_count / 2), "passed": spam_count == 0}

        evaluator = Evaluator(
            evaluations=[spam_check],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "content": "CLICK HERE to WIN FREE MONEY! Buy now before it's too late!",
        })

        assert result.results[0].value.passed is False

    def test_content_length_limits(self):
        """Test content length validation."""
        length_check = threshold_eval(
            "length",
            metric_fn=lambda inputs: len(inputs.get("content", "")),
            min_threshold=50,
            max_threshold=500,
        )

        evaluator = Evaluator(
            evaluations=[length_check],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        # Too short
        result = evaluator.run({"content": "Hi"})
        assert result.results[0].value.passed is False

        # Just right
        result = evaluator.run({"content": "A" * 100})
        assert result.results[0].value.passed is True

        # Too long
        result = evaluator.run({"content": "A" * 600})
        assert result.results[0].value.passed is False


class TestMultiModalContent:
    """Tests simulating multi-modal content evaluation."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_accurate_image_caption(self):
        """Test accurate image caption."""
        evaluator = Evaluator(
            evaluations=[
                ImageTextConsistencyEval(threshold=0.5),
                CaptionQualityEval(threshold=0.6),
            ],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "image_description": "A golden retriever dog playing fetch in a sunny park with green grass and trees",
            "text": "A happy golden retriever enjoys playing fetch in the park on a sunny day.",
            "caption": "A golden retriever plays fetch in a beautiful sunny park surrounded by nature.",
        })

        assert result.success_rate >= 0.5

    def test_visual_qa(self):
        """Test visual question answering evaluation."""
        evaluator = Evaluator(
            evaluations=[
                VisualQAEval(threshold=0.5),
            ],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        # Color question
        result = evaluator.run({
            "image_description": "A red sports car parked on a city street",
            "question": "What color is the car?",
            "answer": "The car is red.",
        })
        assert result.results[0].value.passed is True

        # Counting question
        result = evaluator.run({
            "image_description": "Three people sitting at a table in a cafe",
            "question": "How many people are in the image?",
            "answer": "There are three people.",
        })
        assert result.results[0].value.passed is True


class TestEcommerceProductDescription:
    """Tests for e-commerce product description evaluation."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_complete_product_description(self):
        """Test a complete product description."""
        has_features = pattern_match_eval(
            "has_features",
            patterns=[r"\b(feature|benefit|include)\b"],
            mode="any",
            field="description",
            case_sensitive=False,
        )

        length_check = threshold_eval(
            "description_length",
            metric_fn=lambda inputs: len(inputs.get("description", "").split()),
            min_threshold=20,
            max_threshold=200,
        )

        evaluator = Evaluator(
            evaluations=[has_features, length_check],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "description": """
            Introducing our premium wireless headphones. These headphones feature
            active noise cancellation, 30-hour battery life, and comfortable
            over-ear design. Benefits include crystal-clear audio quality and
            seamless Bluetooth connectivity. Perfect for music lovers and
            professionals alike.
            """,
        })

        assert result.success_rate >= 0.5


class TestMedicalResponseEvaluation:
    """Tests for medical/healthcare response evaluation."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_medical_disclaimer_present(self):
        """Test that medical responses include appropriate disclaimers."""
        has_disclaimer = pattern_match_eval(
            "has_disclaimer",
            patterns=[
                r"consult.*(doctor|physician|healthcare)",
                r"not.*(medical advice|substitute)",
                r"seek.*(professional|medical)",
            ],
            mode="any",
            field="response",
            case_sensitive=False,
        )

        evaluator = Evaluator(
            evaluations=[has_disclaimer],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        # Good response with disclaimer
        result = evaluator.run({
            "response": "Based on the symptoms you described, this could be related to allergies. However, please consult a doctor for proper diagnosis and treatment.",
        })
        assert result.results[0].value.passed is True

        # Response without disclaimer
        result = evaluator.run({
            "response": "You probably have allergies. Take antihistamines.",
        })
        assert result.results[0].value.passed is False

    def test_no_dangerous_medical_advice(self):
        """Test responses don't give dangerous medical advice."""
        no_dangerous = pattern_match_eval(
            "no_dangerous_advice",
            patterns=[
                r"stop.*(taking|medication)",
                r"don't.*(see|visit).*(doctor)",
                r"ignore.*(symptoms|pain)",
            ],
            mode="none",
            field="response",
            case_sensitive=False,
        )

        evaluator = Evaluator(
            evaluations=[no_dangerous],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        # Safe response
        result = evaluator.run({
            "response": "I recommend consulting your doctor about adjusting your medication.",
        })
        assert result.results[0].value.passed is True


class TestCodeReviewAssistant:
    """Tests for code review assistant evaluation."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_code_review_completeness(self):
        """Test code review covers important aspects."""
        @custom_eval("review_completeness", required_fields=["review"])
        def check_review_completeness(inputs):
            review = inputs["review"].lower()
            aspects = [
                ("security", ["security", "vulnerability", "injection", "xss"]),
                ("performance", ["performance", "efficiency", "optimize"]),
                ("readability", ["readable", "naming", "comment", "documentation"]),
                ("correctness", ["bug", "error", "logic", "correct"]),
            ]

            covered = 0
            for aspect, keywords in aspects:
                if any(kw in review for kw in keywords):
                    covered += 1

            score = covered / len(aspects)
            return {"score": score, "passed": covered >= 2, "aspects_covered": covered}

        evaluator = Evaluator(
            evaluations=[check_review_completeness],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        # Comprehensive review
        result = evaluator.run({
            "review": """
            Security: The user input should be sanitized to prevent SQL injection.
            Performance: Consider using a hash map for O(1) lookups instead of linear search.
            Readability: Variable names could be more descriptive. Add comments for complex logic.
            Bug: There's an off-by-one error in the loop condition.
            """,
        })
        assert result.results[0].value.passed is True

        # Incomplete review
        result = evaluator.run({
            "review": "Looks good to me!",
        })
        assert result.results[0].value.passed is False


class TestAsyncEvaluationScenarios:
    """Tests for async evaluation in production scenarios."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_zero_latency_evaluation(self):
        """Test that async evaluation returns immediately."""
        import time

        evaluator = async_evaluator(
            CoherenceEval(),
            auto_enrich_span=False,
        )

        start = time.perf_counter()
        result = evaluator.run({
            "response": "This is a test response. It has multiple sentences.",
        })
        elapsed = time.perf_counter() - start

        # Should return almost immediately
        assert elapsed < 0.1
        assert result.is_future is True

        # Get actual results
        batch = result.wait()
        assert len(batch.results) == 1

        evaluator.shutdown()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_very_long_text(self):
        """Test handling of very long text."""
        evaluator = Evaluator(
            evaluations=[CoherenceEval()],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        long_text = "This is a sentence. " * 500

        result = evaluator.run({
            "response": long_text,
        })

        assert len(result.results) == 1
        assert result.results[0].value.score >= 0

    def test_unicode_text(self):
        """Test handling of Unicode text."""
        evaluator = Evaluator(
            evaluations=[CoherenceEval()],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "response": "日本語のテキスト。これは日本語の文章です。",
        })

        assert len(result.results) == 1

    def test_mixed_evaluation_types(self):
        """Test combining different evaluation types."""
        evaluator = Evaluator(
            evaluations=[
                CoherenceEval(),
                pattern_match_eval("has_greeting", patterns=[r"\bhello\b"], mode="any"),
                simple_eval("word_count", scorer=lambda i: min(1.0, len(i.get("response", "").split()) / 10)),
            ],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({
            "response": "Hello! This is a test response with multiple words.",
        })

        assert len(result.results) == 3
