"""Tests for fi.evals.framework.evals.builder module."""

import pytest
from fi.evals.framework.evals.builder import (
    EvalBuilder,
    CustomEvaluation,
    CustomEvalResult,
    custom_eval,
    simple_eval,
    comparison_eval,
    threshold_eval,
    pattern_match_eval,
)
from fi.evals.framework.protocols import EvalRegistry


class TestCustomEvalResult:
    """Tests for CustomEvalResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = CustomEvalResult(
            score=0.85,
            passed=True,
        )

        assert result.score == 0.85
        assert result.passed is True
        assert result.confidence == 1.0
        assert result.details == {}

    def test_from_dict(self):
        """Test creating from dictionary."""
        result = CustomEvalResult.from_dict({
            "score": 0.9,
            "passed": True,
            "confidence": 0.8,
            "extra_key": "extra_value",
        })

        assert result.score == 0.9
        assert result.passed is True
        assert result.confidence == 0.8
        assert result.details == {"extra_key": "extra_value"}

    def test_from_dict_minimal(self):
        """Test creating from minimal dictionary."""
        result = CustomEvalResult.from_dict({})

        assert result.score == 0.0
        assert result.passed is False


class TestCustomEvaluation:
    """Tests for CustomEvaluation class."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_basic_creation(self):
        """Test basic evaluation creation."""
        eval = CustomEvaluation(
            name="test_eval",
            evaluator_fn=lambda inputs: {"score": 0.8, "passed": True},
        )

        assert eval.name == "test_eval"
        assert eval.version == "1.0.0"
        assert eval.threshold == 0.7

    def test_evaluate_returns_result(self):
        """Test evaluate returns CustomEvalResult."""
        eval = CustomEvaluation(
            name="test_eval",
            evaluator_fn=lambda inputs: {"score": 0.8, "passed": True},
        )

        result = eval.evaluate({"text": "hello"})

        assert isinstance(result, CustomEvalResult)
        assert result.score == 0.8
        assert result.passed is True

    def test_evaluate_with_custom_result(self):
        """Test evaluate when function returns CustomEvalResult."""
        def evaluator(inputs):
            return CustomEvalResult(score=0.9, passed=True, confidence=0.85)

        eval = CustomEvaluation(
            name="test_eval",
            evaluator_fn=evaluator,
        )

        result = eval.evaluate({})
        assert result.score == 0.9
        assert result.confidence == 0.85

    def test_validate_inputs(self):
        """Test input validation."""
        eval = CustomEvaluation(
            name="test_eval",
            evaluator_fn=lambda inputs: {"score": 1.0, "passed": True},
            required_fields=["text", "reference"],
        )

        errors = eval.validate_inputs({"text": "hello"})
        assert len(errors) == 1
        assert "reference" in errors[0]

    def test_get_span_attributes(self):
        """Test span attributes generation."""
        eval = CustomEvaluation(
            name="test_eval",
            evaluator_fn=lambda inputs: {"score": 0.8, "passed": True},
            threshold=0.6,
        )

        result = CustomEvalResult(score=0.8, passed=True, confidence=0.9)
        attrs = eval.get_span_attributes(result)

        assert attrs["score"] == 0.8
        assert attrs["passed"] is True
        assert attrs["threshold"] == 0.6

    def test_custom_span_attributes(self):
        """Test custom span attributes function."""
        def custom_attrs(result):
            return {"custom_score": result.score * 100}

        eval = CustomEvaluation(
            name="test_eval",
            evaluator_fn=lambda inputs: {"score": 0.8, "passed": True},
            span_attributes_fn=custom_attrs,
        )

        result = CustomEvalResult(score=0.8, passed=True)
        attrs = eval.get_span_attributes(result)

        assert attrs == {"custom_score": 80.0}


class TestEvalBuilder:
    """Tests for EvalBuilder class."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_basic_build(self):
        """Test basic builder usage."""
        eval = (
            EvalBuilder("my_eval")
            .evaluator(lambda inputs: {"score": 0.9, "passed": True})
            .build()
        )

        assert eval.name == "my_eval"
        assert eval.version == "1.0.0"

    def test_full_configuration(self):
        """Test full builder configuration."""
        eval = (
            EvalBuilder("full_eval")
            .version("2.0.0")
            .required_fields(["text"])
            .threshold(0.8)
            .description("A full evaluation")
            .evaluator(lambda inputs: {"score": 0.9, "passed": True})
            .build()
        )

        assert eval.name == "full_eval"
        assert eval.version == "2.0.0"
        assert eval.threshold == 0.8
        assert eval.description == "A full evaluation"

    def test_require_method(self):
        """Test require method for adding fields."""
        eval = (
            EvalBuilder("test")
            .require("field1", "field2")
            .require("field3")
            .evaluator(lambda inputs: {"score": 1.0, "passed": True})
            .build()
        )

        errors = eval.validate_inputs({})
        assert len(errors) == 3

    def test_span_attributes_method(self):
        """Test custom span attributes via builder."""
        eval = (
            EvalBuilder("test")
            .evaluator(lambda inputs: {"score": 0.5, "passed": True})
            .span_attributes(lambda r: {"custom": r.score})
            .build()
        )

        result = CustomEvalResult(score=0.5, passed=True)
        attrs = eval.get_span_attributes(result)

        assert attrs == {"custom": 0.5}

    def test_build_without_evaluator_raises(self):
        """Test that build fails without evaluator."""
        with pytest.raises(ValueError, match="Evaluator function must be set"):
            EvalBuilder("test").build()

    def test_chaining(self):
        """Test method chaining."""
        builder = EvalBuilder("test")
        result = builder.version("1.0.0").threshold(0.5).require("a")

        assert result is builder


class TestCustomEvalDecorator:
    """Tests for @custom_eval decorator."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_basic_decorator(self):
        """Test basic decorator usage."""
        @custom_eval("sentiment")
        def evaluate_sentiment(inputs):
            text = inputs.get("text", "")
            score = 0.8 if "good" in text else 0.2
            return {"score": score, "passed": score > 0.5}

        result = evaluate_sentiment.evaluate({"text": "This is good"})

        assert isinstance(result, CustomEvalResult)
        assert result.score == 0.8
        assert result.passed is True

    def test_decorator_with_options(self):
        """Test decorator with all options."""
        @custom_eval(
            "custom",
            version="2.0.0",
            required_fields=["input"],
            threshold=0.9,
            description="Custom evaluation",
        )
        def my_eval(inputs):
            return {"score": 1.0, "passed": True}

        assert my_eval.name == "custom"
        assert my_eval.version == "2.0.0"
        assert my_eval.threshold == 0.9

    def test_decorator_validates_inputs(self):
        """Test that decorated function validates inputs."""
        @custom_eval("test", required_fields=["required_field"])
        def my_eval(inputs):
            return {"score": 1.0, "passed": True}

        errors = my_eval.validate_inputs({})
        assert len(errors) == 1


class TestSimpleEval:
    """Tests for simple_eval factory."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_basic_usage(self):
        """Test basic simple_eval usage."""
        eval = simple_eval(
            "word_count",
            scorer=lambda inputs: min(1.0, len(inputs["text"].split()) / 10),
            threshold=0.5,
            required_fields=["text"],
        )

        result = eval.evaluate({"text": "one two three four five six"})

        assert result.score == 0.6
        assert result.passed is True

    def test_threshold_application(self):
        """Test threshold is applied correctly."""
        eval = simple_eval(
            "test",
            scorer=lambda inputs: 0.4,
            threshold=0.5,
        )

        result = eval.evaluate({})
        assert result.passed is False


class TestComparisonEval:
    """Tests for comparison_eval factory."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_basic_usage(self):
        """Test basic comparison_eval usage."""
        eval = comparison_eval(
            "exact_match",
            comparator=lambda src, tgt: 1.0 if src == tgt else 0.0,
            threshold=1.0,
        )

        # Exact match
        result = eval.evaluate({
            "response": "hello",
            "reference": "hello",
        })
        assert result.score == 1.0
        assert result.passed is True

        # No match
        result = eval.evaluate({
            "response": "hello",
            "reference": "world",
        })
        assert result.score == 0.0
        assert result.passed is False

    def test_custom_fields(self):
        """Test with custom field names."""
        eval = comparison_eval(
            "length_ratio",
            comparator=lambda src, tgt: len(src) / len(tgt) if tgt else 0,
            source_field="output",
            target_field="expected",
        )

        result = eval.evaluate({
            "output": "hello",
            "expected": "hello world",
        })

        assert result.details["source_field"] == "output"
        assert result.details["target_field"] == "expected"


class TestThresholdEval:
    """Tests for threshold_eval factory."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_min_threshold(self):
        """Test minimum threshold."""
        eval = threshold_eval(
            "min_length",
            metric_fn=lambda inputs: len(inputs["text"]),
            min_threshold=10,
            required_fields=["text"],
        )

        # Above threshold
        result = eval.evaluate({"text": "hello world!"})
        assert result.passed is True

        # Below threshold
        result = eval.evaluate({"text": "hi"})
        assert result.passed is False

    def test_max_threshold(self):
        """Test maximum threshold."""
        eval = threshold_eval(
            "max_length",
            metric_fn=lambda inputs: len(inputs["text"]),
            max_threshold=10,
        )

        # Below threshold
        result = eval.evaluate({"text": "hello"})
        assert result.passed is True

        # Above threshold
        result = eval.evaluate({"text": "hello world this is long"})
        assert result.passed is False

    def test_range_threshold(self):
        """Test min and max threshold."""
        eval = threshold_eval(
            "length_range",
            metric_fn=lambda inputs: len(inputs["text"]),
            min_threshold=5,
            max_threshold=15,
        )

        # In range
        result = eval.evaluate({"text": "hello world"})
        assert result.passed is True

        # Below range
        result = eval.evaluate({"text": "hi"})
        assert result.passed is False

        # Above range
        result = eval.evaluate({"text": "this is a very long text"})
        assert result.passed is False


class TestPatternMatchEval:
    """Tests for pattern_match_eval factory."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_any_mode(self):
        """Test 'any' mode."""
        eval = pattern_match_eval(
            "has_greeting",
            patterns=[r"\bhello\b", r"\bhi\b", r"\bhey\b"],
            mode="any",
        )

        # Has one pattern
        result = eval.evaluate({"response": "Hello there!"})
        assert result.passed is True

        # Has no patterns
        result = eval.evaluate({"response": "Goodbye!"})
        assert result.passed is False

    def test_all_mode(self):
        """Test 'all' mode."""
        eval = pattern_match_eval(
            "complete_greeting",
            patterns=[r"\bhello\b", r"\bworld\b"],
            mode="all",
        )

        # Has all patterns
        result = eval.evaluate({"response": "Hello world!"})
        assert result.passed is True

        # Has only some
        result = eval.evaluate({"response": "Hello there!"})
        assert result.passed is False

    def test_none_mode(self):
        """Test 'none' mode."""
        eval = pattern_match_eval(
            "no_profanity",
            patterns=[r"\bbad\b", r"\bworse\b"],
            mode="none",
        )

        # Has no patterns
        result = eval.evaluate({"response": "This is good!"})
        assert result.passed is True

        # Has pattern
        result = eval.evaluate({"response": "This is bad!"})
        assert result.passed is False

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        eval = pattern_match_eval(
            "test",
            patterns=[r"\bhello\b"],
            case_sensitive=False,
        )

        result = eval.evaluate({"response": "HELLO"})
        assert result.passed is True

    def test_case_sensitive(self):
        """Test case sensitive matching."""
        eval = pattern_match_eval(
            "test",
            patterns=[r"\bhello\b"],
            case_sensitive=True,
        )

        result = eval.evaluate({"response": "HELLO"})
        assert result.passed is False


class TestIntegrationWithFramework:
    """Tests for integration with the evaluation framework."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_use_with_evaluator(self):
        """Test using builder evals with Evaluator."""
        from fi.evals.framework import Evaluator, ExecutionMode

        word_count_eval = simple_eval(
            "word_count",
            scorer=lambda inputs: min(1.0, len(inputs["text"].split()) / 10),
            threshold=0.3,
            required_fields=["text"],
        )

        evaluator = Evaluator(
            evaluations=[word_count_eval],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({"text": "one two three four five"})

        assert len(result.results) == 1
        assert result.results[0].value.score == 0.5

    def test_use_with_async_evaluator(self):
        """Test using builder evals with async evaluator."""
        from fi.evals.framework import async_evaluator

        eval = (
            EvalBuilder("async_test")
            .evaluator(lambda inputs: {"score": 0.9, "passed": True})
            .build()
        )

        evaluator = async_evaluator(eval, auto_enrich_span=False)

        result = evaluator.run({})
        batch = result.wait()

        assert len(batch.results) == 1
        assert batch.results[0].value.score == 0.9

        evaluator.shutdown()

    def test_multiple_custom_evals(self):
        """Test multiple custom evals together."""
        from fi.evals.framework import Evaluator, ExecutionMode

        @custom_eval("eval1")
        def eval1(inputs):
            return {"score": 0.8, "passed": True}

        eval2 = simple_eval("eval2", scorer=lambda i: 0.9, threshold=0.5)

        eval3 = (
            EvalBuilder("eval3")
            .evaluator(lambda i: {"score": 0.7, "passed": True})
            .build()
        )

        evaluator = Evaluator(
            evaluations=[eval1, eval2, eval3],
            mode=ExecutionMode.BLOCKING,
            auto_enrich_span=False,
        )

        result = evaluator.run({})

        assert len(result.results) == 3
        assert result.success_rate == 1.0
