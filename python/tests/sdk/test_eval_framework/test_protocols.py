"""Tests for fi.evals.framework.protocols module."""

import pytest
from typing import Dict, Any
from fi.evals.framework.protocols import (
    BaseEvaluation,
    EvalRegistry,
    register_evaluation,
    create_evaluation,
)


# Test evaluation implementations
class SimpleResult:
    """Simple result type for testing."""
    def __init__(self, score: float, passed: bool):
        self.score = score
        self.passed = passed


class SimpleEval:
    """Simple evaluation implementation for testing."""
    name = "simple_eval"
    version = "1.0.0"

    def evaluate(self, inputs: Dict[str, Any]) -> SimpleResult:
        score = len(inputs.get("text", "")) / 100
        return SimpleResult(score=min(score, 1.0), passed=score > 0.5)

    def get_span_attributes(self, result: SimpleResult) -> Dict[str, Any]:
        return {"score": result.score, "passed": result.passed}

    def validate_inputs(self, inputs: Dict[str, Any]) -> str | None:
        if "text" not in inputs:
            return "Missing required input: text"
        return None


class VersionedEval:
    """Evaluation with specific version for testing."""
    name = "versioned_eval"
    version = "2.0.0"

    def evaluate(self, inputs: Dict[str, Any]) -> dict:
        return {"value": inputs.get("x", 0) * 2}

    def get_span_attributes(self, result: dict) -> Dict[str, Any]:
        return result


class TestBaseEvaluation:
    """Tests for BaseEvaluation protocol."""

    def test_protocol_check_valid(self):
        """Test that valid implementation has required attributes/methods."""
        eval_instance = SimpleEval()
        # Check required attributes and methods exist
        assert hasattr(eval_instance, 'name')
        assert hasattr(eval_instance, 'version')
        assert hasattr(eval_instance, 'evaluate')
        assert hasattr(eval_instance, 'get_span_attributes')
        assert callable(eval_instance.evaluate)
        assert callable(eval_instance.get_span_attributes)

    def test_evaluate_method(self):
        """Test evaluate method."""
        eval_instance = SimpleEval()
        result = eval_instance.evaluate({"text": "Hello world" * 10})

        assert isinstance(result, SimpleResult)
        assert 0 <= result.score <= 1.0

    def test_get_span_attributes(self):
        """Test get_span_attributes method."""
        eval_instance = SimpleEval()
        result = SimpleResult(score=0.8, passed=True)
        attrs = eval_instance.get_span_attributes(result)

        assert attrs["score"] == 0.8
        assert attrs["passed"] is True

    def test_validate_inputs_valid(self):
        """Test validate_inputs with valid inputs."""
        eval_instance = SimpleEval()
        error = eval_instance.validate_inputs({"text": "hello"})

        assert error is None

    def test_validate_inputs_invalid(self):
        """Test validate_inputs with invalid inputs."""
        eval_instance = SimpleEval()
        error = eval_instance.validate_inputs({})

        assert error is not None
        assert "text" in error


class TestEvalRegistry:
    """Tests for EvalRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        EvalRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        EvalRegistry.clear()

    def test_register_class(self):
        """Test registering an evaluation class."""
        EvalRegistry.register(SimpleEval)

        assert EvalRegistry.is_registered("simple_eval")
        assert EvalRegistry.is_registered("simple_eval", "1.0.0")

    def test_register_multiple_versions(self):
        """Test registering multiple versions."""
        # Create different versions
        class V1Eval:
            name = "multi_version"
            version = "1.0.0"
            def evaluate(self, inputs): return {}
            def get_span_attributes(self, result): return {}

        class V2Eval:
            name = "multi_version"
            version = "2.0.0"
            def evaluate(self, inputs): return {}
            def get_span_attributes(self, result): return {}

        EvalRegistry.register(V1Eval)
        EvalRegistry.register(V2Eval)

        versions = EvalRegistry.list_all()["multi_version"]
        assert "1.0.0" in versions
        assert "2.0.0" in versions

    def test_get_specific_version(self):
        """Test getting specific version."""
        EvalRegistry.register(SimpleEval)

        cls = EvalRegistry.get("simple_eval", "1.0.0")
        assert cls is SimpleEval

    def test_get_latest_version(self):
        """Test getting latest version."""
        class V1:
            name = "latest_test"
            version = "1.0.0"
            def evaluate(self, inputs): return {}
            def get_span_attributes(self, result): return {}

        class V2:
            name = "latest_test"
            version = "2.0.0"
            def evaluate(self, inputs): return {}
            def get_span_attributes(self, result): return {}

        class V15:
            name = "latest_test"
            version = "1.5.0"
            def evaluate(self, inputs): return {}
            def get_span_attributes(self, result): return {}

        EvalRegistry.register(V1)
        EvalRegistry.register(V15)
        EvalRegistry.register(V2)

        cls = EvalRegistry.get("latest_test", "latest")
        assert cls is V2

    def test_get_not_found(self):
        """Test getting non-existent evaluation."""
        with pytest.raises(ValueError, match="not found"):
            EvalRegistry.get("nonexistent")

    def test_get_version_not_found(self):
        """Test getting non-existent version."""
        EvalRegistry.register(SimpleEval)

        with pytest.raises(ValueError, match="Version.*not found"):
            EvalRegistry.get("simple_eval", "9.9.9")

    def test_get_instance(self):
        """Test getting instantiated evaluation."""
        EvalRegistry.register(SimpleEval)

        instance = EvalRegistry.get_instance("simple_eval")
        assert isinstance(instance, SimpleEval)

    def test_list_all(self):
        """Test listing all registrations."""
        EvalRegistry.register(SimpleEval)
        EvalRegistry.register(VersionedEval)

        all_evals = EvalRegistry.list_all()
        assert "simple_eval" in all_evals
        assert "versioned_eval" in all_evals

    def test_is_registered_true(self):
        """Test is_registered returns True."""
        EvalRegistry.register(SimpleEval)

        assert EvalRegistry.is_registered("simple_eval") is True
        assert EvalRegistry.is_registered("simple_eval", "1.0.0") is True

    def test_is_registered_false(self):
        """Test is_registered returns False."""
        assert EvalRegistry.is_registered("nonexistent") is False

        EvalRegistry.register(SimpleEval)
        assert EvalRegistry.is_registered("simple_eval", "9.9.9") is False

    def test_unregister_all_versions(self):
        """Test unregistering all versions."""
        EvalRegistry.register(SimpleEval)
        assert EvalRegistry.is_registered("simple_eval")

        result = EvalRegistry.unregister("simple_eval")
        assert result is True
        assert EvalRegistry.is_registered("simple_eval") is False

    def test_unregister_specific_version(self):
        """Test unregistering specific version."""
        class V1:
            name = "unregister_test"
            version = "1.0.0"
            def evaluate(self, inputs): return {}
            def get_span_attributes(self, result): return {}

        class V2:
            name = "unregister_test"
            version = "2.0.0"
            def evaluate(self, inputs): return {}
            def get_span_attributes(self, result): return {}

        EvalRegistry.register(V1)
        EvalRegistry.register(V2)

        result = EvalRegistry.unregister("unregister_test", "1.0.0")
        assert result is True
        assert EvalRegistry.is_registered("unregister_test", "1.0.0") is False
        assert EvalRegistry.is_registered("unregister_test", "2.0.0") is True

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent evaluation."""
        result = EvalRegistry.unregister("nonexistent")
        assert result is False

    def test_clear(self):
        """Test clearing all registrations."""
        EvalRegistry.register(SimpleEval)
        EvalRegistry.register(VersionedEval)

        EvalRegistry.clear()

        assert EvalRegistry.list_all() == {}

    def test_version_comparison(self):
        """Test semantic version comparison."""
        class V1:
            name = "semver_test"
            version = "1.0.0"
            def evaluate(self, inputs): return {}
            def get_span_attributes(self, result): return {}

        class V110:
            name = "semver_test"
            version = "1.10.0"
            def evaluate(self, inputs): return {}
            def get_span_attributes(self, result): return {}

        class V19:
            name = "semver_test"
            version = "1.9.0"
            def evaluate(self, inputs): return {}
            def get_span_attributes(self, result): return {}

        EvalRegistry.register(V1)
        EvalRegistry.register(V110)
        EvalRegistry.register(V19)

        # 1.10.0 should be greater than 1.9.0
        cls = EvalRegistry.get("semver_test", "latest")
        assert cls is V110


class TestRegisterDecorator:
    """Tests for register_evaluation decorator."""

    def setup_method(self):
        EvalRegistry.clear()

    def teardown_method(self):
        EvalRegistry.clear()

    def test_decorator_registers_class(self):
        """Test that decorator registers the class."""
        @register_evaluation
        class DecoratedEval:
            name = "decorated"
            version = "1.0.0"
            def evaluate(self, inputs): return {"ok": True}
            def get_span_attributes(self, result): return result

        assert EvalRegistry.is_registered("decorated")
        assert EvalRegistry.get("decorated") is DecoratedEval

    def test_decorator_returns_class(self):
        """Test that decorator returns the class."""
        @register_evaluation
        class DecoratedEval:
            name = "returns_class"
            version = "1.0.0"
            def evaluate(self, inputs): return {}
            def get_span_attributes(self, result): return {}

        # Should be able to instantiate
        instance = DecoratedEval()
        assert instance is not None

    def test_decorator_uses_class_name_if_no_name(self):
        """Test fallback to class name."""
        @register_evaluation
        class NoNameEval:
            version = "1.0.0"
            def evaluate(self, inputs): return {}
            def get_span_attributes(self, result): return {}

        assert EvalRegistry.is_registered("NoNameEval")


class TestCreateEvaluation:
    """Tests for create_evaluation factory."""

    def test_create_basic_evaluation(self):
        """Test creating evaluation from functions."""
        MyEval = create_evaluation(
            name="factory_eval",
            version="1.0.0",
            evaluate_fn=lambda inputs: {"result": inputs["x"] * 2},
            span_attributes_fn=lambda result: result,
        )

        instance = MyEval()
        result = instance.evaluate({"x": 5})

        assert result == {"result": 10}

    def test_create_evaluation_span_attributes(self):
        """Test span attributes from factory eval."""
        MyEval = create_evaluation(
            name="factory_span",
            evaluate_fn=lambda inputs: {"score": 0.8, "category": "good"},
            span_attributes_fn=lambda result: {
                "score": result["score"],
                "category": result["category"],
            },
        )

        instance = MyEval()
        result = instance.evaluate({})
        attrs = instance.get_span_attributes(result)

        assert attrs["score"] == 0.8
        assert attrs["category"] == "good"

    def test_create_evaluation_default_span_attributes(self):
        """Test default span attributes extraction."""
        MyEval = create_evaluation(
            name="default_span",
            evaluate_fn=lambda inputs: {
                "score": 0.9,
                "passed": True,
                "complex": {"nested": "value"},  # Should be filtered out
            },
        )

        instance = MyEval()
        result = instance.evaluate({})
        attrs = instance.get_span_attributes(result)

        assert attrs["score"] == 0.9
        assert attrs["passed"] is True
        assert "complex" not in attrs  # Non-primitive filtered

    def test_create_evaluation_has_name_version(self):
        """Test that created class has name and version."""
        MyEval = create_evaluation(
            name="named_eval",
            version="2.5.0",
            evaluate_fn=lambda inputs: {},
        )

        assert MyEval.name == "named_eval"
        assert MyEval.version == "2.5.0"

    def test_create_evaluation_no_evaluate_fn(self):
        """Test error when evaluate_fn not provided."""
        MyEval = create_evaluation(name="no_fn")
        instance = MyEval()

        with pytest.raises(NotImplementedError):
            instance.evaluate({})

    def test_create_evaluation_validate_inputs(self):
        """Test that validate_inputs returns empty list by default."""
        MyEval = create_evaluation(
            name="validate_test",
            evaluate_fn=lambda inputs: {},
        )

        instance = MyEval()
        assert instance.validate_inputs({}) == []
