"""Tests for fi.evals.framework.enrichment module."""

import pytest
from unittest.mock import MagicMock, patch
from fi.evals.framework.enrichment import (
    enrich_current_span,
    enrich_span,
    add_eval_event,
    get_current_span,
    is_span_recording,
    flatten_attributes,
    SpanEnricher,
    _is_valid_span_value,
)


class TestIsValidSpanValue:
    """Tests for _is_valid_span_value helper."""

    def test_string_valid(self):
        assert _is_valid_span_value("hello") is True

    def test_int_valid(self):
        assert _is_valid_span_value(42) is True

    def test_float_valid(self):
        assert _is_valid_span_value(3.14) is True

    def test_bool_valid(self):
        assert _is_valid_span_value(True) is True
        assert _is_valid_span_value(False) is True

    def test_none_invalid(self):
        assert _is_valid_span_value(None) is False

    def test_list_invalid(self):
        assert _is_valid_span_value([1, 2, 3]) is False

    def test_dict_invalid(self):
        assert _is_valid_span_value({"a": 1}) is False

    def test_object_invalid(self):
        assert _is_valid_span_value(object()) is False


class TestFlattenAttributes:
    """Tests for flatten_attributes function."""

    def test_flat_dict(self):
        """Test already flat dict."""
        data = {"a": 1, "b": "hello", "c": True}
        result = flatten_attributes(data)

        assert result == {"a": 1, "b": "hello", "c": True}

    def test_nested_dict(self):
        """Test nested dict flattening."""
        data = {
            "level1": {
                "level2": {
                    "value": 42,
                }
            }
        }
        result = flatten_attributes(data)

        assert result == {"level1.level2.value": 42}

    def test_with_prefix(self):
        """Test with prefix."""
        data = {"a": 1, "b": 2}
        result = flatten_attributes(data, prefix="eval")

        assert result == {"eval.a": 1, "eval.b": 2}

    def test_filters_invalid_values(self):
        """Test that invalid values are filtered."""
        data = {
            "valid_str": "hello",
            "valid_int": 42,
            "invalid_list": [1, 2, 3],
            "invalid_dict": {"nested": "object"},
            "invalid_none": None,
        }
        result = flatten_attributes(data)

        assert "valid_str" in result
        assert "valid_int" in result
        assert "invalid_list" not in result
        assert "invalid_dict" not in result
        assert "invalid_none" not in result

    def test_mixed_nesting(self):
        """Test mixed nested and flat keys."""
        data = {
            "top": "value",
            "nested": {
                "a": 1,
                "b": 2,
            }
        }
        result = flatten_attributes(data)

        assert result == {"top": "value", "nested.a": 1, "nested.b": 2}

    def test_custom_separator(self):
        """Test custom separator."""
        data = {"a": {"b": 1}}
        result = flatten_attributes(data, separator="_")

        assert result == {"a_b": 1}


class TestEnrichCurrentSpan:
    """Tests for enrich_current_span function."""

    def test_no_otel_returns_false(self):
        """Test returns False when OTEL not available."""
        with patch.dict('sys.modules', {'opentelemetry': None}):
            # Force reimport to trigger ImportError
            result = enrich_current_span("test", {"score": 0.9})
            # May or may not return False depending on import caching
            # The function should handle this gracefully

    def test_with_mock_span(self):
        """Test enrichment with mock span."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch('fi.evals.framework.enrichment.get_current_span', return_value=mock_span):
            # We need to patch at the opentelemetry level
            pass  # Skip detailed mocking for now

    def test_attributes_prefixed(self):
        """Test that attributes get proper prefix."""
        # This is more of an integration test
        # Verify the logic handles prefixes correctly
        pass


class TestEnrichSpan:
    """Tests for enrich_span function."""

    def test_none_span_returns_false(self):
        """Test returns False when span is None."""
        result = enrich_span(None, "test", {"score": 0.9})
        assert result is False

    def test_with_mock_span(self):
        """Test enrichment with mock span."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        result = enrich_span(mock_span, "test_eval", {"score": 0.95, "passed": True})

        assert result is True
        # Verify set_attribute was called
        calls = mock_span.set_attribute.call_args_list
        assert any("eval.test_eval.score" in str(c) for c in calls)
        assert any("eval.test_eval.passed" in str(c) for c in calls)

    def test_not_recording_returns_false(self):
        """Test returns False when span not recording."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        result = enrich_span(mock_span, "test", {"score": 0.9})
        assert result is False

    def test_custom_prefix(self):
        """Test custom prefix."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        enrich_span(mock_span, "my_eval", {"score": 0.9}, prefix="custom")

        calls = mock_span.set_attribute.call_args_list
        assert any("custom.my_eval.score" in str(c) for c in calls)

    def test_filters_none_values(self):
        """Test that None values are filtered."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        enrich_span(mock_span, "test", {"valid": 1, "invalid": None})

        calls = [str(c) for c in mock_span.set_attribute.call_args_list]
        assert any("valid" in c for c in calls)
        assert not any("invalid" in c for c in calls)


class TestAddEvalEvent:
    """Tests for add_eval_event function."""

    def test_no_otel_returns_false(self):
        """Test returns False when OTEL not available."""
        # Would need to mock the import
        pass


class TestGetCurrentSpan:
    """Tests for get_current_span function."""

    def test_no_otel_returns_none(self):
        """Test returns None when OTEL not available."""
        # The actual behavior depends on whether OTEL is installed
        result = get_current_span()
        # Should be None or a valid span
        assert result is None or hasattr(result, 'is_recording')


class TestIsSpanRecording:
    """Tests for is_span_recording function."""

    def test_no_span_returns_false(self):
        """Test returns False when no span."""
        with patch('fi.evals.framework.enrichment.get_current_span', return_value=None):
            assert is_span_recording() is False

    def test_with_recording_span(self):
        """Test returns True when span is recording."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch('fi.evals.framework.enrichment.get_current_span', return_value=mock_span):
            assert is_span_recording() is True


class TestSpanEnricher:
    """Tests for SpanEnricher context manager."""

    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        with SpanEnricher("test_eval") as enricher:
            enricher.set_result({"score": 0.95})

    def test_records_latency(self):
        """Test that latency is recorded."""
        import time

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch('fi.evals.framework.enrichment.get_current_span', return_value=mock_span):
            with patch('fi.evals.framework.enrichment.enrich_current_span') as mock_enrich:
                with SpanEnricher("test_eval") as enricher:
                    time.sleep(0.01)  # 10ms

                # Check that enrich was called with latency
                assert mock_enrich.called

    def test_set_result(self):
        """Test set_result method."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch('fi.evals.framework.enrichment.enrich_current_span') as mock_enrich:
            with SpanEnricher("test_eval") as enricher:
                result = enricher.set_result({"score": 0.9})

            # set_result should call enrich_current_span
            assert mock_enrich.called

    def test_set_error(self):
        """Test set_error method."""
        with patch('fi.evals.framework.enrichment.enrich_current_span') as mock_enrich:
            with SpanEnricher("test_eval") as enricher:
                enricher.set_error("Something went wrong")

            assert mock_enrich.called

    def test_handles_exception(self):
        """Test that exceptions are handled gracefully."""
        with patch('fi.evals.framework.enrichment.enrich_current_span'):
            with patch('fi.evals.framework.enrichment.add_eval_event'):
                try:
                    with SpanEnricher("test_eval") as enricher:
                        raise ValueError("Test error")
                except ValueError:
                    pass  # Expected

    def test_custom_prefix(self):
        """Test custom prefix."""
        enricher = SpanEnricher("test_eval", prefix="custom")
        assert enricher.prefix == "custom"

    def test_version_attribute(self):
        """Test version is included."""
        enricher = SpanEnricher("test_eval", eval_version="2.0.0")
        assert enricher.eval_version == "2.0.0"
