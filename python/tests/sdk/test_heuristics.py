"""Comprehensive tests for heuristic metrics."""

import pytest
from fi.evals.types import TextMetricInput, JsonMetricInput


class TestStringMetrics:
    """Tests for string-based metrics."""

    def test_regex_match_found(self):
        """Test Regex metric when pattern is found."""
        from fi.evals.metrics.heuristics.string_metrics import Regex

        metric = Regex(config={"pattern": r"\d{3}-\d{4}"})
        input_data = TextMetricInput(response="Call me at 555-1234")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0
        assert "found" in result["reason"]

    def test_regex_no_match(self):
        """Test Regex metric when pattern is not found."""
        from fi.evals.metrics.heuristics.string_metrics import Regex

        metric = Regex(config={"pattern": r"\d{3}-\d{4}"})
        input_data = TextMetricInput(response="No phone number here")
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0
        assert "not found" in result["reason"]

    def test_regex_requires_pattern(self):
        """Test Regex metric requires pattern in config."""
        from fi.evals.metrics.heuristics.string_metrics import Regex

        with pytest.raises(ValueError, match="pattern"):
            Regex(config={})

    def test_contains_keyword_found(self):
        """Test Contains metric when keyword is present."""
        from fi.evals.metrics.heuristics.string_metrics import Contains

        metric = Contains(config={"keyword": "hello"})
        input_data = TextMetricInput(response="Hello, world!")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0
        assert "found" in result["reason"]

    def test_contains_keyword_not_found(self):
        """Test Contains metric when keyword is absent."""
        from fi.evals.metrics.heuristics.string_metrics import Contains

        metric = Contains(config={"keyword": "goodbye"})
        input_data = TextMetricInput(response="Hello, world!")
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0
        assert "not found" in result["reason"]

    def test_contains_case_sensitive(self):
        """Test Contains metric with case sensitivity."""
        from fi.evals.metrics.heuristics.string_metrics import Contains

        metric = Contains(config={"keyword": "Hello", "case_sensitive": True})
        input_data = TextMetricInput(response="hello, world!")
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0  # "Hello" != "hello" when case sensitive

    def test_contains_requires_keyword(self):
        """Test Contains metric requires keyword in config."""
        from fi.evals.metrics.heuristics.string_metrics import Contains

        with pytest.raises(ValueError, match="keyword"):
            Contains(config={})

    def test_contains_all_found(self):
        """Test ContainsAll when all keywords are present."""
        from fi.evals.metrics.heuristics.string_metrics import ContainsAll

        metric = ContainsAll(config={"keywords": ["apple", "banana"]})
        input_data = TextMetricInput(response="I like apple and banana")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0
        assert "All" in result["reason"]

    def test_contains_all_partial(self):
        """Test ContainsAll when only some keywords are present."""
        from fi.evals.metrics.heuristics.string_metrics import ContainsAll

        metric = ContainsAll(config={"keywords": ["apple", "banana", "cherry"]})
        input_data = TextMetricInput(response="I like apple and banana")
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0
        assert "cherry" in result["reason"]

    def test_contains_any_found(self):
        """Test ContainsAny when at least one keyword is present."""
        from fi.evals.metrics.heuristics.string_metrics import ContainsAny

        metric = ContainsAny(config={"keywords": ["apple", "orange"]})
        input_data = TextMetricInput(response="I have an orange")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_contains_any_none_found(self):
        """Test ContainsAny when no keywords are present."""
        from fi.evals.metrics.heuristics.string_metrics import ContainsAny

        metric = ContainsAny(config={"keywords": ["apple", "orange"]})
        input_data = TextMetricInput(response="I have a banana")
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0

    def test_contains_none_pass(self):
        """Test ContainsNone when no forbidden keywords are present."""
        from fi.evals.metrics.heuristics.string_metrics import ContainsNone

        metric = ContainsNone(config={"keywords": ["spam", "junk"]})
        input_data = TextMetricInput(response="This is a valid message")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_contains_none_fail(self):
        """Test ContainsNone when forbidden keywords are present."""
        from fi.evals.metrics.heuristics.string_metrics import ContainsNone

        metric = ContainsNone(config={"keywords": ["spam", "junk"]})
        input_data = TextMetricInput(response="This is spam content")
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0

    def test_one_line_pass(self):
        """Test OneLine with single line text."""
        from fi.evals.metrics.heuristics.string_metrics import OneLine

        metric = OneLine()
        input_data = TextMetricInput(response="This is a single line")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_one_line_fail(self):
        """Test OneLine with multi-line text."""
        from fi.evals.metrics.heuristics.string_metrics import OneLine

        metric = OneLine()
        input_data = TextMetricInput(response="Line 1\nLine 2")
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0

    def test_contains_email(self):
        """Test ContainsEmail metric."""
        from fi.evals.metrics.heuristics.string_metrics import ContainsEmail

        metric = ContainsEmail()
        input_data = TextMetricInput(response="Contact me at user@example.com")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_is_email_valid(self):
        """Test IsEmail with valid email."""
        from fi.evals.metrics.heuristics.string_metrics import IsEmail

        metric = IsEmail()
        input_data = TextMetricInput(response="user@example.com")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_is_email_invalid(self):
        """Test IsEmail with invalid email."""
        from fi.evals.metrics.heuristics.string_metrics import IsEmail

        metric = IsEmail()
        input_data = TextMetricInput(response="not an email")
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0

    def test_contains_link(self):
        """Test ContainsLink metric."""
        from fi.evals.metrics.heuristics.string_metrics import ContainsLink

        metric = ContainsLink()
        input_data = TextMetricInput(response="Visit https://example.com for more")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_equals_match(self):
        """Test Equals when strings match."""
        from fi.evals.metrics.heuristics.string_metrics import Equals

        metric = Equals()
        input_data = TextMetricInput(response="Hello World", expected_response="hello world")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0  # case insensitive by default

    def test_equals_no_match_case_sensitive(self):
        """Test Equals with case sensitivity."""
        from fi.evals.metrics.heuristics.string_metrics import Equals

        metric = Equals(config={"case_sensitive": True})
        input_data = TextMetricInput(response="Hello World", expected_response="hello world")
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0

    def test_starts_with_pass(self):
        """Test StartsWith when text starts with prefix."""
        from fi.evals.metrics.heuristics.string_metrics import StartsWith

        metric = StartsWith()
        input_data = TextMetricInput(response="Hello world", expected_response="Hello")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_ends_with_pass(self):
        """Test EndsWith when text ends with suffix."""
        from fi.evals.metrics.heuristics.string_metrics import EndsWith

        metric = EndsWith()
        input_data = TextMetricInput(response="Hello world", expected_response="world")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_length_less_than_pass(self):
        """Test LengthLessThan when length is within limit."""
        from fi.evals.metrics.heuristics.string_metrics import LengthLessThan

        metric = LengthLessThan(config={"max_length": 20})
        input_data = TextMetricInput(response="Short text")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_length_less_than_fail(self):
        """Test LengthLessThan when length exceeds limit."""
        from fi.evals.metrics.heuristics.string_metrics import LengthLessThan

        metric = LengthLessThan(config={"max_length": 5})
        input_data = TextMetricInput(response="This is a longer text")
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0

    def test_length_greater_than_pass(self):
        """Test LengthGreaterThan when length exceeds minimum."""
        from fi.evals.metrics.heuristics.string_metrics import LengthGreaterThan

        metric = LengthGreaterThan(config={"min_length": 5})
        input_data = TextMetricInput(response="This is long enough")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_length_between_pass(self):
        """Test LengthBetween when length is in range."""
        from fi.evals.metrics.heuristics.string_metrics import LengthBetween

        metric = LengthBetween(config={"min_length": 5, "max_length": 20})
        input_data = TextMetricInput(response="Perfect length")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0


class TestJSONMetrics:
    """Tests for JSON-based metrics."""

    def test_contains_json_valid(self):
        """Test ContainsJson with valid JSON in text."""
        from fi.evals.metrics.heuristics.json_metrics import ContainsJson

        metric = ContainsJson()
        input_data = TextMetricInput(response='Here is the data: {"key": "value"}')
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_contains_json_invalid(self):
        """Test ContainsJson with no valid JSON."""
        from fi.evals.metrics.heuristics.json_metrics import ContainsJson

        metric = ContainsJson()
        input_data = TextMetricInput(response="No JSON here")
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0

    def test_is_json_valid(self):
        """Test IsJson with valid JSON."""
        from fi.evals.metrics.heuristics.json_metrics import IsJson

        metric = IsJson()
        input_data = TextMetricInput(response='{"name": "test", "value": 123}')
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_is_json_invalid(self):
        """Test IsJson with invalid JSON."""
        from fi.evals.metrics.heuristics.json_metrics import IsJson

        metric = IsJson()
        input_data = TextMetricInput(response='{invalid json}')
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0

    def test_is_json_array(self):
        """Test IsJson with JSON array."""
        from fi.evals.metrics.heuristics.json_metrics import IsJson

        metric = IsJson()
        input_data = TextMetricInput(response='[1, 2, 3]')
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_json_schema_valid(self):
        """Test JsonSchema with valid JSON matching schema."""
        from fi.evals.metrics.heuristics.json_metrics import JsonSchema

        metric = JsonSchema()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
        input_data = JsonMetricInput(
            response='{"name": "John", "age": 30}',
            schema=schema
        )
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_json_schema_invalid(self):
        """Test JsonSchema with JSON not matching schema."""
        from fi.evals.metrics.heuristics.json_metrics import JsonSchema

        metric = JsonSchema()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
        input_data = JsonMetricInput(
            response='{"name": "John"}',  # missing required "age"
            schema=schema
        )
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0


class TestSimilarityMetrics:
    """Tests for similarity-based metrics."""

    def test_bleu_score_perfect_match(self):
        """Test BLEUScore with identical texts."""
        from fi.evals.metrics.heuristics.similarity_metrics import BLEUScore

        metric = BLEUScore()
        input_data = TextMetricInput(
            response="The quick brown fox",
            expected_response="The quick brown fox"
        )
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_bleu_score_no_match(self):
        """Test BLEUScore with completely different texts."""
        from fi.evals.metrics.heuristics.similarity_metrics import BLEUScore

        metric = BLEUScore()
        input_data = TextMetricInput(
            response="Hello world",
            expected_response="Goodbye universe"
        )
        result = metric.compute_one(input_data)

        assert result["output"] < 0.5

    def test_bleu_score_multiple_references(self):
        """Test BLEUScore with multiple reference translations."""
        from fi.evals.metrics.heuristics.similarity_metrics import BLEUScore

        metric = BLEUScore()
        input_data = TextMetricInput(
            response="The quick brown fox",
            expected_response=["The quick brown fox", "A fast brown fox"]
        )
        result = metric.compute_one(input_data)

        assert result["output"] >= 0.5

    def test_rouge_score_high_overlap(self):
        """Test ROUGEScore with high text overlap."""
        from fi.evals.metrics.heuristics.similarity_metrics import ROUGEScore

        metric = ROUGEScore(config={"rouge_type": "rouge1"})
        input_data = TextMetricInput(
            response="The quick brown fox jumps",
            expected_response="The quick brown fox jumps over"
        )
        result = metric.compute_one(input_data)

        assert result["output"] > 0.5

    def test_rouge_score_types(self):
        """Test different ROUGE score types."""
        from fi.evals.metrics.heuristics.similarity_metrics import ROUGEScore

        for rouge_type in ["rouge1", "rouge2", "rougeL"]:
            metric = ROUGEScore(config={"rouge_type": rouge_type})
            input_data = TextMetricInput(
                response="This is a test",
                expected_response="This is a test"
            )
            result = metric.compute_one(input_data)
            assert result["output"] == 1.0

    def test_rouge_score_invalid_type(self):
        """Test ROUGEScore with invalid rouge_type."""
        from fi.evals.metrics.heuristics.similarity_metrics import ROUGEScore

        with pytest.raises(ValueError, match="Invalid rouge_type"):
            ROUGEScore(config={"rouge_type": "invalid"})

    def test_levenshtein_similarity_identical(self):
        """Test LevenshteinSimilarity with identical strings."""
        from fi.evals.metrics.heuristics.similarity_metrics import LevenshteinSimilarity

        metric = LevenshteinSimilarity()
        input_data = TextMetricInput(
            response="Hello World",
            expected_response="hello world"
        )
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0  # case insensitive by default

    def test_levenshtein_similarity_different(self):
        """Test LevenshteinSimilarity with different strings."""
        from fi.evals.metrics.heuristics.similarity_metrics import LevenshteinSimilarity

        metric = LevenshteinSimilarity()
        input_data = TextMetricInput(
            response="Hello",
            expected_response="World"
        )
        result = metric.compute_one(input_data)

        assert 0 < result["output"] < 1

    def test_numeric_similarity_exact(self):
        """Test NumericSimilarity with exact match."""
        from fi.evals.metrics.heuristics.similarity_metrics import NumericSimilarity

        metric = NumericSimilarity()
        input_data = TextMetricInput(
            response="The answer is 42",
            expected_response="42"
        )
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_numeric_similarity_close(self):
        """Test NumericSimilarity with close values."""
        from fi.evals.metrics.heuristics.similarity_metrics import NumericSimilarity

        metric = NumericSimilarity()
        input_data = TextMetricInput(
            response="100",
            expected_response="95"
        )
        result = metric.compute_one(input_data)

        assert result["output"] > 0.9

    def test_recall_score_perfect(self):
        """Test RecallScore with perfect recall."""
        from fi.evals.metrics.heuristics.similarity_metrics import RecallScore

        metric = RecallScore()
        input_data = TextMetricInput(
            response="[1, 2, 3]",
            expected_response="[1, 2, 3]"
        )
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_recall_score_partial(self):
        """Test RecallScore with partial recall."""
        from fi.evals.metrics.heuristics.similarity_metrics import RecallScore

        metric = RecallScore()
        input_data = TextMetricInput(
            response="[1, 2]",
            expected_response="[1, 2, 3, 4]"
        )
        result = metric.compute_one(input_data)

        assert result["output"] == 0.5  # 2 of 4 items


class TestMetricRequirements:
    """Tests for metric input requirements and error handling."""

    def test_equals_requires_expected_response(self):
        """Test Equals raises error without expected_response."""
        from fi.evals.metrics.heuristics.string_metrics import Equals

        metric = Equals()
        input_data = TextMetricInput(response="test")

        with pytest.raises(ValueError, match="expected_response"):
            metric.compute_one(input_data)

    def test_starts_with_requires_expected_response(self):
        """Test StartsWith raises error without expected_response."""
        from fi.evals.metrics.heuristics.string_metrics import StartsWith

        metric = StartsWith()
        input_data = TextMetricInput(response="test")

        with pytest.raises(ValueError, match="expected_response"):
            metric.compute_one(input_data)

    def test_ends_with_requires_expected_response(self):
        """Test EndsWith raises error without expected_response."""
        from fi.evals.metrics.heuristics.string_metrics import EndsWith

        metric = EndsWith()
        input_data = TextMetricInput(response="test")

        with pytest.raises(ValueError, match="expected_response"):
            metric.compute_one(input_data)

    def test_bleu_requires_expected_response(self):
        """Test BLEUScore raises error without expected_response."""
        from fi.evals.metrics.heuristics.similarity_metrics import BLEUScore

        metric = BLEUScore()
        input_data = TextMetricInput(response="test")

        with pytest.raises(ValueError, match="expected_response"):
            metric.compute_one(input_data)

    def test_rouge_requires_expected_response(self):
        """Test ROUGEScore raises error without expected_response."""
        from fi.evals.metrics.heuristics.similarity_metrics import ROUGEScore

        metric = ROUGEScore()
        input_data = TextMetricInput(response="test")

        with pytest.raises(ValueError, match="expected_response"):
            metric.compute_one(input_data)

    def test_levenshtein_requires_expected_response(self):
        """Test LevenshteinSimilarity raises error without expected_response."""
        from fi.evals.metrics.heuristics.similarity_metrics import LevenshteinSimilarity

        metric = LevenshteinSimilarity()
        input_data = TextMetricInput(response="test")

        with pytest.raises(ValueError, match="expected_response"):
            metric.compute_one(input_data)

    def test_json_schema_requires_schema(self):
        """Test JsonSchema raises error without schema."""
        from fi.evals.metrics.heuristics.json_metrics import JsonSchema

        metric = JsonSchema()
        input_data = JsonMetricInput(response='{"key": "value"}')

        with pytest.raises(ValueError, match="schema"):
            metric.compute_one(input_data)


class TestMetricEdgeCases:
    """Tests for edge cases in metrics."""

    def test_empty_response(self):
        """Test handling of empty response."""
        from fi.evals.metrics.heuristics.string_metrics import Contains

        metric = Contains(config={"keyword": "test"})
        input_data = TextMetricInput(response="")
        result = metric.compute_one(input_data)

        assert result["output"] == 0.0

    def test_whitespace_only_response(self):
        """Test handling of whitespace-only response."""
        from fi.evals.metrics.heuristics.string_metrics import OneLine

        metric = OneLine()
        input_data = TextMetricInput(response="   ")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0  # single line of whitespace

    def test_special_characters(self):
        """Test handling of special characters."""
        from fi.evals.metrics.heuristics.string_metrics import Regex

        metric = Regex(config={"pattern": r"[!@#$%]+"})
        input_data = TextMetricInput(response="Hello! @user #tag $price %off")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        from fi.evals.metrics.heuristics.string_metrics import Contains

        metric = Contains(config={"keyword": "日本語"})
        input_data = TextMetricInput(response="This contains 日本語 text")
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0

    def test_very_long_text(self):
        """Test handling of very long text."""
        from fi.evals.metrics.heuristics.string_metrics import LengthGreaterThan

        long_text = "a" * 10000
        metric = LengthGreaterThan(config={"min_length": 5000})
        input_data = TextMetricInput(response=long_text)
        result = metric.compute_one(input_data)

        assert result["output"] == 1.0
