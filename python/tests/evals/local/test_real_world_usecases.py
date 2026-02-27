"""Real-world use case tests for local evaluation.

These tests validate practical scenarios where local evaluation is used,
matching the use cases documented in docs/local-execution.md.
"""

import json
import pytest

from fi.evals.local import LocalEvaluator, HybridEvaluator, ExecutionMode


class TestRAGPipelineValidation:
    """Test Case 1: RAG Pipeline Validation.

    Validates that RAG responses have proper formatting, citations,
    and don't contain cop-out answers.
    """

    @pytest.fixture
    def evaluator(self):
        return LocalEvaluator()

    def test_rag_response_with_citations_passes(self, evaluator):
        """RAG response with proper citations should pass validation."""
        response = """Based on the documentation [1], the API supports REST endpoints.
        According to the developer guide [2], authentication uses OAuth 2.0.
        The rate limits are described in the API reference [3]."""

        # Check for citation markers
        result = evaluator.evaluate(
            "regex",
            [{"response": response}],
            {"pattern": r"\[\d+\]"}
        )
        assert result.results.eval_results[0].output == 1.0

    def test_rag_response_without_citations_fails(self, evaluator):
        """RAG response without citations should fail citation check."""
        response = "The API supports REST endpoints and uses OAuth 2.0 for authentication."

        result = evaluator.evaluate(
            "regex",
            [{"response": response}],
            {"pattern": r"\[\d+\]"}
        )
        assert result.results.eval_results[0].output == 0.0

    def test_rag_response_length_validation(self, evaluator):
        """RAG response should be within acceptable length range."""
        short_response = "Yes."
        good_response = "Based on the documentation, the API supports multiple authentication methods including OAuth 2.0 and API keys. You can configure these in the settings panel."
        long_response = "x" * 3000

        result = evaluator.evaluate(
            "length_between",
            [
                {"response": short_response},
                {"response": good_response},
                {"response": long_response},
            ],
            {"min_length": 50, "max_length": 2000}
        )

        assert result.results.eval_results[0].output == 0.0  # Too short
        assert result.results.eval_results[1].output == 1.0  # Good
        assert result.results.eval_results[2].output == 0.0  # Too long

    def test_rag_response_no_cop_outs(self, evaluator):
        """RAG response should not contain cop-out phrases."""
        good_response = "The feature you're looking for is available in version 2.0."
        cop_out_response = "I don't know the answer to that question."

        result = evaluator.evaluate(
            "contains_none",
            [
                {"response": good_response},
                {"response": cop_out_response},
            ],
            {"keywords": ["I don't know", "I cannot", "I'm not sure", "no information"]}
        )

        assert result.results.eval_results[0].output == 1.0  # Good
        assert result.results.eval_results[1].output == 0.0  # Has cop-out

    def test_rag_response_mentions_sources(self, evaluator):
        """RAG response should mention provided sources."""
        sources = ["docs.example.com", "api.example.com", "guide.example.com"]
        response = "According to docs.example.com, the feature is supported. See api.example.com for details."

        result = evaluator.evaluate(
            "contains_any",
            [{"response": response}],
            {"keywords": sources}
        )
        assert result.results.eval_results[0].output == 1.0


class TestChatbotQualityGates:
    """Test Case 2: Chatbot Response Quality Gates.

    Ensures chatbot responses meet quality standards including
    length, professionalism, and no PII leakage.
    """

    @pytest.fixture
    def evaluator(self):
        return LocalEvaluator()

    def test_minimum_response_length(self, evaluator):
        """Chatbot response should meet minimum length requirement."""
        too_short = "OK"
        good_length = "I'd be happy to help you with your account settings. You can find them in the top menu."

        result = evaluator.evaluate(
            "length_greater_than",
            [
                {"response": too_short},
                {"response": good_length},
            ],
            {"min_length": 20}
        )

        assert result.results.eval_results[0].output == 0.0  # Too short
        assert result.results.eval_results[1].output == 1.0  # Good

    def test_no_pii_keywords(self, evaluator):
        """Response should not contain PII-related keywords."""
        safe_response = "Your account has been updated successfully."
        unsafe_response = "Your SSN 123-45-6789 has been recorded."

        result = evaluator.evaluate(
            "contains_none",
            [
                {"response": safe_response},
                {"response": unsafe_response},
            ],
            {"keywords": ["SSN", "social security", "credit card", "password"]}
        )

        assert result.results.eval_results[0].output == 1.0  # Safe
        assert result.results.eval_results[1].output == 0.0  # Contains PII keyword

    def test_no_email_leakage(self, evaluator):
        """Response should not leak email addresses."""
        safe_response = "Please contact our support team for assistance."
        leaky_response = "You can reach John at john.doe@company.com for help."

        result = evaluator.evaluate(
            "contains_email",
            [
                {"response": safe_response},
                {"response": leaky_response},
            ]
        )

        # contains_email returns 1.0 if email found, we want 0.0 (no email)
        assert result.results.eval_results[0].output == 0.0  # No email (good)
        assert result.results.eval_results[1].output == 1.0  # Has email (bad)

    def test_professional_tone(self, evaluator):
        """Response should maintain professional tone."""
        professional = "I understand your concern and I'm here to help resolve this issue."
        unprofessional = "lol that's a weird bug, omg let me check"

        result = evaluator.evaluate(
            "contains_none",
            [
                {"response": professional.lower()},
                {"response": unprofessional.lower()},
            ],
            {"keywords": ["lol", "omg", "wtf", "damn", "crap"]}
        )

        assert result.results.eval_results[0].output == 1.0  # Professional
        assert result.results.eval_results[1].output == 0.0  # Unprofessional

    def test_concise_response_for_simple_query(self, evaluator):
        """Simple queries should get concise (single paragraph) responses."""
        concise = "You can change your password in Settings > Security > Change Password."
        verbose = "You can change your password.\n\nFirst, go to Settings.\n\nThen click Security.\n\nFinally, click Change Password."

        result = evaluator.evaluate(
            "one_line",
            [
                {"response": concise},
                {"response": verbose},
            ]
        )

        assert result.results.eval_results[0].output == 1.0  # Concise
        assert result.results.eval_results[1].output == 0.0  # Too verbose


class TestCodeGenerationValidation:
    """Test Case 3: Code Generation Validation.

    Validates LLM-generated code for security issues and proper structure.
    """

    @pytest.fixture
    def evaluator(self):
        return LocalEvaluator()

    def test_no_dangerous_python_patterns(self, evaluator):
        """Generated Python code should not contain dangerous patterns."""
        safe_code = """
def calculate_sum(numbers):
    return sum(numbers)
"""
        dangerous_code = """
def run_command(cmd):
    return eval(cmd)
"""

        result = evaluator.evaluate(
            "contains_none",
            [
                {"response": safe_code},
                {"response": dangerous_code},
            ],
            {"keywords": ["eval(", "exec(", "os.system(", "__import__"]}
        )

        assert result.results.eval_results[0].output == 1.0  # Safe
        assert result.results.eval_results[1].output == 0.0  # Dangerous

    def test_no_hardcoded_secrets(self, evaluator):
        """Generated code should not contain hardcoded secrets."""
        safe_code = """
import os
api_key = os.environ.get("API_KEY")
"""
        unsafe_code = """
api_key = "sk-1234567890abcdef"
password = "supersecret123"
"""

        result = evaluator.evaluate(
            "regex",
            [
                {"response": safe_code},
                {"response": unsafe_code},
            ],
            {"pattern": r"(api[_-]?key|password|secret)\s*=\s*['\"][^'\"]+['\"]"}
        )

        assert result.results.eval_results[0].output == 0.0  # No secrets
        assert result.results.eval_results[1].output == 1.0  # Has secrets

    def test_python_code_structure(self, evaluator):
        """Python code should have proper structure (imports, functions, etc.)."""
        proper_code = """
import json

def parse_data(data):
    return json.loads(data)
"""
        no_structure = "x = 1 + 2"

        result = evaluator.evaluate(
            "contains_any",
            [
                {"response": proper_code},
                {"response": no_structure},
            ],
            {"keywords": ["def ", "class ", "import ", "from "]}
        )

        assert result.results.eval_results[0].output == 1.0  # Has structure
        assert result.results.eval_results[1].output == 0.0  # No structure

    def test_no_sql_injection_patterns(self, evaluator):
        """SQL code should not contain injection vulnerabilities."""
        safe_sql = "SELECT * FROM users WHERE id = ?"
        dangerous_sql = "SELECT * FROM users; DROP TABLE users; --"

        result = evaluator.evaluate(
            "contains_none",
            [
                {"response": safe_sql},
                {"response": dangerous_sql},
            ],
            {"keywords": ["DROP TABLE", "DELETE FROM", "; --"]}
        )

        assert result.results.eval_results[0].output == 1.0  # Safe
        assert result.results.eval_results[1].output == 0.0  # Dangerous

    def test_code_length_reasonable(self, evaluator):
        """Generated code should be within reasonable length."""
        good_code = "def add(a, b):\n    return a + b"
        too_short = "x"
        too_long = "x = 1\n" * 1000

        result = evaluator.evaluate(
            "length_between",
            [
                {"response": good_code},
                {"response": too_short},
                {"response": too_long},
            ],
            {"min_length": 10, "max_length": 5000}
        )

        assert result.results.eval_results[0].output == 1.0  # Good
        assert result.results.eval_results[1].output == 0.0  # Too short
        assert result.results.eval_results[2].output == 0.0  # Too long


class TestAPIResponseContractTesting:
    """Test Case 4: API Response Contract Testing.

    Validates that LLM API responses match expected JSON schemas.
    """

    @pytest.fixture
    def evaluator(self):
        return LocalEvaluator()

    def test_valid_json_response(self, evaluator):
        """API response should be valid JSON."""
        valid_json = '{"message": "Hello", "status": "ok"}'
        invalid_json = '{"message": "Hello", status: "ok"}'  # Missing quotes

        result = evaluator.evaluate(
            "is_json",
            [
                {"response": valid_json},
                {"response": invalid_json},
            ]
        )

        assert result.results.eval_results[0].output == 1.0  # Valid
        assert result.results.eval_results[1].output == 0.0  # Invalid

    def test_chat_endpoint_schema(self, evaluator):
        """Chat endpoint response should match schema."""
        chat_schema = {
            "type": "object",
            "required": ["message", "conversation_id"],
            "properties": {
                "message": {"type": "string", "minLength": 1},
                "conversation_id": {"type": "string"},
                "tokens_used": {"type": "integer"}
            }
        }

        valid_response = '{"message": "Hello!", "conversation_id": "abc123", "tokens_used": 50}'
        missing_field = '{"message": "Hello!"}'  # Missing conversation_id

        result = evaluator.evaluate(
            "json_schema",
            [
                {"response": valid_response, "schema": chat_schema},
                {"response": missing_field, "schema": chat_schema},
            ]
        )

        assert result.results.eval_results[0].output == 1.0  # Valid
        assert result.results.eval_results[1].output == 0.0  # Missing field

    def test_summarize_endpoint_schema(self, evaluator):
        """Summarize endpoint response should match schema."""
        schema = {
            "type": "object",
            "required": ["summary", "key_points"],
            "properties": {
                "summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}}
            }
        }

        valid = '{"summary": "A brief summary", "key_points": ["point 1", "point 2"]}'
        invalid_type = '{"summary": "A brief summary", "key_points": "not an array"}'

        result = evaluator.evaluate(
            "json_schema",
            [
                {"response": valid, "schema": schema},
                {"response": invalid_type, "schema": schema},
            ]
        )

        assert result.results.eval_results[0].output == 1.0  # Valid
        assert result.results.eval_results[1].output == 0.0  # Wrong type

    def test_no_error_indicators(self, evaluator):
        """Response should not contain error indicators."""
        success_response = '{"status": "ok", "data": {"id": 123}}'
        error_response = '{"status": "error", "message": "Something failed"}'

        result = evaluator.evaluate(
            "contains_none",
            [
                {"response": success_response.lower()},
                {"response": error_response.lower()},
            ],
            {"keywords": ["error", "failed", "exception"]}
        )

        assert result.results.eval_results[0].output == 1.0  # No errors
        assert result.results.eval_results[1].output == 0.0  # Has error


class TestTranslationQualityAssessment:
    """Test Case 5: Translation Quality Assessment.

    Evaluates machine translation quality using similarity metrics.
    """

    @pytest.fixture
    def evaluator(self):
        return LocalEvaluator()

    def test_bleu_score_identical_text(self, evaluator):
        """Identical texts should have high BLEU score."""
        reference = "The quick brown fox jumps over the lazy dog."
        translation = "The quick brown fox jumps over the lazy dog."

        result = evaluator.evaluate(
            "bleu_score",
            [{"response": translation, "expected_response": reference}],
            {"mode": "sentence"}
        )

        assert result.results.eval_results[0].output >= 0.9

    def test_bleu_score_similar_text(self, evaluator):
        """Similar texts should have lower BLEU score than identical."""
        reference = "The quick brown fox jumps over the lazy dog."
        translation = "The fast brown fox leaps over the sleepy dog."

        result = evaluator.evaluate(
            "bleu_score",
            [{"response": translation, "expected_response": reference}],
            {"mode": "sentence"}
        )

        score = result.results.eval_results[0].output
        # BLEU is strict on exact n-gram matches, so similar but different words score low
        assert 0.0 <= score <= 0.5  # Lower score due to word substitutions

    def test_bleu_score_different_text(self, evaluator):
        """Completely different texts should have low BLEU score."""
        reference = "The quick brown fox jumps over the lazy dog."
        translation = "Hello world, this is a test."

        result = evaluator.evaluate(
            "bleu_score",
            [{"response": translation, "expected_response": reference}],
            {"mode": "sentence"}
        )

        assert result.results.eval_results[0].output < 0.3

    def test_rouge_score_for_summarization(self, evaluator):
        """ROUGE score for summarization quality."""
        reference = "The cat sat on the mat. It was a sunny day."
        good_summary = "The cat sat on the mat on a sunny day."
        bad_summary = "Dogs like to play fetch in the park."

        result = evaluator.evaluate(
            "rouge_score",
            [
                {"response": good_summary, "expected_response": reference},
                {"response": bad_summary, "expected_response": reference},
            ],
            {"rouge_type": "rougeL"}
        )

        assert result.results.eval_results[0].output > result.results.eval_results[1].output

    def test_levenshtein_similarity(self, evaluator):
        """Levenshtein similarity for character-level comparison."""
        reference = "Hello world"
        similar = "Hello World"  # Just case difference
        different = "Goodbye universe"

        result = evaluator.evaluate(
            "levenshtein_similarity",
            [
                {"response": similar, "expected_response": reference},
                {"response": different, "expected_response": reference},
            ]
        )

        # Similar should have higher score (case insensitive by default)
        assert result.results.eval_results[0].output > result.results.eval_results[1].output

    def test_translation_length_ratio(self, evaluator):
        """Translation should be similar length to reference."""
        reference = "This is a medium length sentence for testing."  # ~45 chars

        good_translation = "This is also a medium length translated text."  # Similar
        too_short = "Short."
        too_long = "This is an extremely long translation that goes on and on with unnecessary details and extra words that make it much longer than the original."

        result = evaluator.evaluate(
            "length_between",
            [
                {"response": good_translation},
                {"response": too_short},
                {"response": too_long},
            ],
            {"min_length": int(len(reference) * 0.5), "max_length": int(len(reference) * 1.5)}
        )

        assert result.results.eval_results[0].output == 1.0  # Good length
        assert result.results.eval_results[1].output == 0.0  # Too short
        assert result.results.eval_results[2].output == 0.0  # Too long


class TestContentModerationPreScreening:
    """Test Case 6: Content Moderation Pre-Screening.

    Fast local pre-screening before expensive cloud moderation.
    """

    @pytest.fixture
    def evaluator(self):
        return LocalEvaluator()

    def test_blocked_keywords_detection(self, evaluator):
        """Content with blocked keywords should be flagged."""
        safe_content = "Here's a great recipe for chocolate cake!"
        blocked_content = "Learn how to hack into computer systems"

        blocked_keywords = ["hack", "exploit", "illegal", "weapon"]

        result = evaluator.evaluate(
            "contains_none",
            [
                {"response": safe_content.lower()},
                {"response": blocked_content.lower()},
            ],
            {"keywords": blocked_keywords}
        )

        assert result.results.eval_results[0].output == 1.0  # Safe
        assert result.results.eval_results[1].output == 0.0  # Blocked

    def test_commercial_content_detection(self, evaluator):
        """Commercial content patterns should be flagged for review."""
        regular_content = "I enjoyed reading this article about cooking."
        commercial_content = "buy now for only $99! Limited time offer!"  # lowercase for regex match

        result = evaluator.evaluate(
            "regex",
            [
                {"response": regular_content},
                {"response": commercial_content},
            ],
            {"pattern": r"(buy|sell|price).*\$\d+"}
        )

        assert result.results.eval_results[0].output == 0.0  # No commercial
        assert result.results.eval_results[1].output == 1.0  # Commercial detected

    def test_external_links_detection(self, evaluator):
        """Content with external links should be flagged for review."""
        no_links = "Check out the documentation for more information."
        has_links = "Visit https://example.com for more details."

        result = evaluator.evaluate(
            "contains_link",
            [
                {"response": no_links},
                {"response": has_links},
            ]
        )

        assert result.results.eval_results[0].output == 0.0  # No links
        assert result.results.eval_results[1].output == 1.0  # Has links

    def test_content_length_limits(self, evaluator):
        """Content should be within acceptable length limits."""
        good_content = "This is a normal piece of content that's reasonable in length."
        empty_content = ""
        spam_content = "buy buy buy " * 1000

        result = evaluator.evaluate(
            "length_between",
            [
                {"response": good_content},
                {"response": empty_content},
                {"response": spam_content},
            ],
            {"min_length": 1, "max_length": 10000}
        )

        assert result.results.eval_results[0].output == 1.0  # Good
        assert result.results.eval_results[1].output == 0.0  # Empty
        assert result.results.eval_results[2].output == 0.0  # Too long


class TestDataExtractionValidation:
    """Test Case 7: Data Extraction Validation.

    Validates that extracted data from documents matches expected formats.
    """

    @pytest.fixture
    def evaluator(self):
        return LocalEvaluator()

    def test_invoice_number_format(self, evaluator):
        """Invoice numbers should match expected format."""
        valid_invoice = "INV-12345"
        valid_invoice_alt = "ABC-1234567890"
        invalid_invoice = "12345"

        result = evaluator.evaluate(
            "regex",
            [
                {"response": valid_invoice},
                {"response": valid_invoice_alt},
                {"response": invalid_invoice},
            ],
            {"pattern": r"^[A-Z]{2,3}-\d{4,10}$"}
        )

        assert result.results.eval_results[0].output == 1.0  # Valid
        assert result.results.eval_results[1].output == 1.0  # Valid alt format
        assert result.results.eval_results[2].output == 0.0  # Invalid

    def test_date_format(self, evaluator):
        """Extracted dates should be in ISO format."""
        valid_date = "2024-01-15"
        invalid_date = "01/15/2024"
        invalid_date_2 = "January 15, 2024"

        result = evaluator.evaluate(
            "regex",
            [
                {"response": valid_date},
                {"response": invalid_date},
                {"response": invalid_date_2},
            ],
            {"pattern": r"^\d{4}-\d{2}-\d{2}$"}
        )

        assert result.results.eval_results[0].output == 1.0  # Valid ISO
        assert result.results.eval_results[1].output == 0.0  # Wrong format
        assert result.results.eval_results[2].output == 0.0  # Wrong format

    def test_currency_format(self, evaluator):
        """Currency values should match expected format."""
        valid_amounts = ["$1234.56", "$99.99", "$1000"]
        invalid_amounts = ["1234.56", "USD 100", "100 dollars"]

        valid_inputs = [{"response": amt} for amt in valid_amounts]
        invalid_inputs = [{"response": amt} for amt in invalid_amounts]

        result = evaluator.evaluate(
            "regex",
            valid_inputs + invalid_inputs,
            {"pattern": r"^\$\d+\.?\d{0,2}$"}
        )

        # First 3 should pass
        for i in range(3):
            assert result.results.eval_results[i].output == 1.0
        # Last 3 should fail
        for i in range(3, 6):
            assert result.results.eval_results[i].output == 0.0

    def test_extracted_json_valid(self, evaluator):
        """Extracted data should be valid JSON."""
        extracted_data = {
            "invoice_number": "INV-12345",
            "date": "2024-01-15",
            "total": "$1234.56",
            "items": [{"name": "Widget", "qty": 10}]
        }

        result = evaluator.evaluate(
            "is_json",
            [{"response": json.dumps(extracted_data)}]
        )

        assert result.results.eval_results[0].output == 1.0

    def test_extracted_data_schema(self, evaluator):
        """Extracted invoice data should match schema."""
        invoice_schema = {
            "type": "object",
            "required": ["invoice_number", "date", "total"],
            "properties": {
                "invoice_number": {"type": "string"},
                "date": {"type": "string"},
                "total": {"type": "string"},
                "vendor": {"type": "string"}
            }
        }

        complete_data = json.dumps({
            "invoice_number": "INV-123",
            "date": "2024-01-15",
            "total": "$100.00",
            "vendor": "Acme"
        })

        incomplete_data = json.dumps({
            "invoice_number": "INV-123"
            # Missing required fields
        })

        result = evaluator.evaluate(
            "json_schema",
            [
                {"response": complete_data, "schema": invoice_schema},
                {"response": incomplete_data, "schema": invoice_schema},
            ]
        )

        assert result.results.eval_results[0].output == 1.0  # Complete
        assert result.results.eval_results[1].output == 0.0  # Incomplete


class TestPromptEngineeringIteration:
    """Test Case 8: Prompt Engineering Iteration.

    Rapidly test prompt variations during development.
    """

    @pytest.fixture
    def evaluator(self):
        return LocalEvaluator()

    def test_response_contains_expected_elements(self, evaluator):
        """Generated response should contain expected elements."""
        response_with_elements = "Here is a summary with the following key points: First, the main idea. Second, supporting details."
        response_missing_elements = "Here is some information about the topic."

        result = evaluator.evaluate(
            "contains_all",
            [
                {"response": response_with_elements},
                {"response": response_missing_elements},
            ],
            {"keywords": ["summary", "key points"]}
        )

        assert result.results.eval_results[0].output == 1.0  # Has elements
        assert result.results.eval_results[1].output == 0.0  # Missing elements

    def test_response_no_forbidden_phrases(self, evaluator):
        """Response should not contain forbidden phrases."""
        good_response = "Based on the available information, the answer is X."
        bad_response = "I cannot answer that question. I don't know the answer."

        result = evaluator.evaluate(
            "contains_none",
            [
                {"response": good_response},
                {"response": bad_response},
            ],
            {"keywords": ["I cannot", "I don't know", "I'm not able"]}
        )

        assert result.results.eval_results[0].output == 1.0  # Good
        assert result.results.eval_results[1].output == 0.0  # Has forbidden

    def test_response_length_constraints(self, evaluator):
        """Response should meet length constraints."""
        responses = [
            "Short.",  # Too short
            "This is a good response that provides enough detail without being excessive.",
            "x" * 1000  # Too long
        ]

        result = evaluator.evaluate(
            "length_between",
            [{"response": r} for r in responses],
            {"min_length": 20, "max_length": 500}
        )

        assert result.results.eval_results[0].output == 0.0  # Too short
        assert result.results.eval_results[1].output == 1.0  # Good
        assert result.results.eval_results[2].output == 0.0  # Too long

    def test_response_format_structured(self, evaluator):
        """Response should follow structured format when required."""
        structured = "1. First point\n2. Second point\n3. Third point"
        unstructured = "Here are some points about the topic mixed together."

        result = evaluator.evaluate(
            "regex",
            [
                {"response": structured},
                {"response": unstructured},
            ],
            {"pattern": r"\d+\.\s"}  # Numbered list pattern
        )

        assert result.results.eval_results[0].output == 1.0  # Structured
        assert result.results.eval_results[1].output == 0.0  # Unstructured


class TestHybridModeRouting:
    """Test hybrid mode routing for mixed evaluation scenarios."""

    @pytest.fixture
    def hybrid(self):
        return HybridEvaluator()

    def test_partition_mixed_evaluations(self, hybrid):
        """Hybrid mode should correctly partition local and cloud metrics."""
        evaluations = [
            {"metric_name": "contains", "inputs": [{"response": "test"}]},
            {"metric_name": "is_json", "inputs": [{"response": "{}"}]},
            {"metric_name": "regex", "inputs": [{"response": "test"}]},
            {"metric_name": "groundedness", "inputs": [{"response": "test"}]},
            {"metric_name": "hallucination", "inputs": [{"response": "test"}]},
            {"metric_name": "context_adherence", "inputs": [{"response": "test"}]},
        ]

        partitions = hybrid.partition_evaluations(evaluations)

        # Local metrics
        local_names = [e["metric_name"] for e in partitions[ExecutionMode.LOCAL]]
        assert "contains" in local_names
        assert "is_json" in local_names
        assert "regex" in local_names

        # Cloud metrics
        cloud_names = [e["metric_name"] for e in partitions[ExecutionMode.CLOUD]]
        assert "groundedness" in cloud_names
        assert "hallucination" in cloud_names
        assert "context_adherence" in cloud_names

    def test_execute_local_partition(self, hybrid):
        """Local partition should execute successfully."""
        evaluations = [
            {
                "metric_name": "contains",
                "inputs": [{"response": "Hello world"}],
                "config": {"keyword": "world"}
            },
            {
                "metric_name": "is_json",
                "inputs": [{"response": '{"key": "value"}'}]
            },
        ]

        result = hybrid.evaluate_local_partition(evaluations)

        assert len(result.results.eval_results) == 2
        assert result.results.eval_results[0].output == 1.0
        assert result.results.eval_results[1].output == 1.0
        assert "contains" in result.executed_locally
        assert "is_json" in result.executed_locally


class TestBatchEvaluationScenarios:
    """Test batch evaluation for real-world scenarios."""

    @pytest.fixture
    def evaluator(self):
        return LocalEvaluator()

    def test_multi_metric_quality_check(self, evaluator):
        """Run multiple quality checks on a single response."""
        response = '{"status": "success", "message": "Operation completed", "data": {"id": 123}}'

        result = evaluator.evaluate_batch([
            # Check it's valid JSON
            {"metric_name": "is_json", "inputs": [{"response": response}]},
            # Check it contains success indicator
            {"metric_name": "contains", "inputs": [{"response": response}], "config": {"keyword": "success"}},
            # Check it's not too short
            {"metric_name": "length_greater_than", "inputs": [{"response": response}], "config": {"min_length": 20}},
            # Check no error indicators
            {"metric_name": "contains_none", "inputs": [{"response": response.lower()}], "config": {"keywords": ["error", "failed"]}},
        ])

        # All checks should pass
        assert all(r.output == 1.0 for r in result.results.eval_results)

    def test_multi_response_validation(self, evaluator):
        """Validate multiple responses with the same criteria."""
        responses = [
            '{"valid": true, "data": "test1"}',
            '{"valid": true, "data": "test2"}',
            'invalid json',
            '{"valid": false}',
        ]

        result = evaluator.evaluate(
            "is_json",
            [{"response": r} for r in responses]
        )

        assert result.results.eval_results[0].output == 1.0  # Valid
        assert result.results.eval_results[1].output == 1.0  # Valid
        assert result.results.eval_results[2].output == 0.0  # Invalid
        assert result.results.eval_results[3].output == 1.0  # Valid (even if data says false)

    def test_comprehensive_content_validation(self, evaluator):
        """Comprehensive content validation pipeline."""
        content = "Welcome to our service! We're here to help you get started. Check out our documentation at docs.example.com for more details."

        validations = evaluator.evaluate_batch([
            # Length check
            {"metric_name": "length_between", "inputs": [{"response": content}], "config": {"min_length": 50, "max_length": 500}},
            # Has greeting
            {"metric_name": "contains_any", "inputs": [{"response": content}], "config": {"keywords": ["Welcome", "Hello", "Hi"]}},
            # Professional (no informal language)
            {"metric_name": "contains_none", "inputs": [{"response": content.lower()}], "config": {"keywords": ["lol", "omg", "gonna"]}},
            # Contains link
            {"metric_name": "contains_link", "inputs": [{"response": content}]},
        ])

        results = validations.results.eval_results
        assert results[0].output == 1.0  # Good length
        assert results[1].output == 1.0  # Has greeting
        assert results[2].output == 1.0  # Professional
        assert results[3].output == 1.0  # Has link
