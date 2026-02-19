# Local Execution Guide

> **Language Support:** Python ✅ | TypeScript 🚧 | Go 📋 | Java 📋

This guide explains how to use the local execution feature to run evaluations without API calls, enabling offline evaluation and faster feedback loops during development.

## Overview

The local execution module allows you to run heuristic metrics (string matching, JSON validation, similarity scores, etc.) locally on your machine without requiring network access or API credentials. This is particularly useful for:

- **Offline development**: Work on evaluations without internet connectivity
- **Fast iteration**: Get immediate feedback without API latency
- **Cost savings**: Reduce API calls during development and testing
- **CI/CD pipelines**: Run quick validation checks before cloud-based evaluations

## Execution Modes

The SDK supports three execution modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `LOCAL` | Run all evaluations locally using heuristic metrics only | Offline development, fast feedback |
| `CLOUD` | Run all evaluations via the cloud API | Production evaluations, LLM-based metrics |
| `HYBRID` | Automatically route each evaluation based on metric type | Best of both worlds |

## Available Local Metrics

### String Metrics

| Metric | Description | Config |
|--------|-------------|--------|
| `regex` | Check if response matches a regex pattern | `pattern: str` |
| `contains` | Check if response contains a keyword | `keyword: str`, `case_sensitive: bool` |
| `contains_all` | Check if response contains all keywords | `keywords: list[str]` |
| `contains_any` | Check if response contains any keyword | `keywords: list[str]` |
| `contains_none` | Check if response contains no forbidden keywords | `keywords: list[str]` |
| `one_line` | Check if response is a single line | - |
| `equals` | Check if response equals expected text | `case_sensitive: bool` |
| `starts_with` | Check if response starts with expected text | `case_sensitive: bool` |
| `ends_with` | Check if response ends with expected text | `case_sensitive: bool` |
| `length_less_than` | Check if response length is below threshold | `max_length: int` |
| `length_greater_than` | Check if response length is above threshold | `min_length: int` |
| `length_between` | Check if response length is within range | `min_length: int`, `max_length: int` |
| `contains_email` | Check if response contains an email address | - |
| `is_email` | Check if response is a valid email address | - |
| `contains_link` | Check if response contains a URL | - |
| `contains_valid_link` | Check if response contains a reachable URL | - |

### JSON Metrics

| Metric | Description | Config |
|--------|-------------|--------|
| `contains_json` | Check if response contains valid JSON | - |
| `is_json` | Check if entire response is valid JSON | - |
| `json_schema` | Validate response against JSON schema | `schema: dict` (in input) |

### Similarity Metrics

| Metric | Description | Config |
|--------|-------------|--------|
| `bleu_score` | Calculate BLEU score for translation quality | `mode: str`, `max_n_gram: int` |
| `rouge_score` | Calculate ROUGE score for summarization | `rouge_type: str`, `use_stemmer: bool` |
| `recall_score` | Calculate recall for retrieved items | - |
| `levenshtein_similarity` | Calculate normalized Levenshtein similarity | `case_insensitive: bool` |
| `numeric_similarity` | Compare numeric values in text | - |
| `embedding_similarity` | Calculate semantic similarity using embeddings | `model_name: str`, `similarity_method: str` |
| `semantic_list_contains` | Check for semantically similar phrases | `similarity_threshold: float` |

## Quick Start

### Basic Usage

```python
from fi.evals.local import LocalEvaluator

# Create a local evaluator
evaluator = LocalEvaluator()

# Run a simple contains check
result = evaluator.evaluate(
    metric_name="contains",
    inputs=[{"response": "The quick brown fox jumps over the lazy dog"}],
    config={"keyword": "fox"}
)

print(result.results.eval_results[0].output)  # 1.0 (found)
print(result.results.eval_results[0].reason)  # "Keyword 'fox' found"
```

### Checking Metric Availability

```python
evaluator = LocalEvaluator()

# Check if specific metrics can run locally
print(evaluator.can_run_locally("contains"))      # True
print(evaluator.can_run_locally("is_json"))       # True
print(evaluator.can_run_locally("groundedness"))  # False (requires LLM)

# List all available local metrics
print(evaluator.list_available_metrics())
# ['bleu_score', 'contains', 'contains_all', 'contains_any', ...]
```

### Batch Evaluation

```python
# Evaluate multiple metrics at once
result = evaluator.evaluate_batch([
    {
        "metric_name": "contains",
        "inputs": [{"response": "Hello world"}],
        "config": {"keyword": "world"},
    },
    {
        "metric_name": "is_json",
        "inputs": [
            {"response": '{"valid": true}'},
            {"response": "not json"},
        ],
    },
])

# Results are combined
for eval_result in result.results.eval_results:
    print(f"{eval_result.name}: {eval_result.output}")
```

## Hybrid Mode

Hybrid mode automatically routes metrics to the appropriate execution environment:

```python
from fi.evals.local import HybridEvaluator, ExecutionMode

hybrid = HybridEvaluator()

# Define a mixed set of evaluations
evaluations = [
    {"metric_name": "contains", "inputs": [{"response": "test"}], "config": {"keyword": "test"}},
    {"metric_name": "is_json", "inputs": [{"response": "{}"}]},
    {"metric_name": "groundedness", "inputs": [{"response": "The sky is blue", "context": "..."}]},
    {"metric_name": "hallucination", "inputs": [{"response": "test"}]},
]

# Partition by execution capability
partitions = hybrid.partition_evaluations(evaluations)

print(f"Local: {len(partitions[ExecutionMode.LOCAL])} evaluations")
# Local: 2 evaluations (contains, is_json)

print(f"Cloud: {len(partitions[ExecutionMode.CLOUD])} evaluations")
# Cloud: 2 evaluations (groundedness, hallucination)

# Run local evaluations immediately
local_results = hybrid.evaluate_local_partition(partitions[ExecutionMode.LOCAL])

# Send cloud evaluations to API (using your existing cloud evaluator)
# cloud_results = cloud_evaluator.evaluate(partitions[ExecutionMode.CLOUD])
```

## Configuration

### LocalEvaluatorConfig

```python
from fi.evals.local import LocalEvaluator, LocalEvaluatorConfig, ExecutionMode

config = LocalEvaluatorConfig(
    execution_mode=ExecutionMode.LOCAL,  # Default mode
    fail_on_unsupported=True,            # Raise error for non-local metrics
    parallel_workers=4,                   # For future parallel execution
    timeout=60,                           # Timeout per evaluation in seconds
)

evaluator = LocalEvaluator(config=config)
```

### Error Handling

By default, unsupported metrics are skipped:

```python
evaluator = LocalEvaluator()

result = evaluator.evaluate(
    metric_name="groundedness",  # LLM metric, can't run locally
    inputs=[{"response": "test"}]
)

# Returns empty result with skip marker
print(result.skipped)  # {'groundedness'}
print(result.results.eval_results[0].output)  # None
print(result.results.eval_results[0].reason)  # "Metric 'groundedness' cannot run locally"
```

To raise an error instead:

```python
config = LocalEvaluatorConfig(fail_on_unsupported=True)
evaluator = LocalEvaluator(config=config)

try:
    evaluator.evaluate("groundedness", [...])
except ValueError as e:
    print(e)  # "Metric 'groundedness' cannot run locally..."
```

## Real-World Use Cases

### 1. RAG Pipeline Validation

Validate that your RAG (Retrieval-Augmented Generation) pipeline returns properly formatted responses with citations:

```python
from fi.evals.local import LocalEvaluator

evaluator = LocalEvaluator()

def validate_rag_response(response: str, sources: list[str]) -> dict:
    """Validate RAG response format and content quality."""

    results = evaluator.evaluate_batch([
        # Check response isn't too short or too long
        {
            "metric_name": "length_between",
            "inputs": [{"response": response}],
            "config": {"min_length": 50, "max_length": 2000}
        },
        # Ensure response contains citation markers like [1], [2]
        {
            "metric_name": "regex",
            "inputs": [{"response": response}],
            "config": {"pattern": r"\[\d+\]"}
        },
        # Check that source URLs are mentioned
        {
            "metric_name": "contains_any",
            "inputs": [{"response": response}],
            "config": {"keywords": sources[:3]}  # Check first 3 sources
        },
        # Ensure no "I don't know" cop-outs
        {
            "metric_name": "contains_none",
            "inputs": [{"response": response}],
            "config": {"keywords": ["I don't know", "I cannot", "I'm not sure", "no information"]}
        },
    ])

    checks = {
        "length_ok": results.results.eval_results[0].output == 1.0,
        "has_citations": results.results.eval_results[1].output == 1.0,
        "mentions_sources": results.results.eval_results[2].output == 1.0,
        "no_cop_outs": results.results.eval_results[3].output == 1.0,
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "score": sum(checks.values()) / len(checks)
    }

# Usage
response = """Based on the documentation [1], the API supports both REST and GraphQL
endpoints. According to the developer guide [2], authentication uses OAuth 2.0..."""

result = validate_rag_response(response, ["docs.example.com", "api.example.com"])
print(f"Validation passed: {result['passed']}, Score: {result['score']:.0%}")
```

### 2. Chatbot Response Quality Gates

Ensure chatbot responses meet quality standards before deployment:

```python
from fi.evals.local import LocalEvaluator
from dataclasses import dataclass

@dataclass
class QualityGate:
    name: str
    passed: bool
    reason: str

def validate_chatbot_response(response: str, user_query: str) -> list[QualityGate]:
    """Run quality gates on chatbot responses."""

    evaluator = LocalEvaluator()
    gates = []

    # Gate 1: Response shouldn't be empty or too short
    result = evaluator.evaluate(
        "length_greater_than",
        [{"response": response}],
        {"min_length": 20}
    )
    gates.append(QualityGate(
        name="Minimum Length",
        passed=result.results.eval_results[0].output == 1.0,
        reason=result.results.eval_results[0].reason
    ))

    # Gate 2: Response should be single paragraph for simple queries
    if len(user_query) < 100:
        result = evaluator.evaluate(
            "one_line",
            [{"response": response.replace('\n\n', '\n')}]  # Normalize
        )
        gates.append(QualityGate(
            name="Concise Response",
            passed=result.results.eval_results[0].output == 1.0,
            reason="Response should be concise for simple queries"
        ))

    # Gate 3: No PII patterns (basic check)
    result = evaluator.evaluate(
        "contains_none",
        [{"response": response}],
        {"keywords": ["SSN", "social security", "credit card", "password"]}
    )
    gates.append(QualityGate(
        name="No PII Keywords",
        passed=result.results.eval_results[0].output == 1.0,
        reason=result.results.eval_results[0].reason
    ))

    # Gate 4: No email addresses leaked
    result = evaluator.evaluate(
        "contains_email",
        [{"response": response}]
    )
    gates.append(QualityGate(
        name="No Email Leakage",
        passed=result.results.eval_results[0].output == 0.0,  # Should NOT contain
        reason="Response should not contain email addresses"
    ))

    # Gate 5: Professional tone indicators
    result = evaluator.evaluate(
        "contains_none",
        [{"response": response.lower()}],
        {"keywords": ["lol", "omg", "wtf", "damn", "crap"]}
    )
    gates.append(QualityGate(
        name="Professional Tone",
        passed=result.results.eval_results[0].output == 1.0,
        reason=result.results.eval_results[0].reason
    ))

    return gates

# Usage
response = "I'd be happy to help you with your account settings. You can access them by clicking on your profile icon in the top right corner."
gates = validate_chatbot_response(response, "How do I change my settings?")

for gate in gates:
    status = "✓" if gate.passed else "✗"
    print(f"{status} {gate.name}: {gate.reason}")
```

### 3. Code Generation Validation

Validate LLM-generated code before execution:

```python
from fi.evals.local import LocalEvaluator
import json

def validate_generated_code(code: str, language: str = "python") -> dict:
    """Validate generated code for common issues."""

    evaluator = LocalEvaluator()
    issues = []

    # Check for dangerous patterns
    dangerous_patterns = {
        "python": ["eval(", "exec(", "os.system(", "__import__", "subprocess.call("],
        "javascript": ["eval(", "Function(", "document.write(", "innerHTML"],
        "sql": ["DROP TABLE", "DELETE FROM", "--", "/*"],
    }

    patterns = dangerous_patterns.get(language, [])
    if patterns:
        result = evaluator.evaluate(
            "contains_none",
            [{"response": code}],
            {"keywords": patterns, "case_sensitive": False}
        )
        if result.results.eval_results[0].output == 0.0:
            issues.append(f"Contains potentially dangerous patterns")

    # Check for hardcoded secrets patterns
    result = evaluator.evaluate(
        "regex",
        [{"response": code}],
        {"pattern": r"(api[_-]?key|password|secret|token)\s*=\s*['\"][^'\"]+['\"]"}
    )
    if result.results.eval_results[0].output == 1.0:
        issues.append("Contains hardcoded secrets")

    # Check code length is reasonable
    result = evaluator.evaluate(
        "length_between",
        [{"response": code}],
        {"min_length": 10, "max_length": 10000}
    )
    if result.results.eval_results[0].output == 0.0:
        issues.append("Code length is unusual")

    # For Python, check for proper function structure
    if language == "python":
        result = evaluator.evaluate(
            "contains_any",
            [{"response": code}],
            {"keywords": ["def ", "class ", "import ", "from "]}
        )
        if result.results.eval_results[0].output == 0.0:
            issues.append("Missing standard Python structures")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "code_length": len(code)
    }

# Usage
generated_code = '''
def calculate_total(items):
    """Calculate total price of items."""
    return sum(item["price"] * item["quantity"] for item in items)
'''

result = validate_generated_code(generated_code, "python")
print(f"Valid: {result['valid']}, Issues: {result['issues']}")
```

### 4. API Response Contract Testing

Validate that your LLM-powered API returns responses matching the expected contract:

```python
from fi.evals.local import LocalEvaluator
import json

def validate_api_response(response_body: str, endpoint: str) -> dict:
    """Validate API response matches contract."""

    evaluator = LocalEvaluator()
    validations = {}

    # Check response is valid JSON
    result = evaluator.evaluate("is_json", [{"response": response_body}])
    validations["valid_json"] = result.results.eval_results[0].output == 1.0

    if not validations["valid_json"]:
        return {"valid": False, "validations": validations, "error": "Invalid JSON"}

    # Parse and validate structure
    data = json.loads(response_body)
    response_str = json.dumps(data)

    # Define schemas per endpoint
    schemas = {
        "/api/chat": {
            "type": "object",
            "required": ["message", "conversation_id"],
            "properties": {
                "message": {"type": "string", "minLength": 1},
                "conversation_id": {"type": "string"},
                "tokens_used": {"type": "integer"}
            }
        },
        "/api/summarize": {
            "type": "object",
            "required": ["summary", "key_points"],
            "properties": {
                "summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}}
            }
        },
        "/api/extract": {
            "type": "object",
            "required": ["entities"],
            "properties": {
                "entities": {"type": "array"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }
    }

    schema = schemas.get(endpoint)
    if schema:
        result = evaluator.evaluate(
            "json_schema",
            [{"response": response_str, "schema": schema}]
        )
        validations["schema_valid"] = result.results.eval_results[0].output == 1.0

    # Check for error indicators in response
    result = evaluator.evaluate(
        "contains_none",
        [{"response": response_str.lower()}],
        {"keywords": ["error", "failed", "exception", "null"]}
    )
    validations["no_error_indicators"] = result.results.eval_results[0].output == 1.0

    return {
        "valid": all(validations.values()),
        "validations": validations
    }

# Usage
response = '{"message": "Here is your summary...", "conversation_id": "abc123", "tokens_used": 150}'
result = validate_api_response(response, "/api/chat")
print(f"API Response Valid: {result['valid']}")
```

### 5. Translation Quality Assessment

Evaluate machine translation quality using similarity metrics:

```python
from fi.evals.local import LocalEvaluator

def assess_translation_quality(
    translation: str,
    reference: str,
    source: str = None
) -> dict:
    """Assess translation quality using multiple metrics."""

    evaluator = LocalEvaluator()

    results = evaluator.evaluate_batch([
        # BLEU score for n-gram precision
        {
            "metric_name": "bleu_score",
            "inputs": [{"response": translation, "expected_response": reference}],
            "config": {"mode": "sentence", "max_n_gram": 4}
        },
        # ROUGE score for recall
        {
            "metric_name": "rouge_score",
            "inputs": [{"response": translation, "expected_response": reference}],
            "config": {"rouge_type": "rougeL"}
        },
        # Levenshtein for character-level similarity
        {
            "metric_name": "levenshtein_similarity",
            "inputs": [{"response": translation, "expected_response": reference}]
        },
        # Length ratio check (translation shouldn't be drastically different length)
        {
            "metric_name": "length_between",
            "inputs": [{"response": translation}],
            "config": {
                "min_length": int(len(reference) * 0.5),
                "max_length": int(len(reference) * 1.5)
            }
        }
    ])

    scores = {
        "bleu": results.results.eval_results[0].output,
        "rouge_l": results.results.eval_results[1].output,
        "levenshtein": results.results.eval_results[2].output,
        "length_ok": results.results.eval_results[3].output == 1.0
    }

    # Weighted overall score
    overall = (
        scores["bleu"] * 0.4 +
        scores["rouge_l"] * 0.4 +
        scores["levenshtein"] * 0.2
    )

    quality_tier = (
        "excellent" if overall >= 0.8 else
        "good" if overall >= 0.6 else
        "acceptable" if overall >= 0.4 else
        "poor"
    )

    return {
        "scores": scores,
        "overall": overall,
        "quality_tier": quality_tier,
        "length_ok": scores["length_ok"]
    }

# Usage
reference = "The quick brown fox jumps over the lazy dog."
translation = "The fast brown fox leaps over the sleepy dog."

result = assess_translation_quality(translation, reference)
print(f"Quality: {result['quality_tier']} (score: {result['overall']:.2f})")
print(f"BLEU: {result['scores']['bleu']:.2f}, ROUGE-L: {result['scores']['rouge_l']:.2f}")
```

### 6. Content Moderation Pre-Screening

Fast local pre-screening before expensive cloud-based moderation:

```python
from fi.evals.local import LocalEvaluator, HybridEvaluator, ExecutionMode

class ContentModerator:
    """Two-stage content moderation with local pre-screening."""

    def __init__(self):
        self.local = LocalEvaluator()
        self.hybrid = HybridEvaluator()

        # Keywords that trigger immediate rejection
        self.blocked_keywords = [
            "hack", "exploit", "illegal", "weapon",
            # Add your blocked terms
        ]

        # Patterns that need review
        self.review_patterns = [
            r"\b(buy|sell|price)\b.*\$\d+",  # Commercial content
            r"http[s]?://",  # External links
        ]

    def pre_screen(self, content: str) -> dict:
        """Fast local pre-screening."""

        # Check for blocked keywords
        result = self.local.evaluate(
            "contains_none",
            [{"response": content.lower()}],
            {"keywords": self.blocked_keywords}
        )

        if result.results.eval_results[0].output == 0.0:
            return {
                "action": "block",
                "reason": "Contains blocked keywords",
                "needs_cloud_review": False
            }

        # Check for patterns needing review
        for pattern in self.review_patterns:
            result = self.local.evaluate(
                "regex",
                [{"response": content}],
                {"pattern": pattern}
            )
            if result.results.eval_results[0].output == 1.0:
                return {
                    "action": "review",
                    "reason": f"Matches pattern: {pattern}",
                    "needs_cloud_review": True
                }

        # Check content length
        result = self.local.evaluate(
            "length_between",
            [{"response": content}],
            {"min_length": 1, "max_length": 10000}
        )

        if result.results.eval_results[0].output == 0.0:
            return {
                "action": "review",
                "reason": "Unusual content length",
                "needs_cloud_review": True
            }

        return {
            "action": "pass",
            "reason": "Passed local pre-screening",
            "needs_cloud_review": False  # Or True for additional safety
        }

    def moderate(self, content: str) -> dict:
        """Full moderation pipeline."""

        # Stage 1: Local pre-screening (fast, no API)
        pre_screen_result = self.pre_screen(content)

        if pre_screen_result["action"] == "block":
            return pre_screen_result

        if not pre_screen_result["needs_cloud_review"]:
            return pre_screen_result

        # Stage 2: Cloud moderation (for flagged content only)
        # This would use your cloud evaluator for content_moderation, toxicity, etc.
        # cloud_result = cloud_evaluator.evaluate(...)

        return {
            "action": "pending_cloud_review",
            "local_result": pre_screen_result,
            # "cloud_result": cloud_result
        }

# Usage
moderator = ContentModerator()

contents = [
    "Check out this amazing recipe for chocolate cake!",
    "Learn how to hack into systems...",
    "Buy now for only $99.99 at http://example.com",
]

for content in contents:
    result = moderator.pre_screen(content)
    print(f"Content: {content[:50]}...")
    print(f"Action: {result['action']}, Reason: {result['reason']}\n")
```

### 7. Data Extraction Validation

Validate that extracted data from documents matches expected formats:

```python
from fi.evals.local import LocalEvaluator
import json

def validate_extracted_data(
    extracted: dict,
    document_type: str
) -> dict:
    """Validate extracted data from documents."""

    evaluator = LocalEvaluator()
    validations = []

    # Define validation rules per document type
    rules = {
        "invoice": {
            "required_fields": ["invoice_number", "date", "total", "vendor"],
            "patterns": {
                "invoice_number": r"^[A-Z]{2,3}-\d{4,10}$",
                "date": r"^\d{4}-\d{2}-\d{2}$",
                "total": r"^\$?\d+\.?\d{0,2}$"
            }
        },
        "receipt": {
            "required_fields": ["store", "date", "items", "total"],
            "patterns": {
                "date": r"^\d{4}-\d{2}-\d{2}$",
                "total": r"^\$?\d+\.?\d{0,2}$"
            }
        },
        "contract": {
            "required_fields": ["parties", "effective_date", "terms"],
            "patterns": {
                "effective_date": r"^\d{4}-\d{2}-\d{2}$"
            }
        }
    }

    doc_rules = rules.get(document_type, {})

    # Check required fields exist
    for field in doc_rules.get("required_fields", []):
        exists = field in extracted and extracted[field] is not None
        validations.append({
            "check": f"required_field_{field}",
            "passed": exists,
            "message": f"Field '{field}' {'present' if exists else 'missing'}"
        })

    # Validate field patterns
    for field, pattern in doc_rules.get("patterns", {}).items():
        if field in extracted and extracted[field]:
            result = evaluator.evaluate(
                "regex",
                [{"response": str(extracted[field])}],
                {"pattern": pattern}
            )
            passed = result.results.eval_results[0].output == 1.0
            validations.append({
                "check": f"pattern_{field}",
                "passed": passed,
                "message": f"Field '{field}' {'matches' if passed else 'does not match'} expected pattern"
            })

    # Validate extracted JSON is well-formed
    result = evaluator.evaluate(
        "is_json",
        [{"response": json.dumps(extracted)}]
    )
    validations.append({
        "check": "valid_json",
        "passed": result.results.eval_results[0].output == 1.0,
        "message": "Extracted data is valid JSON"
    })

    passed_count = sum(1 for v in validations if v["passed"])

    return {
        "valid": all(v["passed"] for v in validations),
        "score": passed_count / len(validations) if validations else 0,
        "validations": validations
    }

# Usage
extracted_invoice = {
    "invoice_number": "INV-12345",
    "date": "2024-01-15",
    "total": "$1,234.56",
    "vendor": "Acme Corp",
    "items": [{"name": "Widget", "qty": 10, "price": 123.45}]
}

result = validate_extracted_data(extracted_invoice, "invoice")
print(f"Valid: {result['valid']}, Score: {result['score']:.0%}")
for v in result["validations"]:
    status = "✓" if v["passed"] else "✗"
    print(f"  {status} {v['check']}: {v['message']}")
```

### 8. Prompt Engineering Iteration

Rapidly test prompt variations during development:

```python
from fi.evals.local import LocalEvaluator
from typing import Callable
import time

class PromptTester:
    """Test prompt variations with fast local evaluation."""

    def __init__(self, llm_call: Callable[[str], str]):
        self.evaluator = LocalEvaluator()
        self.llm_call = llm_call
        self.results = []

    def test_prompt(
        self,
        prompt: str,
        test_inputs: list[str],
        expected_patterns: list[str] = None,
        forbidden_patterns: list[str] = None,
        max_length: int = None,
        min_length: int = None
    ) -> dict:
        """Test a prompt variation against criteria."""

        responses = []
        for test_input in test_inputs:
            full_prompt = prompt.format(input=test_input)
            response = self.llm_call(full_prompt)
            responses.append(response)

        metrics = {"responses": len(responses), "checks": {}}

        # Check expected patterns
        if expected_patterns:
            result = self.evaluator.evaluate(
                "contains_all",
                [{"response": r} for r in responses],
                {"keywords": expected_patterns}
            )
            passed = sum(1 for r in result.results.eval_results if r.output == 1.0)
            metrics["checks"]["expected_patterns"] = f"{passed}/{len(responses)}"

        # Check forbidden patterns
        if forbidden_patterns:
            result = self.evaluator.evaluate(
                "contains_none",
                [{"response": r} for r in responses],
                {"keywords": forbidden_patterns}
            )
            passed = sum(1 for r in result.results.eval_results if r.output == 1.0)
            metrics["checks"]["no_forbidden"] = f"{passed}/{len(responses)}"

        # Check length constraints
        if max_length or min_length:
            config = {}
            if min_length:
                config["min_length"] = min_length
            if max_length:
                config["max_length"] = max_length

            result = self.evaluator.evaluate(
                "length_between",
                [{"response": r} for r in responses],
                config
            )
            passed = sum(1 for r in result.results.eval_results if r.output == 1.0)
            metrics["checks"]["length_ok"] = f"{passed}/{len(responses)}"

        # Calculate overall score
        total_checks = sum(
            int(v.split("/")[0])
            for v in metrics["checks"].values()
        )
        total_possible = sum(
            int(v.split("/")[1])
            for v in metrics["checks"].values()
        )
        metrics["score"] = total_checks / total_possible if total_possible > 0 else 0

        self.results.append({"prompt": prompt[:100], "metrics": metrics})
        return metrics

    def compare_prompts(self) -> None:
        """Compare all tested prompts."""
        print("\nPrompt Comparison:")
        print("-" * 60)
        for i, r in enumerate(sorted(self.results, key=lambda x: -x["metrics"]["score"])):
            print(f"{i+1}. Score: {r['metrics']['score']:.0%}")
            print(f"   Prompt: {r['prompt']}...")
            print(f"   Checks: {r['metrics']['checks']}")
            print()

# Usage (with mock LLM for example)
def mock_llm(prompt: str) -> str:
    return f"Based on the input, here is a summary with key points: Point 1, Point 2, Point 3."

tester = PromptTester(mock_llm)

# Test different prompt variations
prompts = [
    "Summarize this: {input}",
    "Provide a brief summary with key points: {input}",
    "As an expert, summarize the following text concisely: {input}",
]

test_inputs = ["Long document text here...", "Another document..."]

for prompt in prompts:
    result = tester.test_prompt(
        prompt,
        test_inputs,
        expected_patterns=["summary", "key points"],
        forbidden_patterns=["I cannot", "I don't know"],
        min_length=50,
        max_length=500
    )
    print(f"Prompt: {prompt[:40]}... Score: {result['score']:.0%}")

tester.compare_prompts()
```

## Integration Examples

### CI/CD Pipeline

```yaml
# .github/workflows/evaluation.yml
- name: Run local evaluations
  run: |
    python -c "
    from fi.evals.local import LocalEvaluator

    evaluator = LocalEvaluator()
    result = evaluator.evaluate_batch([
        {'metric_name': 'is_json', 'inputs': test_cases},
        {'metric_name': 'length_between', 'inputs': test_cases, 'config': {'min_length': 10, 'max_length': 1000}},
    ])

    # Check all passed
    passed = all(r.output == 1.0 for r in result.results.eval_results if r.output is not None)
    exit(0 if passed else 1)
    "
```

### Pre-commit Hook

```python
# .pre-commit-config.yaml hook script
from fi.evals.local import LocalEvaluator

def check_outputs():
    evaluator = LocalEvaluator()

    # Load your test outputs
    outputs = load_test_outputs()

    # Run quick validation
    result = evaluator.evaluate(
        metric_name="is_json",
        inputs=[{"response": out} for out in outputs]
    )

    failures = [r for r in result.results.eval_results if r.output == 0.0]
    if failures:
        print(f"Found {len(failures)} invalid JSON outputs")
        return False
    return True
```

### Development Workflow

```python
from fi.evals.local import LocalEvaluator, HybridEvaluator

# During development, use local evaluator for quick iteration
local = LocalEvaluator()

# Test your responses quickly
responses = generate_responses(prompts)
for response in responses:
    result = local.evaluate(
        "contains_json",
        [{"response": response}]
    )
    if result.results.eval_results[0].output == 0.0:
        print(f"Warning: Response doesn't contain JSON: {response[:100]}...")

# Before deployment, use hybrid to run full evaluation
hybrid = HybridEvaluator()
partitions = hybrid.partition_evaluations(full_eval_suite)

# Quick local checks pass
local_results = hybrid.evaluate_local_partition(partitions[ExecutionMode.LOCAL])

# Then run expensive cloud evaluations
# cloud_results = evaluator.evaluate(partitions[ExecutionMode.CLOUD])
```

## Metric Input Format

All metrics expect inputs as dictionaries with the following fields:

### TextMetricInput

```python
{
    "response": str,                    # Required: The text to evaluate
    "expected_response": str | list,    # Optional: Reference text(s) for comparison
}
```

### JsonMetricInput

```python
{
    "response": str | dict | list,      # Required: JSON to validate
    "expected_response": str | dict,    # Optional: Expected JSON
    "schema": dict,                     # Optional: JSON schema for validation
}
```

## Performance Considerations

1. **Lazy Loading**: Similarity metrics with heavy dependencies (sentence-transformers, NLTK) are loaded only when first used
2. **Batch Processing**: Use `evaluate_batch()` to process multiple inputs efficiently
3. **Memory**: Embedding models are cached after first load

## Troubleshooting

### "Metric cannot run locally"

The metric requires LLM evaluation. Use `CLOUD` or `HYBRID` mode, or check `evaluator.list_available_metrics()` for local alternatives.

### "Input validation failed"

Check that your input dictionary has the required fields (`response` for most metrics). See the Metric Input Format section.

### Similarity metrics are slow

First invocation loads ML models. Subsequent calls are faster. Consider pre-warming in production:

```python
evaluator = LocalEvaluator()
# Pre-load embedding model
evaluator.evaluate("embedding_similarity", [
    {"response": "warmup", "expected_response": "warmup"}
])
```
