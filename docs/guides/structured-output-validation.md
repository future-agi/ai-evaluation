# Structured Output Validation Guide

> **Language Support:** Python ✅ | TypeScript 🚧 | Go 📋 | Java 📋

This guide explains how to use the structured output validation metrics to evaluate LLM-generated structured data (JSON, YAML, Pydantic models).

## Overview

Structured output validation is critical for applications where LLMs must generate machine-readable data. These metrics help you evaluate:

- **Syntax validity**: Is the output parseable?
- **Schema compliance**: Does it match the expected structure?
- **Field completeness**: Are all required fields present?
- **Type correctness**: Do values have the correct types?
- **Structural similarity**: How similar is the structure to expected output?

## Available Metrics

### JSON Validation Metrics

| Metric | Description | Score Range |
|--------|-------------|-------------|
| `JSONValidation` | Comprehensive JSON validation with schema | 0.0 - 1.0 |
| `JSONSyntaxOnly` | Simple syntax check only | 0.0 or 1.0 |

### Schema Compliance Metrics

| Metric | Description | Score Range |
|--------|-------------|-------------|
| `SchemaCompliance` | Generic schema validation with breakdown | 0.0 - 1.0 |
| `TypeCompliance` | Type-only validation (lenient) | 0.0 - 1.0 |

### Field Metrics

| Metric | Description | Score Range |
|--------|-------------|-------------|
| `FieldCompleteness` | Required + optional field presence | 0.0 - 1.0 |
| `RequiredFieldsOnly` | Only required field check | 0.0 - 1.0 |
| `FieldCoverage` | Coverage vs expected output | 0.0 - 1.0 |

### Hierarchy Metrics

| Metric | Description | Score Range |
|--------|-------------|-------------|
| `HierarchyScore` | Tree-based structural similarity | 0.0 - 1.0 |
| `TreeEditDistance` | Normalized edit distance (lower = better) | 0.0 - 1.0 |

### Composite Metrics

| Metric | Description | Score Range |
|--------|-------------|-------------|
| `StructuredOutputScore` | Comprehensive weighted score | 0.0 - 1.0 |
| `QuickStructuredCheck` | Fast lightweight validation | 0.0 - 1.0 |

## Quick Start

### Basic JSON Validation

```python
from fi.evals.metrics.structured import JSONValidation

metric = JSONValidation()
result = metric.evaluate([{
    "response": '{"name": "Alice", "age": 30}',
    "schema": {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
}])

print(result.eval_results[0].output)  # 1.0 (fully valid)
```

### Comprehensive Validation

```python
from fi.evals.metrics.structured import StructuredOutputScore

metric = StructuredOutputScore()
result = metric.evaluate([{
    "response": '{"user": {"name": "Alice"}, "status": "active"}',
    "format": "json",
    "schema": {
        "type": "object",
        "required": ["user", "status"],
        "properties": {
            "user": {
                "type": "object",
                "required": ["name", "email"],
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"}
                }
            },
            "status": {"type": "string"}
        }
    }
}])

# Access score breakdown
print(result.eval_results[0].output)  # 0.85 (partial compliance)
```

## Real-World Examples

### 1. REST API Response Validation

Validate that LLM-generated API responses match your schema:

```python
from fi.evals.metrics.structured import JSONValidation
import json

metric = JSONValidation()

# LLM-generated user API response
api_response = json.dumps({
    "id": 12345,
    "username": "john_doe",
    "email": "john@example.com",
    "profile": {
        "first_name": "John",
        "last_name": "Doe",
        "avatar_url": "https://example.com/avatar.jpg"
    },
    "settings": {
        "notifications_enabled": True,
        "theme": "dark"
    },
    "created_at": "2024-01-15T10:30:00Z"
})

user_schema = {
    "type": "object",
    "required": ["id", "username", "email"],
    "properties": {
        "id": {"type": "integer"},
        "username": {"type": "string", "minLength": 3},
        "email": {"type": "string"},
        "profile": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "avatar_url": {"type": "string"}
            }
        },
        "settings": {
            "type": "object",
            "properties": {
                "notifications_enabled": {"type": "boolean"},
                "theme": {"type": "string"}
            }
        },
        "created_at": {"type": "string"}
    }
}

result = metric.evaluate([{
    "response": api_response,
    "schema": user_schema
}])

print(f"Score: {result.eval_results[0].output}")  # 1.0
```

### 2. LLM Function Calling Validation

Validate function call outputs from LLMs:

```python
from fi.evals.metrics.structured import JSONValidation
import json

metric = JSONValidation()

function_call = json.dumps({
    "name": "get_weather",
    "arguments": {
        "location": "San Francisco, CA",
        "unit": "celsius"
    }
})

function_schema = {
    "type": "object",
    "required": ["name", "arguments"],
    "properties": {
        "name": {"type": "string"},
        "arguments": {
            "type": "object",
            "required": ["location"],
            "properties": {
                "location": {"type": "string"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            }
        }
    }
}

result = metric.evaluate([{
    "response": function_call,
    "schema": function_schema
}])

print(f"Valid function call: {result.eval_results[0].output == 1.0}")
```

### 3. E-Commerce Product Catalog

Validate product listing responses:

```python
from fi.evals.metrics.structured import StructuredOutputScore
import json

metric = StructuredOutputScore()

product_response = json.dumps({
    "products": [
        {
            "id": "prod_123",
            "name": "Wireless Headphones",
            "price": {"amount": 99.99, "currency": "USD"},
            "in_stock": True,
            "categories": ["electronics", "audio"],
            "ratings": {"average": 4.5, "count": 128}
        }
    ],
    "pagination": {
        "page": 1,
        "per_page": 20,
        "total": 156
    }
})

product_schema = {
    "type": "object",
    "required": ["products", "pagination"],
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "price"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "price": {
                        "type": "object",
                        "required": ["amount", "currency"],
                        "properties": {
                            "amount": {"type": "number"},
                            "currency": {"type": "string"}
                        }
                    },
                    "in_stock": {"type": "boolean"},
                    "categories": {"type": "array", "items": {"type": "string"}},
                    "ratings": {"type": "object"}
                }
            }
        },
        "pagination": {
            "type": "object",
            "required": ["page", "total"],
            "properties": {
                "page": {"type": "integer"},
                "per_page": {"type": "integer"},
                "total": {"type": "integer"}
            }
        }
    }
}

result = metric.evaluate([{
    "response": product_response,
    "format": "json",
    "schema": product_schema
}])

print(f"Catalog score: {result.eval_results[0].output}")  # ~0.95+
```

### 4. ML Model Prediction Output

Validate classification outputs from ML models:

```python
from fi.evals.metrics.structured import SchemaCompliance
import json

metric = SchemaCompliance()

classification = json.dumps({
    "label": "positive",
    "confidence": 0.87,
    "all_scores": {
        "positive": 0.87,
        "negative": 0.08,
        "neutral": 0.05
    }
})

prediction_schema = {
    "type": "object",
    "required": ["label", "confidence"],
    "properties": {
        "label": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        },
        "all_scores": {
            "type": "object",
            "additionalProperties": {"type": "number"}
        }
    }
}

result = metric.evaluate([{
    "response": classification,
    "format": "json",
    "schema": prediction_schema
}])

print(f"Prediction valid: {result.eval_results[0].output}")  # 1.0
```

### 5. YAML Configuration Validation

Validate Kubernetes-style YAML configurations:

```python
from fi.evals.metrics.structured import SchemaCompliance

metric = SchemaCompliance()

k8s_config = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    spec:
      containers:
      - name: web
        image: nginx:latest
        ports:
        - containerPort: 80
"""

k8s_schema = {
    "type": "object",
    "required": ["apiVersion", "kind", "metadata", "spec"],
    "properties": {
        "apiVersion": {"type": "string"},
        "kind": {"type": "string"},
        "metadata": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "labels": {"type": "object"}
            }
        },
        "spec": {
            "type": "object",
            "properties": {
                "replicas": {"type": "integer"},
                "selector": {"type": "object"},
                "template": {"type": "object"}
            }
        }
    }
}

result = metric.evaluate([{
    "response": k8s_config,
    "format": "yaml",
    "schema": k8s_schema
}])

print(f"K8s config valid: {result.eval_results[0].output >= 0.9}")
```

### 6. Entity Extraction Output

Validate NER/entity extraction outputs:

```python
from fi.evals.metrics.structured import HierarchyScore
import json

metric = HierarchyScore()

llm_output = json.dumps({
    "entities": [
        {"text": "Apple Inc.", "type": "ORG", "start": 0, "end": 10},
        {"text": "Tim Cook", "type": "PERSON", "start": 15, "end": 23},
        {"text": "California", "type": "LOC", "start": 40, "end": 50}
    ],
    "relationships": [
        {"subject": "Tim Cook", "predicate": "CEO_OF", "object": "Apple Inc."}
    ]
})

# Expected structure (values don't matter, just structure)
expected_structure = {
    "entities": [
        {"text": "", "type": "", "start": 0, "end": 0}
    ],
    "relationships": [
        {"subject": "", "predicate": "", "object": ""}
    ]
}

result = metric.evaluate([{
    "response": llm_output,
    "expected": expected_structure
}])

print(f"Structure similarity: {result.eval_results[0].output}")  # ~0.7+
```

### 7. Chain-of-Thought Structured Output

Validate structured CoT reasoning:

```python
from fi.evals.metrics.structured import FieldCompleteness
import json

metric = FieldCompleteness()

cot_output = json.dumps({
    "thinking": [
        "First, I need to understand the problem.",
        "The key insight is that we can use dynamic programming.",
        "Time complexity will be O(n^2)."
    ],
    "answer": 42,
    "confidence": 0.95,
    "reasoning_type": "mathematical"
})

cot_schema = {
    "type": "object",
    "required": ["thinking", "answer", "confidence"],
    "properties": {
        "thinking": {
            "type": "array",
            "items": {"type": "string"}
        },
        "answer": {},
        "confidence": {"type": "number"},
        "reasoning_type": {"type": "string"}
    }
}

result = metric.evaluate([{
    "response": cot_output,
    "format": "json",
    "schema": cot_schema
}])

print(f"CoT completeness: {result.eval_results[0].output}")  # ~0.9+
```

### 8. RAG Retrieval Output Validation

Validate RAG pipeline outputs:

```python
from fi.evals.metrics.structured import SchemaCompliance
import json

metric = SchemaCompliance()

rag_output = json.dumps({
    "query": "What is machine learning?",
    "retrieved_documents": [
        {
            "id": "doc_1",
            "content": "Machine learning is a subset of AI...",
            "score": 0.95,
            "metadata": {"source": "wikipedia", "date": "2024-01-01"}
        },
        {
            "id": "doc_2",
            "content": "ML algorithms learn from data...",
            "score": 0.88,
            "metadata": {"source": "textbook"}
        }
    ],
    "generated_answer": "Machine learning is a branch of AI that enables computers to learn from data."
})

rag_schema = {
    "type": "object",
    "required": ["query", "retrieved_documents", "generated_answer"],
    "properties": {
        "query": {"type": "string"},
        "retrieved_documents": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "content", "score"],
                "properties": {
                    "id": {"type": "string"},
                    "content": {"type": "string"},
                    "score": {"type": "number"},
                    "metadata": {"type": "object"}
                }
            }
        },
        "generated_answer": {"type": "string"}
    }
}

result = metric.evaluate([{
    "response": rag_output,
    "format": "json",
    "schema": rag_schema
}])

print(f"RAG output valid: {result.eval_results[0].output}")  # 1.0
```

## Pydantic Model Validation

For Python applications, you can validate against Pydantic models:

```python
from pydantic import BaseModel
from typing import List, Optional
from fi.evals.metrics.structured import PydanticValidator

class Address(BaseModel):
    city: str
    country: str
    zip_code: Optional[str] = None

class Person(BaseModel):
    name: str
    age: int
    addresses: List[Address]

validator = PydanticValidator(model_class=Person)

result = validator.validate_model(
    '{"name": "Alice", "age": 30, "addresses": [{"city": "NYC", "country": "USA"}]}',
    Person
)

print(f"Valid: {result.valid}")
print(f"Parsed: {result.parsed}")
```

## Validation Modes

Three validation modes are available:

| Mode | Description |
|------|-------------|
| `STRICT` | Exact match - correct types, no extra fields |
| `COERCE` | Allow type coercion (e.g., "30" → 30) |
| `LENIENT` | Allow extra fields, flexible types |

```python
from fi.evals.metrics.structured import JSONValidation, ValidationMode

metric = JSONValidation()

# Strict mode - type must be exact
result_strict = metric.evaluate([{
    "response": '{"age": "30"}',  # String instead of int
    "schema": {"type": "object", "properties": {"age": {"type": "integer"}}},
    "mode": "strict"
}])

# Coerce mode - allows "30" to be coerced to 30
result_coerce = metric.evaluate([{
    "response": '{"age": "30"}',
    "schema": {"type": "object", "properties": {"age": {"type": "integer"}}},
    "mode": "coerce"
}])
```

## Batch Processing

Process multiple outputs efficiently:

```python
from fi.evals.metrics.structured import JSONValidation

metric = JSONValidation()

inputs = [
    {"response": '{"valid": true}'},
    {"response": '{"also": "valid"}'},
    {"response": 'invalid json'},
    {"response": '{"another": 1}'}
]

result = metric.evaluate(inputs)

for i, eval_result in enumerate(result.eval_results):
    print(f"Input {i}: score={eval_result.output}")
# Input 0: score=1.0
# Input 1: score=1.0
# Input 2: score=0.0
# Input 3: score=1.0
```

## Configuration Options

### JSONValidation

```python
metric = JSONValidation(config={
    "syntax_weight": 0.3,       # Weight for syntax validity
    "schema_weight": 0.5,       # Weight for schema compliance
    "completeness_weight": 0.2  # Weight for field completeness
})
```

### StructuredOutputScore

```python
metric = StructuredOutputScore(config={
    "syntax_weight": 0.2,
    "schema_weight": 0.3,
    "completeness_weight": 0.25,
    "type_weight": 0.15,
    "value_weight": 0.1
})
```

### HierarchyScore

```python
metric = HierarchyScore(config={
    "key_weight": 0.5,   # Weight for key matching
    "type_weight": 0.3,  # Weight for type matching
    "depth_weight": 0.2, # Weight for depth similarity
    "max_depth": 10      # Maximum recursion depth
})
```

## Error Handling

All metrics return detailed error information:

```python
from fi.evals.metrics.structured import JSONValidation

metric = JSONValidation()
result = metric.evaluate([{
    "response": '{"name": 123}',  # Wrong type
    "schema": {
        "type": "object",
        "properties": {"name": {"type": "string"}}
    }
}])

eval_result = result.eval_results[0]
print(f"Score: {eval_result.output}")
print(f"Reason: {eval_result.reason}")
# Access detailed errors if available in the metric output
```

## Best Practices

1. **Use appropriate strictness**: Start with `COERCE` mode and tighten to `STRICT` only when needed.

2. **Define complete schemas**: Include all expected fields in your schema for accurate completeness scoring.

3. **Validate early**: Use `QuickStructuredCheck` for fast validation in high-throughput scenarios.

4. **Batch similar inputs**: Group similar validation requests for efficient processing.

5. **Use HierarchyScore for flexible matching**: When exact values don't matter, HierarchyScore compares structure only.

6. **Handle nested structures**: All metrics support deeply nested objects and arrays.

## Dependencies

Required:
- `pydantic>=2.0.0` (for type definitions)

Optional:
- `jsonschema>=4.0.0` (for advanced JSON Schema validation)
- `PyYAML>=6.0` (for YAML support)
