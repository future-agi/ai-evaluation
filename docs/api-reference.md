# API Reference

> REST API reference for AI Evaluation

---

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Response Formats](#response-formats)
- [Error Handling](#error-handling)
- [Rate Limits](#rate-limits)

---

## Overview

The AI Evaluation API provides programmatic access to LLM evaluation capabilities. While the SDK is recommended for most use cases, the REST API enables direct integration.

**Base URL:**
```
https://api.futureagi.com
```

**Content Type:**
```
Content-Type: application/json
```

---

## Authentication

All API requests require authentication using API key and secret key.

### Headers

```http
Authorization: Bearer <api_key>
X-Secret-Key: <secret_key>
```

### Example

```bash
curl -X POST "https://api.futureagi.com/v2/evaluate" \
  -H "Authorization: Bearer sk-fi-your-api-key" \
  -H "X-Secret-Key: sk-secret-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

---

## Endpoints

### POST /v2/evaluate

Run an evaluation.

**Request Body:**

```json
{
  "eval_name": "groundedness",
  "inputs": {
    "context": ["The sky is blue due to Rayleigh scattering."],
    "output": ["The sky appears blue because of light scattering."]
  },
  "model": "turing_flash",
  "span_id": null,
  "custom_eval_name": null,
  "trace_eval": false,
  "is_async": false,
  "error_localizer": false
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `eval_name` | string | Yes | Evaluation template name |
| `inputs` | object | Yes | Input data (values as arrays) |
| `model` | string | Yes | Model to use |
| `span_id` | string | No | OpenTelemetry span ID |
| `custom_eval_name` | string | No | Custom name for tracing |
| `trace_eval` | boolean | No | Enable tracing |
| `is_async` | boolean | No | Run asynchronously |
| `error_localizer` | boolean | No | Enable error localization |

**Response:**

```json
{
  "result": [
    {
      "evaluations": [
        {
          "name": "groundedness",
          "output": "GROUNDED",
          "reason": "The response accurately reflects the context about light scattering.",
          "runtime": 1.234,
          "outputType": "string",
          "evalId": "eval_abc123",
          "metadata": {
            "usage": {
              "prompt_tokens": 150,
              "completion_tokens": 50,
              "total_tokens": 200
            },
            "cost": {
              "total": 0.001
            }
          }
        }
      ]
    }
  ]
}
```

**Example:**

```bash
curl -X POST "https://api.futureagi.com/v2/evaluate" \
  -H "Authorization: Bearer $FI_API_KEY" \
  -H "X-Secret-Key: $FI_SECRET_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "eval_name": "groundedness",
    "inputs": {
      "context": ["Paris is the capital of France."],
      "output": ["The capital of France is Paris."]
    },
    "model": "turing_flash"
  }'
```

---

### GET /v2/evaluate/templates

List all available evaluation templates.

**Response:**

```json
{
  "result": [
    {
      "name": "groundedness",
      "eval_id": "47",
      "description": "Check if response is grounded in context",
      "eval_tags": ["rag", "hallucination"],
      "config": {
        "required_keys": ["context", "output"],
        "output": "string",
        "eval_type_id": "llm"
      }
    },
    {
      "name": "toxicity",
      "eval_id": "15",
      "description": "Detect toxic content",
      "eval_tags": ["safety"],
      "config": {
        "required_keys": ["text"],
        "output": "string",
        "eval_type_id": "llm"
      }
    }
  ]
}
```

**Example:**

```bash
curl -X GET "https://api.futureagi.com/v2/evaluate/templates" \
  -H "Authorization: Bearer $FI_API_KEY" \
  -H "X-Secret-Key: $FI_SECRET_KEY"
```

---

### GET /v2/evaluate/result

Get result of an async evaluation.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `eval_id` | string | Yes | Evaluation ID |

**Response:**

```json
{
  "result": {
    "status": "completed",
    "evaluations": [
      {
        "name": "groundedness",
        "output": "GROUNDED",
        "reason": "...",
        "runtime": 1.5
      }
    ]
  }
}
```

**Example:**

```bash
curl -X GET "https://api.futureagi.com/v2/evaluate/result?eval_id=eval_abc123" \
  -H "Authorization: Bearer $FI_API_KEY" \
  -H "X-Secret-Key: $FI_SECRET_KEY"
```

---

### POST /v2/evaluate/pipeline

Evaluate a pipeline version.

**Request Body:**

```json
{
  "project_name": "my-chatbot",
  "version": "v1.0.0",
  "eval_data": [
    {
      "eval_name": "groundedness",
      "inputs": {
        "context": ["..."],
        "output": ["..."]
      }
    }
  ]
}
```

**Example:**

```bash
curl -X POST "https://api.futureagi.com/v2/evaluate/pipeline" \
  -H "Authorization: Bearer $FI_API_KEY" \
  -H "X-Secret-Key: $FI_SECRET_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "my-chatbot",
    "version": "v1.0.0",
    "eval_data": [...]
  }'
```

---

### GET /v2/evaluate/pipeline

Get pipeline evaluation results.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_name` | string | Yes | Project name |
| `versions` | string | Yes | Comma-separated versions |

**Example:**

```bash
curl -X GET "https://api.futureagi.com/v2/evaluate/pipeline?project_name=my-chatbot&versions=v1.0.0,v1.1.0" \
  -H "Authorization: Bearer $FI_API_KEY" \
  -H "X-Secret-Key: $FI_SECRET_KEY"
```

---

## Response Formats

### Success Response

```json
{
  "result": [
    {
      "evaluations": [
        {
          "name": "template_name",
          "output": "RESULT",
          "reason": "Explanation of the result",
          "runtime": 1.234,
          "outputType": "string",
          "evalId": "unique_id",
          "metadata": {
            "usage": {...},
            "cost": {...}
          }
        }
      ]
    }
  ]
}
```

### Evaluation Result Object

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Template name |
| `output` | any | Evaluation result (varies by template) |
| `reason` | string | Explanation |
| `runtime` | number | Execution time (seconds) |
| `outputType` | string | Type of output |
| `evalId` | string | Unique ID |
| `metadata` | object | Additional metadata |

### Output Types by Template

| Template | Output Type | Possible Values |
|----------|-------------|-----------------|
| `groundedness` | string | `GROUNDED`, `NOT_GROUNDED` |
| `toxicity` | string | `SAFE`, `TOXIC` |
| `tone` | string | `FORMAL`, `INFORMAL` |
| `is_json` | boolean | `true`, `false` |
| `factual_accuracy` | number | 0.0 - 1.0 |

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {...}
  }
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid API key |
| 403 | Forbidden - Invalid secret key |
| 404 | Not Found - Template not found |
| 429 | Too Many Requests - Rate limited |
| 500 | Internal Server Error |

### Common Errors

**Invalid Authentication (403):**
```json
{
  "error": {
    "code": "INVALID_AUTH",
    "message": "Invalid API key or secret key"
  }
}
```

**Invalid Template (400):**
```json
{
  "error": {
    "code": "INVALID_TEMPLATE",
    "message": "Evaluation template 'unknown_template' not found"
  }
}
```

**Missing Required Field (400):**
```json
{
  "error": {
    "code": "MISSING_FIELD",
    "message": "Required field 'context' is missing from inputs"
  }
}
```

---

## Rate Limits

| Plan | Requests/Minute | Requests/Day |
|------|-----------------|--------------|
| Free | 60 | 1,000 |
| Pro | 300 | 10,000 |
| Enterprise | Custom | Custom |

### Rate Limit Headers

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1609459200
```

### Handling Rate Limits

When rate limited, implement exponential backoff:

```python
import time

def evaluate_with_retry(evaluator, max_retries=3):
    for attempt in range(max_retries):
        try:
            return evaluator.evaluate(...)
        except Exception as e:
            if "429" in str(e):
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")
```

---

## SDK vs API

While the REST API provides full access, the SDK is recommended because it:

- Handles authentication automatically
- Provides type safety
- Implements retry logic
- Offers convenient helper methods
- Manages connection pooling

**Use the SDK:**
```python
from fi.evals import Evaluator

evaluator = Evaluator()
result = evaluator.evaluate(
    eval_templates="groundedness",
    inputs={"context": "...", "output": "..."},
    model_name="turing_flash"
)
```

**Use the API when:**
- Integrating with non-Python/TypeScript languages
- Building custom tooling
- Debugging or testing

---

## See Also

- [Python SDK](./python-sdk.md)
- [TypeScript SDK](./typescript-sdk.md)
- [Getting Started](./getting-started.md)
- [Templates Reference](./templates-reference.md)
