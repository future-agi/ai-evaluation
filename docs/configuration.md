# Configuration Reference

> Complete reference for AI Evaluation configuration

---

## Table of Contents

- [Environment Variables](#environment-variables)
- [YAML Configuration](#yaml-configuration)
- [Configuration File Discovery](#configuration-file-discovery)
- [Test Data Formats](#test-data-formats)
- [Assertions](#assertions)
- [Output Configuration](#output-configuration)

---

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `FI_API_KEY` | Future AGI API key | `sk-fi-...` |
| `FI_SECRET_KEY` | Future AGI secret key | `sk-secret-...` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `FI_BASE_URL` | API base URL | `https://api.futureagi.com` |
| `FI_DEFAULT_MODEL` | Default model for evaluations | `turing_flash` |

### Integration Variables

| Variable | Description |
|----------|-------------|
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_HOST` | Langfuse host URL |

### Setting Environment Variables

**Bash/Zsh:**
```bash
export FI_API_KEY="your_api_key"
export FI_SECRET_KEY="your_secret_key"
```

**Fish:**
```fish
set -x FI_API_KEY "your_api_key"
set -x FI_SECRET_KEY "your_secret_key"
```

**Windows PowerShell:**
```powershell
$env:FI_API_KEY = "your_api_key"
$env:FI_SECRET_KEY = "your_secret_key"
```

**`.env` file:**
```env
FI_API_KEY=your_api_key
FI_SECRET_KEY=your_secret_key
FI_BASE_URL=https://api.futureagi.com
FI_DEFAULT_MODEL=turing_flash
```

---

## YAML Configuration

The CLI uses a YAML configuration file (`fi-evaluation.yaml`) for declarative evaluation setup.

### Full Schema

```yaml
# fi-evaluation.yaml - AI Evaluation Configuration
version: "1.0"

# API Configuration (optional - prefer environment variables)
api:
  base_url: "https://api.futureagi.com"

# Default settings for all evaluations
defaults:
  model: "turing_flash"          # Default model
  timeout: 200                   # Timeout in seconds
  parallel_workers: 8            # Parallel execution

# Evaluation definitions
evaluations:
  # Simple single-template evaluation
  - name: "rag_quality"
    template: "groundedness"
    data: "./data/test_cases.json"

  # Multiple templates on same data
  - name: "comprehensive"
    templates:
      - "context_adherence"
      - "completeness"
      - "factual_accuracy"
    data: "./data/test_cases.json"

  # With custom model
  - name: "safety_check"
    template: "content_moderation"
    data: "./data/safety_tests.json"
    model: "protect_flash"

# Output configuration
output:
  format: "json"                 # json, table, csv, html
  path: "./results/"             # Output directory
  include_metadata: true         # Include execution metadata

# Assertions (optional)
assertions:
  - template: "groundedness"
    condition: "score >= 0.7"
    on_fail: "warn"              # warn, error, skip
```

---

### Configuration Sections

#### version

Required. Configuration version.

```yaml
version: "1.0"
```

---

#### api

Optional. API configuration (prefer environment variables).

```yaml
api:
  base_url: "https://api.futureagi.com"
```

---

#### defaults

Optional. Default settings applied to all evaluations.

```yaml
defaults:
  model: "turing_flash"      # Default model
  timeout: 200               # Default timeout (seconds)
  parallel_workers: 8        # Parallel workers
```

**Available Models:**

| Model | Use Case | Speed |
|-------|----------|-------|
| `turing_flash` | General evaluations | Fast |
| `turing_pro` | Complex evaluations | Moderate |
| `protect_flash` | Safety evaluations | Fast |
| `protect_pro` | Detailed safety | Moderate |

---

#### evaluations

Required. List of evaluation definitions.

**Single Template:**
```yaml
evaluations:
  - name: "groundedness_check"
    template: "groundedness"
    data: "./data/rag_tests.json"
```

**Multiple Templates:**
```yaml
evaluations:
  - name: "full_rag_eval"
    templates:
      - "groundedness"
      - "context_adherence"
      - "completeness"
    data: "./data/rag_tests.json"
```

**With Custom Settings:**
```yaml
evaluations:
  - name: "safety_eval"
    template: "content_moderation"
    data: "./data/safety_tests.json"
    model: "protect_flash"
    timeout: 300
```

---

#### output

Optional. Output configuration.

```yaml
output:
  format: "json"              # Output format
  path: "./results/"          # Output directory
  include_metadata: true      # Include metadata
```

**Formats:**
- `table` - Rich terminal table
- `json` - JSON format
- `csv` - CSV format
- `html` - HTML report

---

#### assertions

Optional. Pass/fail conditions for evaluations.

```yaml
assertions:
  - template: "groundedness"
    condition: "score >= 0.8"
    on_fail: "error"

  - template: "toxicity"
    condition: "output == 'SAFE'"
    on_fail: "warn"
```

**on_fail Options:**
- `warn` - Log warning, continue
- `error` - Fail the run
- `skip` - Skip remaining assertions

---

## Configuration File Discovery

The CLI automatically discovers configuration files in this order:

1. `fi-evaluation.yaml` (current directory)
2. `fi-evaluation.yml` (current directory)
3. `.fi-evaluation.yaml` (hidden file)
4. `.fi-evaluation.yml` (hidden file)
5. Parent directories (walks up to find config)

### Specifying Config Path

```bash
fi run --config /path/to/custom-config.yaml
fi validate --config ./configs/production.yaml
```

---

## Test Data Formats

The CLI supports multiple data formats.

### JSON

```json
[
  {
    "query": "What is machine learning?",
    "response": "Machine learning is a subset of AI...",
    "context": "Machine learning is a branch of artificial intelligence..."
  },
  {
    "query": "Explain neural networks",
    "response": "Neural networks are...",
    "context": "A neural network is a computational model..."
  }
]
```

### JSONL (JSON Lines)

```jsonl
{"query": "Question 1", "response": "Answer 1", "context": "Context 1"}
{"query": "Question 2", "response": "Answer 2", "context": "Context 2"}
{"query": "Question 3", "response": "Answer 3", "context": "Context 3"}
```

### CSV

```csv
query,response,context
"What is ML?","Machine learning is...","Machine learning is a branch..."
"Explain AI","Artificial intelligence is...","AI refers to..."
```

### Field Mapping

Different templates require different fields. Common mappings:

| Template | Fields |
|----------|--------|
| `groundedness` | `context`, `output` |
| `context_adherence` | `context`, `output` |
| `factual_accuracy` | `input`, `output`, `context` |
| `tone` | `input` |
| `content_moderation` | `text` |
| `is_json` | `text` |

Map your data fields to template requirements:

```yaml
evaluations:
  - name: "rag_eval"
    template: "groundedness"
    data: "./data/tests.json"
    # Data should have: context, output fields
```

---

## Assertions

Define pass/fail conditions for evaluations.

### Condition Syntax

```yaml
assertions:
  # Numeric comparison
  - template: "groundedness"
    condition: "score >= 0.7"

  # Equality check
  - template: "toxicity"
    condition: "output == 'SAFE'"

  # Not equal
  - template: "pii"
    condition: "output != 'DETECTED'"
```

### Multiple Assertions

```yaml
assertions:
  - template: "groundedness"
    condition: "score >= 0.8"
    on_fail: "error"

  - template: "completeness"
    condition: "score >= 0.7"
    on_fail: "warn"

  - template: "toxicity"
    condition: "output == 'SAFE'"
    on_fail: "error"
```

---

## Output Configuration

### Terminal Output

```bash
fi run -o table              # Rich table (default)
fi run -o json               # JSON output
fi run -o csv                # CSV output
fi run -o html               # HTML report
```

### File Output

```bash
fi run -o json -O results.json
fi run -o csv -O results.csv
fi run -o html -O report.html
```

### In Configuration

```yaml
output:
  format: "json"
  path: "./results/"
  include_metadata: true
```

---

## Starter Templates

Initialize projects with pre-configured templates:

```bash
fi init --template basic     # Basic setup
fi init --template rag       # RAG evaluation setup
fi init --template safety    # Safety evaluation setup
fi init --template agent     # Agent evaluation setup
```

### Basic Template

```yaml
version: "1.0"

evaluations:
  - name: "basic_eval"
    template: "groundedness"
    data: "./data/test_cases.json"
```

### RAG Template

```yaml
version: "1.0"

defaults:
  model: "turing_flash"

evaluations:
  - name: "rag_quality"
    templates:
      - "groundedness"
      - "context_adherence"
      - "completeness"
      - "chunk_utilization"
    data: "./data/rag_tests.json"

assertions:
  - template: "groundedness"
    condition: "score >= 0.8"
    on_fail: "warn"
```

### Safety Template

```yaml
version: "1.0"

defaults:
  model: "protect_flash"

evaluations:
  - name: "safety_check"
    templates:
      - "content_moderation"
      - "toxicity"
      - "pii"
      - "prompt_injection"
    data: "./data/safety_tests.json"

assertions:
  - template: "toxicity"
    condition: "output == 'SAFE'"
    on_fail: "error"
```

### Agent Template

```yaml
version: "1.0"

defaults:
  model: "turing_flash"

evaluations:
  - name: "agent_quality"
    templates:
      - "task_completion"
      - "is_helpful"
      - "factual_accuracy"
    data: "./data/agent_tests.json"
```

---

## See Also

- [CLI Guide](./cli-guide.md)
- [Getting Started](./getting-started.md)
- [Templates Reference](./templates-reference.md)
