# AI Evaluation CLI Guide

> Command-line interface for running LLM evaluations with 60+ pre-built templates

---

## Installation

```bash
pip install ai-evaluation
```

After installation, the `fi` command will be available in your terminal.

## Quick Start

### 1. Initialize a Project

```bash
fi init my-evals
cd my-evals
```

This creates:
```
my-evals/
├── fi-evaluation.yaml  # Configuration file
├── data/
│   └── test_cases.json # Sample test data
└── results/            # Evaluation results
```

### 2. Configure Your Evaluations

Edit `fi-evaluation.yaml`:

```yaml
version: "1.0"

evaluations:
  - name: "rag_quality"
    template: "groundedness"
    data: "./data/test_cases.json"
```

### 3. Set API Keys

```bash
export FI_API_KEY="your_api_key"
export FI_SECRET_KEY="your_secret_key"
```

### 4. Run Evaluations

```bash
fi run
```

---

## Commands

### `fi init`

Initialize a new evaluation project.

```bash
fi init [directory] [--template <name>] [--force]
```

**Arguments:**
- `directory` - Target directory (default: current directory)

**Options:**
- `--template, -t` - Template to use: `basic`, `rag`, `safety`, `agent`
- `--force, -f` - Overwrite existing configuration file

**Examples:**
```bash
fi init                        # Initialize in current directory
fi init my-project             # Initialize in my-project/
fi init --template rag         # Use RAG evaluation template
fi init my-project --force     # Overwrite existing config
```

---

### `fi run`

Run evaluations from config file or CLI arguments.

```bash
fi run [--config <path>] [--eval <name>] [--data <path>] [--output <format>]
```

**Options:**
- `--config, -c` - Path to configuration file (default: auto-discover)
- `--eval, -e` - Evaluation template to run (overrides config)
- `--data, -d` - Path to test data file (overrides config)
- `--output, -o` - Output format: `table`, `json`, `csv`, `html`
- `--parallel, -p` - Number of parallel workers (default: 8)
- `--timeout, -T` - Timeout per evaluation in seconds (default: 200)
- `--model, -m` - Model for LLM-as-judge evaluations
- `--dry-run` - Validate config without running
- `--output-file, -O` - Path to save output file
- `--quiet, -q` - Suppress progress output

**Examples:**
```bash
fi run                                    # Use default config
fi run -c custom.yaml                     # Use custom config
fi run -e groundedness -d data.json       # Run single evaluation
fi run -o json > results.json             # Output as JSON
fi run --dry-run                          # Validate only
fi run -o html -O report.html             # Save HTML report
```

---

### `fi list`

List available evaluation resources.

```bash
fi list [resource] [--format <format>] [--category <category>]
```

**Resources:**
- `templates` - List all evaluation templates (default)
- `categories` - List template categories

**Options:**
- `--format, -f` - Output format: `table`, `json`
- `--category, -c` - Filter templates by category

**Examples:**
```bash
fi list                            # List all templates
fi list templates                  # Same as above
fi list templates --category rag   # List RAG templates only
fi list templates -f json          # Output as JSON
fi list categories                 # List all categories
```

**Available Categories:**
- `conversation` - Conversation quality evaluations
- `rag` - RAG pipeline evaluations
- `safety` - Safety and moderation
- `bias` - Bias detection
- `quality` - Response quality
- `format` - Format validation
- `tone` - Tone analysis
- `translation` - Translation quality
- `audio` - Audio evaluations
- `function_calling` - Function calling accuracy
- `hallucination` - Hallucination detection

---

### `fi validate`

Validate configuration file and test data.

```bash
fi validate [--config <path>] [--strict]
```

**Options:**
- `--config, -c` - Path to configuration file
- `--strict, -s` - Enable strict validation (fail on warnings)

**Examples:**
```bash
fi validate                        # Validate auto-discovered config
fi validate -c custom.yaml         # Validate specific config
fi validate --strict               # Fail on warnings
```

**Validation Checks:**
- YAML syntax validity
- Template name existence
- Required input fields for templates
- Data file accessibility
- API key presence (warning if missing)

---

### `fi config`

Manage CLI configuration.

```bash
fi config <subcommand>
```

**Subcommands:**

#### `fi config show`
Display current configuration settings.

```bash
fi config show
```

#### `fi config get <key>`
Get a specific configuration value.

```bash
fi config get FI_API_KEY
```

#### `fi config set <key> <value>`
Set a configuration value.

```bash
fi config set FI_DEFAULT_MODEL gpt-4o
fi config set FI_API_KEY your_key --save  # Save to .env
```

**Options:**
- `--save, -s` - Save to .env file

#### `fi config init`
Create a default configuration file.

```bash
fi config init
fi config init --force  # Overwrite existing
```

#### `fi config env`
Show environment variable template.

```bash
fi config env > .env
```

---

## Configuration File

### `fi-evaluation.yaml`

```yaml
# fi-evaluation.yaml - AI Evaluation Configuration
version: "1.0"

# API Configuration (optional - use environment variables)
api:
  base_url: "https://api.futureagi.com"

# Default settings
defaults:
  model: "gpt-4o"
  timeout: 200
  parallel_workers: 8

# Evaluation definitions
evaluations:
  # Simple evaluation
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

# Output configuration
output:
  format: "json"
  path: "./results/"
  include_metadata: true

# Assertions (optional)
assertions:
  - template: "groundedness"
    condition: "score >= 0.7"
    on_fail: "warn"  # warn, error, skip
```

---

## Test Data Format

### JSON

```json
[
  {
    "query": "What is machine learning?",
    "response": "Machine learning is a subset of AI...",
    "context": "Machine learning is a branch of artificial intelligence..."
  }
]
```

### JSONL

```jsonl
{"query": "Question 1", "response": "Answer 1"}
{"query": "Question 2", "response": "Answer 2"}
```

### CSV

```csv
query,response,context
"What is ML?","Machine learning is...","Machine learning is a branch..."
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FI_API_KEY` | Future AGI API key | Required |
| `FI_SECRET_KEY` | Future AGI secret key | Required |
| `FI_BASE_URL` | API base URL | `https://api.futureagi.com` |
| `FI_DEFAULT_MODEL` | Default model for LLM evaluations | `gpt-4o` |

---

## Available Templates

Run `fi list templates` to see all available templates. Here are some commonly used ones:

### RAG Evaluations
- `groundedness` - Check if response is grounded in context
- `context_adherence` - Check if response adheres to context
- `context_relevance` - Check if context is relevant to query
- `completeness` - Check if response is complete
- `chunk_attribution` - Check chunk attribution
- `chunk_utilization` - Check chunk utilization

### Safety Evaluations
- `content_moderation` - Content moderation check
- `toxicity` - Toxicity detection
- `pii` - PII detection
- `prompt_injection` - Prompt injection detection
- `no_racial_bias` - Racial bias check
- `no_gender_bias` - Gender bias check

### Quality Evaluations
- `factual_accuracy` - Factual accuracy check
- `summary_quality` - Summary quality assessment
- `is_helpful` - Helpfulness check
- `is_concise` - Conciseness check

---

## CI/CD Integration

### GitHub Actions

```yaml
name: LLM Evaluation

on:
  push:
    branches: [main]
  pull_request:

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install ai-evaluation

      - name: Validate config
        run: fi validate --strict

      - name: Run evaluations
        env:
          FI_API_KEY: ${{ secrets.FI_API_KEY }}
          FI_SECRET_KEY: ${{ secrets.FI_SECRET_KEY }}
        run: fi run -o json > results.json

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-results
          path: results.json
```

---

## Troubleshooting

### API Keys Not Found

```
Warning: API keys not found in environment.
Set FI_API_KEY and FI_SECRET_KEY environment variables.
```

**Solution:** Set environment variables:
```bash
export FI_API_KEY="your_api_key"
export FI_SECRET_KEY="your_secret_key"
```

### Configuration File Not Found

```
Error: No configuration file found.
```

**Solution:** Create a config file or specify path:
```bash
fi init                    # Create new project
fi run -c path/to/config.yaml  # Specify path
```

### Unknown Template

```
Error: Unknown template 'my_template'
```

**Solution:** Check available templates:
```bash
fi list templates
```

---

*For more information, visit [Future AGI Documentation](https://docs.futureagi.com)*
