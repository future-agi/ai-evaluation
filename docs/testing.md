# Running Tests

> Guide for running tests in the AI Evaluation project

---

## Prerequisites

- Python 3.10+
- Node.js 18+ (for TypeScript tests)
- uv, poetry, or pip for Python package management

---

## Python Tests

### Using uv (Recommended)

```bash
cd python

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test directories
uv run pytest tests/sdk/          # SDK tests
uv run pytest tests/cli/          # CLI tests

# Run a specific test file
uv run pytest tests/sdk/test_evaluator.py

# Run with coverage
uv run pytest --cov=fi

# Run tests matching a pattern
uv run pytest -k "test_eval"
```

### Using Poetry

```bash
cd python

# Install dependencies
poetry install

# Run all tests
poetry run pytest

# Run specific tests
poetry run pytest tests/sdk/ -v
poetry run pytest tests/cli/ -v

# Run with coverage
poetry run pytest --cov=fi
```

### Using pip

```bash
cd python

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package and dev dependencies
pip install -e .
pip install pytest pytest-cov

# Run tests
pytest
pytest tests/sdk/ -v
pytest tests/cli/ -v
```

---

## TypeScript Tests

```bash
cd typescript/ai-evaluation

# Install dependencies
npm install

# Run all tests
npm test

# Run with coverage
npm test -- --coverage

# Run specific test file
npm test -- types.test.ts

# Run in watch mode
npm test -- --watch
```

---

## Test Structure

```
python/
└── tests/
    ├── sdk/                    # SDK unit tests (mocked)
    │   ├── conftest.py         # Shared fixtures
    │   ├── test_types.py       # Type tests
    │   ├── test_templates.py   # Template tests
    │   ├── test_heuristics.py  # Heuristic metric tests
    │   └── test_evaluator.py   # Evaluator tests
    ├── cli/                    # CLI tests (mocked)
    │   ├── conftest.py         # CLI fixtures
    │   ├── test_init.py        # init command tests
    │   ├── test_run.py         # run command tests
    │   ├── test_list.py        # list command tests
    │   ├── test_validate.py    # validate command tests
    │   └── test_config_loader.py # Config loader tests
    └── integration/            # Integration tests (real backend)
        ├── conftest.py         # Integration fixtures
        └── test_evaluator_integration.py  # Real API tests

typescript/ai-evaluation/
└── src/__tests__/
    ├── evaluator.test.ts       # Evaluator tests
    ├── protect.test.ts         # Protect class tests
    ├── types.test.ts           # Type tests
    └── templates.test.ts       # Template tests
```

---

## Environment Variables

Tests use mock API keys by default. The test fixtures automatically set:

```bash
FI_API_KEY=test_api_key
FI_SECRET_KEY=test_secret_key
```

To run integration tests with real API calls, set your actual keys:

```bash
export FI_API_KEY="your_real_api_key"
export FI_SECRET_KEY="your_real_secret_key"
```

---

## Common Issues

### Import Errors

If you see import errors, ensure the package is installed in development mode:

```bash
uv pip install -e .
# or
pip install -e .
```

### Missing Dependencies

Install dev dependencies:

```bash
uv pip install pytest pytest-cov
# or
pip install pytest pytest-cov
```

### Test Discovery Issues

Ensure test files follow the naming convention:
- Files: `test_*.py`
- Functions: `test_*`

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - name: Run tests
        working-directory: python
        run: |
          uv venv
          uv pip install -e ".[dev]"
          uv run pytest --cov=fi

  typescript-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
      - name: Run tests
        working-directory: typescript/ai-evaluation
        run: |
          npm install
          npm test
```

---

## Integration Tests (with Real Backend)

Integration tests run against the actual backend to test real API calls.

### Prerequisites

1. **Start backend test services** (from core-backend repo):

```bash
cd /path/to/core-backend

# Start Docker services
docker compose -f docker-compose.test.yml -p futureagi-test up -d

# Load test environment and run migrations
set -a && source .env.test.local && set +a
python manage.py migrate

# Create test data (organization, user, API key, templates)
# See core-backend/docs/TESTING.md for the setup script

# Run backend dev server
python manage.py runserver 0.0.0.0:8001
```

### Running Integration Tests

```bash
cd python

# Set environment variables
export FI_API_KEY="test_api_key_12345"
export FI_SECRET_KEY="test_secret_key_67890"
export FI_BASE_URL="http://localhost:8001"

# Run integration tests only
pytest tests/integration/ -v -m integration

# Run tests that require model serving (when available)
pytest tests/integration/ -v -m integration --run-model-serving
```

### Test Structure

```
python/tests/integration/
├── conftest.py                      # Integration fixtures
├── test_evaluator_integration.py    # Evaluator integration tests
└── test_guardrails_integration.py   # Guardrails integration tests
```

### Test Categories

**Evaluator Tests:**

| Test Class | Description | Requires Model Serving |
|------------|-------------|------------------------|
| `TestEvaluatorConnection` | Basic connectivity, list templates | No |
| `TestAPIRequestValidation` | Invalid inputs, missing fields | No |
| `TestAsyncEvaluation` | Async evaluation mode | No |
| `TestGroundednessEvaluation` | RAG groundedness checks | Yes |
| `TestSafetyEvaluations` | Toxicity, PII detection | Yes |
| `TestToneEvaluations` | Politeness, helpfulness | Yes |

**Guardrails Tests:**

| Test Class | Description | Requires Model Serving |
|------------|-------------|------------------------|
| `TestGuardrailsConnection` | Basic connectivity | No |
| `TestEdgeCases` | Empty input, whitespace, long input | No |
| `TestConfiguration` | Custom categories | No |
| `TestHarmfulContentDetection` | Block harmful content | Yes |
| `TestJailbreakDetection` | Block jailbreak attempts | Yes |
| `TestPromptInjectionDetection` | Block prompt injections | Yes |
| `TestOutputRails` | Screen LLM output | Yes |
| `TestRetrievalRails` | Screen RAG chunks | Yes |
| `TestRealWorldScenarios` | E2E flows | Yes |
| `TestMultilingual` | Multi-language detection | Yes |

Tests marked `requires_model_serving` are skipped by default (the model inference service isn't running in local test environments). Use `--run-model-serving` to run them.

### Skipping Integration Tests

```bash
# Run all tests (integration tests will be skipped if backend unavailable)
pytest tests/ -v

# Explicitly skip integration tests
pytest tests/ -v --ignore=tests/integration/
```

---

## See Also

- [Python SDK](./python-sdk.md)
- [TypeScript SDK](./typescript-sdk.md)
- [CLI Guide](./cli-guide.md)
- [Backend Testing](../../core-backend/docs/TESTING.md)
