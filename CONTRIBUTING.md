# Contributing to AI-Evaluation

Thank you for your interest in contributing to AI-Evaluation! This document provides guidelines and instructions for contributing.

## 🌟 Ways to Contribute

- 🐛 **Bug Reports**: Report bugs via [GitHub Issues](https://github.com/future-agi/ai-evaluation/issues)
- 💡 **Feature Requests**: Suggest new evaluation templates or features
- 📝 **Documentation**: Improve documentation, examples, or tutorials
- 🧪 **Evaluation Templates**: Contribute new evaluation templates
- 🔧 **Code**: Fix bugs or implement new features
- 🌍 **Integrations**: Add integrations with other frameworks

## 🚀 Getting Started

### Prerequisites

**For Python:**
- Python 3.10 or higher
- Poetry (recommended) or pip

**For TypeScript:**
- Node.js 18.0.0 or higher
- pnpm (recommended), npm, or yarn

### Setting Up Development Environment

#### Python Setup

```bash
# Clone the repository
git clone https://github.com/future-agi/ai-evaluation.git
cd ai-evaluation/python

# Install dependencies
poetry install

# Or with pip
pip install -e ".[dev]"

# Set up environment variables
export FI_API_KEY=your_api_key
export FI_SECRET_KEY=your_secret_key
```

#### TypeScript Setup

```bash
# Clone the repository
git clone https://github.com/future-agi/ai-evaluation.git
cd ai-evaluation/typescript/ai-evaluation

# Install dependencies
pnpm install

# Build
pnpm run build

# Run tests
pnpm test
```

## 📋 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features or enhancements
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `test/` - Test additions or fixes
- `refactor/` - Code refactoring

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style and conventions
- Add tests for new features
- Update documentation as needed

### 3. Test Your Changes

**Python:**
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=fi.evals

# Run linter
black .
flake8 .
mypy fi/
```

**TypeScript:**
```bash
# Run tests
pnpm test

# Run linter
pnpm lint

# Type check
pnpm typecheck

# Format code
pnpm format
```

### 4. Commit Your Changes

Follow conventional commit messages:

```bash
git commit -m "feat: add new evaluation template for code quality"
git commit -m "fix: resolve issue with groundedness evaluation"
git commit -m "docs: update integration examples"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test changes
- `refactor`: Code refactoring
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Detailed description of what and why
- Reference any related issues
- Screenshots/examples if applicable

## 🧪 Adding New Evaluation Templates

### Template Structure

Create a new template in the appropriate category:

**Python:**
```python
# python/fi/evals/templates/your_category.py

from fi.evals.metrics.base_llm_metric import BaseLLMMetric

class YourNewMetric(BaseLLMMetric):
    """
    Evaluates [what it evaluates].
    
    Input Schema:
        - field1: Description
        - field2: Description
    
    Output:
        - PASS/FAIL or score 0-1
    """
    
    def __init__(self):
        super().__init__(
            name="your_metric_name",
            description="What this metric measures",
            input_schema={
                "field1": "string",
                "field2": "string"
            }
        )
    
    def evaluate(self, inputs: dict) -> dict:
        # Your evaluation logic
        pass
```

**TypeScript:**
```typescript
// typescript/ai-evaluation/src/templates/yourCategory.ts

export class YourNewMetric extends BaseLLMMetric {
    constructor() {
        super({
            name: "your_metric_name",
            description: "What this metric measures",
            inputSchema: {
                field1: "string",
                field2: "string"
            }
        });
    }
    
    async evaluate(inputs: Record<string, any>): Promise<EvalResult> {
        // Your evaluation logic
    }
}
```

### Template Requirements

1. **Clear Documentation**: Include docstrings with input/output schema
2. **Input Validation**: Validate required fields
3. **Error Handling**: Handle edge cases gracefully
4. **Unit Tests**: Add comprehensive test cases
5. **Example Usage**: Provide example in documentation

### Testing New Templates

```python
# Add to tests/test_templates.py
def test_your_new_metric():
    evaluator = Evaluator()
    result = evaluator.evaluate(
        eval_templates="your_metric_name",
        inputs={
            "field1": "test value",
            "field2": "test value"
        },
        model_name="turing_flash"
    )
    assert result.eval_results[0].metrics[0].value is not None
```

## 📖 Documentation Guidelines

- Use clear, concise language
- Include code examples
- Add type hints (Python) / TypeScript types
- Update README.md if adding major features
- Add entries to appropriate documentation sections

## 🐛 Reporting Bugs

When reporting bugs, include:

1. **Clear Title**: Brief description of the issue
2. **Steps to Reproduce**: Detailed steps to reproduce the bug
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**:
   - OS and version
   - Python/Node.js version
   - Package version
   - Relevant dependencies
6. **Code Sample**: Minimal reproducible example
7. **Error Messages**: Full stack trace if applicable

## 💡 Feature Requests

When requesting features:

1. **Use Case**: Describe the problem you're trying to solve
2. **Proposed Solution**: Your idea for implementation
3. **Alternatives**: Other solutions you've considered
4. **Additional Context**: Examples, mockups, or references

## 🔍 Code Review Process

1. Maintainer reviews your PR
2. Address any feedback or requested changes
3. Once approved, maintainer will merge
4. Your contribution will be included in the next release

## 📜 Code Style

### Python

- Follow PEP 8
- Use Black for formatting
- Use type hints
- Maximum line length: 88 characters
- Use docstrings for classes and functions

### TypeScript

- Follow Airbnb style guide
- Use Prettier for formatting
- Use ESLint for linting
- Prefer async/await over promises
- Use strong typing (avoid `any`)

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information requested
- `wontfix`: This will not be worked on

## 🤝 Community

- Be respectful and inclusive
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)
- Help others in discussions
- Share your use cases and examples

## 📄 License

By contributing, you agree that your contributions will be licensed under the GPL-3.0 License.

## ❓ Questions?

- Open a [GitHub Discussion](https://github.com/future-agi/ai-evaluation/discussions)
- Join our [Discord Community](https://discord.gg/futureagi)
- Email us at opensource@futureagi.com

---

Thank you for contributing to AI-Evaluation! 🎉

