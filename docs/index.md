# AI Evaluation Documentation

> Evaluate LLM outputs with 60+ pre-built templates across Python, TypeScript, and CLI

---

## Quick Start

Choose your language to get started:

<div class="grid-3">

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](./getting-started/python.md)

[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](./getting-started/typescript.md)

[![CLI](https://img.shields.io/badge/CLI-4EAA25?style=for-the-badge&logo=gnu-bash&logoColor=white)](./getting-started/cli.md)

</div>

---

## What is AI Evaluation?

AI Evaluation is a comprehensive SDK for evaluating LLM outputs. It provides:

- **60+ Pre-built Templates** - RAG, safety, quality, bias, tone, and more
- **Multi-language SDKs** - Python, TypeScript, with Go/Java coming soon
- **CLI Tool** - Run evaluations from the command line
- **CI/CD Integration** - Integrate into your deployment pipeline
- **Platform Integration** - Works with Langfuse, LangChain, and more

---

## Documentation Sections

### [Getting Started](./getting-started/index.md)
Quick installation and first evaluation for each language.

### [Core Concepts](./concepts/index.md)
Understand evaluations, templates, models, and best practices.

### [SDK Reference](./sdks/index.md)
Detailed API documentation for each language SDK.

### [CLI Guide](./cli/index.md)
Command-line interface documentation and CI/CD setup.

### [Templates](./templates/index.md)
Complete reference for all 60+ evaluation templates.

### [API Reference](./api/index.md)
REST API documentation for direct integration.

### [Integrations](./integrations/index.md)
Connect with Langfuse, LangChain, OpenTelemetry, and more.

### [Tutorials](./tutorials/index.md)
Step-by-step guides for common use cases.

### [Cookbook](./cookbook/index.md)
Copy-paste code recipes for specific tasks.

---

## SDK Availability

| Feature | Python | TypeScript | Go | Java |
|---------|:------:|:----------:|:--:|:----:|
| Core Evaluation | ✅ | ✅ | 🚧 | 📋 |
| Batch Processing | ✅ | ✅ | 🚧 | 📋 |
| Async Evaluation | ✅ | ✅ | 🚧 | 📋 |
| CLI Tool | ✅ | - | - | - |
| OpenTelemetry | ✅ | ✅ | 🚧 | 📋 |
| Langfuse | ✅ | 🚧 | - | - |

✅ Available | 🚧 In Progress | 📋 Planned | - Not Planned

---

## Installation

**Python:**
```bash
pip install ai-evaluation
```

**TypeScript:**
```bash
npm install @future-agi/ai-evaluation
```

**CLI (included with Python):**
```bash
pip install ai-evaluation
fi --help
```

---

## Quick Example

**Python:**
```python
from fi.evals import Evaluator

evaluator = Evaluator()
result = evaluator.evaluate(
    eval_templates="groundedness",
    inputs={
        "context": "The sky is blue due to Rayleigh scattering.",
        "output": "The sky appears blue because of light scattering."
    },
    model_name="turing_flash"
)
print(result.eval_results[0].output)  # "GROUNDED"
```

**TypeScript:**
```typescript
import { Evaluator } from "@future-agi/ai-evaluation";

const evaluator = new Evaluator();
const result = await evaluator.evaluate(
  "groundedness",
  {
    context: "The sky is blue due to Rayleigh scattering.",
    output: "The sky appears blue because of light scattering."
  },
  { modelName: "turing_flash" }
);
console.log(result.eval_results[0].data);  // "GROUNDED"
```

**CLI:**
```bash
fi init my-evals
cd my-evals
fi run
```

---

## Support

- **Platform**: [app.futureagi.com](https://app.futureagi.com)
- **Documentation**: [docs.futureagi.com](https://docs.futureagi.com)
- **GitHub**: [github.com/future-agi/ai-evaluation](https://github.com/future-agi/ai-evaluation)
- **Issues**: [Report bugs](https://github.com/future-agi/ai-evaluation/issues)

---

*Built by [Future AGI](https://futureagi.com)*
