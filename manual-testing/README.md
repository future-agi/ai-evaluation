# Manual Testing

> Feature-by-feature manual testing for AI Evaluation SDK

## Structure

```
manual-testing/
├── 01-python-sdk-core/      # Core Python SDK functionality
│   ├── initialization/      # Evaluator init tests
│   ├── single-evaluation/   # Single template evaluation
│   ├── batch-evaluation/    # Multi-template evaluation
│   ├── model-selection/     # Model selection tests
│   └── error-handling/      # Error scenarios
├── 02-templates/            # All 60+ templates
│   ├── rag-context/         # RAG & context templates
│   ├── safety/              # Safety templates
│   ├── quality/             # Quality templates
│   ├── format/              # Format validation templates
│   ├── tone-behavior/       # Tone & behavior templates
│   ├── bias/                # Bias detection templates
│   ├── summarization/       # Summarization templates
│   └── similarity/          # Similarity/heuristic templates
├── 03-local-evaluation/     # Local evaluation (no API)
├── 04-protect-guardrails/   # Protection & guardrails
├── 05-streaming/            # Streaming evaluation
├── 06-autoeval/             # AutoEval pipeline
├── 07-cli/                  # CLI commands
│   ├── init/                # fi init
│   ├── run/                 # fi run
│   ├── list/                # fi list
│   ├── validate/            # fi validate
│   ├── output-formats/      # Output format tests
│   └── assertions/          # CI/CD assertions
├── 08-typescript-sdk/       # TypeScript SDK
├── 09-integrations/         # OTEL, Langfuse, etc.
└── 10-edge-cases/           # Edge cases & stress tests
```

## How to Run Tests

Each folder contains:
- `test_*.py` - Python test scripts
- `test_*.ts` - TypeScript test scripts (where applicable)
- `RESULTS.md` - Document your test results here

## Progress Tracker

| Section | Status | Notes |
|---------|--------|-------|
| 01-python-sdk-core | 🔄 In Progress | |
| 02-templates | ⏳ Pending | |
| 03-local-evaluation | ⏳ Pending | |
| 04-protect-guardrails | ⏳ Pending | |
| 05-streaming | ⏳ Pending | |
| 06-autoeval | ⏳ Pending | |
| 07-cli | ⏳ Pending | |
| 08-typescript-sdk | ⏳ Pending | |
| 09-integrations | ⏳ Pending | |
| 10-edge-cases | ⏳ Pending | |

Legend: ✅ Complete | 🔄 In Progress | ⏳ Pending | ❌ Failed
