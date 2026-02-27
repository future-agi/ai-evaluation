# fi-evals Cookbooks

Each cookbook solves a **real problem** you'll face when building AI applications.

| # | Cookbook | Problem It Solves | API Keys? |
|---|---------|-------------------|-----------|
| 01 | [Catch a Hallucinating Medical Chatbot](01_local_metrics.py) | Your chatbot makes up dosages and contradicts source material | No |
| 02 | [When Heuristics Aren't Enough: LLM-as-Judge](02_llm_as_judge.py) | Local metrics miss paraphrases — use Gemini to judge accuracy | Yes (`GOOGLE_API_KEY`) |
| 03 | [Is Your RAG Pipeline Lying to Users?](03_rag_evaluation.py) | Figure out WHERE your RAG fails: retrieval or generation? | No (optional for augmented) |
| 04 | [Protect Your LLM from Prompt Injection](04_guardrails.py) | Block jailbreaks, SQL injection, PII leaks, secret exposure | No |
| 05 | [Stop Toxic Output Mid-Stream](05_streaming.py) | Cut off LLM output the instant it turns toxic or off-topic | No |
| 06 | [Auto-Configure Your Testing Pipeline](06_autoeval.py) | "What should we test?" — describe your app, get a pipeline | No |
| 07 | [See Every LLM Call in Your Observability Stack](07_otel_tracing.py) | Trace calls with quality scores in Jaeger/Datadog/Grafana | No |
| 08 | [Teach Your Judge from Past Mistakes](feedback_loop_demo.py) | LLM judge keeps getting the same cases wrong — fix it with feedback | Yes (`GOOGLE_API_KEY`) |

## Quick Start

```bash
cd python

# Run any cookbook (no API keys needed for 01, 03-07)
uv run python -m examples.01_local_metrics

# For cookbooks that need an LLM (02, 08)
export GOOGLE_API_KEY=your-key
uv run python -m examples.02_llm_as_judge
```

## What You'll Learn

- **Cookbook 01**: Build a validation layer that catches hallucinations, wrong dosages, and contradictions — all locally in <1 second
- **Cookbook 02**: When local heuristics fail on paraphrases, use an LLM judge with `augment=True` for production-grade accuracy
- **Cookbook 03**: Diagnose RAG failures by measuring retrieval quality (recall, precision) separately from generation quality (faithfulness, groundedness)
- **Cookbook 04**: Build a <10ms security middleware that blocks jailbreaks, code injection, PII exposure, and secret leaks
- **Cookbook 05**: Monitor streaming LLM output token-by-token and kill the stream when safety thresholds are breached
- **Cookbook 06**: Auto-generate test pipelines from app descriptions, customize thresholds, export YAML for CI/CD
- **Cookbook 07**: Wire quality scores into your OTEL traces so you can search for bad responses in Jaeger/Datadog
- **Cookbook 08**: Store developer corrections in ChromaDB, retrieve them as few-shot examples, and teach your LLM judge to not repeat mistakes
