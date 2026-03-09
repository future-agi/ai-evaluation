# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.1.0] - 2026-03-09

### Python

#### Added
- **`eval_config` parameter** — pass eval-specific configuration to `evaluator.evaluate()`, e.g. `eval_config={"k": 3}` for retrieval metrics (recall_at_k, precision_at_k, ndcg_at_k). Config is forwarded to the backend as `config.params` in the API payload.

## [1.0.0] - 2026-02-27

### Python

#### Added
- **`evaluate()` unified API** — single entrypoint for local, cloud, and LLM-as-Judge evaluations with automatic engine routing
- **Multimodal LLM Judge** — pass `image_url`, `audio_url`, `input_image_url`, `output_image_url` for vision/audio evaluation with Gemini, GPT-4o, etc.
- **Auto-generate grading criteria** — `generate_prompt=True` converts a short description into a detailed rubric via LLM
- **LLM augmentation** — `augment=True` runs local heuristic first, then LLM refines the score (faithfulness, hallucination_score, task_completion, etc.)
- **Feedback loop system** — submit corrections, retrieve as few-shot examples via ChromaDB, calibrate pass/fail thresholds
- **72+ local metrics** — string checks, JSON validation, similarity, NLI-based hallucination detection, RAG evaluation, function calling, agent trajectory, structured output, security guardrails
- **OpenTelemetry integration** — `enable_auto_enrichment()` emits `gen_ai.evaluation.*` spans for Jaeger/Datadog/Grafana
- **Streaming evaluation** — token-by-token monitoring with configurable early stopping
- **9 cookbooks** — local metrics, LLM judge, RAG evaluation, guardrails, streaming, autoeval, OTEL tracing, feedback loop, multimodal judge

#### Changed
- **Poetry to uv** — migrated build system for 10x faster dependency resolution
- Widened LLM provider type signatures from `Dict[str, str]` to `Dict[str, Any]` for multimodal content parts

#### Fixed
- 6 code security scanner bugs (Phase 2)
- Guardrails ensemble scoring and scanner edge cases
- NLI consolidation and empty-input handling in RAG metrics
- K8s backend JSON log parsing
- Temporal Docker healthcheck and DB config
- Celery serialization for closures

## [0.2.2] - 2025-10-27

- Introducing LLM As A Judge
- Heuristics for JSON, Similarity, String, and Aggregation metrics
- Minor Bug Fixes

## [0.2.1] - 2025-10-9

### Python
#### Added
- Support for batch evaluation
- New evaluation templates for bias detection
- Improved error handling and logging

#### Fixed
- Issue with context adherence evaluation
- Memory leak in long-running evaluations

## [0.1.5] - 2025-10-01

### TypeScript
#### Added
- Initial TypeScript SDK release
- Core evaluation functionality
- Support for all evaluation templates
- ESM and CommonJS module support

### Python
#### Added
- Initial Python SDK release
- 50+ evaluation templates across multiple categories
- Support for RAG, Safety, Function Calling, and Summarization evaluations
- Integration with Future AGI platform
- Batch evaluation support

#### Features
- **RAG Evaluations**: groundedness, context_adherence, answer_relevance
- **Safety**: content_moderation, prompt_injection, harmful_advice detection
- **Function Calling**: JSON validation, schema validation
- **Summarization**: quality assessment, factual consistency
- **Behavioral**: tone analysis, helpfulness, politeness
- **Metrics**: ROUGE, embedding similarity, fuzzy matching

---

## Release Notes Format

### Types of Changes
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` in case of vulnerabilities

### Versioning
- **Major version (X.0.0)**: Breaking changes
- **Minor version (0.X.0)**: New features, backward compatible
- **Patch version (0.0.X)**: Bug fixes, backward compatible

---

[Unreleased]: https://github.com/future-agi/ai-evaluation/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/future-agi/ai-evaluation/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/future-agi/ai-evaluation/compare/v0.2.2...v1.0.0
[0.2.2]: https://github.com/future-agi/ai-evaluation/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/future-agi/ai-evaluation/compare/v0.1.0...v0.2.1
[0.1.0]: https://github.com/future-agi/ai-evaluation/releases/tag/v0.1.0
