# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.1.0] (Python) / [0.2.0] (TypeScript) - 2026-04-18

Major update to align the SDK with the api-side Turing eval revamp.

### Python

#### Added
- **Dynamic cloud eval registry** (`fi.evals.core.cloud_registry`). Fetches `required_keys` from the live api on first use and maps user kwargs to the exact backend shape. New backend templates work without an SDK release.
- **`TuringEngine` string fallback** — `evaluate("customer_agent_query_handling", ...)` works with only a string, no `EvalTemplate` class needed.
- **14 new template classes** for the revamp: full `customer_agent_*` family (11 classes), plus `TextToSQL`, `ContainsCode`, `NoLLMReference`, `DetectHallucination`, `GroundTruthMatch`, `PromptInstructionAdherence`, `ProtectFlash`, `ImageInstructionAdherence`, `SyntheticImageEvaluator`, `OCREvaluation`, `ASRAccuracy`, `TTSAccuracy`.
- **Failed `EvalResult` sentinel** on backend 4xx/5xx. Previously the SDK returned `BatchRunResult(eval_results=[])` silently, crashing downstream consumers. Now every failure surfaces as a concrete `EvalResult` with the api error text in `.reason`.
- **`EvalTemplateManager` is usable** — previously threw `AttributeError` because the routes weren't shipped. All eval-template / composite / ground-truth / playground routes are now in `fi.utils.routes.Routes`.
- **Contract + release test tiers** (`tests/contract/`, `tests/release/`) + `.github/workflows/dev-to-main.yml` CI gate.

#### Changed
- **Python version constraint relaxed to `>=3.10`** (was `>=3.10,<3.14`).
- **`templates.py` no longer hardcodes Pydantic `Input` models** — schemas come from the live registry. `OutputOnly`, `OutputWithContext`, `OutputWithExpected`, `InputOnly`, `OutputWithInput`, `ConversationMessages`, `ImageInput`, `AudioInput` base classes are removed.
- **Decoupled from the `futureagi` package.** `fi.api.auth`, `fi.api.types`, `fi.utils.routes`, `fi.utils.errors`, `fi.utils.executor`, `fi.utils.constants`, `fi.utils.utils` are vendored in-repo. `futureagi` is no longer a dependency.
- New direct deps: `requests-futures`, `pydantic>=2`, `levenshtein`, `nltk`, `rouge-score` (previously transitive via `futureagi`).

#### Removed
- **5 upstream-removed templates** raise `ImportError`: `SafeForWorkText`, `NotGibberishText`, `NoValidLinks`, `IsCode`, `IsCSV`.
- Dead `evaluate = lambda ...` shim at `evaluator.py` bottom — use `from fi.evals import evaluate`.
- `Evaluator._validate_inputs` stub and `_get_eval_configs` method.
- `ApiKeyName` enum in `fi.utils.utils` — unused internal.
- 6 stale test files (~3,700 LOC) referencing symbols renamed in earlier phases.

#### Deprecated — BC aliases (to be removed in 2.0)
- `NoOpenAIReference` → `NoLLMReference`
- `DetectHallucinationMissingInfo` → `DetectHallucination`
- `LLMFunctionCalling` → `EvaluateFunctionCalling`
- `AudioTranscriptionEvaluator` → `ASRAccuracy`

#### Fixed
- `Evaluator.evaluate()` no longer drops results silently on backend 4xx/5xx — always returns a concrete `EvalResult`.
- Response parser handles both legacy `outputType`/`evalId` (camelCase) and revamped `output_type`/`eval_id` (snake_case).

### TypeScript

#### Added
- **Dynamic cloud eval registry** (`src/core/cloudRegistry.ts`) — mirrors the Python implementation. Maps user inputs to backend `required_keys`.
- **23 new template entries** for the revamp (same list as Python).
- **Failed `EvalResult` sentinel** on 4xx/5xx — `Evaluator.evaluate()` returns a concrete failed result instead of throwing.
- **`EvalTemplateManager` works** — routes were missing from `@future-agi/sdk`. All routes inlined in `src/core/routes.ts`.
- **Contract test suite** (`src/__tests__/contract/`) — drift, input mapping, response parsing, silent-empty, routes-enum.

#### Changed
- **Decoupled from `@future-agi/sdk`.** `APIKeyAuth`, `ResponseHandler`, `HttpMethod`, `RequestConfig`, `Routes`, `BoundedExecutor`, errors, constants are vendored in `src/core/`. `@future-agi/sdk` is no longer a dependency.

#### Removed
- **5 upstream-removed templates** — `Templates.SafeForWorkText`, `NotGibberishText`, `NoValidLinks`, `IsCode`, `IsCSV` are gone.

#### Deprecated — BC aliases (to be removed in 1.0)
- `Templates.NoOpenAIReference` → `Templates.NoLLMReference`
- `Templates.DetectHallucinationMissingInfo` → `Templates.DetectHallucination`
- `Templates.LLMFunctionCalling` → `Templates.EvaluateFunctionCalling`
- `Templates.AudioTranscriptionEvaluator` → `Templates.ASRAccuracy`

### Known issues

- Async `Evaluator.submit()` + `Execution.wait()` — submit works, but completion depends on the api's temporal worker being healthy (see [TH-4305](https://linear.app/future-agi/issue/TH-4305)). Sync `.evaluate()` is the safe default.
- Audio/PDF evals require `turing_large`; `turing_flash` rejects them with a clear error.


## [1.0.2] - 2026-04-02

### Python

#### Fixed
- **Inline evals failing with "Custom eval configuration already exists"** — removed client-side `check_custom_eval_config_exists` call from `evaluator.evaluate()`. This check was incorrectly blocking `trace_eval` when a config with the same `custom_eval_name` already existed in the project (e.g. from a previous run), causing all inline eval results to silently not attach to spans. The backend handles config creation/deduplication on its own; the client-side check is only needed in `register()` for eval_tags.

## [1.0.1] - 2026-03-09

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

[Unreleased]: https://github.com/future-agi/ai-evaluation/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/future-agi/ai-evaluation/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/future-agi/ai-evaluation/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/future-agi/ai-evaluation/compare/v0.2.2...v1.0.0
[0.2.2]: https://github.com/future-agi/ai-evaluation/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/future-agi/ai-evaluation/compare/v0.1.0...v0.2.1
[0.1.0]: https://github.com/future-agi/ai-evaluation/releases/tag/v0.1.0
