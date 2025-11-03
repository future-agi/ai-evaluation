# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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

[Unreleased]: https://github.com/future-agi/ai-evaluation/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/future-agi/ai-evaluation/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/future-agi/ai-evaluation/compare/v0.1.0...v0.2.1
[0.1.0]: https://github.com/future-agi/ai-evaluation/releases/tag/v0.1.0
