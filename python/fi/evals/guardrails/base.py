"""
Main Guardrails Class.

Provides the primary interface for content screening with support for:
- Multiple models (Turing, local models, third-party APIs)
- Input, output, and retrieval rails
- Ensemble mode with configurable aggregation
- Async and sync APIs
- Fast scanner pipeline for quick threat detection
"""

import asyncio
import functools
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Type

from fi.evals.guardrails.config import (
    GuardrailsConfig,
    GuardrailModel,
    RailType,
    AggregationStrategy,
)
from fi.evals.guardrails.types import GuardrailResult, GuardrailsResponse
from fi.evals.guardrails.backends.base import BaseBackend
from fi.evals.guardrails.scanners import ScannerPipeline, create_default_pipeline

logger = logging.getLogger(__name__)


def _trace_guardrail(fn):
    """Decorator that wraps guardrail screen methods with OTEL spans."""
    @functools.wraps(fn)
    def wrapper(self, content, *args, **kwargs):
        try:
            from fi.evals.otel.enrichment import is_auto_enrichment_enabled
            if not is_auto_enrichment_enabled():
                raise ImportError
            from fi.evals.otel.conventions import (
                GenAIAttributes,
                GuardrailAttributes,
            )
            from opentelemetry import trace as _trace
        except ImportError:
            return fn(self, content, *args, **kwargs)

        tracer = _trace.get_tracer("fi.evals.guardrails")
        rail_type = args[0] if args else kwargs.get("rail_type", RailType.INPUT)
        span_name = f"guardrail.screen_{rail_type.value}"

        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute(GenAIAttributes.SPAN_KIND, "GUARDRAIL")
            span.set_attribute(GuardrailAttributes.GEN_AI_NAME, span_name)
            span.set_attribute(GuardrailAttributes.GEN_AI_TYPE, rail_type.value)
            span.set_attribute(GenAIAttributes.INPUT_MESSAGES, content)

            response = fn(self, content, *args, **kwargs)

            result_str = "allow" if response.passed else "block"
            span.set_attribute(GuardrailAttributes.GEN_AI_RESULT, result_str)
            if response.blocked_categories:
                span.set_attribute(
                    GuardrailAttributes.GEN_AI_CATEGORIES,
                    str(response.blocked_categories),
                )
            if response.total_latency_ms:
                span.set_attribute(GuardrailAttributes.LATENCY_MS, response.total_latency_ms)

            return response
    return wrapper


class Guardrails:
    """
    Comprehensive guardrails system supporting multiple backends.

    Extends the existing Protect system with:
    - Local model support (Qwen3Guard, Granite Guardian, etc.)
    - Turing model integration
    - Ensemble mode with configurable aggregation
    - Input, output, and retrieval rails
    - Async and sync APIs

    Usage:
        # Quick start with defaults (uses Turing Flash)
        guardrails = Guardrails()
        result = guardrails.screen_input("user message")

        # Advanced: Ensemble with multiple models
        guardrails = Guardrails(
            config=GuardrailsConfig(
                models=[
                    GuardrailModel.TURING_FLASH,
                    GuardrailModel.QWEN3GUARD_8B,
                ],
                aggregation=AggregationStrategy.MAJORITY,
            )
        )
    """

    def __init__(
        self,
        config: Optional[GuardrailsConfig] = None,
        fi_api_key: Optional[str] = None,
        fi_secret_key: Optional[str] = None,
        fi_base_url: Optional[str] = None,
    ):
        """
        Initialize the Guardrails system.

        Args:
            config: Configuration for the guardrails system
            fi_api_key: FutureAGI API key (for Turing models)
            fi_secret_key: FutureAGI secret key
            fi_base_url: Base URL for FutureAGI API
        """
        self.config = config or GuardrailsConfig()
        self._fi_api_key = fi_api_key
        self._fi_secret_key = fi_secret_key
        self._fi_base_url = fi_base_url
        self.backends: Dict[GuardrailModel, BaseBackend] = {}
        self.scanner_pipeline: Optional[ScannerPipeline] = None

        self._load_backends()
        self._load_scanners()

    @classmethod
    def discover_backends(cls) -> List[GuardrailModel]:
        """
        Discover available backends based on environment.

        Checks for API keys, VLLM servers, and GPU availability.

        Returns:
            List of available GuardrailModel values

        Usage:
            available = Guardrails.discover_backends()
            print(f"Available: {[m.value for m in available]}")
        """
        from fi.evals.guardrails.discovery import discover_backends
        return discover_backends()

    @classmethod
    def get_backend_details(cls) -> Dict[str, Dict]:
        """
        Get detailed information about all backends.

        Returns:
            Dict mapping model names to availability details

        Usage:
            details = Guardrails.get_backend_details()
            for model, info in details.items():
                print(f"{model}: {info['status']} - {info['reason']}")
        """
        from fi.evals.guardrails.discovery import get_backend_details
        return get_backend_details()

    def _load_backends(self):
        """Initialize model backends based on configuration."""
        for model in self.config.models:
            self.backends[model] = self._create_backend(model)

    def _load_scanners(self):
        """Initialize scanner pipeline based on configuration."""
        scanner_config = self.config.scanners
        if scanner_config is None or not scanner_config.enabled:
            self.scanner_pipeline = None
            return

        # Build scanner pipeline from config
        from fi.evals.guardrails.scanners import (
            JailbreakScanner,
            CodeInjectionScanner,
            SecretsScanner,
            MaliciousURLScanner,
            InvisibleCharScanner,
            LanguageScanner,
            TopicRestrictionScanner,
            RegexScanner,
            RegexPattern,
        )

        scanners = []

        if scanner_config.jailbreak:
            scanners.append(JailbreakScanner(
                threshold=scanner_config.jailbreak_threshold
            ))

        if scanner_config.code_injection:
            scanners.append(CodeInjectionScanner(
                threshold=scanner_config.code_injection_threshold
            ))

        if scanner_config.secrets:
            scanners.append(SecretsScanner(
                threshold=scanner_config.secrets_threshold
            ))

        if scanner_config.urls:
            scanners.append(MaliciousURLScanner(
                threshold=scanner_config.urls_threshold
            ))

        if scanner_config.invisible_chars:
            scanners.append(InvisibleCharScanner())

        if scanner_config.language:
            scanners.append(LanguageScanner(
                allowed_languages=scanner_config.language.allowed,
                blocked_languages=scanner_config.language.blocked,
                allowed_scripts=scanner_config.language.allowed_scripts,
            ))

        if scanner_config.topics:
            scanners.append(TopicRestrictionScanner(
                allowed_topics=scanner_config.topics.allowed,
                denied_topics=scanner_config.topics.denied,
                custom_topics=scanner_config.topics.custom_topics,
                min_keyword_matches=scanner_config.topics.min_keyword_matches,
            ))

        if scanner_config.regex_patterns or scanner_config.predefined_patterns:
            custom_patterns = []
            for pattern_config in scanner_config.regex_patterns:
                custom_patterns.append(RegexPattern(
                    name=pattern_config.name,
                    pattern=pattern_config.pattern,
                    confidence=pattern_config.confidence,
                    description=pattern_config.description,
                ))
            scanners.append(RegexScanner(
                patterns=scanner_config.predefined_patterns,
                custom_patterns=custom_patterns if custom_patterns else None,
            ))

        if scanners:
            self.scanner_pipeline = ScannerPipeline(
                scanners=scanners,
                parallel=scanner_config.parallel,
                fail_fast=scanner_config.fail_fast,
            )

    def _create_backend(self, model: GuardrailModel) -> BaseBackend:
        """
        Create a backend for the specified model.

        Uses the model registry to find and instantiate the correct backend.

        Args:
            model: The model to create a backend for

        Returns:
            Backend instance

        Raises:
            ValueError: If model not found in registry
            RuntimeError: If backend cannot be instantiated
        """
        from fi.evals.guardrails.registry import get_model_info, get_backend_class

        info = get_model_info(model)
        if not info:
            raise ValueError(f"Model {model.value} not found in registry")

        try:
            backend_class = get_backend_class(model)
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import backend for {model.value}: {e}. "
                "Make sure required dependencies are installed."
            )

        # Build kwargs based on model type
        kwargs = {"model": model}

        if info.model_type == "api":
            if model.value.startswith("turing"):
                kwargs.update({
                    "fi_api_key": self._fi_api_key,
                    "fi_secret_key": self._fi_secret_key,
                    "fi_base_url": self._fi_base_url,
                })
            elif model.value == "openai-moderation":
                kwargs["api_key"] = os.environ.get("OPENAI_API_KEY")
            elif model.value == "azure-content-safety":
                kwargs["endpoint"] = os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT")
                kwargs["api_key"] = os.environ.get("AZURE_CONTENT_SAFETY_KEY")

        elif info.model_type == "local":
            # Check for VLLM server URL
            env_var = f"VLLM_{model.value.upper().replace('-', '_')}_URL"
            vllm_url = os.environ.get(env_var) or os.environ.get("VLLM_SERVER_URL")
            if vllm_url:
                kwargs["vllm_url"] = vllm_url

            # HuggingFace token for gated models
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if hf_token:
                kwargs["hf_token"] = hf_token

        try:
            return backend_class(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create backend for {model.value}: {e}")

    # =========================================================================
    # Input Rails - Screen user input before LLM
    # =========================================================================

    def screen_input(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GuardrailsResponse:
        """
        Screen user input BEFORE sending to LLM.

        Args:
            content: User input to screen
            metadata: Optional context (user_id, session_id, etc.)

        Returns:
            GuardrailsResponse with pass/fail and details
        """
        return self._screen_sync(content, RailType.INPUT, metadata=metadata)

    async def screen_input_async(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GuardrailsResponse:
        """
        Async version of screen_input.

        Args:
            content: User input to screen
            metadata: Optional context

        Returns:
            GuardrailsResponse with pass/fail and details
        """
        return await self._screen_async(content, RailType.INPUT, metadata=metadata)

    # =========================================================================
    # Output Rails - Screen LLM response before user
    # =========================================================================

    def screen_output(
        self,
        content: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GuardrailsResponse:
        """
        Screen LLM output BEFORE sending to user.

        Args:
            content: LLM response to screen
            context: Optional context (for hallucination check)
            metadata: Optional metadata
        """
        return self._screen_sync(content, RailType.OUTPUT, context=context, metadata=metadata)

    async def screen_output_async(
        self,
        content: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GuardrailsResponse:
        """Async version of screen_output."""
        return await self._screen_async(content, RailType.OUTPUT, context=context, metadata=metadata)

    # =========================================================================
    # Retrieval Rails - Screen RAG chunks
    # =========================================================================

    def screen_retrieval(
        self,
        chunks: List[str],
        query: Optional[str] = None,
    ) -> List[GuardrailsResponse]:
        """
        Screen retrieved chunks in RAG pipeline.

        Args:
            chunks: Retrieved document chunks
            query: Original user query

        Returns:
            List of responses, one per chunk
        """
        results = []
        for chunk in chunks:
            result = self._screen_sync(
                chunk,
                RailType.RETRIEVAL,
                metadata={"query": query} if query else None,
            )
            results.append(result)
        return results

    async def screen_retrieval_async(
        self,
        chunks: List[str],
        query: Optional[str] = None,
    ) -> List[GuardrailsResponse]:
        """Async version of screen_retrieval."""
        tasks = [
            self._screen_async(
                chunk,
                RailType.RETRIEVAL,
                metadata={"query": query} if query else None,
            )
            for chunk in chunks
        ]
        return await asyncio.gather(*tasks)

    # =========================================================================
    # Batch Processing
    # =========================================================================

    async def screen_batch_async(
        self,
        contents: List[str],
        rail_type: RailType = RailType.INPUT,
    ) -> List[GuardrailsResponse]:
        """Process multiple inputs in parallel."""
        tasks = [self._screen_async(c, rail_type) for c in contents]
        return await asyncio.gather(*tasks)

    # =========================================================================
    # Internal Implementation
    # =========================================================================

    @_trace_guardrail
    def _screen_sync(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GuardrailsResponse:
        """
        Synchronous screening using scanners and backends.

        Flow:
        1. Run fast scanners first (<10ms each)
        2. If scanners block, optionally skip slow model backends
        3. Run model backends
        4. Aggregate all results

        Args:
            content: Content to screen
            rail_type: Type of rail
            context: Optional context
            metadata: Optional metadata

        Returns:
            Aggregated GuardrailsResponse
        """
        start_time = time.time()

        # Handle empty content
        if not content or not content.strip():
            return GuardrailsResponse.create_passed(
                content=content,
                latency_ms=(time.time() - start_time) * 1000,
                models_used=[m.value for m in self.backends.keys()],
            )

        all_results: List[GuardrailResult] = []
        errors: List[str] = []
        scanner_blocked = False

        # Step 1: Run fast scanners first
        if self.scanner_pipeline:
            try:
                pipeline_result = self.scanner_pipeline.scan(content, context)

                # Convert scanner results to GuardrailResults
                for scan_result in pipeline_result.results:
                    guardrail_result = GuardrailResult(
                        category=scan_result.category,
                        score=scan_result.score,
                        passed=scan_result.passed,
                        model=f"scanner:{scan_result.scanner_name}",
                        reason=scan_result.reason,
                        latency_ms=scan_result.latency_ms,
                    )
                    all_results.append(guardrail_result)

                scanner_blocked = not pipeline_result.passed

                # If fail_fast is enabled and scanners blocked, skip model backends
                if scanner_blocked and self.config.scanners and self.config.scanners.fail_fast:
                    elapsed_ms = (time.time() - start_time) * 1000
                    return self._aggregate_results(content, all_results, elapsed_ms)

            except Exception as e:
                errors.append(f"scanner_pipeline: {str(e)}")

        # Step 2: Run model backends
        use_weighted_early_exit = (
            self.config.aggregation == AggregationStrategy.WEIGHTED
            and len(self.backends) > 1
        )

        if self.config.parallel and len(self.backends) > 1:
            # Run backends in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_model = {
                    executor.submit(
                        backend.classify,
                        content,
                        rail_type,
                        context,
                        metadata,
                    ): model
                    for model, backend in self.backends.items()
                }

                timeout_seconds = self.config.timeout_ms / 1000.0
                weights = self.config.model_weights
                total_weight = sum(
                    weights.get(m.value, 1.0) for m in self.backends
                )
                failed_weight = 0.0
                passed_weight = 0.0

                try:
                    for future in as_completed(future_to_model, timeout=timeout_seconds):
                        model = future_to_model[future]
                        try:
                            results = future.result()
                            all_results.extend(results)

                            # Early exit for WEIGHTED: if accumulated weight
                            # already decides the outcome, cancel remaining futures
                            if use_weighted_early_exit:
                                w = weights.get(model.value, 1.0)
                                has_fail = any(not r.passed for r in results)
                                if has_fail:
                                    failed_weight += w
                                else:
                                    passed_weight += w
                                threshold = self.config.weighted_threshold
                                if failed_weight > threshold * total_weight:
                                    # Enough weight to block — cancel the rest
                                    for f in future_to_model:
                                        f.cancel()
                                    break
                                if passed_weight >= (1 - threshold) * total_weight:
                                    # Remaining models can't tip the balance — pass early
                                    for f in future_to_model:
                                        f.cancel()
                                    break

                        except Exception as e:
                            errors.append(f"{model.value}: {str(e)}")
                except FuturesTimeoutError:
                    errors.append("Timeout waiting for backends")
        else:
            # Run backends sequentially
            weights = self.config.model_weights
            total_weight = sum(
                weights.get(m.value, 1.0) for m in self.backends
            )
            failed_weight = 0.0
            passed_weight = 0.0

            for model, backend in self.backends.items():
                try:
                    results = backend.classify(content, rail_type, context, metadata)
                    all_results.extend(results)

                    # Early exit for WEIGHTED sequential
                    if use_weighted_early_exit:
                        w = weights.get(model.value, 1.0)
                        has_fail = any(not r.passed for r in results)
                        if has_fail:
                            failed_weight += w
                        else:
                            passed_weight += w
                        threshold = self.config.weighted_threshold
                        if failed_weight > threshold * total_weight:
                            break
                        if passed_weight >= (1 - threshold) * total_weight:
                            break

                except Exception as e:
                    errors.append(f"{model.value}: {str(e)}")

        elapsed_ms = (time.time() - start_time) * 1000

        # Handle errors
        if not all_results and errors:
            return GuardrailsResponse.create_error(
                content=content,
                error="; ".join(errors),
                fail_open=self.config.fail_open,
            )

        # Aggregate results
        return self._aggregate_results(content, all_results, elapsed_ms)

    async def _screen_async(
        self,
        content: str,
        rail_type: RailType,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GuardrailsResponse:
        """
        Asynchronous screening using scanners and backends.

        Flow:
        1. Run fast scanners first (<10ms each)
        2. If scanners block, optionally skip slow model backends
        3. Run model backends in parallel
        4. Aggregate all results

        Args:
            content: Content to screen
            rail_type: Type of rail
            context: Optional context
            metadata: Optional metadata

        Returns:
            Aggregated GuardrailsResponse
        """
        start_time = time.time()

        # Handle empty content
        if not content or not content.strip():
            return GuardrailsResponse.create_passed(
                content=content,
                latency_ms=(time.time() - start_time) * 1000,
                models_used=[m.value for m in self.backends.keys()],
            )

        all_results: List[GuardrailResult] = []
        scanner_blocked = False

        # Step 1: Run fast scanners first (sync, as they're already fast)
        if self.scanner_pipeline:
            try:
                pipeline_result = await self.scanner_pipeline.scan_async(content, context)

                # Convert scanner results to GuardrailResults
                for scan_result in pipeline_result.results:
                    guardrail_result = GuardrailResult(
                        category=scan_result.category,
                        score=scan_result.score,
                        passed=scan_result.passed,
                        model=f"scanner:{scan_result.scanner_name}",
                        reason=scan_result.reason,
                        latency_ms=scan_result.latency_ms,
                    )
                    all_results.append(guardrail_result)

                scanner_blocked = not pipeline_result.passed

                # If fail_fast is enabled and scanners blocked, skip model backends
                if scanner_blocked and self.config.scanners and self.config.scanners.fail_fast:
                    elapsed_ms = (time.time() - start_time) * 1000
                    return self._aggregate_results(content, all_results, elapsed_ms)

            except Exception:
                pass  # Continue with model backends even if scanners fail

        # Step 2: Run all backends in parallel
        tasks = [
            backend.classify_async(content, rail_type, context, metadata)
            for backend in self.backends.values()
        ]

        try:
            timeout_seconds = self.config.timeout_ms / 1000.0
            results_list = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            if self.config.fail_open:
                return GuardrailsResponse.create_passed(
                    content=content,
                    latency_ms=(time.time() - start_time) * 1000,
                    models_used=[m.value for m in self.backends.keys()],
                )
            else:
                return GuardrailsResponse.create_error(
                    content=content,
                    error="Timeout waiting for backends",
                    fail_open=False,
                )

        elapsed_ms = (time.time() - start_time) * 1000

        # Flatten backend results
        for result in results_list:
            if isinstance(result, Exception):
                continue
            all_results.extend(result)

        return self._aggregate_results(content, all_results, elapsed_ms)

    def _aggregate_results(
        self,
        content: str,
        results: List[GuardrailResult],
        elapsed_ms: float,
    ) -> GuardrailsResponse:
        """
        Aggregate results from multiple backends using configured strategy.

        Args:
            content: Original content
            results: Results from all backends
            elapsed_ms: Total elapsed time

        Returns:
            Aggregated GuardrailsResponse
        """
        models_used = [m.value for m in self.backends.keys()]

        if not results:
            return GuardrailsResponse.create_passed(
                content=content,
                latency_ms=elapsed_ms,
                models_used=models_used,
            )

        # Check for error results - if all results are errors, fail closed (unless fail_open)
        error_results = [r for r in results if r.category == "error"]
        non_error_results = [r for r in results if r.category != "error"]

        if not non_error_results:
            # All results are errors
            if self.config.fail_open:
                return GuardrailsResponse.create_passed(
                    content=content,
                    latency_ms=elapsed_ms,
                    models_used=models_used,
                    results=results,
                )
            else:
                error_msg = "; ".join(r.reason or "Unknown error" for r in error_results)
                return GuardrailsResponse(
                    passed=False,
                    results=results,
                    blocked_categories=["error"],
                    original_content=content,
                    total_latency_ms=elapsed_ms,
                    models_used=models_used,
                    error=error_msg,
                )

        # Apply category thresholds
        processed_results = self._apply_thresholds(results)

        # Count total distinct models/scanners that produced results.
        # Models that didn't flag a category are implicit "pass" votes.
        all_model_names = set(r.model for r in processed_results)
        total_voters = len(all_model_names)

        # Group results by category
        category_results: Dict[str, List[GuardrailResult]] = {}
        for result in processed_results:
            if result.category not in category_results:
                category_results[result.category] = []
            category_results[result.category].append(result)

        # Apply aggregation strategy
        blocked_categories: List[str] = []
        flagged_categories: List[str] = []

        for category, cat_results in category_results.items():
            if category in ("safe", "empty", "error"):
                continue

            should_block = self._should_block(cat_results, total_voters)

            if should_block:
                category_config = self.config.categories.get(category)
                if category_config:
                    if category_config.action == "block":
                        blocked_categories.append(category)
                    elif category_config.action == "flag":
                        flagged_categories.append(category)
                else:
                    blocked_categories.append(category)

        # Create response
        passed = len(blocked_categories) == 0
        models_used = [m.value for m in self.backends.keys()]

        return GuardrailsResponse(
            passed=passed,
            results=processed_results,
            blocked_categories=blocked_categories,
            flagged_categories=flagged_categories,
            original_content=content,
            total_latency_ms=elapsed_ms,
            models_used=models_used,
        )

    def _apply_thresholds(
        self,
        results: List[GuardrailResult],
    ) -> List[GuardrailResult]:
        """
        Apply category-specific thresholds to results.

        Args:
            results: Raw results from backends

        Returns:
            Results with thresholds applied
        """
        processed = []
        for result in results:
            category_config = self.config.categories.get(result.category)
            if category_config and category_config.enabled:
                if result.score >= category_config.threshold:
                    result.passed = False
                    result.action = category_config.action
                else:
                    result.passed = True
                    result.action = "pass"
            processed.append(result)
        return processed

    def _should_block(self, cat_results: List[GuardrailResult], total_voters: int = 0) -> bool:
        """
        Determine if content should be blocked based on aggregation strategy.

        Args:
            cat_results: Results for a single category from all models.
            total_voters: Total distinct models/scanners that ran.
                Models that didn't flag this category are implicit passes.

        Returns:
            True if content should be blocked
        """
        if not cat_results:
            return False

        failed_count = sum(1 for r in cat_results if not r.passed)
        # Use total_voters (all models that ran) as denominator, not just
        # models that flagged this specific category. Models that returned
        # "safe" instead are implicit passes for any category they didn't flag.
        total_count = max(total_voters, len(cat_results))
        strategy = self.config.aggregation

        if strategy == AggregationStrategy.ANY:
            return failed_count > 0
        elif strategy == AggregationStrategy.ALL:
            return failed_count == total_count
        elif strategy == AggregationStrategy.MAJORITY:
            return failed_count > total_count / 2
        elif strategy == AggregationStrategy.WEIGHTED:
            weights = self.config.model_weights
            # Total weight includes ALL models that ran (from backends dict),
            # not just those that flagged this specific category.
            total_weight = sum(
                weights.get(m.value, 1.0) for m in self.backends
            )
            if total_weight == 0:
                # Fallback: just use weights from cat_results
                total_weight = sum(weights.get(r.model, 1.0) for r in cat_results)
            if total_weight == 0:
                return False
            failed_weight = 0.0
            for r in cat_results:
                if not r.passed:
                    failed_weight += weights.get(r.model, 1.0)
            return failed_weight > self.config.weighted_threshold * total_weight
        else:
            return failed_count > 0

    def __repr__(self) -> str:
        models = [m.value for m in self.config.models]
        return f"Guardrails(models={models}, aggregation={self.config.aggregation.value})"
