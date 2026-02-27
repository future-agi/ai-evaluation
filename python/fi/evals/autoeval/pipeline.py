"""AutoEval Pipeline - Main API for automatic evaluation pipelines.

Provides the core AutoEvalPipeline class for creating and running
evaluation pipelines from natural language descriptions or templates.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Type, Union

from .types import AutoEvalResult, AppAnalysis
from .config import AutoEvalConfig, EvalConfig, ScannerConfig
from .analyzer import AppAnalyzer
from .recommender import EvalRecommender
from .templates import get_template, list_templates

logger = logging.getLogger(__name__)


# Registry for eval/scanner class lookups
_EVAL_CLASS_REGISTRY: Dict[str, Type] = {}
_SCANNER_CLASS_REGISTRY: Dict[str, Type] = {}


def register_eval_class(name: str, cls: Type) -> None:
    """Register an evaluation class for AutoEval lookup."""
    _EVAL_CLASS_REGISTRY[name] = cls


def register_scanner_class(name: str, cls: Type) -> None:
    """Register a scanner class for AutoEval lookup."""
    _SCANNER_CLASS_REGISTRY[name] = cls


def _get_eval_class(name: str) -> Optional[Type]:
    """Get evaluation class by name."""
    # Try direct registry lookup first
    if name in _EVAL_CLASS_REGISTRY:
        return _EVAL_CLASS_REGISTRY[name]

    # Try to import from framework
    try:
        from fi.evals.framework import EvalRegistry

        if EvalRegistry.is_registered(name):
            return EvalRegistry.get(name)
    except (ImportError, ValueError):
        pass

    # Try to import from evals module
    try:
        from fi.evals import templates as eval_templates

        if hasattr(eval_templates, name):
            return getattr(eval_templates, name)
    except ImportError:
        pass

    return None


def _get_scanner_class(name: str) -> Optional[Type]:
    """Get scanner class by name."""
    # Try direct registry lookup first
    if name in _SCANNER_CLASS_REGISTRY:
        return _SCANNER_CLASS_REGISTRY[name]

    # Try to import from guardrails.scanners
    try:
        from fi.evals.guardrails import scanners as scanner_module

        if hasattr(scanner_module, name):
            return getattr(scanner_module, name)
    except ImportError:
        pass

    return None


class AutoEvalPipeline:
    """
    Automatic evaluation pipeline builder.

    Creates evaluation pipelines from natural language descriptions,
    pre-built templates, or manual configuration.

    Example:
        # From natural language description
        pipeline = AutoEvalPipeline.from_description(
            "A RAG-based customer support chatbot for healthcare. "
            "Retrieves patient records and answers questions about appointments."
        )

        # From pre-built template
        pipeline = AutoEvalPipeline.from_template("rag_system")

        # Run evaluation
        result = pipeline.evaluate({
            "query": "When is my appointment?",
            "response": "Your appointment is Monday at 2pm.",
            "context": "Patient has appointment on 2024-01-15 14:00",
        })

        print(result.passed)  # True/False
        print(result.explain())  # Detailed breakdown

        # Export configuration
        pipeline.export_yaml("eval_config.yaml")
    """

    def __init__(
        self,
        config: AutoEvalConfig,
        analysis: Optional[AppAnalysis] = None,
    ):
        """
        Initialize the pipeline with configuration.

        Args:
            config: AutoEvalConfig with evaluations and scanners
            analysis: Optional AppAnalysis for explanation context
        """
        self.config = config
        self.analysis = analysis
        self._evaluator = None
        self._scanner_pipeline = None
        self._eval_instances: List[Any] = []
        self._scanner_instances: List[Any] = []

    @classmethod
    def from_description(
        cls,
        description: str,
        llm_provider: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> "AutoEvalPipeline":
        """
        Create pipeline from natural language description.

        Uses LLM-powered analysis when available, falls back to
        rule-based analysis otherwise.

        Args:
            description: Natural language application description
            llm_provider: Optional LLM provider for intelligent analysis
            name: Optional name for the pipeline

        Returns:
            Configured AutoEvalPipeline

        Example:
            pipeline = AutoEvalPipeline.from_description(
                "A customer support chatbot for a healthcare company. "
                "It retrieves patient information and answers questions."
            )
        """
        # Analyze the description
        analyzer = AppAnalyzer(llm_provider=llm_provider)
        analysis = analyzer.analyze(description)

        # Generate recommendations
        recommender = EvalRecommender()
        evals, scanners = recommender.recommend(analysis)

        # Build config
        config = AutoEvalConfig(
            name=name or f"autoeval_{analysis.category.value}",
            description=description[:200] if len(description) > 200 else description,
            app_category=analysis.category.value,
            risk_level=analysis.risk_level.value,
            domain_sensitivity=analysis.domain_sensitivity.value,
            evaluations=evals,
            scanners=scanners,
        )

        return cls(config, analysis)

    @classmethod
    def from_template(cls, template_name: str) -> "AutoEvalPipeline":
        """
        Create pipeline from pre-built template.

        Available templates:
        - customer_support: Customer service chatbots
        - rag_system: RAG-based document Q&A
        - code_assistant: Code generation and review
        - content_moderation: Content filtering and safety
        - agent_workflow: Autonomous agents with tool use
        - healthcare: Healthcare applications (HIPAA)
        - financial: Financial services

        Args:
            template_name: Name of the template to use

        Returns:
            Configured AutoEvalPipeline

        Raises:
            ValueError: If template not found

        Example:
            pipeline = AutoEvalPipeline.from_template("rag_system")
        """
        config = get_template(template_name)
        if config is None:
            available = list(list_templates().keys())
            raise ValueError(
                f"Template '{template_name}' not found. "
                f"Available templates: {available}"
            )
        return cls(config)

    @classmethod
    def from_config(cls, config: AutoEvalConfig) -> "AutoEvalPipeline":
        """
        Create pipeline from existing configuration.

        Args:
            config: AutoEvalConfig instance

        Returns:
            Configured AutoEvalPipeline
        """
        return cls(config)

    @classmethod
    def from_yaml(cls, path: str) -> "AutoEvalPipeline":
        """
        Load pipeline from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Configured AutoEvalPipeline
        """
        from .export import load_config

        config = load_config(path)
        return cls(config)

    def _is_core_metric(self, name: str) -> bool:
        """Check if a name corresponds to a core metric in the local registry."""
        try:
            from fi.evals.local.registry import get_registry
            return get_registry().is_registered(name)
        except ImportError:
            return False

    def _get_metric_configs(self) -> List[EvalConfig]:
        """Get eval configs that are core metrics (routed through evaluate())."""
        return [
            e for e in self.config.evaluations
            if e.enabled and self._is_core_metric(e.name)
        ]

    def _get_class_configs(self) -> List[EvalConfig]:
        """Get eval configs that are framework classes (routed through Evaluator)."""
        return [
            e for e in self.config.evaluations
            if e.enabled and not self._is_core_metric(e.name)
        ]

    def _run_core_metrics(self, inputs: Dict[str, Any]) -> List[Any]:
        """Run core metrics via evaluate() API and return EvalResults."""
        from fi.evals import evaluate as core_evaluate

        metric_configs = self._get_metric_configs()
        if not metric_configs:
            return []

        results = []
        for eval_config in metric_configs:
            try:
                result = core_evaluate(
                    eval_config.name,
                    model=eval_config.model,
                    augment=eval_config.augment,
                    **inputs,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Core metric '{eval_config.name}' failed: {e}")
        return results

    def _build_evaluator(self) -> None:
        """Build the evaluator with framework-class evaluations only."""
        if self._evaluator is not None:
            return

        from fi.evals.framework import Evaluator, ExecutionMode

        # Only build for non-metric (framework class) evals
        self._eval_instances = []
        for eval_config in self._get_class_configs():
            eval_class = _get_eval_class(eval_config.name)
            if eval_class is None:
                logger.warning(f"Evaluation class not found: {eval_config.name}")
                continue

            try:
                instance = eval_class(**eval_config.params)
                self._eval_instances.append(instance)
            except Exception as e:
                logger.warning(f"Failed to instantiate {eval_config.name}: {e}")

        if self._eval_instances:
            mode = (
                ExecutionMode.NON_BLOCKING
                if self.config.execution_mode == "non_blocking"
                else ExecutionMode.BLOCKING
            )
            self._evaluator = Evaluator(
                evaluations=self._eval_instances,
                mode=mode,
                max_workers=self.config.parallel_workers,
                fail_fast=self.config.fail_fast,
            )

    def _build_scanner_pipeline(self) -> None:
        """Build the scanner pipeline with configured scanners."""
        if self._scanner_pipeline is not None:
            return

        from fi.evals.guardrails.scanners import ScannerPipeline

        # Instantiate scanner classes
        self._scanner_instances = []
        for scanner_config in self.config.scanners:
            if not scanner_config.enabled:
                continue

            scanner_class = _get_scanner_class(scanner_config.name)
            if scanner_class is None:
                logger.warning(f"Scanner class not found: {scanner_config.name}")
                continue

            try:
                instance = scanner_class(**scanner_config.params)
                # Set threshold and action if the scanner supports them
                if hasattr(instance, "threshold"):
                    instance.threshold = scanner_config.threshold
                if hasattr(instance, "action"):
                    from fi.evals.guardrails.scanners.base import ScannerAction

                    instance.action = ScannerAction(scanner_config.action)
                self._scanner_instances.append(instance)
            except Exception as e:
                logger.warning(f"Failed to instantiate {scanner_config.name}: {e}")

        if self._scanner_instances:
            self._scanner_pipeline = ScannerPipeline(
                scanners=self._scanner_instances,
                parallel=True,
                fail_fast=self.config.fail_fast,
            )

    def evaluate(
        self,
        inputs: Dict[str, Any],
        scan_content: Optional[str] = None,
    ) -> AutoEvalResult:
        """
        Run the full evaluation pipeline.

        Executes scanners first (fast), then evaluations.
        If scanners block, evaluations may be skipped.

        Args:
            inputs: Input data for evaluations (e.g., query, response, context)
            scan_content: Content to scan (defaults to response from inputs)

        Returns:
            AutoEvalResult with combined results

        Example:
            result = pipeline.evaluate({
                "query": "What is the patient's blood type?",
                "response": "The patient's blood type is O+.",
                "context": "Medical record: Blood type O+",
            })

            if result.passed:
                print("Evaluation passed!")
            else:
                print(f"Failed: {result.explain()}")
        """
        start_time = time.perf_counter()

        # Build components lazily
        self._build_evaluator()
        self._build_scanner_pipeline()

        scan_result = None
        eval_result = None
        metric_results = []
        blocked_by_scanner = False

        # Run scanners first (fast)
        if self._scanner_pipeline:
            content = scan_content or inputs.get("response", "")
            context = inputs.get("context")
            scan_result = self._scanner_pipeline.scan(content, context)
            blocked_by_scanner = not scan_result.passed

        # Run evaluations if not blocked
        if not blocked_by_scanner:
            # Core metrics via evaluate() API
            metric_results = self._run_core_metrics(inputs)

            # Framework class evals via Evaluator
            if self._evaluator:
                eval_result = self._evaluator.run(inputs)

        # Determine overall pass/fail
        passed = True
        if scan_result and not scan_result.passed:
            passed = False

        # Check core metric results against thresholds
        if metric_results:
            metric_configs = {e.name: e for e in self._get_metric_configs()}
            failed_count = 0
            total_count = len(metric_results)
            for r in metric_results:
                config = metric_configs.get(getattr(r, "eval_name", ""))
                threshold = config.threshold if config else 0.5
                score = getattr(r, "score", None)
                if score is not None and score < threshold:
                    failed_count += 1
            if total_count > 0:
                success_rate = (total_count - failed_count) / total_count
                if success_rate < self.config.global_pass_rate:
                    passed = False

        if eval_result:
            # Check if framework evaluations meet threshold
            batch = eval_result.wait() if eval_result.is_future else eval_result.batch
            if batch and batch.success_rate < self.config.global_pass_rate:
                passed = False

        total_latency = (time.perf_counter() - start_time) * 1000

        return AutoEvalResult(
            passed=passed,
            scan_result=scan_result,
            eval_result=eval_result,
            metric_results=metric_results,
            blocked_by_scanner=blocked_by_scanner,
            total_latency_ms=total_latency,
        )

    def add(self, item: Union[EvalConfig, ScannerConfig]) -> "AutoEvalPipeline":
        """
        Add an evaluation or scanner to the pipeline.

        Args:
            item: EvalConfig or ScannerConfig to add

        Returns:
            Self for chaining

        Example:
            pipeline.add(EvalConfig("CustomEval", threshold=0.8))
            pipeline.add(ScannerConfig("CustomScanner", action="flag"))
        """
        if isinstance(item, EvalConfig):
            self.config.evaluations.append(item)
            self._evaluator = None  # Reset to rebuild
        elif isinstance(item, ScannerConfig):
            self.config.scanners.append(item)
            self._scanner_pipeline = None  # Reset to rebuild
        return self

    def remove(self, name: str) -> "AutoEvalPipeline":
        """
        Remove an evaluation or scanner by name.

        Args:
            name: Name of the evaluation or scanner to remove

        Returns:
            Self for chaining

        Example:
            pipeline.remove("CoherenceEval")
        """
        # Try to remove from evaluations
        original_count = len(self.config.evaluations)
        self.config.evaluations = [
            e for e in self.config.evaluations if e.name != name
        ]
        if len(self.config.evaluations) < original_count:
            self._evaluator = None

        # Try to remove from scanners
        original_count = len(self.config.scanners)
        self.config.scanners = [s for s in self.config.scanners if s.name != name]
        if len(self.config.scanners) < original_count:
            self._scanner_pipeline = None

        return self

    def set_threshold(self, name: str, threshold: float) -> "AutoEvalPipeline":
        """
        Set threshold for an evaluation or scanner.

        Args:
            name: Name of the evaluation or scanner
            threshold: New threshold value (0.0-1.0)

        Returns:
            Self for chaining

        Example:
            pipeline.set_threshold("CoherenceEval", 0.9)
        """
        for e in self.config.evaluations:
            if e.name == name:
                e.threshold = threshold
                self._evaluator = None

        for s in self.config.scanners:
            if s.name == name:
                s.threshold = threshold
                self._scanner_pipeline = None

        return self

    def enable(self, name: str) -> "AutoEvalPipeline":
        """Enable an evaluation or scanner."""
        for e in self.config.evaluations:
            if e.name == name:
                e.enabled = True
                self._evaluator = None

        for s in self.config.scanners:
            if s.name == name:
                s.enabled = True
                self._scanner_pipeline = None

        return self

    def disable(self, name: str) -> "AutoEvalPipeline":
        """Disable an evaluation or scanner."""
        for e in self.config.evaluations:
            if e.name == name:
                e.enabled = False
                self._evaluator = None

        for s in self.config.scanners:
            if s.name == name:
                s.enabled = False
                self._scanner_pipeline = None

        return self

    def export_yaml(self, path: str) -> None:
        """
        Export pipeline configuration to YAML file.

        Args:
            path: Path to write YAML file

        Example:
            pipeline.export_yaml("eval_config.yaml")
        """
        from .export import export_yaml

        export_yaml(self.config, path)

    def export_json(self, path: str) -> None:
        """
        Export pipeline configuration to JSON file.

        Args:
            path: Path to write JSON file
        """
        from .export import export_json

        export_json(self.config, path)

    def explain(self) -> str:
        """
        Get human-readable explanation of the pipeline.

        Returns:
            Detailed explanation string

        Example:
            print(pipeline.explain())
        """
        lines = [self.config.summary()]

        if self.analysis:
            lines.append("")
            lines.append("Analysis Details:")
            lines.append(f"  Confidence: {self.analysis.confidence:.0%}")
            lines.append(f"  Explanation: {self.analysis.explanation}")
            if self.analysis.detected_features:
                lines.append(f"  Detected Features: {', '.join(self.analysis.detected_features)}")

        return "\n".join(lines)

    def summary(self) -> str:
        """Get brief summary of the pipeline."""
        return (
            f"AutoEvalPipeline: {self.config.name}\n"
            f"  Evaluations: {len(self.config.evaluations)}\n"
            f"  Scanners: {len(self.config.scanners)}\n"
            f"  Risk Level: {self.config.risk_level}"
        )

    def __repr__(self) -> str:
        return (
            f"AutoEvalPipeline(name={self.config.name!r}, "
            f"evaluations={len(self.config.evaluations)}, "
            f"scanners={len(self.config.scanners)})"
        )
