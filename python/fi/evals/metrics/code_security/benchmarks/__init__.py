"""
Benchmark Test Suites for AI Code Security Evaluation.

Provides curated test cases for consistent, reproducible evaluation
of AI-generated code security across multiple languages and modes.

Features:
- 100+ Instruct mode prompts per language
- 100+ Autocomplete scenarios per language
- 50+ Repair scenarios per language
- Ground truth labels
- Expected CWE mappings

Usage:
    from fi.evals.metrics.code_security.benchmarks import (
        SecurityBenchmark,
        InstructTest,
        AutocompleteTest,
        RepairTest,
    )

    # Load benchmark
    benchmark = SecurityBenchmark()

    # Run evaluation
    result = benchmark.evaluate_model(
        model_fn=my_model,
        language="python",
        mode=EvaluationMode.INSTRUCT,
    )

    print(f"func@k: {result.func_at_k}")
    print(f"sec@k: {result.sec_at_k}")
    print(f"func-sec@k: {result.func_sec_at_k}")
"""

from .types import (
    InstructTest,
    AutocompleteTest,
    RepairTest,
    BenchmarkResult,
    CWEBreakdown,
)

from .loader import (
    SecurityBenchmark,
    load_benchmark,
    list_available_benchmarks,
)

from .builtin import (
    PYTHON_INSTRUCT_TESTS,
    PYTHON_AUTOCOMPLETE_TESTS,
    PYTHON_REPAIR_TESTS,
)


__all__ = [
    # Types
    "InstructTest",
    "AutocompleteTest",
    "RepairTest",
    "BenchmarkResult",
    "CWEBreakdown",
    # Loader
    "SecurityBenchmark",
    "load_benchmark",
    "list_available_benchmarks",
    # Built-in tests
    "PYTHON_INSTRUCT_TESTS",
    "PYTHON_AUTOCOMPLETE_TESTS",
    "PYTHON_REPAIR_TESTS",
]
