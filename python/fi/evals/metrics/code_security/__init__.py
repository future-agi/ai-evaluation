"""
Code Security Evaluation for AI Agents.

A comprehensive framework for evaluating the security of AI-generated code.
Unlike traditional SAST tools, this is purpose-built for AI code evaluation with:

- **Multiple Evaluation Modes**: Instruct, Autocomplete, Repair, Adversarial
- **Joint Metrics**: func@k, sec@k, func-sec@k (code that is BOTH correct AND secure)
- **CWE-Based Detection**: 25+ vulnerability detectors covering OWASP Top 10
- **Dual-Judge System**: Pattern-based + LLM semantic analysis
- **Benchmark Support**: Curated test suites for consistent evaluation

Research Context:
- 40% of GitHub Copilot completions contain vulnerabilities
- 30% of LLM code suggestions are insecure (CyberSecEval)
- Only 9-15% of AI code passes func-sec@1 (both correct AND secure)

Quick Start:
    from fi.evals.metrics.code_security import (
        CodeSecurityScore,
        QuickSecurityCheck,
        CodeSecurityInput,
    )

    # Simple security check
    metric = CodeSecurityScore()
    result = metric.compute_one(CodeSecurityInput(
        response='''
def get_user(name):
    query = f"SELECT * FROM users WHERE name = '{name}'"
    cursor.execute(query)
        ''',
        language="python",
    ))

    print(f"Security Score: {result['output']}")
    print(f"Findings: {result['findings']}")

Evaluation Modes:
    # Instruct mode (default): Evaluate code from instructions
    input = CodeSecurityInput(
        response=generated_code,
        instruction="Write a function to query users",
        mode=EvaluationMode.INSTRUCT,
    )

    # Autocomplete mode: Evaluate code completion
    input = CodeSecurityInput(
        response=completion,
        code_prefix="def get_user(id):\\n    query = ",
        mode=EvaluationMode.AUTOCOMPLETE,
    )

    # Repair mode: Evaluate vulnerability fixes
    input = CodeSecurityInput(
        response=fixed_code,
        vulnerable_code=original_code,
        mode=EvaluationMode.REPAIR,
    )

Available Metrics:
    - CodeSecurityScore: Comprehensive evaluation (recommended)
    - QuickSecurityCheck: Fast pattern-only check (<10ms)
    - InjectionSecurityScore: Focus on injection vulnerabilities
    - CryptographySecurityScore: Focus on crypto issues
    - SecretsSecurityScore: Focus on hardcoded credentials

CWE Coverage:
    Tier 1 (Critical): CWE-89, 78, 79, 798, 22, 502, 327, 330
    Tier 2 (High): CWE-20, 287, 306, 352, 434, 611, 918
    Tier 3 (Medium): CWE-120, 476, 190, 209, 400, 404

References:
    - CyberSecEval: https://github.com/meta-llama/PurpleLlama
    - OWASP Top 10: https://owasp.org/Top10/
    - CWE Top 25: https://cwe.mitre.org/top25/
"""

from .types import (
    # Enums
    Severity,
    EvaluationMode,
    VulnerabilityCategory,
    # Types
    CodeLocation,
    SecurityFinding,
    FunctionalTestCase,
    TestCase,  # Alias for FunctionalTestCase
    CodeSecurityInput,
    CodeSecurityOutput,
    # Mappings
    CWE_CATEGORIES,
    CWE_METADATA,
    SEVERITY_WEIGHTS,
    # Helper functions
    get_cwe_metadata,
    get_cwe_severity,
    get_cwe_category,
    # Aliases
    Finding,
    Location,
    Input,
    Output,
)

from .analyzer import (
    CodeAnalyzer,
    AnalysisResult,
    FunctionInfo,
    ImportInfo,
    StringLiteral,
    # Language-specific analyzers
    PythonAnalyzer,
    JavaScriptAnalyzer,
    JavaAnalyzer,
    GoAnalyzer,
)

from .detectors import (
    BaseDetector,
    PatternBasedDetector,
    CompositeDetector,
    register_detector,
    get_detector,
    list_detectors,
    get_all_detectors,
    get_detectors_by_category,
    get_detectors_by_cwe,
)

# Import metrics
from .metrics import (
    CodeSecurityScore,
    QuickSecurityCheck,
    InjectionSecurityScore,
    CryptographySecurityScore,
    SecretsSecurityScore,
    SerializationSecurityScore,
)

# Import joint metrics
from .joint_metrics import (
    JointSecurityMetrics,
    JointMetricsResult,
    FunctionalTestResult,
    compute_func_at_k,
    compute_sec_at_k,
    compute_func_sec_at_k,
)

# Import evaluation modes
from .modes import (
    InstructModeEvaluator,
    AutocompleteModeEvaluator,
    RepairModeEvaluator,
    AdversarialModeEvaluator,
    InstructModeResult,
    AutocompleteModeResult,
    RepairModeResult,
    AdversarialModeResult,
)

# Import judges (Dual-Judge System)
from .judges import (
    BaseJudge,
    JudgeResult,
    JudgeFinding,
    ConsensusMode,
    PatternJudge,
    PatternRule,
    LLMJudge,
    MockLLMJudge,
    DualJudge,
)

# Import benchmarks
from .benchmarks import (
    SecurityBenchmark,
    InstructTest,
    AutocompleteTest,
    RepairTest,
    BenchmarkResult,
    CWEBreakdown,
    load_benchmark,
    list_available_benchmarks,
    PYTHON_INSTRUCT_TESTS,
    PYTHON_AUTOCOMPLETE_TESTS,
    PYTHON_REPAIR_TESTS,
)

# Import reports/leaderboard
from .reports import (
    SecurityLeaderboard,
    ModelEntry,
    LeaderboardReport,
    CWEComparison,
    LanguageComparison,
    ReportGenerator,
    generate_security_report,
)


__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    # Enums
    "Severity",
    "EvaluationMode",
    "VulnerabilityCategory",
    # Types
    "CodeLocation",
    "SecurityFinding",
    "FunctionalTestCase",
    "TestCase",  # Alias
    "CodeSecurityInput",
    "CodeSecurityOutput",
    # Mappings
    "CWE_CATEGORIES",
    "CWE_METADATA",
    "SEVERITY_WEIGHTS",
    # Helper functions
    "get_cwe_metadata",
    "get_cwe_severity",
    "get_cwe_category",
    # Aliases
    "Finding",
    "Location",
    "Input",
    "Output",
    # Analyzer
    "CodeAnalyzer",
    "AnalysisResult",
    "FunctionInfo",
    "ImportInfo",
    "StringLiteral",
    "PythonAnalyzer",
    "JavaScriptAnalyzer",
    "JavaAnalyzer",
    "GoAnalyzer",
    # Detectors
    "BaseDetector",
    "PatternBasedDetector",
    "CompositeDetector",
    "register_detector",
    "get_detector",
    "list_detectors",
    "get_all_detectors",
    "get_detectors_by_category",
    "get_detectors_by_cwe",
    # Metrics
    "CodeSecurityScore",
    "QuickSecurityCheck",
    "InjectionSecurityScore",
    "CryptographySecurityScore",
    "SecretsSecurityScore",
    "SerializationSecurityScore",
    # Joint metrics
    "JointSecurityMetrics",
    "JointMetricsResult",
    "FunctionalTestResult",
    "compute_func_at_k",
    "compute_sec_at_k",
    "compute_func_sec_at_k",
    # Evaluation modes
    "InstructModeEvaluator",
    "AutocompleteModeEvaluator",
    "RepairModeEvaluator",
    "AdversarialModeEvaluator",
    "InstructModeResult",
    "AutocompleteModeResult",
    "RepairModeResult",
    "AdversarialModeResult",
    # Judges (Dual-Judge System)
    "BaseJudge",
    "JudgeResult",
    "JudgeFinding",
    "ConsensusMode",
    "PatternJudge",
    "PatternRule",
    "LLMJudge",
    "MockLLMJudge",
    "DualJudge",
    # Benchmarks
    "SecurityBenchmark",
    "InstructTest",
    "AutocompleteTest",
    "RepairTest",
    "BenchmarkResult",
    "CWEBreakdown",
    "load_benchmark",
    "list_available_benchmarks",
    "PYTHON_INSTRUCT_TESTS",
    "PYTHON_AUTOCOMPLETE_TESTS",
    "PYTHON_REPAIR_TESTS",
    # Reports/Leaderboard
    "SecurityLeaderboard",
    "ModelEntry",
    "LeaderboardReport",
    "CWEComparison",
    "LanguageComparison",
    "ReportGenerator",
    "generate_security_report",
]
