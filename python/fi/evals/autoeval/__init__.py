"""AutoEval - Automatic Evaluation Pipeline Builder.

AutoEval automatically designs evaluation pipelines from natural language
application descriptions. It analyzes the app type, risk level, and domain
sensitivity to recommend appropriate evaluations and scanners.

Example:
    # From natural language description
    from fi.evals.autoeval import AutoEvalPipeline

    pipeline = AutoEvalPipeline.from_description(
        "A RAG-based customer support chatbot for healthcare. "
        "Retrieves patient records and answers questions about appointments."
    )

    result = pipeline.evaluate({
        "query": "When is my appointment?",
        "response": "Your appointment is Monday at 2pm.",
        "context": "Patient has appointment on 2024-01-15 14:00",
    })

    print(result.passed)  # True/False
    print(pipeline.explain())  # Detailed breakdown

    # From pre-built template
    pipeline = AutoEvalPipeline.from_template("rag_system")

    # Export for version control
    pipeline.export_yaml("eval_config.yaml")

Available Templates:
    - customer_support: Customer service chatbots
    - rag_system: RAG-based document Q&A
    - code_assistant: Code generation and review
    - content_moderation: Content filtering and safety
    - agent_workflow: Autonomous agents with tool use
    - healthcare: Healthcare applications (HIPAA)
    - financial: Financial services
"""

# Core types
from .types import (
    AppCategory,
    RiskLevel,
    DomainSensitivity,
    AppRequirement,
    AppAnalysis,
    AutoEvalResult,
)

# Configuration
from .config import (
    EvalConfig,
    ScannerConfig,
    AutoEvalConfig,
)

# Main pipeline
from .pipeline import (
    AutoEvalPipeline,
    register_eval_class,
    register_scanner_class,
)

# Templates
from .templates import (
    get_template,
    list_templates,
    get_template_names,
    TEMPLATES,
)

# Analysis components
from .analyzer import AppAnalyzer
from .recommender import EvalRecommender
from .rules import RuleBasedAnalyzer

# Export/Import
from .export import (
    export_yaml,
    export_json,
    load_yaml,
    load_json,
    load_config,
    to_yaml_string,
    to_json_string,
    from_yaml_string,
    from_json_string,
)

# Interactive mode
from .interactive import (
    InteractiveConfigurator,
    InteractiveSession,
    ClarificationQuestion,
)

__all__ = [
    # Types
    "AppCategory",
    "RiskLevel",
    "DomainSensitivity",
    "AppRequirement",
    "AppAnalysis",
    "AutoEvalResult",
    # Configuration
    "EvalConfig",
    "ScannerConfig",
    "AutoEvalConfig",
    # Pipeline
    "AutoEvalPipeline",
    "register_eval_class",
    "register_scanner_class",
    # Templates
    "get_template",
    "list_templates",
    "get_template_names",
    "TEMPLATES",
    # Analyzers
    "AppAnalyzer",
    "EvalRecommender",
    "RuleBasedAnalyzer",
    # Export/Import
    "export_yaml",
    "export_json",
    "load_yaml",
    "load_json",
    "load_config",
    "to_yaml_string",
    "to_json_string",
    "from_yaml_string",
    "from_json_string",
    # Interactive
    "InteractiveConfigurator",
    "InteractiveSession",
    "ClarificationQuestion",
]
